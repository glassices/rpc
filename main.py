import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import torch.distributed.rpc as rpc

import argparse
from collections import namedtuple
import math
import logging
import os
import socket
import sys
import queue
from contextlib import contextmanager
from collections import OrderedDict
from functools import partial

import utils
from utils import adjust_lr, ensure_dir, get_free_port
from typing import Optional, List

from torch import Tensor
import gc

import dag
import config

import rdkit
from rdkit import Chem
import numpy as np
import time
import threading
import network
import shutil
import random
import config
import pickle
import copy


class MultipleDict:
    def __init__(self):
        self.d = []

    def get(self, key):
        for k, v in self.d:
            if k == key:
                return v
        return None

    def override(self, key, value):
        for i in range(len(self.d)):
            if self.d[i][0] == key:
                self.d[i] = (key, value)
                break

    def __getitem__(self, key):
        for k, v in self.d:
            if k == key:
                return v
        raise KeyError

    def __setitem__(self, key, value):
        self.d.append((key, value))

    def __contains__(self, key):
        return any(k == key for k, _ in self.d)

    def to_string(self):
        parts = []
        for k, v in self.d:
            if k == 'peaks':
                parts.append('\n'.join(f"{m} {i}" for m, i in v))
            elif k == 'Num peaks':
                parts.append(f"{k}: {len(self['peaks'])}")
            else:
                parts.append(f"{k}: {v}")
        return '\n'.join(parts)

def parse_nisp_msp(item):
    lines = item.split('\n')
    dt = dict()
    for i, line in enumerate(lines):
        if line.startswith('Num peaks: '):
            num_peaks = int(line.split()[-1])
            assert i + num_peaks + 1 == len(lines)
            data = []
            for j in range(i + 1, len(lines)):
                parts = lines[j].split(maxsplit = 2)
                data.append((float(parts[0]), float(parts[1])))
            dt['Num peaks'] = num_peaks
            dt['peaks'] = np.asarray(data)
            break
        else:
            parts = line.split(maxsplit = 1)
            dt[parts[0][:-1]] = parts[1]
    return dt

@contextmanager
def sync_ddp(ddp, to_sync):
    old_require_backward_grad_sync = ddp.require_backward_grad_sync
    ddp.require_backward_grad_sync = to_sync
    try:
        yield
    finally:
        ddp.require_backward_grad_sync = old_require_backward_grad_sync

def parse_collision_energy(s):
    if s.startswith('NCE=') and s.endswith('eV'):
        a, b = s.split()
        nce = int(a[4:-1])
        ev = int(b[:-2])
        assert f"NCE={nce}% {ev}eV" == s
        return nce, ev
    elif s.startswith('NCE=') and s.endswith('%'):
        nce = int(s[4:-1])
        assert f"NCE={nce}%" == s
        return nce, None
    elif s.isdigit():
        ev = int(s)
        return None, ev
    else:
        assert False

def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(x) for x in obj)
    elif isinstance(obj, list):
        return list(move_to_cpu(x) for x in obj)
    elif isinstance(obj, dict):
        return dict((k, move_to_cpu(v)) for k, v in obj.items())
    else:
        return obj

def move_to_cuda(obj, device, non_blocking = False):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking = non_blocking)
    elif isinstance(obj, tuple):
        return tuple(move_to_cuda(x, device, non_blocking) for x in obj)
    elif isinstance(obj, list):
        return list(move_to_cuda(x, device, non_blocking) for x in obj)
    elif isinstance(obj, dict):
        return dict((k, move_to_cuda(v, device, non_blocking)) for k, v in obj.items())
    else:
        return obj

def move_to_pin_memory(obj):
    if isinstance(obj, torch.Tensor):
        return obj.pin_memory()
    elif isinstance(obj, tuple):
        return tuple(move_to_pin_memory(x) for x in obj)
    elif isinstance(obj, list):
        return list(move_to_pin_memory(x) for x in obj)
    elif isinstance(obj, dict):
        return dict((k, move_to_pin_memory(v)) for k, v in obj.items())
    else:
        return obj

def _gloo_free_tensorpipe_init_backend_handler(store, name, rank, world_size, rpc_backend_options):
    from torch.distributed.rpc import TensorPipeAgent
    from torch.distributed.rpc import TensorPipeRpcBackendOptions
    if not isinstance(store, dist.Store):
        raise TypeError("`store` must be a c10d::Store. {}".format(store))

    if not isinstance(
        rpc_backend_options, TensorPipeRpcBackendOptions
    ):
        raise TypeError(
            "`rpc_backend_options` must be a `TensorPipeRpcBackendOptions`. {}".format(
                rpc_backend_options
            )
        )

    if torch.cuda.is_available():
        torch.cuda.init()
        device_count = torch.cuda.device_count()
    else:
        device_count = 0

    agent = TensorPipeAgent(
        store,
        name,
        rank,
        world_size,
        rpc_backend_options,
        {},
        [],
    )
    rpc.backend_registry.api._init_rpc_states(agent)
    return agent


"""
Instrument: {'Agilent QTOF 6530': 4255,
             'Thermo Finnigan Velos Orbitrap': 7520,
             'Thermo Finnigan Elite Orbitrap': 272483,
             'Orbitrap Fusion Lumos': 82700},
Instrument_type: {'Q-TOF': 4255,
                  'HCD': 334843,
                  'IT-FT/ion trap with FTMS': 27860},
Collision_gas: {'N2': 364696,
                'He': 2262},
Sample_inlet: {'direct flow injection': 366301,
               'LC': 4,
               'HPLC': 653},
In-source_voltage: {'150': 1036,
                    '175': 2679,
                    '200': 244,
                    '250': 36,
                    '100': 119,
                    '130': 112,
                    None: 362703,
                    '120': 29}}
"""
mm_precursor_type = {'[M+H]+': 0,
                     '[M-H]-': 1}
mm_instrument = {'Agilent QTOF 6530': 0,
                 'Thermo Finnigan Velos Orbitrap': 1,
                 'Thermo Finnigan Elite Orbitrap': 2,
                 'Orbitrap Fusion Lumos': 3}
mm_instrument_type = {'Q-TOF': 0,
                      'HCD': 1,
                      'IT-FT/ion trap with FTMS': 2}
mm_collision_gas = {'N2': 0,
                    'He': 1}
mm_sample_inlet = {'direct flow injection': 0,
                   'LC': 1,
                   'HPLC': 2}

def decode(d, is_train):
    r"""meta_feature of 23 balabala
        meta_feature[no_nce | has_nce | nce | no_ev | has_ev | ev | is_m+h+ | is_m-h-    [0:8)
                     | instrument_1 | instrument_2 | instrument_2 | instrument_3    [8:12)
                     | type_1 | type_2 | type_3    [12:15)
                     | N2 | He    [15:17)
                     | inlet1 | inlet2 | inlet 3    [17:20)
                     | no_in_voltage | has_in_voltage | voltage]    [20:23)
    """
    r"""
        meta_long[has_nce | has_ev | precursor_type | instrument | instrument_type | collision_gas | sample_inlet | has_insource_voltage]
        meta_real[nce | ev | in_source_voltage]
    """
    mol = Chem.MolFromSmiles(d['Smiles'])
    meta_long = torch.zeros(config._meta_long_dim).long()
    meta_real = torch.zeros(config._meta_real_dim).float()

    nce, ev = parse_collision_energy(d['Collision_energy'])
    if nce is None:
        meta_long[0] = 0
    else:
        meta_long[0] = 1
        if is_train:
            meta_real[0] = nce / 100 * (random.random() * 0.2 + 0.9)
        else:
            meta_real[0] = nce / 100
    if ev is None:
        meta_long[1] = 0
    else:
        meta_long[1] = 1
        if is_train:
            meta_real[1] = ev / 10 * (random.random() * 0.2 + 0.9)
        else:
            meta_real[1] = ev / 10

    if is_train and nce is not None and ev is not None and random.randint(0, 1) == 0:
        if random.randint(0, 1) == 0:
            meta_long[0] = 0
            meta_real[0] = 0.0
        else:
            meta_long[1] = 0
            meta_real[1] = 0.0

    meta_long[2] = mm_precursor_type[d.get('Precursor_type')]
    meta_long[3] = mm_instrument[d.get('Instrument')]
    meta_long[4] = mm_instrument_type[d.get('Instrument_type')]
    meta_long[5] = mm_collision_gas[d.get('Collision_gas')]
    meta_long[6] = mm_sample_inlet[d.get('Sample_inlet')]
    vol = d.get('In-source_voltage')
    if vol is None:
        meta_long[7] = 0
    else:
        meta_long[7] = 1
        if is_train:
            meta_real[2] = int(vol) / 100 * (random.random() * 0.2 + 0.9)
        else:
            meta_real[2] = int(vol) / 100

    return mol, meta_long, meta_real


class CPUWorker:

    def __init__(self, args, main_worker_rref, gpu_worker_rref, cpu_rank, tot_cpu_workers):
        self.args = args
        self.main_worker_rref = main_worker_rref
        self.gpu_worker_rref = gpu_worker_rref
        self.cpu_rank = cpu_rank
        self.tot_cpu_workers = tot_cpu_workers

    def initialize(self, self_rref, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self_rref.rpc_async()._run()
        return None

    def _run(self):
        torch.set_num_threads(1)
        idx = 0
        train_keys = ['kl', 'cosine', 'l1', 'l2', 'sim_error', 'num_dag_nodes', 'num_sample_nodes', 'num_tot_nodes', 'hit', 'solved', 'residual_prob', 'p_waste', 'tree_entropy', 'avg_dist', 'avg_one_bond', 'avg_ring_bond', 'avg_excision', 'max_q_p', 'eval_sizes']
        test_keys = ['kl', 'cosine', 'l1', 'l2', 'sim_error', 'num_dag_nodes', 'num_sample_nodes', 'num_tot_nodes', 'hit', 'solved', 'residual_prob', 'p_waste', 'tree_entropy', 'avg_dist', 'avg_one_bond', 'avg_ring_bond', 'avg_excision']

        func = partial(dag.generate_training_data,
            gpu_worker_rref = self.gpu_worker_rref,
            ppm = self.args.dag_ppm,
            num_sim = self.args.dag_num_sim,
            max_node_force_halt = self.args.dag_max_node_force_halt,
            train_max_nodes = self.args.forward_train_max_nodes,
            test_max_nodes = self.args.forward_test_max_nodes,
            max_excision_group = config._max_excision_group,
            weight_dist = self.args.dag_weight_dist,
            weight_one = self.args.dag_weight_one,
            weight_ring = self.args.dag_weight_ring,
            weight_exn = self.args.dag_weight_exn,
            remove_peak_thres = self.args.dag_remove_peak_thres,
        )

        if self.args.debug_test_with_full_log:
            for d in self.test_data:
                ret = func(d = d, mode = 'test')
                self.main_worker_rref.rpc_async().log(f"[debug] {', '.join(['ID: ' + str(d['ID'])] + [key + ': ' + str(ret[key]) for key in test_keys])}")
            return

        while True:
            d = self.train_data[torch.randint(len(self.train_data), size = (1,))]
            d = copy.deepcopy(d)
            d['Smiles'] = d['tauts'][torch.randint(len(d['tauts']), size = (1,))]

            ret = func(d = d, mode = 'train')
            self.main_worker_rref.rpc_sync().report_trained(int(len(ret) > 1))
            self.gpu_worker_rref.rpc_sync().doit(ret['batch_datas'])
            if len(ret) > 1:
                self.main_worker_rref.rpc_sync().log(f"[train] {', '.join(['ID: ' + str(d['ID'])] + [key + ': ' + str(ret[key]) for key in train_keys])}")

            idx += 1
            if idx % self.args.forward_test_every_batch == 0:
                rec = OrderedDict((key, 0.0) for key in test_keys)
                for d in self.test_data:
                    ret = func(d = d, mode = 'test')
                    for key in test_keys:
                        rec[key] += ret[key]
                self.main_worker_rref.rpc_sync().report_test(len(self.test_data), rec)

class GPUWorker:

    def __init__(self, args, main_worker_rref, device_id, ranks, num_cpu_workers_per_gpu, is_head):
        self.args = args
        self.main_worker_rref = main_worker_rref
        self.device_id = device_id
        self.ranks = ranks
        self.num_cpu_workers_per_gpu = num_cpu_workers_per_gpu
        self.is_head = is_head

        self.infer_queue = queue.Queue()
        self.train_queue = queue.Queue()
        self.set_queue = queue.Queue()
        self.infer_done = 0
        self.train_cv = threading.Condition()

        self.model_lock = threading.Lock()


    def initialize(self, self_rref, ckpt):
        r"""
            initialize the model and optimizer from state_dict if key provided
        """
        torch.set_num_threads(8)
        self.gpu_group = dist.new_group(ranks = self.ranks, backend = 'nccl')

        self.model = network.Network(
            x_long_dim = config._x_long_dim,
            x_real_dim = config._x_real_dim,
            e_dim = config._e_dim,
            num_encoder_layers = self.args.arch_num_encoder_layers,
            embedding_dim = self.args.arch_embedding_dim,
            ffn_embedding_dim = self.args.arch_ffn_embedding_dim,
            num_attention_heads = self.args.arch_num_attention_heads,
            resid_pdrop = self.args.arch_resid_pdrop,
            attn_pdrop = self.args.arch_attn_pdrop,
            num_obk_actions = config._num_obk_actions,
            num_rbk_actions = config._num_rbk_actions,
            num_exn_actions = config._num_exn_actions,
            num_global_neg_hs_actions = config._max_global_neg_hs + 2,
        ).to(self.device_id)

        weight_list = [p for n, p in self.model.named_parameters() if n.endswith('weight')]
        non_weight_list = [p for n, p in self.model.named_parameters() if not n.endswith('weight')]
        if self.args.forward_optim == 'AdamW':
            self.optim = torch.optim.AdamW(
                [
                    {'params': weight_list, 'weight_decay': self.args.forward_wd},
                    {'params': non_weight_list, 'weight_decay': 0.},
                ],
                lr = self.args.forward_lr,
                betas = (self.args.forward_adamw_beta1, self.args.forward_adamw_beta2),
            )
        elif self.args.forward_optim == 'SGD':
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr = self.args.forward_lr,
                weight_decay = self.args.forward_wd,
                momentum = 0.9,
            )
        else:
            assert False
        self.scaler = torch.cuda.amp.GradScaler()

        model_state_dict = ckpt.get('model_state_dict')
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        optim_state_dict = ckpt.get('optim_state_dict')
        if optim_state_dict is not None:
            self.optim.load_state_dict(optim_state_dict)
        scaler_state_dict = ckpt.get('scaler_state_dict')
        if scaler_state_dict is not None:
            self.scaler.load_state_dict(scaler_state_dict)

        self.step = ckpt['step']
        # XXX adjust lr here, load_state_dict might override lr if set earlier
        for g in self.optim.param_groups:
            g['lr'] = self.args.forward_lr

        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids = [self.device_id],
            process_group = self.gpu_group,
        )
        #self.model = torch.compile(self.model)

        self.model.eval()
        self_rref.rpc_async()._run_train()
        self_rref.rpc_async()._run_infer()
        self_rref.rpc_async()._run_set()
        return None

    @rpc.functions.async_execution
    def doit(self, data):
        r"""
            A normal inference task if label is None, otherwise will do a backward and compute the gradient.
            Will only compute the gradient if self.num_cpu_workers_per_gpu requests have been collected
        """
        fut = torch.futures.Future()
        if data is None:
            fut.set_result(None)
            with self.train_cv:
                self.infer_done += 1
                if self.infer_done == self.num_cpu_workers_per_gpu:
                    self.train_cv.notify()
        elif isinstance(data, list):
            self.train_queue.put((move_to_pin_memory(data), fut))
        else:
            self.infer_queue.put((move_to_cuda(data, self.device_id, non_blocking = True), fut))
        return fut

    def fetch_ckpt(self):
        with self.model_lock:
            step = self.step
            model_state_dict = move_to_cpu(self.model.module.state_dict())
            optim_state_dict = move_to_cpu(self.optim.state_dict())
            scaler_state_dict = move_to_cpu(self.scaler.state_dict())
        return {
            'step': step,
            'model_state_dict': model_state_dict,
            'optim_state_dict': optim_state_dict,
            'scaler_state_dict': scaler_state_dict,
        }

    def _run_train(self):
        torch.set_num_threads(8)
        while True:
            with self.train_cv:
                while self.infer_done != self.num_cpu_workers_per_gpu:
                    self.train_cv.wait()

            # No need to keep the lock when doing backward and updating the parameters
            # All cpu_workers are guaranteed to NOT submit any inference task
            self.model.train()
            futs = []
            for i in range(self.num_cpu_workers_per_gpu):
                data, fut = self.train_queue.get()
                futs.append(fut)
                while len(data) > 0:
                    batch_data = move_to_cuda(data[-1], self.device_id, non_blocking = True)
                    del data[-1]
                    with sync_ddp(self.model, i + 1 == self.num_cpu_workers_per_gpu and len(data) == 0):
                        with torch.autocast(device_type = 'cuda', dtype = torch.float16):
                            logit = self.model(batch_data)
                            loss = (logit[batch_data['action_mask']] * batch_data['grad']).sum() / self.num_cpu_workers_per_gpu
                        self.scaler.scale(loss).backward()

            if self.step < self.args.forward_warm_up_steps:
                lr = (self.step + 1) / self.args.forward_warm_up_steps * self.args.forward_lr
            else:
                lr = self.args.forward_lr
            self.step += 1

            for g in self.optim.param_groups:
                g['lr'] = lr

            self.scaler.unscale_(self.optim)
            gs = [(name[7:], param.grad.square().mean().sqrt().item()) for name, param in self.model.named_parameters()]
            gs.sort(key = lambda x: x[1])
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.forward_clip_norm)

            with self.model_lock:
                self.scaler.step(self.optim)
                self.scaler.update()
            self.optim.zero_grad(set_to_none = True)

            self.model.eval()
            self.infer_done = 0
            for fut in futs:
                fut.set_result(None)

            if self.is_head:
                self.main_worker_rref.rpc_async().log(f"[debug] grad_norm: {norm.item()}")
                #self.main_worker_rref.rpc_sync().log(f"[debug] {gs}")

    
    def _run_infer(self):
        torch.set_num_threads(8)
        with torch.no_grad():
            while True:
                batch_data, fut = self.infer_queue.get()
                with torch.autocast(device_type = 'cuda', dtype = torch.float16):
                    logit = self.model(batch_data)
                self.set_queue.put((fut, logit))

    def _run_set(self):
        torch.set_num_threads(8)
        while True:
            fut, logit = self.set_queue.get()
            fut.set_result(logit.cpu())

class MainWorker:

    def __init__(self, args, index_book, node_infos, pre_messages):
        self.args = args
        self.index_book = index_book
        
        """ initialize all log-related stuff """
        self.ckpt_dir = os.path.join(args.config_log_dir, 'ckpt')
        ensure_dir(self.ckpt_dir)

        logFormatter = logging.Formatter("%(asctime)s  %(message)s")
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        fileHandler = logging.FileHandler(os.path.join(args.config_log_dir, 'log.txt'))
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)

        self.logger.info(' '.join(sys.argv))
        for pre_message in pre_messages:
            self.logger.info(pre_message)

        self.test_cv = threading.Lock()
        self.test_n = 0
        self.test_size = 0
        self.test_rec = OrderedDict()

        self.trained_cv = threading.Lock()
        self.n_trained = 0
        self.n_untrained = 0

    def run(self, main_worker_rref):
        self.main_worker_rref = main_worker_rref

        if self.args.config_resume is not None and os.path.exists(self.args.config_resume):
            self.logger.info("resume from checkpoint...")
            ckpt = torch.load(self.args.config_resume, map_location = torch.device('cpu'))
        else:
            ckpt = {'step': 0}

        futs = []
        """ initialize all gpu_workers """
        ranks = list(range(sum(pi.name.startswith('gpu_worker') for pi in self.index_book.values())))
        self.gpu_worker_rrefs = []
        num_cpu_workers_per_gpu = self.args.forward_batch_size // len(ranks)
        is_head = True
        for i, pi in enumerate(self.index_book.values()):
            if pi.name.startswith('gpu_worker'):
                self.gpu_worker_rrefs.append(rpc.remote(pi.name, GPUWorker,
                                                        args = (self.args, main_worker_rref, pi.device_id, ranks, num_cpu_workers_per_gpu, is_head)))
                is_head = False
        for gpu_worker_rref in self.gpu_worker_rrefs:
            futs.append(gpu_worker_rref.rpc_async().initialize(gpu_worker_rref, ckpt))

        all_data = {
            'train': pickle.load(open(self.args.config_data_train, 'rb')),
            'test': pickle.load(open(self.args.config_data_test, 'rb')),
        }
        # filter train nodes according to curriculum setting
        def _tot_atoms(formula):
            ct = utils.formula_to_element_count(formula)
            if 'H' in ct:
                del ct['H']
            return sum(v for v in ct.values())
        all_data['train'] = [item for item in all_data['train'] if _tot_atoms(item['Formula']) <= self.args.forward_cirriculum_max_nodes]
        all_data['test'] = [item for item in all_data['test'] if _tot_atoms(item['Formula']) <= self.args.forward_cirriculum_max_nodes]

        # shuffle is crucial to prevent a chunk of large molecules to go into the same cpu worker
        random.shuffle(all_data['train'])
        random.shuffle(all_data['test'])

        """ initialize all cpu_workers """
        self.cpu_worker_rrefs = []
        for pi in self.index_book.values():
            if pi.name.startswith('cpu_worker'):
                cpu_rank = int(pi.name.split(':')[1])
                self.cpu_worker_rrefs.append(rpc.remote(pi.name, CPUWorker,
                                                        args = (self.args, main_worker_rref, self.gpu_worker_rrefs[cpu_rank // num_cpu_workers_per_gpu],
                                                                cpu_rank, self.args.forward_batch_size)))
        train_len, test_len = len(all_data['train']), len(all_data['test'])
        self.log(f"number of train instances: {train_len}, number of test instances: {test_len}")
        for r, cpu_worker_rref in enumerate(self.cpu_worker_rrefs):
            futs.append(cpu_worker_rref.rpc_async().initialize(
                cpu_worker_rref,
                all_data['train'][r * train_len // len(self.cpu_worker_rrefs):(r + 1) * train_len // len(self.cpu_worker_rrefs)],
                all_data['test'][r * test_len // len(self.cpu_worker_rrefs):(r + 1) * test_len // len(self.cpu_worker_rrefs)]))
        
        for fut in futs:
            fut.wait()
        self.log("all workers have been initialized...")

        main_worker_rref.rpc_async().save_checkpoint()

    def log(self, message):
        self.logger.info(message)

    def report_test(self, size, rec):
        message = None
        with self.test_cv:
            self.test_n += 1
            self.test_size += size
            for k, v in rec.items():
                if k not in self.test_rec:
                    self.test_rec[k] = v
                else:
                    self.test_rec[k] += v
            if self.test_n == len(self.cpu_worker_rrefs):
                message = f"[test] {', '.join([k + ': ' + str(v / self.test_size) for k, v in self.test_rec.items()])}"
                self.test_n = 0
                self.test_size = 0
                for k in self.test_rec:
                    self.test_rec[k] = 0.0
        if message is not None:
            self.log(message)


    def report_trained(self, trained):
        message = None
        with self.trained_cv:
            if trained:
                self.n_trained += 1
            else:
                self.n_untrained += 1
            if self.n_trained + self.n_untrained == len(self.cpu_worker_rrefs):
                message = f"[info] trained: {self.n_trained / (self.n_trained + self.n_untrained)}, dag_num_sim: {self.args.dag_num_sim}, dag_max_node_force_halt: {self.args.dag_max_node_force_halt}, lr: {self.args.forward_lr}, max_nodes: {self.args.forward_cirriculum_max_nodes}"
                self.n_trained = 0
                self.n_untrained = 0
        if message is not None:
            self.log(message)

    def save_checkpoint(self):
        torch.set_num_threads(1)
        while True:
            time.sleep(self.args.config_save_every_min * 60)
            ckpt = self.gpu_worker_rrefs[0].rpc_sync().fetch_ckpt()
            # checkpoint step can be ruined, but latest MUST be protected
            # the following codes make the possiblity to ruin latest to minimum
            torch.save(ckpt, os.path.join(self.ckpt_dir, str(ckpt['step'])))
            shutil.copyfile(os.path.join(self.ckpt_dir, str(ckpt['step'])), os.path.join(self.ckpt_dir, 'latest.tmp'))
            shutil.move(os.path.join(self.ckpt_dir, 'latest.tmp'), os.path.join(self.ckpt_dir, 'latest'))
            self.log(f"done saving checkpoint at step {ckpt['step']}")


ProcessInfo = namedtuple('ProcessInfo', ['name', 'node_rank', 'device_id'])
    
def run_rpc(args, rank, name, index_book, node_infos, pg_addr, rpc_addr, pre_messages):
    r"""
        1) main_worker
        2) cpu_worker.{0,1,2,3,...}
        3) gpu_worker.{0,1,2,3,...}
    """
    rpc.backend_registry.register_backend(
        "GLOOFREETENSORPIPE",
        rpc.backend_registry._tensorpipe_construct_rpc_backend_options_handler,
        _gloo_free_tensorpipe_init_backend_handler,
    )

    world_size = len(index_book)
    
    # Since the total number of processes might be large, the init_process_group is only invoked on
    # all GPU processes. Use rpc to transfer data between arbitrary processes
    if name.startswith('gpu_worker'):
        alls = [pi.name for pi in index_book.values() if pi.name.startswith('gpu_worker')]
        dist.init_process_group(backend = 'gloo',
                                rank = alls.index(name), world_size = len(alls),
                                init_method = f"tcp://{pg_addr}")

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads = 256 if name.startswith('main_worker') else 16,
        # num_worker_threads = 16,
        rpc_timeout = 0,
        _transports = ['uv'],
        init_method = f"tcp://{rpc_addr}",
    )
    # use self-defined GLOOFREETENSORPIPE. The downside is that we can no longer use rpc.shutdown()
    # and always need to kill the program with abrupt system calls
    rpc.init_rpc(name, backend = rpc.backend_registry.BackendType.GLOOFREETENSORPIPE,
                 rank = rank, world_size = world_size,
                 rpc_backend_options = options)

    if name == 'main_worker':
        main_worker_rref = rpc.RRef(MainWorker(args, index_book, node_infos, pre_messages))
        main_worker_rref.rpc_sync().run(main_worker_rref)

    # we modified pytorch source codes to remove the dependancy of gloo
    # instead of using rpc.shutdown, we use time.sleep
    time.sleep(1 << 31)



NodeInfo = namedtuple('NodeInfo', ['hostname', 'num_cpu', 'gpu_type', 'num_gpu', 'pg_port', 'rpc_port'])
ProcessInfo = namedtuple('ProcessInfo', ['name', 'node_rank', 'device_id'])

def initialize(args):
    r"""
    Initialize the distributed communication protocol
    """
    if 'SLURM_PROCID' in os.environ and 'SLURM_NPROCS' in os.environ:
        # multi-node mode
        node_rank = int(os.environ['SLURM_PROCID'])
        node_size = int(os.environ['SLURM_NPROCS'])
    else:
        # single-node mode
        node_rank = 0
        node_size = 1

    if node_size == 1:
        dist.init_process_group(backend = 'gloo', rank = node_rank, world_size = node_size,
                                init_method = f"tcp://localhost:{get_free_port()}")
    else:
        assert args.config_head_addr is not None, "config_head_addr must be specified for multi-node training"
        dist.init_process_group(backend = 'gloo', rank = node_rank, world_size = node_size,
                                init_method = f"tcp://{args.config_head_addr}")

    node_infos = [None] * node_size
    if 'SLURM_CPUS_ON_NODE' in os.environ:
        num_cpu = int(os.environ['SLURM_CPUS_ON_NODE'])
    else:
        num_cpu = len(os.sched_getaffinity(0))
    dist.all_gather_object(node_infos, NodeInfo(hostname = socket.gethostname(),
                                                num_cpu = num_cpu,
                                                gpu_type = torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                                                num_gpu = torch.cuda.device_count(),
                                                pg_port = get_free_port(),
                                                rpc_port = get_free_port()))
    dist.destroy_process_group()

    # The logger is only initialized in the MainWorker but not here. Save all messages to print
    # temporarily and print when the logger is initialized in the future
    pre_messages = []
    hostnames, num_cpus, gpu_types, num_gpus, _, _ = zip(*node_infos)
    pre_messages.append("The following is the node information...")
    for node_info in node_infos:
        if node_info.num_gpu == 0:
            pre_messages.append(f"hostname: {node_info.hostname}, num_cpu: {node_info.num_cpu}")
        else:
            pre_messages.append(f"hostname: {node_info.hostname}, num_cpu: {node_info.num_cpu}, gpu_type: {node_info.gpu_type}, num_gpu: {node_info.num_gpu}")

    tot_num_cpus = sum(num_cpus)
    tot_num_gpus = sum(num_gpus)
    # adjust the forward batch size to be multiplier of the number of gpus
    args.forward_batch_size = math.ceil(args.forward_batch_size / tot_num_gpus) * tot_num_gpus

    index_book = {0: ProcessInfo(name = "main_worker", node_rank = 0, device_id = None)}

    idx = 0
    for r, node_info in enumerate(node_infos):
        st = idx * args.forward_batch_size // tot_num_cpus
        idx += node_info.num_cpu
        ed = idx * args.forward_batch_size // tot_num_cpus
        for i in range(st, ed):
            index_book[len(index_book)] = ProcessInfo(name = f"cpu_worker:{i}",
                                                      node_rank = r, device_id = None)
    
    idx_gpu_worker = 0
    for r, node_info in enumerate(node_infos):
        for i in range(node_info.num_gpu):
            index_book[len(index_book)] = ProcessInfo(name = f"gpu_worker:{idx_gpu_worker}",
                                                      node_rank = r, device_id = i)
            idx_gpu_worker += 1
    # pg_addr must be on the node with GPU device and cannot naively be set to the node 0 (in case node 0 does not have GPU)
    for node_info in node_infos:
        if node_info.num_gpu > 0:
            pg_addr = f"{node_info.hostname}:{node_info.pg_port}"
            break

    ps = []
    for rank, process_info in index_book.items():
        if process_info.node_rank == node_rank:
            ps.append(multiprocessing.Process(target = run_rpc,
                                              args = (args, rank, process_info.name, index_book, node_infos, pg_addr,
                                                      f"{node_infos[0].hostname}:{node_infos[0].rpc_port}",
                                                      pre_messages)))
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    parser.add_argument('--config-head-addr', type = str, help = 'addresss (host:port) of the head node for multi-node communication')
    parser.add_argument('--config-log-dir', type = str, default = 'log', help = 'path of the log directory')
    parser.add_argument('--config-data-train', type = str, default = '/mnt/beegfs/bulk/mirror/df394/data/metabolite-nist-20-ms-ms/acc_ppm5.0_merge_ppm20.0_large_peak_ratio0.1_tautomer0.95.train.pkl')
    parser.add_argument('--config-data-test', type = str, default = '/mnt/beegfs/bulk/mirror/df394/data/metabolite-nist-20-ms-ms/acc_ppm5.0_merge_ppm20.0_large_peak_ratio0.1_tautomer0.95.test.pkl')
    parser.add_argument('--config-resume', type = str, help = '')
    parser.add_argument('--config-save-every-min', type = int, default = 5)

    parser.add_argument('--forward-batch-size', type = int, help = '')
    parser.add_argument('--forward-optim', type = str, default = 'AdamW', choices = ['AdamW', 'SGD'])
    parser.add_argument('--forward-adamw-beta1', type = float, default = 0.9)
    parser.add_argument('--forward-adamw-beta2', type = float, default = 0.999)
    parser.add_argument('--forward-lr', type = float, default = 1e-3, help = '')
    parser.add_argument('--forward-warm-up-steps', type = int, default = 500, help = '')
    parser.add_argument('--forward-wd', type = float, default = 1e-4, help = '')
    parser.add_argument('--forward-test-every-batch', type = int, default = 10, help = '')
    parser.add_argument('--forward-train-max-nodes', type = int, default = 4096, help = '')
    parser.add_argument('--forward-test-max-nodes', type = int, default = 2048, help = '')
    parser.add_argument('--forward-cirriculum-max-nodes', type = int, default = 1000000, help = '')
    parser.add_argument('--forward-clip-norm', type = float, default = 100.0)

    parser.add_argument('--arch-num-encoder-layers', type = int, default = 12)
    parser.add_argument('--arch-embedding-dim', type = int, default = 80)
    parser.add_argument('--arch-ffn-embedding-dim', type = int, default = 80)
    parser.add_argument('--arch-num-attention-heads', type = int, default = 8)
    parser.add_argument('--arch-resid-pdrop', type = float, default = 0.1)
    parser.add_argument('--arch-attn-pdrop', type = float, default = 0.1)

    parser.add_argument('--dag-ppm', type = float, default = 10.0)
    parser.add_argument('--dag-num-sim', type = int, default = 1000000)
    parser.add_argument('--dag-max-node-force-halt', type = int, default = 1000000000)
    parser.add_argument('--dag-weight-dist', type = float, default = 0.0)
    parser.add_argument('--dag-weight-one', type = float, default = 0.0)
    parser.add_argument('--dag-weight-ring', type = float, default = 0.0)
    parser.add_argument('--dag-weight-exn', type = float, default = 0.0)
    parser.add_argument('--dag-remove-peak-thres', type = float, default = 0.0)

    parser.add_argument('--debug-test-with-full-log', action = 'store_true')

    args = parser.parse_args()

    initialize(args)
