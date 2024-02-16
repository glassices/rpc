#!/bin/bash

#SBATCH --job-name=a100
#SBATCH --output=slurm_log/slurm.out
#SBATCH --error=slurm_log/slurm.err
#SBATCH --time=120:00:00

#SBATCH --partition=aida --gpus=a100:18 --cpus-per-gpu=20 --mem-per-gpu=40G --exclude=c0072

set -x

ulimit -Sn $(ulimit -Hn)
ulimit -Su $(ulimit -Hu)
ulimit -Ss $(ulimit -Hs)

HOST=$(hostname)
PORT=$(python -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

srun python main.py \
    --config-data-train /mnt/beegfs/bulk/mirror/df394/data/metabolite-nist-20-ms-ms/train_val_test.acc_ppm5.0_merge_ppm20.0_large_peak_ratio0.1_tautomer0.95_dist1.train.pkl \
    --config-data-test /mnt/beegfs/bulk/mirror/df394/data/metabolite-nist-20-ms-ms/train_val_test.acc_ppm5.0_merge_ppm20.0_large_peak_ratio0.1_tautomer0.95_dist1.val.pkl \
    --config-head-addr $HOST:$PORT --config-log-dir log --config-resume log/ckpt/latest --config-save-every-min 5 \
    --forward-batch-size 128 --forward-optim AdamW --forward-lr 1e-5 --forward-warm-up-steps 100 --forward-wd 1e-2 \
    --forward-adamw-beta1 0.9 --forward-adamw-beta2 0.99 \
    --forward-test-every-batch 1024 \
    --forward-train-max-nodes 2048 --forward-test-max-nodes 8192 \
    --forward-cirriculum-max-nodes 20 \
    --forward-clip-norm 20.0 \
    --dag-ppm 20.0 --dag-num-sim 10000 --dag-max-node-force-halt 10000 \
    --dag-weight-dist 0.0 --dag-weight-one 0.0 --dag-weight-ring 0.0 --dag-weight-exn 0.0 \
    --dag-remove-peak-thres 0.01 \
    --arch-num-encoder-layers 12 --arch-embedding-dim 256 --arch-ffn-embedding-dim 1024 --arch-num-attention-heads 16 \
    --arch-resid-pdrop 0.0 --arch-attn-pdrop 0.0 \
