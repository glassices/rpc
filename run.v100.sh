#!/bin/bash

#SBATCH --job-name=v100
#SBATCH --output=slurm_log/slurm.out
#SBATCH --error=slurm_log/slurm.err
#SBATCH --time=120:00:00

#SBATCH --partition=aida --gpus=v100:15 --cpus-per-gpu=14 --mem-per-gpu=30G

set -x

ulimit -Sn $(ulimit -Hn)
ulimit -Su $(ulimit -Hu)
ulimit -Ss $(ulimit -Hs)

HOST=$(hostname)
PORT=$(python -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

srun python main.py \
    --config-data-train /mnt/beegfs/bulk/mirror/df394/data/metabolite-nist-20-ms-ms/acc_ppm5.0_merge_ppm20.0_large_peak_ratio0.1_tautomer0.95_dist1.train.pkl \
    --config-data-test /mnt/beegfs/bulk/mirror/df394/data/metabolite-nist-20-ms-ms/acc_ppm5.0_merge_ppm20.0_large_peak_ratio0.1_tautomer0.95_dist1.test.pkl \
    --config-head-addr $HOST:$PORT --config-log-dir log --config-resume log/ckpt/latest --config-save-every-min 5 \
    --forward-batch-size 128 --forward-optim AdamW --forward-lr 3e-5 --forward-warm-up-steps 100 --forward-wd 1e-2 \
    --forward-adamw-beta1 0.9 --forward-adamw-beta2 0.99 \
    --forward-test-every-batch 250 \
    --forward-train-max-nodes 4096 --forward-test-max-nodes 16384 \
    --forward-cirriculum-max-nodes 25 \
    --forward-clip-norm 20.0 \
    --dag-ppm 20.0 --dag-num-sim 1000000 --dag-max-node-force-halt 15000 --dag-weight-tree-entropy 0.0 --dag-weight-avg-depth 0.0 \
    --dag-weight-kl 0.01 --dag-weight-l2 0.99 \
    --arch-num-encoder-layers 12 --arch-embedding-dim 256 --arch-ffn-embedding-dim 1024 --arch-num-attention-heads 16 \
    --arch-resid-pdrop 0.0 --arch-attn-pdrop 0.0
