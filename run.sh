#!/bin/bash

#SBATCH --job-name=banbanban
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --time=168:00:00
#SBATCH --qos=low

#SBATCH --partition=aida --gpus=a100:14 --cpus-per-gpu=24 --mem-per-gpu=220G

#SBATCH hetjob

#SBATCH --partition=aida --gpus=v100:21 --cpus-per-gpu=8 --mem-per-gpu=40G

set -x

ulimit -Sn $(ulimit -Hn)
ulimit -Su $(ulimit -Hu)
ulimit -Ss $(ulimit -Hs)

RDZV_HOST=$(hostname)
RDZV_PORT=$(python -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

log_dir="log_dq"

# RDZV_HOST must be in the training group!!! Really important

srun --het-group=0,1 python main.py --config-gloo-addr $RDZV_HOST:$RDZV_PORT --config-log-dir $log_dir --config-resume $log_dir/ckpt/latest --config-num-threads 32 --config-raw-cases /mnt/beegfs/bulk/mirror/df394/data/crawler/raw_cases --config-init-plans n100_m100.plans --optimizer-report-reuse-every-min 30 --optimizer-buffer-size-min 1000000 --optimizer-buffer-size-max 6000000 --optimizer-max-train-size 2440 --optimizer-learning-rate 3e-5 --optimizer-warm-up-steps 1000 --rollout-ground-truth-ratio 0.0 --rollout-noise-epsilon 0.0 --rollout-search-steps 50000 --rollout-max-total-cost 500 --arch-block 50 --arch-channel 256 --arch-channel-pool 84 --arch-channel-head 84 --arch-channel-val 128 --arch-channel-attn 80 --arch-channel-in 24 --arch-num-heads 8 --arch-global-every 7 --arch-attn-from 25 --rollout-beam-size 24 --rollout-concurrent-size 4 --optimizer-concurrent-size 4 --rollout-procs-per-gpu 16 --optimizer-update-every-batch 2 --config-save-every-batch 64 --optimizer-batch-size 4096 --optimizer-num-gpus 12
