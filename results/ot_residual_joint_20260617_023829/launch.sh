#!/usr/bin/env bash
set -euo pipefail
cd /data/uavlab/maneuverable-target
PYTHONPATH=/data/uavlab/maneuverable-target HYDRA_FULL_ERROR=1 conda run --no-capture-output -n sim python scripts/train.py \
  headless=true \
  wandb.mode=disabled \
  wandb.run_name=ot_residual_joint \
  total_frames=262144 \
  max_iters=32 \
  eval_interval=-1 \
  save_interval=8 \
  task=HideAndSeek \
  task.env.num_envs=64 \
  algo.train_every=128
