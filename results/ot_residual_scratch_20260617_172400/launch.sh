#!/bin/bash
# OT Residual from-scratch training with 3.0m/s config
# All 6 handover issues verified resolved before launch
set -e
export PYTHONPATH=/data/uavlab/maneuverable-target
export HYDRA_FULL_ERROR=1
conda run --no-capture-output -n sim python scripts/train.py \
  task=HideAndSeek headless=true wandb.mode=disabled \
  total_frames=524288 max_iters=64 eval_interval=-1 save_interval=8 \
  task.env.num_envs=64 algo.train_every=128 \
  2>&1 | tee results/ot_residual_scratch_20260617_172400/train.log
