#!/usr/bin/env bash
cd /data/uavlab/maneuverable-target || exit 1
PYTHONPATH=/data/uavlab/maneuverable-target HYDRA_FULL_ERROR=1 conda run --no-capture-output -n sim python scripts/train.py task=HideAndSeek headless=true wandb.mode=disabled total_frames=1048576 max_iters=128 eval_interval=-1 save_interval=8 task.env.num_envs=64 algo.train_every=128
