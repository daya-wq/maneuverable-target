#!/usr/bin/env bash
set -eo pipefail
export ISAACSIM_PATH=/data/uavlab/isaac_sim_2022.2.0
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source /data/uavlab/isaac_sim_2022.2.0/setup_conda_env.sh
cd /data/uavlab/maneuverable-target
exec python scripts/test_goal_arrival.py task=HideAndSeek70Isolated task.env.num_envs=24 \
 task.sim.device=cuda:1 task.sim.physics_gpu=1 task.sim.active_gpu=1 \
 +arrival_output=/data/uavlab/maneuverable-target/results/goal_arrival_20261022_verified \
 headless=true wandb.mode=disabled
