#!/usr/bin/env bash
# Finite integration check only; does not resume the overnight training job.
set -eo pipefail
export ISAACSIM_PATH=/data/uavlab/isaac_sim_2022.2.0
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source /data/uavlab/isaac_sim_2022.2.0/setup_conda_env.sh
cd /data/uavlab/maneuverable-target
exec python scripts/train.py --config-name=train_pv task.env.num_envs=16 \
 task.sim.device=cuda:1 task.sim.physics_gpu=1 task.sim.active_gpu=1 \
 algo.num_minibatches=4 algo.predictor.num_minibatches=2 max_iters=3 \
 eval_interval=-1 save_interval=3 wandb.mode=disabled headless=true +diagnostic_run=true
