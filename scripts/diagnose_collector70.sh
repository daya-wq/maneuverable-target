#!/usr/bin/env bash
set -eo pipefail
export ISAACSIM_PATH=/data/uavlab/isaac_sim_2022.2.0
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source /data/uavlab/isaac_sim_2022.2.0/setup_conda_env.sh
cd /data/uavlab/maneuverable-target
logstd="${1:--1.5}"
output="${2:-/data/uavlab/maneuverable-target/results/overnight70_20261021/collector_diagnostic_final}"
task_name="${3:-HideAndSeek70}"
actor_lr="${4:-0.000001}"
extra_args=()
if [ "$#" -ge 5 ]; then extra_args+=("task.m3_terminal_collision_penalty=$5"); fi
if [ "$#" -ge 6 ]; then extra_args+=("+task.m3_safety_env_dist=$6"); fi
exec python scripts/train.py "task=$task_name" task.env.num_envs=128 \
  task.sim.device=cuda:1 task.sim.physics_gpu=1 task.sim.active_gpu=1 \
  algo.train_every=32 algo.num_minibatches=4 algo.predictor.num_minibatches=2 \
  algo.predictor.max_speed=17 algo.predictor.max_accel=12 \
  "algo.actor.log_std_init=$logstd" "algo.actor.lr=$actor_lr" algo.critic.lr=0.00005 \
  algo.actor.lr_scheduler=null algo.critic.lr_scheduler=null \
  max_iters=12 save_interval=12 eval_interval=-1 wandb.mode=disabled \
  +sampling_diagnostic=true +trajectory_evaluation=true \
  "+evaluation_output=$output" "${extra_args[@]}"
