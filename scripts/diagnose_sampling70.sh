#!/usr/bin/env bash
set -eo pipefail
export ISAACSIM_PATH=/data/uavlab/isaac_sim_2022.2.0
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source /data/uavlab/isaac_sim_2022.2.0/setup_conda_env.sh
cd /data/uavlab/maneuverable-target
mode="${1:-stochastic}"
output="${2:-/data/uavlab/maneuverable-target/results/overnight70_20261021/attempt_02/diagnostic_projected_stochastic_00250}"
model="${3:-/data/uavlab/maneuverable-target/checkpoints/HideAndSeek_20261021_232636/checkpoint_1024000.pt}"
envs="${4:-64}"
logstd="${5:--1.5}"
repeats="${6:-1}"
task_name="${7:-HideAndSeek70}"
reset_finished="${8:-false}"
reset_integrity="${9:-false}"
case "$mode" in
  stochastic) mode_args=(+evaluation_stochastic=true);;
  deterministic) mode_args=(+evaluation_stochastic=false);;
  rule) mode_args=(+rule_baseline=true);;
  hover) mode_args=(+hover_test=true);;
  *) echo "Mode must be stochastic, deterministic, rule, or hover" >&2; exit 2;;
esac
exec python scripts/train.py "task=$task_name" "task.env.num_envs=$envs" \
  task.sim.device=cuda:1 task.sim.physics_gpu=1 task.sim.active_gpu=1 \
  algo.num_minibatches=4 algo.predictor.num_minibatches=2 \
  algo.predictor.max_speed=17 algo.predictor.max_accel=12 \
  "algo.actor.log_std_init=$logstd" algo.actor.lr_scheduler=null algo.critic.lr_scheduler=null \
  eval_interval=-1 wandb.mode=disabled \
  "model_dir=$model" \
  +evaluate_only=true +trajectory_evaluation=true "${mode_args[@]}" \
  "+evaluation_repeats=$repeats" \
  "+evaluation_reset_finished=$reset_finished" \
  "+reset_integrity_audit=$reset_integrity" \
  "+evaluation_output=$output"
