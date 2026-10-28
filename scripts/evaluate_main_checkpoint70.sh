#!/usr/bin/env bash
# Read-only independent evaluation with the exact attempt_06 scene/reward.
set -eo pipefail
checkpoint="${1:?Provide an existing main checkpoint absolute path}"
folder="${2:?Provide a new evaluation output directory absolute path}"
envs="${3:-128}"
test -f "$checkpoint"
export ISAACSIM_PATH=/data/uavlab/isaac_sim_2022.2.0
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source /data/uavlab/isaac_sim_2022.2.0/setup_conda_env.sh
cd /data/uavlab/maneuverable-target
exec 9>results/overnight70_20261021/independent_gpu1_eval.lock
flock -n 9
# mkdir intentionally fails for any prior output, including an interrupted run.
mkdir "$folder"
printf '%s\n' "$$" > "$folder/evaluation.pid"
printf '%s\n' "$checkpoint" > "$folder/checkpoint.txt"
exec python scripts/train.py task=HideAndSeek70Isolated "task.env.num_envs=$envs" \
  task.sim.device=cuda:1 task.sim.physics_gpu=1 task.sim.active_gpu=1 \
  algo.train_every=32 algo.num_minibatches=4 algo.predictor.num_minibatches=2 \
  algo.predictor.max_speed=17 algo.predictor.max_accel=12 \
  algo.actor.lr=0.000003 algo.critic.lr=0.00005 \
  algo.actor.log_std_init=-3 algo.actor.log_std_min=-3 \
  algo.actor.lr_scheduler=null algo.critic.lr_scheduler=null \
  task.m3_terminal_collision_penalty=40 +task.m3_safety_env_dist=5 \
  "model_dir=$checkpoint" eval_interval=-1 wandb.mode=disabled \
  +evaluate_only=true +trajectory_evaluation=true +evaluation_stochastic=false \
  +evaluation_reset_finished=false +diagnostic_run=true \
  "+evaluation_output=$folder"
