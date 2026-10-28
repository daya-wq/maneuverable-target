#!/usr/bin/env bash
# Bounded full-initialization comparison after the genuine attempt_06@5000.
set -eo pipefail
export ISAACSIM_PATH=/data/uavlab/isaac_sim_2022.2.0
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source /data/uavlab/isaac_sim_2022.2.0/setup_conda_env.sh
cd /data/uavlab/maneuverable-target
folder=/data/uavlab/maneuverable-target/results/overnight70_20261021/fresh_initialization_ab
mkdir -p "$folder"
exec 9>"$folder/experiment.lock"
flock -n 9
if [ -e "$folder/fresh500.log" ] || [ -e "$folder/warm500.log" ]; then
  echo 'Existing initialization experiment; refusing duplicate/overwrite' >&2
  exit 2
fi
printf '%s\n' "$$" > "$folder/experiment.pid"
bash scripts/evaluate_main_checkpoint70.sh \
  /data/uavlab/maneuverable-target/checkpoints/HideAndSeek_20261022_055552/checkpoint_2048000.pt \
  "$folder/warm500_eval" 128 > "$folder/warm500.log" 2>&1 &
baseline_pid=$!
printf '%s\n' "$baseline_pid" > "$folder/baseline_evaluation.pid"
python scripts/train.py task=HideAndSeek70Isolated task.env.num_envs=128 \
  task.sim.device=cuda:0 task.sim.physics_gpu=0 task.sim.active_gpu=0 \
  algo.train_every=32 algo.num_minibatches=4 algo.predictor.num_minibatches=2 \
  algo.predictor.max_speed=17 algo.predictor.max_accel=12 \
  algo.actor.lr=0.000003 algo.critic.lr=0.00005 \
  algo.actor.log_std_init=-3 algo.actor.log_std_min=-3 \
  algo.actor.lr_scheduler=null algo.critic.lr_scheduler=null \
  task.m3_terminal_collision_penalty=40 +task.m3_safety_env_dist=5 \
  model_dir=null max_iters=500 save_interval=250 eval_interval=-1 wandb.mode=disabled \
  +diagnostic_run=true +sampling_diagnostic=true +trajectory_evaluation=true \
  "+evaluation_output=$folder/fresh500_eval" > "$folder/fresh500.log" 2>&1
wait "$baseline_pid"
printf '%s\n' complete > "$folder/completed.txt"
