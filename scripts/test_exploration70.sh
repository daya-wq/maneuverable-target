#!/usr/bin/env bash
# Noise-floor-only screening; do not replace the main overnight run.
set -eo pipefail
export ISAACSIM_PATH=/data/uavlab/isaac_sim_2022.2.0
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source /data/uavlab/isaac_sim_2022.2.0/setup_conda_env.sh
cd /data/uavlab/maneuverable-target
folder=/data/uavlab/maneuverable-target/results/overnight70_20261021/attempt_06/exploration_ab
mkdir -p "$folder"
exec 9>"$folder/experiment.lock"
flock -n 9
if [ -e "$folder/std_floor_m2.log" ]; then
  echo 'Existing exploration test log; refusing duplicate/overwrite' >&2
  exit 2
fi
printf '%s\n' "$$" > "$folder/experiment.pid"
# Keep loaded log_std weights and init unchanged. Production forward clamps
# the loaded -3 parameter at the new -2 floor; changing init alone is ineffective.
python scripts/train.py task=HideAndSeek70Isolated task.env.num_envs=128 \
  task.sim.device=cuda:1 task.sim.physics_gpu=1 task.sim.active_gpu=1 \
  algo.train_every=32 algo.num_minibatches=4 algo.predictor.num_minibatches=2 \
  algo.predictor.max_speed=17 algo.predictor.max_accel=12 \
  algo.actor.lr=0.000003 algo.critic.lr=0.00005 \
  algo.actor.log_std_init=-3 algo.actor.log_std_min=-2 \
  algo.actor.lr_scheduler=null algo.critic.lr_scheduler=null \
  +algo.actor.reset_optimizer_on_load=true \
  task.m3_terminal_collision_penalty=40 +task.m3_safety_env_dist=5 \
  model_dir=/data/uavlab/maneuverable-target/checkpoints/HideAndSeek_20261022_055552/checkpoint_1024000.pt \
  max_iters=128 save_interval=128 eval_interval=-1 wandb.mode=disabled \
  +diagnostic_run=true +sampling_diagnostic=true +trajectory_evaluation=true \
  "+evaluation_output=$folder/std_floor_m2" > "$folder/std_floor_m2.log" 2>&1
printf '%s\n' complete > "$folder/completed.txt"
