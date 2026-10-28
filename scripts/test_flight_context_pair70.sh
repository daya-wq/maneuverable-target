#!/usr/bin/env bash
set -eo pipefail
export ISAACSIM_PATH=/data/uavlab/isaac_sim_2022.2.0
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source /data/uavlab/isaac_sim_2022.2.0/setup_conda_env.sh
cd /data/uavlab/maneuverable-target
folder=/data/uavlab/maneuverable-target/results/overnight70_20261021/attempt_06/flight_context_ab
mkdir -p "$folder"
exec 9>"$folder/experiment.lock"
flock -n 9
printf '%s\n' "$$" > "$folder/experiment.pid"
case "${1:-both}" in
  both) variants=(baseline context);;
  context_retry1) variants=(context_retry1);;
  *) echo "Expected both or context_retry1" >&2; exit 2;;
esac
for variant in "${variants[@]}"; do
  if [ -f "$folder/$variant.log" ]; then
    echo "Existing comparison log; refusing to overwrite $variant" >&2
    exit 2
  fi
  task_name=HideAndSeek70Isolated
  if [ "$variant" != baseline ]; then task_name=HideAndSeek70Context; fi
  python scripts/train.py "task=$task_name" task.env.num_envs=128 \
    task.sim.device=cuda:1 task.sim.physics_gpu=1 task.sim.active_gpu=1 \
    algo.train_every=32 algo.num_minibatches=4 algo.predictor.num_minibatches=2 \
    algo.predictor.max_speed=17 algo.predictor.max_accel=12 \
    algo.actor.lr=0.000003 algo.critic.lr=0.00005 \
    algo.actor.log_std_init=-3 algo.actor.log_std_min=-3 \
    algo.actor.lr_scheduler=null algo.critic.lr_scheduler=null \
    +algo.actor.reset_optimizer_on_load=true \
    task.m3_terminal_collision_penalty=40 +task.m3_safety_env_dist=5 \
    model_dir=/data/uavlab/maneuverable-target/checkpoints/HideAndSeek_20261022_055552/checkpoint_1024000.pt \
    max_iters=128 save_interval=128 eval_interval=-1 wandb.mode=disabled \
    +diagnostic_run=true +sampling_diagnostic=true +trajectory_evaluation=true \
    "+evaluation_output=$folder/$variant" > "$folder/$variant.log" 2>&1
done
printf '%s\n' complete > "$folder/completed.txt"
