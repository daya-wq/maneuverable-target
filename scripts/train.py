import logging
import math
import os
import time

import hydra
import torch
import numpy as np
try:
    import wandb
except ImportError:
    class _NoOpWandb:
        class Video:
            def __init__(self, *args, **kwargs):
                pass

        @staticmethod
        def save(*args, **kwargs):
            return None

        @staticmethod
        def finish(*args, **kwargs):
            return None

    wandb = _NoOpWandb()
import datetime
from torch.utils.tensorboard import SummaryWriter

from functorch import vmap
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import SyncDataCollector, AgentSpec
from omni_drones.utils.torchrl.transforms import (
    LogOnEpisode, 
    FromMultiDiscreteAction, 
    FromDiscreteAction,
    ravel_composite,
    History
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import (
    MAPPOPolicy, 
    Policy,
    PPOPolicy,
    PPOAdaptivePolicy, PPORNNPolicy
)

try:
    from setproctitle import setproctitle
except ImportError:
    def setproctitle(*args, **kwargs):
        return None
from torchrl.envs.transforms import (
    TransformedEnv, 
    InitTracker, 
    Compose,
)

from tqdm import tqdm

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1

from typing import Sequence
from tensordict import TensorDictBase


def checkpoint_actor_rnn_hidden_size(state_dict):
    if int(state_dict.get("actor_rnn_hidden_size", 0)) > 0:
        return int(state_dict["actor_rnn_hidden_size"])
    actor_params = state_dict.get("actor_params", None)
    if actor_params is None:
        return None
    actor_td = actor_params.to_tensordict()
    for key in actor_td.keys(True, True):
        key_str = "/".join(map(str, key))
        if "rnn" not in key_str or not key_str.endswith("cell.weight_ih"):
            continue
        weight = actor_td.get(key)
        if weight.ndim == 2 and weight.shape[0] % 3 == 0:
            return int(weight.shape[0] // 3)
    return None


def checkpoint_prev_action_condition_hidden_size(state_dict):
    if bool(state_dict.get("actor_has_prev_action_conditioning", False)):
        hidden_dim = int(state_dict.get("actor_prev_action_condition_hidden_dim", 0))
        return hidden_dim if hidden_dim > 0 else 128
    actor_params = state_dict.get("actor_params", None)
    if actor_params is None:
        return None
    actor_td = actor_params.to_tensordict()
    for key in actor_td.keys(True, True):
        key_str = "/".join(map(str, key))
        if "prev_action_conditioner" not in key_str or not key_str.endswith("0.weight"):
            continue
        weight = actor_td.get(key)
        if weight.ndim == 2:
            return int(weight.shape[0])
    return None


def enable_actor_rnn_cfg(cfg, hidden_size: int, train_seq_len: int = 8):
    cfg.algo.actor.rnn = OmegaConf.create(
        {
            "cls": "gru",
            "kwargs": {"hidden_size": int(hidden_size)},
            "train_seq_len": int(train_seq_len),
        }
    )


def enable_prev_action_conditioning_cfg(cfg, hidden_dim: int):
    cfg.algo.actor.prev_action_conditioning = OmegaConf.create(
        {
            "enabled": True,
            "hidden_dim": int(hidden_dim),
        }
    )


def load_torch_checkpoint(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

class EpisodeStats:
    def __init__(self, in_keys: Sequence[str] = None):
        self.in_keys = in_keys
        self._stats = []
        self._episodes = 0

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        done = tensordict.get(("next", "done"))
        truncated = tensordict.get(("next", "truncated"), None)
        done_or_truncated = (
            (done | truncated) if truncated is not None else done.clone()
        )
        if done_or_truncated.any():
            done_or_truncated = done_or_truncated.squeeze(-1) # [env_num, 1, 1]
            self._episodes += done_or_truncated.sum().item()
            self._stats.extend(
                # [env, n, 1]
                tensordict.select(*self.in_keys)[:, 1:][done_or_truncated[:, :-1]].clone().unbind(0)
            )
    
    def pop(self):
        stats: TensorDictBase = torch.stack(self._stats).to_tensordict()
        self._stats.clear()
        return stats

    def __len__(self):
        return len(self._stats)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    set_seed(cfg.seed)

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    # Project-local output directory with timestamp
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_output_dir = os.path.join(project_root, "checkpoints", f"{cfg.task.name}_{time_str}")
    os.makedirs(run_output_dir, exist_ok=True)
    logging.info(f"Run output dir: {run_output_dir}")

    # TensorBoard writer — run name is the timestamp
    tb_log_dir = os.path.join(project_root, "runs", f"{cfg.task.name}_{time_str}")
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    logging.info(f"TensorBoard log dir: {tb_log_dir}")

    from omni_drones.envs import HideAndSeek, HideAndSeek_envgen, HideAndSeek_deploy  # noqa: F401
    from omni_drones.envs.isaac_env import IsaacEnv
    algos = {
        "ppo": PPOPolicy,
        "ppo_adaptive": PPOAdaptivePolicy,
        "ppo_rnn": PPORNNPolicy,
        "mappo": MAPPOPolicy, 
        "test": Policy
    }

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    transforms = [InitTracker()]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # flatten it to use a MLP encoder instead
    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "state")))
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
    ):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

    if cfg.task.get("history", False):
        transforms.append(History([("agents", "observation")], steps=4))
    
    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    action_controller = None
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform == "velocity":
            from omni_drones.controllers import LeePositionController
            from omni_drones.utils.torchrl.transforms import VelController
            controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
            action_controller = controller
            transform = VelController(controller)
            transforms.append(transform)
        elif action_transform == "attitude":
            from omni_drones.controllers import AttitudeController as Controller
            from omni_drones.utils.torchrl.transforms import AttitudeController
            controller = Controller(9.81, base_env.drone.params).to(base_env.device)
            action_controller = controller
            transform = AttitudeController(controller)
            transforms.append(transform)
        elif action_transform == "rate":
            from omni_drones.controllers import RateController as _RateController
            from omni_drones.utils.torchrl.transforms import RateController
            # from torch.distributions.transforms import TanhTransform
            controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
            action_controller = controller
            transform = RateController(controller)
            # transforms.append(TanhTransform)
            transforms.append(transform)
        elif action_transform == "PIDrate":
            from omni_drones.controllers import PIDRateController as _PIDRateController
            from omni_drones.utils.torchrl.transforms import PIDRateController
            controller = _PIDRateController(cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
            action_controller = controller
            actor_has_tanh = bool(cfg.algo.actor.get("tanh", False))
            transform = PIDRateController(controller, actor_has_tanh=actor_has_tanh)
            # transforms.append(TanhTransform)
            transforms.append(transform)
        elif not action_transform.lower() == "none":
            raise NotImplementedError(f"Unknown action transform: {action_transform}")
    
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    checkpoint_state = None
    if cfg.model_dir is not None:
        checkpoint_state = load_torch_checkpoint(cfg.model_dir, map_location="cpu")
        checkpoint_rnn_hidden = checkpoint_actor_rnn_hidden_size(checkpoint_state)
        if checkpoint_rnn_hidden is not None:
            enable_actor_rnn_cfg(cfg, hidden_size=int(checkpoint_rnn_hidden))
        checkpoint_prev_action_hidden = checkpoint_prev_action_condition_hidden_size(checkpoint_state)
        if checkpoint_prev_action_hidden is not None:
            enable_prev_action_conditioning_cfg(cfg, hidden_dim=int(checkpoint_prev_action_hidden))

    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](
        cfg.algo,
        agent_spec=agent_spec,
        device=cfg.sim.device,
        predictor_net=getattr(base_env, "ot_predictor", None),
        predictor_stride=getattr(base_env, "ot_prediction_stride", 1),
    )

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    if checkpoint_state is not None:
        policy.load_state_dict(checkpoint_state)
        print("Successfully load model!")

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    entropy_schedule_cfg = cfg.algo.get("entropy_schedule", None)
    entropy_schedule_enabled = bool(
        entropy_schedule_cfg is not None and entropy_schedule_cfg.get("enabled", False)
    )
    entropy_success_ema = None
    if entropy_schedule_enabled:
        entropy_hold_coef = float(entropy_schedule_cfg.get("hold_coef", cfg.algo.entropy_coef))
        entropy_release_coef = float(entropy_schedule_cfg.get("release_coef", cfg.algo.entropy_coef))
        entropy_success_threshold = float(entropy_schedule_cfg.get("success_threshold", 0.5))
        entropy_success_target = float(entropy_schedule_cfg.get("success_target", entropy_success_threshold))
        entropy_ema_alpha = float(entropy_schedule_cfg.get("ema_alpha", 0.05))
        entropy_min_env_frames = int(entropy_schedule_cfg.get("min_env_frames", 0))
        if hasattr(policy, "set_entropy_coef"):
            policy.set_entropy_coef(entropy_hold_coef)
        else:
            policy.entropy_coef = entropy_hold_coef
    else:
        entropy_hold_coef = float(cfg.algo.entropy_coef)
        entropy_release_coef = float(cfg.algo.entropy_coef)
        entropy_success_threshold = 0.5
        entropy_success_target = 0.5
        entropy_ema_alpha = 0.05
        entropy_min_env_frames = 0

    def update_entropy_schedule(current_frames: int, train_stats: dict):
        nonlocal entropy_success_ema

        if not entropy_schedule_enabled:
            return {
                "drone/effective_entropy_coef": float(getattr(policy, "entropy_coef", cfg.algo.entropy_coef))
            }

        batch_success = train_stats.get("train/stats.success")
        if batch_success is not None:
            batch_success = float(batch_success)
            if entropy_success_ema is None:
                entropy_success_ema = batch_success
            else:
                entropy_success_ema = (
                    (1.0 - entropy_ema_alpha) * entropy_success_ema
                    + entropy_ema_alpha * batch_success
                )

        effective_entropy_coef = entropy_hold_coef
        ready_to_release = (
            entropy_success_ema is not None
            and current_frames >= entropy_min_env_frames
            and entropy_success_ema >= entropy_success_threshold
        )
        if ready_to_release:
            success_span = max(entropy_success_target - entropy_success_threshold, 1e-6)
            release_progress = min(
                max((entropy_success_ema - entropy_success_threshold) / success_span, 0.0),
                1.0,
            )
            effective_entropy_coef = entropy_hold_coef + (
                entropy_release_coef - entropy_hold_coef
            ) * release_progress

        if hasattr(policy, "set_entropy_coef"):
            policy.set_entropy_coef(effective_entropy_coef)
        else:
            policy.entropy_coef = effective_entropy_coef

        return {
            "drone/effective_entropy_coef": float(effective_entropy_coef),
            "train/stats.success_ema_for_entropy": float(entropy_success_ema or 0.0),
        }

    warmstart_cfg = cfg.algo.get("warmstart", None)
    actor_freeze_cfg = warmstart_cfg.get("actor_freeze", None) if warmstart_cfg is not None else None
    actor_lr_warmup_cfg = warmstart_cfg.get("actor_lr_warmup", None) if warmstart_cfg is not None else None
    critic_lr_warmup_cfg = warmstart_cfg.get("critic_lr_warmup", None) if warmstart_cfg is not None else None
    actor_log_std_schedule_cfg = (
        warmstart_cfg.get("actor_log_std_schedule", None) if warmstart_cfg is not None else None
    )
    kl_reg_cfg = warmstart_cfg.get("kl_reg", None) if warmstart_cfg is not None else None
    anchor_cfg = warmstart_cfg.get("anchor", None) if warmstart_cfg is not None else None

    actor_freeze_enabled = bool(actor_freeze_cfg is not None and actor_freeze_cfg.get("enabled", False))
    actor_freeze_frames = int(actor_freeze_cfg.get("frames", 0)) if actor_freeze_enabled else 0
    curriculum_cfg = getattr(cfg.task, "curriculum", None)
    curriculum_eval_promote_cfg = (
        getattr(curriculum_cfg, "eval_promote", None) if curriculum_cfg is not None else None
    )
    curriculum_eval_promote_enabled = bool(
        curriculum_eval_promote_cfg is not None
        and curriculum_eval_promote_cfg.get("enabled", False)
    )
    curriculum_eval_promote_after_actor_freeze = bool(
        curriculum_eval_promote_cfg.get("after_actor_freeze", True)
        if curriculum_eval_promote_cfg is not None
        else True
    )
    curriculum_eval_success_threshold = float(
        curriculum_eval_promote_cfg.get("success_threshold", 0.80)
        if curriculum_eval_promote_cfg is not None
        else 0.80
    )
    curriculum_eval_max_pursuer_collisions_count = float(
        curriculum_eval_promote_cfg.get("max_pursuer_collisions_count", 1.0)
        if curriculum_eval_promote_cfg is not None
        else 1.0
    )
    curriculum_eval_promote_consecutive = int(
        curriculum_eval_promote_cfg.get("consecutive_evals", 1)
        if curriculum_eval_promote_cfg is not None
        else 1
    )
    curriculum_eval_promote_consecutive = max(curriculum_eval_promote_consecutive, 1)
    curriculum_eval_promote_streak = 0
    curriculum_stage_start_frames = 0

    actor_lr_warmup_enabled = bool(actor_lr_warmup_cfg is not None and actor_lr_warmup_cfg.get("enabled", False))
    actor_lr_warmup_frames = int(actor_lr_warmup_cfg.get("frames", 0)) if actor_lr_warmup_enabled else 0
    actor_lr_warmup_init = float(actor_lr_warmup_cfg.get("lr_init", cfg.algo.actor.lr)) if actor_lr_warmup_enabled else float(cfg.algo.actor.lr)
    actor_lr_warmup_final = float(actor_lr_warmup_cfg.get("lr_final", cfg.algo.actor.lr)) if actor_lr_warmup_enabled else float(cfg.algo.actor.lr)

    critic_lr_warmup_enabled = bool(critic_lr_warmup_cfg is not None and critic_lr_warmup_cfg.get("enabled", False))
    critic_lr_warmup_frames = int(critic_lr_warmup_cfg.get("frames", 0)) if critic_lr_warmup_enabled else 0
    critic_lr_warmup_init = float(critic_lr_warmup_cfg.get("lr_init", cfg.algo.critic.lr)) if critic_lr_warmup_enabled else float(cfg.algo.critic.lr)
    critic_lr_warmup_final = float(critic_lr_warmup_cfg.get("lr_final", cfg.algo.critic.lr)) if critic_lr_warmup_enabled else float(cfg.algo.critic.lr)

    actor_log_std_schedule_enabled = bool(
        actor_log_std_schedule_cfg is not None
        and actor_log_std_schedule_cfg.get("enabled", False)
    )
    actor_log_std_schedule_reset_on_stage = bool(
        actor_log_std_schedule_cfg is not None
        and actor_log_std_schedule_cfg.get("reset_on_curriculum_stage", False)
    )
    actor_log_std_schedule_frames = (
        [int(x) for x in actor_log_std_schedule_cfg.get("frames", [0])]
        if actor_log_std_schedule_enabled
        else [0]
    )
    actor_log_std_schedule_values = (
        [float(x) for x in actor_log_std_schedule_cfg.get("values", [cfg.algo.actor.log_std_init])]
        if actor_log_std_schedule_enabled
        else [float(cfg.algo.actor.log_std_init)]
    )
    if len(actor_log_std_schedule_frames) != len(actor_log_std_schedule_values):
        raise ValueError("warmstart.actor_log_std_schedule frames and values must have the same length.")
    actor_log_std_adaptive_cfg = (
        actor_log_std_schedule_cfg.get("adaptive_release", None)
        if actor_log_std_schedule_cfg is not None
        else None
    )
    actor_log_std_adaptive_enabled = bool(
        actor_log_std_schedule_enabled
        and actor_log_std_adaptive_cfg is not None
        and actor_log_std_adaptive_cfg.get("enabled", False)
    )
    actor_log_std_adaptive_min_frames = int(
        actor_log_std_adaptive_cfg.get("min_frames", 100_000_000)
        if actor_log_std_adaptive_cfg is not None
        else 100_000_000
    )
    actor_log_std_adaptive_success_threshold = float(
        actor_log_std_adaptive_cfg.get("success_threshold", 0.70)
        if actor_log_std_adaptive_cfg is not None
        else 0.70
    )
    actor_log_std_adaptive_consecutive = int(
        actor_log_std_adaptive_cfg.get("consecutive_evals", 1)
        if actor_log_std_adaptive_cfg is not None
        else 1
    )
    actor_log_std_adaptive_consecutive = max(actor_log_std_adaptive_consecutive, 1)
    actor_log_std_adaptive_step_frames = int(
        actor_log_std_adaptive_cfg.get("step_frames", 20_000_000)
        if actor_log_std_adaptive_cfg is not None
        else 20_000_000
    )
    actor_log_std_adaptive_step_frames = max(actor_log_std_adaptive_step_frames, 1)
    actor_log_std_adaptive_init_std = float(
        actor_log_std_adaptive_cfg.get("init_std", math.exp(actor_log_std_schedule_values[0]))
        if actor_log_std_adaptive_cfg is not None
        else math.exp(actor_log_std_schedule_values[0])
    )
    actor_log_std_adaptive_std_increment = float(
        actor_log_std_adaptive_cfg.get("std_increment", 0.01)
        if actor_log_std_adaptive_cfg is not None
        else 0.01
    )
    actor_log_std_adaptive_max_std = float(
        actor_log_std_adaptive_cfg.get("max_std", 0.05)
        if actor_log_std_adaptive_cfg is not None
        else 0.05
    )
    actor_log_std_adaptive_streak = 0
    actor_log_std_adaptive_unlocked_frame = None if actor_log_std_adaptive_enabled else 0

    kl_reg_enabled = bool(kl_reg_cfg is not None and kl_reg_cfg.get("enabled", False))
    kl_target = float(kl_reg_cfg.get("target_kl", 0.01)) if kl_reg_enabled else None
    kl_coef = float(kl_reg_cfg.get("coef", 0.0)) if kl_reg_enabled else 0.0
    kl_hard_stop = bool(kl_reg_cfg.get("hard_stop", False)) if kl_reg_enabled else False
    if hasattr(policy, "set_kl_reg"):
        policy.set_kl_reg(kl_target, kl_coef, kl_hard_stop)

    anchor_enabled = bool(anchor_cfg is not None and anchor_cfg.get("enabled", False))
    anchor_coef_init = float(anchor_cfg.get("coef_init", 0.0)) if anchor_enabled else 0.0
    anchor_coef_final = float(anchor_cfg.get("coef_final", 0.0)) if anchor_enabled else 0.0
    anchor_decay_frames = int(anchor_cfg.get("decay_frames", 0)) if anchor_enabled else 0
    anchor_loss_type = str(anchor_cfg.get("loss_type", "mse")) if anchor_enabled else "mse"
    if anchor_enabled and checkpoint_state is not None and hasattr(policy, "capture_anchor_actor"):
        policy.capture_anchor_actor()
        policy.set_anchor_loss(anchor_coef_init, anchor_loss_type)

    def training_progress(frames: int, iteration: int) -> float:
        if total_frames > 0:
            return min(float(frames) / float(total_frames), 1.0)
        if max_iters > 0:
            return min(float(iteration + 1) / float(max_iters), 1.0)
        return 0.0

    def interpolate_schedule(current_frames: int, frames: list, values: list) -> float:
        if not frames:
            raise ValueError("Schedule frames cannot be empty.")
        if current_frames <= frames[0] or len(frames) == 1:
            return float(values[0])
        for idx in range(1, len(frames)):
            if current_frames <= frames[idx]:
                left_frame = float(frames[idx - 1])
                right_frame = float(frames[idx])
                span = max(right_frame - left_frame, 1.0)
                progress = (float(current_frames) - left_frame) / span
                return float(values[idx - 1]) + (
                    float(values[idx]) - float(values[idx - 1])
                ) * progress
        return float(values[-1])

    def update_warmstart_controls(current_frames: int):
        info = {}
        if hasattr(policy, "set_actor_frozen"):
            actor_frozen = bool(actor_freeze_enabled and current_frames < actor_freeze_frames)
            policy.set_actor_frozen(actor_frozen)
            info["drone/actor_frozen"] = float(actor_frozen)

        if hasattr(policy, "set_actor_lr"):
            if actor_lr_warmup_enabled and actor_lr_warmup_frames > 0 and current_frames < actor_lr_warmup_frames:
                progress = float(current_frames) / float(max(actor_lr_warmup_frames, 1))
                actor_lr = actor_lr_warmup_init + (actor_lr_warmup_final - actor_lr_warmup_init) * progress
            else:
                actor_lr = actor_lr_warmup_final
            policy.set_actor_lr(actor_lr)
            info["drone/effective_actor_lr"] = float(actor_lr)

        if hasattr(policy, "set_critic_lr"):
            if critic_lr_warmup_enabled and critic_lr_warmup_frames > 0 and current_frames < critic_lr_warmup_frames:
                progress = float(current_frames) / float(max(critic_lr_warmup_frames, 1))
                critic_lr = critic_lr_warmup_init + (critic_lr_warmup_final - critic_lr_warmup_init) * progress
            else:
                critic_lr = critic_lr_warmup_final
            policy.set_critic_lr(critic_lr)
            info["drone/effective_critic_lr"] = float(critic_lr)

        if actor_log_std_schedule_enabled and hasattr(policy, "set_actor_log_std"):
            schedule_frames = (
                max(0, current_frames - curriculum_stage_start_frames)
                if actor_log_std_schedule_reset_on_stage
                else current_frames
            )
            if actor_log_std_adaptive_enabled:
                if actor_log_std_adaptive_unlocked_frame is None:
                    release_frames = 0
                    actor_std = actor_log_std_adaptive_init_std
                else:
                    release_frames = max(0, current_frames - int(actor_log_std_adaptive_unlocked_frame))
                    release_steps = release_frames // actor_log_std_adaptive_step_frames
                    actor_std = min(
                        actor_log_std_adaptive_init_std
                        + actor_log_std_adaptive_std_increment * release_steps,
                        actor_log_std_adaptive_max_std,
                    )
                actor_log_std = math.log(max(actor_std, 1e-6))
            else:
                release_frames = schedule_frames
                actor_std = math.exp(actor_log_std_schedule_values[0])
                actor_log_std = interpolate_schedule(
                    schedule_frames,
                    actor_log_std_schedule_frames,
                    actor_log_std_schedule_values,
                )
            applied_log_std = policy.set_actor_log_std(actor_log_std)
            if applied_log_std is not None:
                info["drone/effective_actor_log_std_target"] = float(applied_log_std)
                info["drone/effective_actor_log_std_stage_frames"] = float(schedule_frames)
                info["drone/effective_actor_std_target"] = float(math.exp(applied_log_std))
                if actor_log_std_adaptive_enabled:
                    info["drone/actor_log_std_adaptive_unlocked"] = float(
                        actor_log_std_adaptive_unlocked_frame is not None
                    )
                    info["drone/actor_log_std_adaptive_release_frames"] = float(release_frames)
                    info["drone/actor_log_std_adaptive_gate_streak"] = float(actor_log_std_adaptive_streak)

        if anchor_enabled and hasattr(policy, "set_anchor_loss"):
            if anchor_decay_frames > 0:
                decay_progress = min(float(current_frames) / float(anchor_decay_frames), 1.0)
            else:
                decay_progress = 1.0
            anchor_coef = anchor_coef_init + (anchor_coef_final - anchor_coef_init) * decay_progress
            policy.set_anchor_loss(anchor_coef, anchor_loss_type)
            info["drone/effective_anchor_coef"] = float(anchor_coef)

        if kl_reg_enabled:
            info["drone/kl_target"] = float(kl_target)
            info["drone/kl_coef"] = float(kl_coef)
        return info

    def update_actor_log_std_adaptive_gate(eval_info: dict, current_frames: int):
        nonlocal actor_log_std_adaptive_streak, actor_log_std_adaptive_unlocked_frame

        if not actor_log_std_adaptive_enabled:
            return {}

        success = float(eval_info.get("eval/stats.success", float("nan")))
        if math.isnan(success):
            success = 0.0

        eligible = current_frames >= actor_log_std_adaptive_min_frames
        passed = (
            eligible
            and success >= actor_log_std_adaptive_success_threshold
        )
        if actor_log_std_adaptive_unlocked_frame is None:
            actor_log_std_adaptive_streak = actor_log_std_adaptive_streak + 1 if passed else 0
            if actor_log_std_adaptive_streak >= actor_log_std_adaptive_consecutive:
                actor_log_std_adaptive_unlocked_frame = int(current_frames)
                logging.info(
                    "[Actor std] adaptive release unlocked at %d frames | success=%.3f",
                    current_frames,
                    success,
                )

        return {
            "drone/actor_log_std_adaptive_gate_eligible": float(eligible),
            "drone/actor_log_std_adaptive_gate_pass": float(passed),
            "drone/actor_log_std_adaptive_gate_streak": float(actor_log_std_adaptive_streak),
            "drone/actor_log_std_adaptive_gate_unlocked": float(
                actor_log_std_adaptive_unlocked_frame is not None
            ),
            "drone/actor_log_std_adaptive_gate_success_threshold": float(
                actor_log_std_adaptive_success_threshold
            ),
        }

    def update_curriculum_from_eval(eval_info: dict, current_frames: int):
        nonlocal curriculum_eval_promote_streak, curriculum_stage_start_frames
        nonlocal actor_log_std_adaptive_streak, actor_log_std_adaptive_unlocked_frame

        info = {}
        current_stage = int(getattr(base_env, "curriculum_stage", 0))
        stages = getattr(base_env, "curriculum_stages", [])
        num_stages = len(stages)
        info["drone/curriculum_stage_after_eval"] = float(current_stage + 1)

        if not curriculum_eval_promote_enabled or not hasattr(base_env, "_set_curriculum_stage"):
            return info
        if num_stages <= 1 or current_stage >= num_stages - 1:
            info["drone/curriculum_eval_promote_eligible"] = 0.0
            info["drone/curriculum_eval_promote_pass"] = 0.0
            info["drone/curriculum_eval_promote_streak"] = float(curriculum_eval_promote_streak)
            info["drone/curriculum_eval_promoted"] = 0.0
            return info

        freeze_done = (not curriculum_eval_promote_after_actor_freeze) or (
            current_frames >= actor_freeze_frames
        )
        success = float(eval_info.get("eval/stats.success", float("nan")))
        pursuer_collision_count = float(
            eval_info.get("eval/stats.pursuer_collisions_count", float("inf"))
        )
        if math.isnan(success):
            success = 0.0
        if math.isnan(pursuer_collision_count):
            pursuer_collision_count = float("inf")

        passed = (
            freeze_done
            and success >= curriculum_eval_success_threshold
            and pursuer_collision_count < curriculum_eval_max_pursuer_collisions_count
        )
        curriculum_eval_promote_streak = (
            curriculum_eval_promote_streak + 1 if passed else 0
        )
        promoted = 0.0
        if curriculum_eval_promote_streak >= curriculum_eval_promote_consecutive:
            next_stage = current_stage + 1
            base_env._set_curriculum_stage(next_stage, reset_metrics=True, announce=True)
            current_stage = int(getattr(base_env, "curriculum_stage", next_stage))
            curriculum_stage_start_frames = int(current_frames)
            curriculum_eval_promote_streak = 0
            if actor_log_std_adaptive_enabled and actor_log_std_schedule_reset_on_stage:
                actor_log_std_adaptive_streak = 0
                actor_log_std_adaptive_unlocked_frame = None
            promoted = 1.0
            logging.info(
                "[Curriculum] eval-promoted to stage %d at %d frames | success=%.3f | pursuer_collisions_count=%.3f",
                current_stage + 1,
                current_frames,
                success,
                pursuer_collision_count,
            )

        info["drone/curriculum_eval_promote_eligible"] = float(freeze_done)
        info["drone/curriculum_eval_promote_pass"] = float(passed)
        info["drone/curriculum_eval_promote_streak"] = float(curriculum_eval_promote_streak)
        info["drone/curriculum_eval_promoted"] = float(promoted)
        info["drone/curriculum_eval_success_threshold"] = float(curriculum_eval_success_threshold)
        info["drone/curriculum_eval_max_pursuer_collisions_count"] = float(
            curriculum_eval_max_pursuer_collisions_count
        )
        info["drone/curriculum_stage_after_eval"] = float(current_stage + 1)
        return info

    if hasattr(base_env, "set_training_progress"):
        base_env.set_training_progress(0.0)
    update_warmstart_controls(0)

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        progress: float=1.0,
    ):
        frames = []
        record_eval_video = bool(cfg.get("record_eval_video", False))

        base_env.enable_render(record_eval_video)
        base_env.eval()
        env.eval()
        env.set_seed(seed)
        prev_progress = getattr(base_env, "training_progress", 0.0)
        if hasattr(base_env, "set_training_progress"):
            base_env.set_training_progress(progress)

        from tqdm import tqdm
        t = tqdm(total=base_env.max_episode_length)
        
        def record_frame(*args, **kwargs):
            frame = env.base_env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)
            t.update(2)

        if hasattr(policy, "reset_prev_action_cache"):
            policy.reset_prev_action_cache()

        td = env.reset()
        n_envs = int(base_env.num_envs)
        n_agents = int(base_env.drone.n)
        done_mask = torch.zeros(n_envs, dtype=torch.bool, device=base_env.device)
        action_dim = int(base_env.action_spec[("agents", "action")].shape[-1])
        zero_action = torch.zeros(n_envs, n_agents, action_dim, device=base_env.device)
        eval_stats = []
        latest_stats = None
        for step_idx in range(base_env.max_episode_length):
            if record_eval_video and step_idx % 2 == 0:
                record_frame()
            action_td = policy(td, deterministic=True)
            if done_mask.any() and ("agents", "action") in action_td.keys(True):
                action_td["agents", "action"][done_mask] = zero_action[done_mask]
            td = env.step(action_td)
            td_next = td.get("next")
            latest_stats = td_next.get("stats")
            done_vec = td_next.get("done").reshape(n_envs).bool()
            newly_done = done_vec & ~done_mask
            if newly_done.any():
                eval_stats.extend(latest_stats[newly_done].detach().clone().unbind(0))
                done_mask |= newly_done
            if done_mask.all():
                break
            td = td_next

        if latest_stats is not None and (~done_mask).any():
            eval_stats.extend(latest_stats[~done_mask].detach().clone().unbind(0))
        if hasattr(policy, "reset_prev_action_cache"):
            policy.reset_prev_action_cache()

        base_env.enable_render(not cfg.headless)
        if hasattr(base_env, "set_training_progress"):
            base_env.set_training_progress(prev_progress)
        env.reset()

        traj_stats = torch.stack(eval_stats).to_tensordict().cpu()

        info = {
            "eval/stats." + k: torch.nanmean(v.float()).item() 
            for k, v in traj_stats.items()
        }

        if record_eval_video and len(frames):
            # video_array = torch.stack(frames)
            video_array = np.stack(frames).transpose(0, 3, 1, 2)
            frames.clear()
            info["recording"] = wandb.Video(
                video_array, fps=0.5 / cfg.sim.dt, format="mp4"
            )
        
        return info

    pbar = tqdm(collector)
    env.train()
    base_env.train()
    fps = []
    for i, data in enumerate(pbar):
        # fps.append(collector._fps)
        data_td = data.to_tensordict()
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats(data_td)
        stats = {}

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        info.update(update_entropy_schedule(collector._frames, stats))
        info.update(update_warmstart_controls(collector._frames))

        info.update(policy.train_op(data_td))

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            eval_info = evaluate(progress=training_progress(collector._frames, i))
            info.update(eval_info)
            info.update(update_actor_log_std_adaptive_gate(eval_info, collector._frames))
            info.update(update_curriculum_from_eval(eval_info, collector._frames))
            env.train()
            base_env.train()

        if save_interval > 0 and i % save_interval == 0:
            if hasattr(policy, "state_dict"):
                ckpt_path = os.path.join(run_output_dir, f"checkpoint_{collector._frames}.pt")
                logging.info(f"Save checkpoint to {str(ckpt_path)}")
                torch.save(policy.state_dict(), ckpt_path)

        run.log(info)
        # Log scalars to TensorBoard
        global_step = collector._frames
        for k, v in info.items():
            if isinstance(v, (int, float)):
                tb_writer.add_scalar(k, v, global_step)
        tb_writer.flush()
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames,
        })

        if hasattr(base_env, "set_training_progress"):
            base_env.set_training_progress(training_progress(collector._frames, i))

        if max_iters > 0 and i >= max_iters - 1:
            break 

        # if len(fps) > 50:
        #     fps = np.array(fps)[10:]
        #     print(fps.mean(), fps.std())
        #     exit()
    
    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate(progress=training_progress(collector._frames, i if 'i' in locals() else 0)))
    run.log(info)

    if hasattr(policy, "state_dict"):
        ckpt_path = os.path.join(run_output_dir, "checkpoint_final.pt")
        logging.info(f"Save checkpoint to {str(ckpt_path)}")
        torch.save(policy.state_dict(), ckpt_path)

    wandb.save(os.path.join(run_output_dir, "checkpoint*"))
    wandb.finish()
    tb_writer.close()
    
    simulation_app.close()


if __name__ == "__main__":
    main()
