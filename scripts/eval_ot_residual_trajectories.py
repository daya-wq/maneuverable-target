#!/usr/bin/env python3
"""Evaluate an OT-residual checkpoint and save 3D trajectory diagnostics."""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.learning import MAPPOPolicy
from omni_drones.utils.torchrl import AgentSpec
from omni_drones.utils.torchrl.transforms import (
    FromDiscreteAction,
    FromMultiDiscreteAction,
    History,
    ravel_composite,
)


_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model_dir", required=True)
_parser.add_argument("--out_dir", default="")
_parser.add_argument("--num_envs", type=int, default=32)
_parser.add_argument("--max_steps", type=int, default=1200)
_parser.add_argument("--seed", type=int, default=20260617)
_parser.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=True)
_parser.add_argument("--plot_envs", type=int, default=8)
_parser.add_argument("--action_mode", choices=["policy", "ot_only"], default="policy")
_args, _hydra_argv = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _hydra_argv


STRATEGIC_NAMES = ["Far_Rush", "Close_Evade", "Goal_Priority", "Goal_Sprint"]
TACTICAL_NAMES = STRATEGIC_NAMES
MANEUVER_NAMES = [
    "forward",
    "left",
    "right",
    "up",
    "down",
    "left_up",
    "left_down",
    "right_up",
    "right_down",
]
PURSUER_COLORS = ["#d1495b", "#00798c", "#edae49"]


def load_torch_checkpoint(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_has_bc_aux(state_dict) -> bool:
    if bool(state_dict.get("actor_has_bc_aux", False)):
        return True
    actor_params = state_dict.get("actor_params", None)
    if actor_params is None:
        return False
    keys = actor_params.to_tensordict().keys(True, True)
    return any("aux_heads" in "/".join(map(str, key)) for key in keys)


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


def point_in_cylinder_np(points: np.ndarray, center: np.ndarray, radius: float, height: float) -> np.ndarray:
    radial = np.linalg.norm(points[..., :2] - center[:2], axis=-1)
    vertical_ok = np.abs(points[..., 2] - center[2]) <= height / 2.0
    return (radial <= radius) & vertical_ok


def point_to_cylinder_distance_np(points: np.ndarray, center: np.ndarray, radius: float, height: float) -> np.ndarray:
    radial = np.linalg.norm(points[..., :2] - center[:2], axis=-1)
    radial_delta = np.maximum(radial - radius, 0.0)
    vertical_delta = np.maximum(np.abs(points[..., 2] - center[2]) - height / 2.0, 0.0)
    return np.sqrt(radial_delta * radial_delta + vertical_delta * vertical_delta)


def clip_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_norm) / (norm + 1e-6), max=1.0)
    return x * scale


def make_env_and_policy(cfg, checkpoint_state):
    from omni_drones.envs import HideAndSeek, HideAndSeek_deploy, HideAndSeek_envgen  # noqa: F401
    from omni_drones.envs.isaac_env import IsaacEnv

    if checkpoint_has_bc_aux(checkpoint_state):
        cfg.algo.actor.bc_aux.enabled = True
    rnn_hidden = checkpoint_actor_rnn_hidden_size(checkpoint_state)
    if rnn_hidden is not None:
        enable_actor_rnn_cfg(cfg, hidden_size=int(rnn_hidden))
    prev_action_hidden = checkpoint_prev_action_condition_hidden_size(checkpoint_state)
    if prev_action_hidden is not None:
        enable_prev_action_conditioning_cfg(cfg, hidden_dim=int(prev_action_hidden))

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=True)
    transforms = [InitTracker()]
    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "state")))
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
    ):
        transforms.append(
            ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1)
        )
    if cfg.task.get("history", False):
        transforms.append(History([("agents", "observation")], steps=4))

    action_transform = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            transforms.append(FromMultiDiscreteAction(int(action_transform.split(":")[1])))
        elif action_transform.startswith("discrete"):
            transforms.append(FromDiscreteAction(int(action_transform.split(":")[1])))
        elif action_transform.lower() != "none":
            raise NotImplementedError(
                f"This diagnostic script expects OT residual direct actions, got {action_transform!r}."
            )

    env = TransformedEnv(base_env, Compose(*transforms)).eval()
    env.set_seed(int(_args.seed))
    base_env.enable_render(False)
    base_env.eval()
    if hasattr(base_env, "set_training_progress"):
        base_env.set_training_progress(1.0)

    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = MAPPOPolicy(
        cfg.algo,
        agent_spec=agent_spec,
        device=cfg.sim.device,
        TP_net=base_env.TP,
        predictor_net=getattr(base_env, "ot_predictor", None),
        predictor_stride=getattr(base_env, "ot_prediction_stride", 1),
    )
    policy.load_state_dict(checkpoint_state)
    if hasattr(policy, "actor"):
        policy.actor.eval()
    if hasattr(policy, "critic"):
        policy.critic.eval()
    if hasattr(policy, "predictor_net") and policy.predictor_net is not None:
        policy.predictor_net.eval()
    if hasattr(policy, "reset_prev_action_cache"):
        policy.reset_prev_action_cache()
    return env, base_env, policy


@torch.no_grad()
def collect_rollout(env, base_env, policy, max_steps: int, deterministic: bool, action_mode: str):
    td = env.reset()
    n_envs = int(base_env.num_envs)
    n_agents = int(base_env.drone.n)
    action_dim = int(base_env.action_spec[("agents", "action")].shape[-1])
    zero_action = torch.zeros(n_envs, n_agents, action_dim, device=base_env.device)
    done_mask = torch.zeros(n_envs, dtype=torch.bool, device=base_env.device)

    records = []
    done_records = [None for _ in range(n_envs)]
    for step_idx in range(max_steps):
        action_td = policy(td, deterministic=deterministic)
        if action_mode == "ot_only":
            action_td[("agents", "action")].zero_()
        if done_mask.any():
            action_td[("agents", "action")][done_mask] = zero_action[done_mask]

        td = env.step(action_td)
        td_next = td.get("next")
        info = td_next.get("info")
        stats = td_next.get("stats")

        drone_state = info["drone_state"].detach().clone()
        target_state = info["target_state"].detach().clone()
        action = action_td[("agents", "action")].detach().clone()
        residual_acc = action.clamp(-1.0, 1.0) * float(base_env.ot_residual_accel_limit)
        ot_acc = getattr(base_env, "ot_base_acc_cmd", torch.zeros_like(residual_acc)).detach().clone()
        cmd_acc = clip_norm(ot_acc + residual_acc, float(base_env.pursuer_accel_limit))
        desired_acc = getattr(base_env, "pursuer_desired_acc_cmd", cmd_acc).detach().clone()
        floor_safe_z = getattr(
            base_env,
            "pursuer_safe_z",
            torch.zeros(n_envs, n_agents, 1, device=base_env.device),
        ).detach().clone()
        floor_active = getattr(
            base_env,
            "pursuer_floor_active",
            torch.zeros(n_envs, n_agents, 1, device=base_env.device),
        ).detach().clone()

        record = {
            "step": step_idx,
            "drone_state": drone_state.cpu().numpy(),
            "target_state": target_state.cpu().numpy(),
            "action": action.cpu().numpy(),
            "residual_acc": residual_acc.cpu().numpy(),
            "ot_acc": ot_acc.cpu().numpy(),
            "cmd_acc": cmd_acc.cpu().numpy(),
            "desired_acc": desired_acc.cpu().numpy(),
            "floor_safe_z": floor_safe_z.cpu().numpy(),
            "floor_active": floor_active.cpu().numpy(),
            "ot_barycenters": getattr(base_env, "ot_barycenters", torch.zeros_like(residual_acc)).detach().cpu().numpy(),
            "ot_ref_dist": getattr(base_env, "ot_ref_dist", torch.zeros(n_envs, n_agents, device=base_env.device)).detach().cpu().numpy(),
            "ot_w2_error": getattr(base_env, "ot_w2_error", torch.zeros(n_envs, 1, device=base_env.device)).detach().cpu().numpy(),
            "ot_mode_coverage": getattr(base_env, "ot_mode_coverage", torch.zeros(n_envs, 1, device=base_env.device)).detach().cpu().numpy(),
            "ot_line_block_score": getattr(base_env, "ot_line_block_score", torch.zeros(n_envs, 1, device=base_env.device)).detach().cpu().numpy(),
            "strategic_probs": getattr(base_env, "ot_strategic_probs", torch.zeros(n_envs, 4, device=base_env.device)).detach().cpu().numpy(),
            "strategic_label": getattr(base_env, "ot_predictor_strategic_label", torch.zeros(n_envs, 1, device=base_env.device)).detach().cpu().numpy(),
            "target_tactical_intent": getattr(base_env, "target_tactical_intent", torch.zeros(n_envs, device=base_env.device)).reshape(n_envs, 1).detach().cpu().numpy(),
            "target_maneuver_id": getattr(base_env, "target_maneuver_id", torch.zeros(n_envs, device=base_env.device)).reshape(n_envs, 1).detach().cpu().numpy(),
            "done": td_next.get("done").reshape(n_envs).detach().cpu().numpy(),
            "stats_success": stats["success"].reshape(n_envs).detach().cpu().numpy(),
            "stats_goal_reached": stats["goal_reached"].reshape(n_envs).detach().cpu().numpy(),
            "stats_any_landed": stats["any_landed"].reshape(n_envs).detach().cpu().numpy(),
            "stats_collision": stats["collision"].reshape(n_envs).detach().cpu().numpy(),
            "stats_collision_floor": stats["collision_floor"].reshape(n_envs).detach().cpu().numpy(),
            "stats_out_of_arena": stats["out_of_arena"].reshape(n_envs).detach().cpu().numpy(),
            "stats_return": stats["return"].reshape(n_envs).detach().cpu().numpy(),
        }
        records.append(record)

        done_vec = torch.as_tensor(record["done"], dtype=torch.bool, device=base_env.device)
        newly_done = done_vec & ~done_mask
        if newly_done.any():
            drone_pos = drone_state[..., :3]
            target_pos = target_state[..., :3].squeeze(1)
            dist = torch.norm(drone_pos - target_pos.unsqueeze(1), dim=-1)
            min_dist = dist.min(dim=-1).values
            goal_center = base_env.goal_region_center.to(drone_pos.device)
            radial = torch.norm(target_pos[..., :2] - goal_center[:2], dim=-1)
            goal_now = (
                (radial <= float(base_env.goal_region_radius))
                & (torch.abs(target_pos[..., 2] - goal_center[2]) <= float(base_env.goal_region_height) / 2.0)
            )
            drone_rpos = base_env.drone_rpos.detach()
            drone_collision = (torch.norm(drone_rpos, dim=-1) < 2.0 * float(base_env.collision_radius)).float().sum(-1)
            drone_collision_done = (drone_collision > 0).any(dim=1)
            speed = torch.norm(drone_state[..., 7:10], dim=-1)
            floor_collision = (drone_pos[..., 2] < float(base_env.ground_collision_height)).any(dim=1)
            landed = (
                (drone_pos[..., 2] < float(base_env.landed_z_threshold))
                & (speed < float(base_env.landed_speed_threshold))
            ).any(dim=1)
            capture_done = min_dist < float(base_env.catch_radius)
            capture_before_goal = capture_done & ~goal_now & ~floor_collision & ~landed & ~drone_collision_done
            for env_idx in newly_done.nonzero(as_tuple=False).reshape(-1).tolist():
                if bool(capture_before_goal[env_idx]):
                    event = "capture"
                elif bool(goal_now[env_idx]):
                    event = "goal"
                elif bool(floor_collision[env_idx]):
                    event = "floor_collision"
                elif bool(landed[env_idx]):
                    event = "landed"
                elif bool(drone_collision_done[env_idx]):
                    event = "drone_collision"
                else:
                    event = "timeout_or_other"
                done_records[env_idx] = {
                    "env": env_idx,
                    "event": event,
                    "step": step_idx,
                    "min_dist": float(min_dist[env_idx].item()),
                    "goal_now": bool(goal_now[env_idx].item()),
                    "capture_done": bool(capture_done[env_idx].item()),
                    "floor_collision": bool(floor_collision[env_idx].item()),
                    "landed": bool(landed[env_idx].item()),
                    "drone_collision": bool(drone_collision_done[env_idx].item()),
                    "stats_success": float(record["stats_success"][env_idx]),
                    "stats_goal_reached": float(record["stats_goal_reached"][env_idx]),
                    "stats_any_landed": float(record["stats_any_landed"][env_idx]),
                    "stats_collision": float(record["stats_collision"][env_idx]),
                    "stats_collision_floor": float(record["stats_collision_floor"][env_idx]),
                    "stats_out_of_arena": float(record["stats_out_of_arena"][env_idx]),
                    "stats_return": float(record["stats_return"][env_idx]),
                }
            done_mask |= newly_done
        if done_mask.all():
            break
        td = td_next

    for env_idx in range(n_envs):
        if done_records[env_idx] is None:
            done_records[env_idx] = {
                "env": env_idx,
                "event": "not_done",
                "step": len(records) - 1,
                "min_dist": float("nan"),
                "goal_now": False,
                "capture_done": False,
                "floor_collision": False,
                "landed": False,
                "drone_collision": False,
                "stats_success": float("nan"),
                "stats_goal_reached": float("nan"),
                "stats_any_landed": float("nan"),
                "stats_collision": float("nan"),
                "stats_collision_floor": float("nan"),
                "stats_out_of_arena": float("nan"),
                "stats_return": float("nan"),
            }
    return records, done_records


def stack_records(records, key):
    return np.stack([record[key] for record in records], axis=0)


def summarize_rollout(records, done_records, base_env):
    drone_state = stack_records(records, "drone_state")
    target_state = stack_records(records, "target_state")
    target_pos = target_state[..., 0, :3]
    drone_pos = drone_state[..., :3]
    dist = np.linalg.norm(drone_pos - target_pos[:, :, None, :], axis=-1)
    min_dist_series = dist.min(axis=-1)
    goal_center = base_env.goal_region_center.detach().cpu().numpy()
    goal_dist = point_to_cylinder_distance_np(
        target_pos,
        goal_center,
        float(base_env.goal_region_radius),
        float(base_env.goal_region_height),
    )
    ot_ref_dist = stack_records(records, "ot_ref_dist")
    ot_w2_error = stack_records(records, "ot_w2_error")[..., 0]
    mode_coverage = stack_records(records, "ot_mode_coverage")[..., 0]
    line_block = stack_records(records, "ot_line_block_score")[..., 0]
    residual_norm = np.linalg.norm(stack_records(records, "residual_acc"), axis=-1)
    ot_acc_norm = np.linalg.norm(stack_records(records, "ot_acc"), axis=-1)
    cmd_acc_norm = np.linalg.norm(stack_records(records, "cmd_acc"), axis=-1)
    desired_acc_norm = np.linalg.norm(stack_records(records, "desired_acc"), axis=-1)
    floor_active = stack_records(records, "floor_active")[..., 0]
    target_speed = np.linalg.norm(target_state[..., 0, 3:6], axis=-1)
    drone_speed = np.linalg.norm(drone_state[..., 7:10], axis=-1)

    rows = []
    for item in done_records:
        env_idx = int(item["env"])
        end = int(item["step"]) + 1
        end = max(1, min(end, len(records)))
        strategic = stack_records(records[:end], "strategic_probs")[:, env_idx]
        mean_strategic = strategic.mean(axis=0)
        dominant_strategy = STRATEGIC_NAMES[int(np.argmax(mean_strategic))]
        strategic_label = stack_records(records[:end], "strategic_label")[:, env_idx, 0].astype(np.int64)
        strategic_label = np.clip(strategic_label, 0, len(STRATEGIC_NAMES) - 1)
        true_counts = np.bincount(strategic_label, minlength=len(STRATEGIC_NAMES)).astype(np.float64)
        true_frac = true_counts / max(float(true_counts.sum()), 1.0)
        dominant_true_strategy = STRATEGIC_NAMES[int(np.argmax(true_frac))]
        tactical_label = stack_records(records[:end], "target_tactical_intent")[:, env_idx, 0].astype(np.int64)
        tactical_label = np.clip(tactical_label, 0, len(TACTICAL_NAMES) - 1)
        tactical_counts = np.bincount(tactical_label, minlength=len(TACTICAL_NAMES)).astype(np.float64)
        tactical_frac = tactical_counts / max(float(tactical_counts.sum()), 1.0)
        dominant_tactical = TACTICAL_NAMES[int(np.argmax(tactical_frac))]
        maneuver_id = stack_records(records[:end], "target_maneuver_id")[:, env_idx, 0].astype(np.int64)
        maneuver_id = np.clip(maneuver_id, 0, len(MANEUVER_NAMES) - 1)
        maneuver_counts = np.bincount(maneuver_id, minlength=len(MANEUVER_NAMES)).astype(np.float64)
        maneuver_frac = maneuver_counts / max(float(maneuver_counts.sum()), 1.0)
        dominant_maneuver = MANEUVER_NAMES[int(np.argmax(maneuver_frac))]
        rows.append(
            {
                **item,
                "duration_s": end * float(base_env.dt),
                "min_target_dist": float(np.nanmin(min_dist_series[:end, env_idx])),
                "final_target_dist": float(min_dist_series[end - 1, env_idx]),
                "min_goal_dist": float(np.nanmin(goal_dist[:end, env_idx])),
                "final_goal_dist": float(goal_dist[end - 1, env_idx]),
                "mean_ot_ref_dist": float(np.nanmean(ot_ref_dist[:end, env_idx])),
                "mean_ot_w2_error": float(np.nanmean(ot_w2_error[:end, env_idx])),
                "mean_mode_coverage": float(np.nanmean(mode_coverage[:end, env_idx])),
                "mean_line_block_score": float(np.nanmean(line_block[:end, env_idx])),
                "mean_residual_acc_norm": float(np.nanmean(residual_norm[:end, env_idx])),
                "mean_ot_acc_norm": float(np.nanmean(ot_acc_norm[:end, env_idx])),
                "mean_cmd_acc_norm": float(np.nanmean(cmd_acc_norm[:end, env_idx])),
                "mean_desired_acc_norm": float(np.nanmean(desired_acc_norm[:end, env_idx])),
                "mean_floor_active": float(np.nanmean(floor_active[:end, env_idx])),
                "mean_target_speed": float(np.nanmean(target_speed[:end, env_idx])),
                "max_target_speed": float(np.nanmax(target_speed[:end, env_idx])),
                "mean_drone_speed": float(np.nanmean(drone_speed[:end, env_idx])),
                "max_drone_speed": float(np.nanmax(drone_speed[:end, env_idx])),
                "dominant_strategy": dominant_strategy,
                "dominant_true_strategy": dominant_true_strategy,
                "dominant_tactical": dominant_tactical,
                "dominant_maneuver": dominant_maneuver,
                **{f"mean_prob_{name}": float(mean_strategic[i]) for i, name in enumerate(STRATEGIC_NAMES)},
                **{f"true_frac_{name}": float(true_frac[i]) for i, name in enumerate(STRATEGIC_NAMES)},
                **{f"tactical_frac_{name}": float(tactical_frac[i]) for i, name in enumerate(TACTICAL_NAMES)},
                **{f"maneuver_frac_{name}": float(maneuver_frac[i]) for i, name in enumerate(MANEUVER_NAMES)},
            }
        )
    return rows


def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = limits.mean(axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([max(0.0, centers[2] - radius), centers[2] + radius])


def draw_goal_cylinder(ax, center, radius, height):
    theta = np.linspace(0, 2 * np.pi, 80)
    z0 = center[2] - height / 2.0
    z1 = center[2] + height / 2.0
    for z in (z0, z1):
        ax.plot(
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            np.full_like(theta, z),
            color="#2a9d8f",
            linewidth=1.0,
            alpha=0.7,
        )
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x = center[0] + radius * np.cos(ang)
        y = center[1] + radius * np.sin(ang)
        ax.plot([x, x], [y, y], [z0, z1], color="#2a9d8f", linewidth=0.7, alpha=0.4)


def plot_episode(records, summary_row, base_env, out_dir: Path):
    env_idx = int(summary_row["env"])
    end = int(summary_row["step"]) + 1
    end = max(1, min(end, len(records)))
    drone_state = stack_records(records[:end], "drone_state")[:, env_idx]
    target_state = stack_records(records[:end], "target_state")[:, env_idx, 0]
    bary = stack_records(records[:end], "ot_barycenters")[:, env_idx]
    min_dist = np.linalg.norm(drone_state[..., :3] - target_state[:, None, :3], axis=-1).min(axis=-1)
    goal_center = base_env.goal_region_center.detach().cpu().numpy()
    goal_dist = point_to_cylinder_distance_np(
        target_state[:, :3],
        goal_center,
        float(base_env.goal_region_radius),
        float(base_env.goal_region_height),
    )
    ot_ref_dist = stack_records(records[:end], "ot_ref_dist")[:, env_idx]
    strategic = stack_records(records[:end], "strategic_probs")[:, env_idx]
    tactical = stack_records(records[:end], "target_tactical_intent")[:, env_idx, 0]
    maneuver = stack_records(records[:end], "target_maneuver_id")[:, env_idx, 0]
    t = np.arange(end) * float(base_env.dt)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(
        target_state[:, 0],
        target_state[:, 1],
        target_state[:, 2],
        color="black",
        linewidth=2.5,
        label="Target",
    )
    ax.scatter(target_state[0, 0], target_state[0, 1], target_state[0, 2], color="black", marker="^", s=60)
    ax.scatter(target_state[-1, 0], target_state[-1, 1], target_state[-1, 2], color="black", marker="v", s=60)
    for agent_idx, color in enumerate(PURSUER_COLORS[: drone_state.shape[1]]):
        pos = drone_state[:, agent_idx, :3]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, linewidth=1.8, label=f"P{agent_idx}")
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color=color, marker="o", s=45)
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color=color, marker="s", s=45)
        ax.plot(
            bary[:, agent_idx, 0],
            bary[:, agent_idx, 1],
            bary[:, agent_idx, 2],
            color=color,
            linestyle=":",
            linewidth=1.1,
            alpha=0.7,
        )
    draw_goal_cylinder(ax, goal_center, float(base_env.goal_region_radius), float(base_env.goal_region_height))
    theta = np.linspace(0, 2 * np.pi, 120)
    arena = float(base_env.arena_size)
    ax.plot(arena * np.cos(theta), arena * np.sin(theta), np.zeros_like(theta), color="gray", alpha=0.25)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("z / m")
    ax.set_title(
        f"Env {env_idx} | {summary_row['event']} | {summary_row['duration_s']:.2f}s | "
        f"min d={summary_row['min_target_dist']:.2f}m"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=24, azim=-55)
    set_axes_equal(ax)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, min_dist, color="#d1495b", label="min target-pursuer dist")
    ax2.axhline(float(base_env.catch_radius), color="#d1495b", linestyle="--", alpha=0.5, label="catch radius")
    ax2.plot(t, goal_dist, color="#2a9d8f", label="target-goal boundary dist")
    ax2.set_xlabel("time / s")
    ax2.set_ylabel("distance / m")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(t, ot_ref_dist.mean(axis=-1), color="#264653", label="mean OT ref dist")
    for i, name in enumerate(STRATEGIC_NAMES):
        ax3.plot(t, strategic[:, i], linewidth=1.0, alpha=0.75, label=name)
    ax3.step(
        t,
        np.clip(tactical, 0, len(TACTICAL_NAMES) - 1) / max(len(TACTICAL_NAMES) - 1, 1),
        where="post",
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.85,
        label="true tactical / 3",
    )
    ax3.step(
        t,
        np.clip(maneuver, 0, len(MANEUVER_NAMES) - 1) / max(len(MANEUVER_NAMES) - 1, 1),
        where="post",
        color="#9d4edd",
        linewidth=1.0,
        linestyle=":",
        alpha=0.85,
        label="maneuver id / 8",
    )
    ax3.set_xlabel("time / s")
    ax3.set_ylabel("OT dist / prob")
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    save_path = out_dir / f"trajectory_env{env_idx:03d}_{summary_row['event']}.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def write_summary_csv(rows, path: Path):
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    args = _args
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.headless = True
    cfg.wandb.mode = "disabled"
    cfg.task.env.num_envs = int(args.num_envs)
    cfg.task.num_envs = int(args.num_envs)
    if hasattr(cfg, "env"):
        cfg.env.num_envs = int(args.num_envs)
        cfg.env.max_episode_length = int(args.max_steps)
    cfg.task.env.max_episode_length = int(args.max_steps)
    cfg.task.max_episode_length = int(args.max_steps)

    checkpoint_state = load_torch_checkpoint(args.model_dir, map_location="cpu")
    simulation_app = init_simulation_app(cfg)
    env, base_env, policy = make_env_and_policy(cfg, checkpoint_state)

    out_dir = Path(args.out_dir or "results/ot_residual_trajectory_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    records, done_records = collect_rollout(
        env,
        base_env,
        policy,
        max_steps=int(args.max_steps),
        deterministic=bool(args.deterministic),
        action_mode=str(args.action_mode),
    )
    rows = summarize_rollout(records, done_records, base_env)
    csv_path = out_dir / "trajectory_summary.csv"
    write_summary_csv(rows, csv_path)

    npz_path = out_dir / "trajectory_rollout.npz"
    np.savez_compressed(
        npz_path,
        drone_state=stack_records(records, "drone_state"),
        target_state=stack_records(records, "target_state"),
        action=stack_records(records, "action"),
        residual_acc=stack_records(records, "residual_acc"),
        ot_acc=stack_records(records, "ot_acc"),
        cmd_acc=stack_records(records, "cmd_acc"),
        desired_acc=stack_records(records, "desired_acc"),
        floor_safe_z=stack_records(records, "floor_safe_z"),
        floor_active=stack_records(records, "floor_active"),
        ot_barycenters=stack_records(records, "ot_barycenters"),
        ot_ref_dist=stack_records(records, "ot_ref_dist"),
        ot_w2_error=stack_records(records, "ot_w2_error"),
        ot_mode_coverage=stack_records(records, "ot_mode_coverage"),
        ot_line_block_score=stack_records(records, "ot_line_block_score"),
        strategic_probs=stack_records(records, "strategic_probs"),
        strategic_label=stack_records(records, "strategic_label"),
        target_tactical_intent=stack_records(records, "target_tactical_intent"),
        target_maneuver_id=stack_records(records, "target_maneuver_id"),
    )

    event_priority = {
        "capture": 0,
        "goal": 1,
        "floor_collision": 2,
        "landed": 3,
        "drone_collision": 4,
        "timeout_or_other": 5,
        "not_done": 6,
    }
    sorted_rows = sorted(rows, key=lambda r: (event_priority.get(r["event"], 99), r["env"]))
    plot_rows = []
    seen_events = set()
    # Always include at least one example from each terminal class before filling
    # the remaining slots with the earliest captures.
    for row in sorted_rows:
        if row["event"] in seen_events:
            continue
        plot_rows.append(row)
        seen_events.add(row["event"])
        if len(plot_rows) >= int(args.plot_envs):
            break
    if len(plot_rows) < int(args.plot_envs):
        used_envs = {int(row["env"]) for row in plot_rows}
        for row in sorted_rows:
            if int(row["env"]) in used_envs:
                continue
            plot_rows.append(row)
            if len(plot_rows) >= int(args.plot_envs):
                break
    plot_paths = []
    for row in plot_rows[: int(args.plot_envs)]:
        plot_paths.append(plot_episode(records, row, base_env, out_dir))

    counts = {}
    for row in rows:
        counts[row["event"]] = counts.get(row["event"], 0) + 1
    print("\nOT residual trajectory evaluation")
    print(f"model_dir={args.model_dir}")
    print(f"action_mode={args.action_mode}")
    print(f"out_dir={out_dir}")
    print(f"episodes={len(rows)} steps_recorded={len(records)} dt={base_env.dt}")
    print("event_counts=" + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    print(f"csv={csv_path}")
    print(f"npz={npz_path}")
    for path in plot_paths:
        print(f"plot={path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
