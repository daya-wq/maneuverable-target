#!/usr/bin/env python3
"""
Record Isaac Sim videos for rule-vs-rule hide-and-seek battles.

Target:
  - 6DoF UAV dynamics from the Isaac HideAndSeek environment
  - rule-based dual-intent controller already implemented in the env

Pursuers:
  - rule-based proportional navigation converted to velocity commands

Outputs:
  - 5 MP4 videos in results/dual_intent_6dof_battles
  - isaac_rule_vs_rule_summary.txt in the same folder
"""

import argparse
import os
import sys
import types
from pathlib import Path
from typing import List, Tuple


BAD_PYTHONPATH_TOKENS = ("omni.pip.torch", "pip_prebundle")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "dual_intent_6dof_battles"
SUMMARY_PATH = RESULTS_DIR / "isaac_rule_vs_rule_summary.txt"
DEFAULT_SEEDS = [101, 123, 207, 314, 409]
RUNNING_IN_ISAAC_PYTHON = sys.version_info[:2] == (3, 7)


def filter_isaac_pythonpath() -> None:
    def _keep(path: str) -> bool:
        return path and not any(token in path for token in BAD_PYTHONPATH_TOKENS)

    pythonpath_parts = os.environ.get("PYTHONPATH", "").split(":")
    os.environ["PYTHONPATH"] = ":".join([p for p in pythonpath_parts if _keep(p)])
    sys.path[:] = [p for p in sys.path if _keep(p)]

if not RUNNING_IN_ISAAC_PYTHON:
    filter_isaac_pythonpath()
sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torchrl.data as torchrl_data
import torchrl.data.tensor_specs as torchrl_tensor_specs


def patch_torchrl_compat() -> None:
    alias_map = {
        "CompositeSpec": "Composite",
        "BoundedTensorSpec": "Bounded",
        "UnboundedContinuousTensorSpec": "UnboundedContinuous",
        "DiscreteTensorSpec": "Categorical",
        "BinaryDiscreteTensorSpec": "Binary",
    }
    for alias, modern_name in alias_map.items():
        if not hasattr(torchrl_data, alias) and hasattr(torchrl_data, modern_name):
            setattr(torchrl_data, alias, getattr(torchrl_data, modern_name))
        if not hasattr(torchrl_tensor_specs, alias) and hasattr(torchrl_tensor_specs, modern_name):
            setattr(torchrl_tensor_specs, alias, getattr(torchrl_tensor_specs, modern_name))


patch_torchrl_compat()

if "wandb" not in sys.modules:
    sys.modules["wandb"] = types.ModuleType("wandb")

from omni_drones import init_simulation_app

def proportional_navigation_3d(
    pursuer_pos: torch.Tensor,
    pursuer_vel: torch.Tensor,
    target_pos: torch.Tensor,
    target_vel: torch.Tensor,
    N: float = 4.0,
    max_accel: float = 3.0,
) -> torch.Tensor:
    if target_pos.dim() < pursuer_pos.dim():
        target_pos = target_pos.unsqueeze(-2)
        target_vel = target_vel.unsqueeze(-2)

    r = target_pos - pursuer_pos
    r_norm = torch.norm(r, dim=-1, keepdim=True).clamp(min=1e-4)
    r_hat = r / r_norm
    v_rel = target_vel - pursuer_vel
    closing_speed = -(v_rel * r_hat).sum(dim=-1, keepdim=True)
    omega = torch.cross(r, v_rel, dim=-1) / (r_norm.square() + 1e-6)
    a_cmd = N * closing_speed * torch.cross(omega, r_hat, dim=-1)

    is_diverging = (closing_speed < 0).squeeze(-1)
    pursuit_acc = r_hat * max_accel * 0.8
    a_cmd = torch.where(
        is_diverging.unsqueeze(-1).expand_as(a_cmd),
        pursuit_acc,
        a_cmd,
    )

    a_norm = torch.norm(a_cmd, dim=-1, keepdim=True).clamp(min=1e-6)
    return a_cmd * (max_accel / a_norm).clamp(max=1.0)


def pn_velocity_command(
    pursuer_pos: torch.Tensor,
    pursuer_vel: torch.Tensor,
    target_pos: torch.Tensor,
    target_vel: torch.Tensor,
    current_vel_cmd: torch.Tensor,
    N: float = 4.0,
    max_accel: float = 3.0,
    max_speed: float = 3.0,
    dt: float = 0.01,
) -> torch.Tensor:
    a_cmd = proportional_navigation_3d(
        pursuer_pos=pursuer_pos,
        pursuer_vel=pursuer_vel,
        target_pos=target_pos,
        target_vel=target_vel,
        N=N,
        max_accel=max_accel,
    )
    new_vel_cmd = current_vel_cmd + a_cmd * dt
    speed = torch.norm(new_vel_cmd, dim=-1, keepdim=True).clamp(min=1e-6)
    return new_vel_cmd * (max_speed / speed).clamp(max=1.0)


def build_cfg(headless: bool) -> OmegaConf:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "cfg"),
    ):
        cfg = compose(config_name="train")

    OmegaConf.set_struct(cfg, False)
    cfg.headless = headless
    cfg.use_local_usd = True
    cfg.seed = 0
    cfg.algo.use_TP_net = 0
    exp_path = os.environ.get("EXP_PATH")
    if exp_path:
        cfg.app_experience = str(Path(exp_path) / "omni.isaac.sim.python.kit")

    cfg.env.num_envs = 1
    cfg.env.max_episode_length = 2000
    cfg.env.env_spacing = 5

    cfg.viewer.resolution = [1280, 720]
    cfg.viewer.eye = [0.0, -16.0, 10.0]
    cfg.viewer.lookat = [0.5, 0.0, 2.5]

    cfg.task.use_eval = 1
    cfg.task.action_transform = "velocity"
    cfg.task.num_agents = 3
    cfg.task.max_agents = 4
    cfg.task.scenario_flag = "goal_defense"

    cfg.task.arena_size = 6.0
    cfg.task.max_height = 5.0
    cfg.task.goal_region_center = [5.0, 0.0, 2.5]
    cfg.task.goal_region_radius = 0.5
    cfg.task.goal_region_height = 1.0

    cfg.task.v_drone = 3.0
    cfg.task.v_prey = 3.0
    cfg.task.target_accel_limit = 3.0
    cfg.task.target_physics_max_velocity = 5.0

    cfg.task.curriculum.enabled = False
    cfg.task.curriculum.eval_uses_fixed_layout = False
    cfg.task.curriculum.pursuer_spawn_mode = "front_box"
    cfg.task.curriculum.front_box_target_x = [-4.8, -4.1]
    cfg.task.curriculum.front_box_target_y = [-1.4, 1.4]
    cfg.task.curriculum.front_box_drone_x = [3.0, 4.8]
    cfg.task.curriculum.front_box_drone_y = [-1.8, 1.8]
    cfg.task.curriculum.front_box_max_z_diff = 0.8
    cfg.task.curriculum.target_z_jitter = 0.2
    cfg.task.curriculum.drone_z_jitter = 0.2
    cfg.task.curriculum.drone_min_separation = 0.7
    cfg.task.curriculum.max_sample_attempts = 128
    cfg.task.curriculum.stages = [
        {
            "pursuer_target_dist": [7.4, 8.6],
            "pursuer_speed": 3.0,
            "target_speed": 3.0,
        }
    ]
    return cfg


def get_status(stats_td) -> str:
    stats = stats_td[0]
    if float(stats["success"].item()) > 0.5:
        return "CAPTURED"
    if float(stats["goal_reached"].item()) > 0.5:
        return "GOAL_REACHED"
    if float(stats["any_landed"].item()) > 0.5:
        return "LANDED"
    if float(stats["out_of_arena"].item()) > 0.5:
        return "OUT_OF_ARENA"
    return "TERMINATED"


def compute_nearest_distance(drone_pos: torch.Tensor, target_pos: torch.Tensor) -> float:
    dists = torch.norm(drone_pos - target_pos, dim=-1)
    return float(dists.min().item())


def get_initial_obs(base_env) -> Tuple[torch.Tensor, float]:
    td = base_env.reset()
    drone_pos = td["info", "drone_state"][0, :, :3]
    target_pos = td["info", "target_state"][0, 0, :3].unsqueeze(0)
    nearest = compute_nearest_distance(drone_pos, target_pos)
    return td, nearest


def reset_with_distance_window(base_env, seed: int, min_dist: float = 7.2, max_dist: float = 8.8):
    base_env.set_seed(seed)
    last_td = None
    last_nearest = None
    for offset in range(24):
        base_env.set_seed(seed + offset)
        td, nearest = get_initial_obs(base_env)
        last_td, last_nearest = td, nearest
        if min_dist <= nearest <= max_dist:
            return td, nearest, seed + offset
    return last_td, last_nearest, seed + 23


def make_video_writer(path: Path, fps: int):
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=None,
    )


def write_video(path: Path, frames: List) -> None:
    with make_video_writer(path, fps=30) as writer:
        for frame in frames:
            writer.append_data(frame)


def run_episode(base_env, controller, seed: int, frame_stride: int) -> dict:
    td, init_nearest, actual_seed = reset_with_distance_window(base_env, seed)
    frames = [base_env.render(mode="rgb_array")]
    dt = float(base_env.dt)
    max_speed = 3.0
    max_accel = 3.0
    pn_gain = 4.0
    vel_cmd = torch.zeros(1, base_env.num_agents, 3, device=base_env.device)

    tactical_prev = int(base_env.target_tactical_intent[0].item())
    maneuver_prev = int(base_env.target_maneuver_id[0].item())
    tactical_changes = []
    maneuver_changes = []
    min_dist = init_nearest
    status = "TIMEOUT"
    steps_taken = 0

    for step in range(base_env.max_episode_length):
        drone_state = td["info", "drone_state"][..., :13]
        target_state = td["info", "target_state"]
        drone_pos = drone_state[..., :3]
        drone_vel = drone_state[..., 7:10]
        target_pos = target_state[..., :3]
        target_vel = target_state[..., 3:6]

        min_dist = min(min_dist, compute_nearest_distance(drone_pos[0], target_pos[0]))

        vel_cmd = pn_velocity_command(
            pursuer_pos=drone_pos,
            pursuer_vel=drone_vel,
            target_pos=target_pos,
            target_vel=target_vel,
            current_vel_cmd=vel_cmd,
            N=pn_gain,
            max_accel=max_accel,
            max_speed=max_speed,
            dt=dt,
        )

        los = target_pos - drone_pos
        target_yaw = torch.atan2(los[..., 1], los[..., 0]).unsqueeze(-1)
        action = controller(
            drone_state,
            target_vel=vel_cmd,
            target_yaw=target_yaw,
        )
        torch.nan_to_num_(action, 0.0)
        td.set(("agents", "action"), action)

        next_td = base_env.step(td)
        td = next_td["next"]
        steps_taken = step + 1

        tactical_now = int(base_env.target_tactical_intent[0].item())
        if tactical_now != tactical_prev:
            tactical_changes.append((steps_taken * dt, tactical_now))
            tactical_prev = tactical_now

        maneuver_now = int(base_env.target_maneuver_id[0].item())
        if maneuver_now != maneuver_prev:
            maneuver_changes.append((steps_taken * dt, maneuver_now))
            maneuver_prev = maneuver_now

        if steps_taken % frame_stride == 0:
            frames.append(base_env.render(mode="rgb_array"))

        if bool(next_td["next", "done"].any().item()):
            status = get_status(next_td["next", "stats"])
            break
    else:
        status = "TIMEOUT"

    if steps_taken % frame_stride != 0:
        frames.append(base_env.render(mode="rgb_array"))

    final_drone_pos = td["info", "drone_state"][0, :, :3]
    final_target_pos = td["info", "target_state"][0, 0, :3]

    return {
        "frames": frames,
        "status": status,
        "seed": actual_seed,
        "duration": steps_taken * dt,
        "steps": steps_taken,
        "init_nearest": init_nearest,
        "min_dist": min_dist,
        "tactical_switches": tactical_changes,
        "maneuver_switches": maneuver_changes,
        "target_final_pos": final_target_pos.detach().cpu(),
        "drone_final_pos": final_drone_pos.detach().cpu(),
    }


def format_switches(changes: List[Tuple[float, int]]) -> str:
    if not changes:
        return "none"
    return ", ".join(f"{t:.2f}s->{idx}" for t, idx in changes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--headless", action="store_true", default=False)
    args = parser.parse_args()

    if "ISAAC_PATH" not in os.environ or "EXP_PATH" not in os.environ:
        raise RuntimeError(
            "Isaac Sim environment variables are missing. "
            "Please launch after sourcing setup_conda_env.sh."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(headless=args.headless)
    print("Launching SimulationApp...", flush=True)
    simulation_app = init_simulation_app(cfg)
    print("SimulationApp ready, entering env setup...", flush=True)
    base_env = None

    summary_lines = []
    try:
        from omni_drones.controllers import LeePositionController
        from omni_drones.envs import IsaacEnv
        from omni_drones.envs.hide_and_seek.hideandseek import HideAndSeek  # noqa: F401
        print("Imported IsaacEnv and controllers.", flush=True)

        env_class = IsaacEnv.REGISTRY[cfg.task.name]
        base_env = env_class(cfg, headless=cfg.headless)
        print("Base env constructed.", flush=True)
        base_env.enable_render(True)
        controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
        print("Controller ready.", flush=True)

        seeds = DEFAULT_SEEDS[: args.num_videos]
        for idx, seed in enumerate(seeds, start=1):
            print(f"Running episode {idx}/{len(seeds)} with seed {seed}...", flush=True)
            result = run_episode(base_env, controller, seed=seed, frame_stride=args.frame_stride)
            filename = f"isaac_rule_vs_rule_{idx:02d}_{result['status'].lower()}.mp4"
            video_path = RESULTS_DIR / filename
            write_video(video_path, result["frames"])

            summary_lines.append(
                f"{idx:02d} | seed={result['seed']} | status={result['status']} | "
                f"duration={result['duration']:.2f}s | steps={result['steps']} | "
                f"init_nearest={result['init_nearest']:.2f}m | min_dist={result['min_dist']:.2f}m | "
                f"tactical_switches={len(result['tactical_switches'])} [{format_switches(result['tactical_switches'])}] | "
                f"maneuver_switches={len(result['maneuver_switches'])} [{format_switches(result['maneuver_switches'])}] | "
                f"video={filename}"
            )
            print(summary_lines[-1], flush=True)

        SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"Saved summary to {SUMMARY_PATH}", flush=True)
    finally:
        try:
            base_env.close()
        except Exception:
            pass
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
