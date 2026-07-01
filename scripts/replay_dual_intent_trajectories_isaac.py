#!/usr/bin/env python3
"""
Replay saved dual-intent 6DoF battle trajectories in Isaac Sim.

This version intentionally follows the project's original viewport style:
1. use the native HideAndSeek environment viewport renderer
2. keep only UAV models, without helper spheres or solid floor geometry
3. reuse the environment's cylindrical arena and goal-region debug drawing
4. render a few settle frames before capture to reduce flicker/ghosting
"""

import argparse
import os
import sys
import types
from pathlib import Path


BAD_PYTHONPATH_TOKENS = ("omni.pip.torch", "pip_prebundle")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "dual_intent_6dof_battles"
TRAJ_DIR = RESULTS_DIR / "trajectory_npz"
SUMMARY_PATH = RESULTS_DIR / "isaac_replay_summary.txt"
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
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torchrl.data as torchrl_data
import torchrl.data.tensor_specs as torchrl_tensor_specs

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None


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
from omni_drones.utils.torch import euler_to_quaternion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--settle-renders", type=int, default=3)
    return parser.parse_args()


def list_trajectory_files(limit: int):
    files = sorted(TRAJ_DIR.glob("battle_dual_intent_*.npz"))
    if len(files) < limit:
        raise FileNotFoundError(
            f"Need at least {limit} trajectory files in {TRAJ_DIR}, found {len(files)}."
        )
    return files[:limit]


def load_episode(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "seed": int(data["seed"][0]),
        "status": str(data["status"][0]),
        "end_step": int(data["end_step"][0]),
        "dt": float(data["dt"][0]),
        "arena_radius": float(data["arena_radius"][0]),
        "max_height": float(data["max_height"][0]),
        "goal_center": data["goal_center"].astype(np.float32),
        "goal_radius": float(data["goal_radius"][0]),
        "goal_height": float(data["goal_height"][0]),
        "target_pos": data["target_pos"].astype(np.float32),
        "pursuer_pos": data["pursuer_pos"].astype(np.float32),
    }


def build_cfg(headless: bool, width: int, height: int, ref: dict) -> OmegaConf:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "cfg"),
    ):
        cfg = compose(config_name="train")

    OmegaConf.set_struct(cfg, False)
    cfg.headless = headless
    cfg.use_local_usd = False
    cfg.seed = 0
    cfg.algo.use_TP_net = 0

    exp_path = os.environ.get("EXP_PATH")
    if exp_path:
        cfg.app_experience = str(Path(exp_path) / "omni.isaac.sim.python.kit")

    cfg.env.num_envs = 1
    cfg.env.max_episode_length = 2000
    cfg.env.env_spacing = 5

    cfg.viewer.resolution = [width, height]
    cfg.viewer.eye = [0.0, -17.2, 11.4]
    cfg.viewer.lookat = [0.5, 0.0, 2.6]

    cfg.task.use_eval = 1
    cfg.task.action_transform = "velocity"
    cfg.task.num_agents = 3
    cfg.task.max_agents = 4
    cfg.task.scenario_flag = "goal_defense"
    cfg.task.drone_model = "Crazyflie"
    cfg.task.target_drone_model = "Crazyflie"
    cfg.task.target_dynamics_mode = "uav"

    cfg.task.arena_size = float(ref["arena_radius"])
    cfg.task.max_height = float(ref["max_height"])
    cfg.task.goal_region_center = [float(v) for v in ref["goal_center"].tolist()]
    cfg.task.goal_region_radius = float(ref["goal_radius"])
    cfg.task.goal_region_height = float(ref["goal_height"])

    cfg.task.v_drone = 3.0
    cfg.task.v_prey = 3.0
    cfg.task.target_accel_limit = 3.0
    cfg.task.target_physics_max_velocity = 5.0

    cfg.task.curriculum.enabled = False
    cfg.task.curriculum.eval_uses_fixed_layout = False
    return cfg


def compute_yaw_orientations(positions: np.ndarray) -> np.ndarray:
    diffs = np.zeros_like(positions)
    diffs[:-1] = positions[1:] - positions[:-1]
    diffs[-1] = diffs[-2] if len(positions) > 1 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    yaw = np.arctan2(diffs[:, 1], diffs[:, 0])
    for idx in range(1, len(yaw)):
        if np.linalg.norm(diffs[idx, :2]) < 1e-4:
            yaw[idx] = yaw[idx - 1]

    euler = torch.zeros(len(yaw), 3, dtype=torch.float32)
    euler[:, 2] = torch.from_numpy(yaw.astype(np.float32))
    return euler_to_quaternion(euler).cpu().numpy()


def compute_pose_and_velocity_tracks(positions: np.ndarray, dt: float):
    quats = compute_yaw_orientations(positions)
    linear_vel = np.zeros_like(positions)
    linear_vel[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
    if len(positions) > 1:
        linear_vel[0] = (positions[1] - positions[0]) / dt
        linear_vel[-1] = (positions[-1] - positions[-2]) / dt
    vel6 = np.zeros((len(positions), 6), dtype=np.float32)
    vel6[:, :3] = linear_vel.astype(np.float32)
    return quats.astype(np.float32), vel6


def add_overlay(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return frame

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, 600, 92), radius=12, fill=(16, 22, 28, 190))
    draw.text((34, 30), title, fill=(245, 245, 245))
    draw.text((34, 58), subtitle, fill=(210, 220, 230))
    return np.asarray(image)


def make_writer(path: Path, fps: int):
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=None,
    )


def prepare_episode_tracks(episode: dict):
    target_quat, target_vel6 = compute_pose_and_velocity_tracks(episode["target_pos"], episode["dt"])
    pursuer_quat = []
    pursuer_vel6 = []
    for idx in range(episode["pursuer_pos"].shape[1]):
        quat, vel6 = compute_pose_and_velocity_tracks(episode["pursuer_pos"][:, idx, :], episode["dt"])
        pursuer_quat.append(quat)
        pursuer_vel6.append(vel6)
    return {
        "target_quat": target_quat,
        "target_vel6": target_vel6,
        "pursuer_quat": np.stack(pursuer_quat, axis=1),
        "pursuer_vel6": np.stack(pursuer_vel6, axis=1),
    }


def create_replay_drone_prim(path: str, usd_path: str, scale: float):
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.utils.prims import create_prim

    create_prim(
        prim_path=path,
        usd_path=usd_path,
        position=np.array([0.0, 0.0, -10.0], dtype=np.float32),
        scale=np.array([scale, scale, scale], dtype=np.float32),
    )
    prim = XFormPrim(path, name=Path(path).name.lower())
    prim.initialize()
    return prim


def set_env_models_far_away(base_env):
    env_ids = torch.tensor([0], device=base_env.device, dtype=torch.long)
    far_pos = torch.tensor([[[999.0, 999.0, -50.0]]], device=base_env.device)
    far_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=base_env.device)
    base_env.target.set_world_poses(far_pos, far_quat, env_ids)
    base_env.target.set_velocities(torch.zeros(1, 1, 6, device=base_env.device), env_ids)

    drone_pos = far_pos.repeat(1, base_env.num_agents, 1)
    drone_quat = far_quat.repeat(1, base_env.num_agents, 1)
    base_env.drone.set_world_poses(drone_pos, drone_quat, env_ids)
    base_env.drone.set_velocities(torch.zeros(1, base_env.num_agents, 6, device=base_env.device), env_ids)
    base_env.sim._physics_sim_view.flush()


def draw_replay_scene(base_env):
    from omni_drones.envs.hide_and_seek.draw_circle import (
        Float3,
        _carb_float3_add,
        draw_court_circle,
    )

    base_env.draw.clear_lines()

    arena_p1, arena_p2, arena_colors, arena_sizes = draw_court_circle(
        base_env.arena_size,
        base_env.max_height,
        color_edge=(0.86, 0.86, 0.86, 1.0),
        color_wall=(0.86, 0.86, 0.86, 1.0),
        line_size=3.0,
        wall_alpha=0.018,
    )
    arena_p1 = [_carb_float3_add(p, base_env.central_env_pos) for p in arena_p1]
    arena_p2 = [_carb_float3_add(p, base_env.central_env_pos) for p in arena_p2]

    goal_origin = _carb_float3_add(
        base_env.central_env_pos,
        Float3(
            float(base_env.goal_region_center[0].item()),
            float(base_env.goal_region_center[1].item()),
            float((base_env.goal_region_center[2] - base_env.goal_region_height / 2.0).item()),
        ),
    )
    goal_p1, goal_p2, goal_colors, goal_sizes = draw_court_circle(
        base_env.goal_region_radius,
        base_env.goal_region_height,
        color_edge=(1.0, 0.65, 0.20, 0.95),
        color_wall=(1.0, 0.65, 0.20, 0.95),
        line_size=5.0,
        wall_alpha=0.22,
    )
    goal_p1 = [_carb_float3_add(p, goal_origin) for p in goal_p1]
    goal_p2 = [_carb_float3_add(p, goal_origin) for p in goal_p2]

    base_env.draw.draw_lines(
        arena_p1 + goal_p1,
        arena_p2 + goal_p2,
        arena_colors + goal_colors,
        arena_sizes + goal_sizes,
    )


def set_frame_pose(base_env, replay_drones: dict, episode: dict, tracks: dict, step_idx: int, settle_renders: int):
    env_ids = torch.tensor([0], device=base_env.device, dtype=torch.long)
    env_offset = base_env.envs_positions[env_ids].unsqueeze(1)

    drone_pos = torch.from_numpy(episode["pursuer_pos"][step_idx]).to(base_env.device).unsqueeze(0) + env_offset
    drone_quat = torch.from_numpy(tracks["pursuer_quat"][step_idx]).to(base_env.device).unsqueeze(0)

    target_pos = torch.from_numpy(episode["target_pos"][step_idx]).to(base_env.device).view(1, 1, 3) + env_offset
    target_quat = torch.from_numpy(tracks["target_quat"][step_idx]).to(base_env.device).view(1, 1, 4)

    base_env.sim._physics_sim_view.flush()

    replay_drones["target"].set_world_pose(
        position=target_pos[0, 0].detach().cpu().numpy(),
        orientation=target_quat[0, 0].detach().cpu().numpy(),
    )
    for idx, drone in enumerate(replay_drones["pursuers"]):
        drone.set_world_pose(
            position=drone_pos[0, idx].detach().cpu().numpy(),
            orientation=drone_quat[0, idx].detach().cpu().numpy(),
        )

    for _ in range(max(1, settle_renders)):
        draw_replay_scene(base_env)
        base_env.sim.render()


def main() -> int:
    args = parse_args()
    if "ISAAC_PATH" not in os.environ or "EXP_PATH" not in os.environ:
        raise RuntimeError(
            "Isaac Sim environment variables are missing. "
            "Please launch after sourcing setup_conda_env.sh."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    traj_files = list_trajectory_files(args.num_videos)
    episodes = [load_episode(path) for path in traj_files]
    cfg = build_cfg(
        headless=args.headless,
        width=args.width,
        height=args.height,
        ref=episodes[0],
    )

    simulation_app = init_simulation_app(cfg)
    base_env = None
    summary_lines = []

    try:
        import carb
        from omni_drones.envs import IsaacEnv
        from omni_drones.envs.hide_and_seek.hideandseek import HideAndSeek  # noqa: F401
        from omni_drones.robots import ASSET_PATH

        settings = carb.settings.get_settings()
        settings.set_bool("/rtx/post/motionblur/enabled", False)
        settings.set_int("/rtx/post/aa/op", 0)

        env_class = IsaacEnv.REGISTRY[cfg.task.name]
        base_env = env_class(cfg, headless=cfg.headless)
        base_env.enable_render(True)
        base_env.reset()
        set_env_models_far_away(base_env)

        drone_usd = str(Path(ASSET_PATH) / "usd" / "cf2x_pybullet.usd")
        replay_drones = {
            "target": create_replay_drone_prim("/World/ReplayTargetDrone", drone_usd, scale=8.0),
            "pursuers": [
                create_replay_drone_prim(f"/World/ReplayPursuerDrone_{idx+1}", drone_usd, scale=8.0)
                for idx in range(3)
            ],
        }

        for _ in range(4):
            draw_replay_scene(base_env)
            base_env.sim.render()

        for episode_idx, episode in enumerate(episodes, start=1):
            tracks = prepare_episode_tracks(episode)
            video_name = f"isaac_replay_{episode_idx:02d}_{episode['status'].lower()}.mp4"
            video_path = RESULTS_DIR / video_name
            duration = (len(episode["target_pos"]) - 1) * episode["dt"]
            title = f"Scenario {episode_idx:02d} | Seed {episode['seed']} | {episode['status']}"

            with make_writer(video_path, fps=args.fps) as writer:
                for step_idx in range(0, len(episode["target_pos"]), args.frame_stride):
                    set_frame_pose(base_env, replay_drones, episode, tracks, step_idx, args.settle_renders)
                    frame = base_env.render(mode="rgb_array")
                    subtitle = f"time={step_idx * episode['dt']:.2f}s | arena_r={episode['arena_radius']:.1f}m"
                    writer.append_data(add_overlay(frame, title, subtitle))

                final_idx = len(episode["target_pos"]) - 1
                set_frame_pose(base_env, replay_drones, episode, tracks, final_idx, args.settle_renders)
                for _ in range(12):
                    frame = base_env.render(mode="rgb_array")
                    subtitle = f"final state | duration={duration:.2f}s"
                    writer.append_data(add_overlay(frame, title, subtitle))

            summary_line = (
                f"{episode_idx:02d} | seed={episode['seed']} | status={episode['status']} | "
                f"duration={duration:.2f}s | frames={int(np.ceil(len(episode['target_pos']) / args.frame_stride)) + 12} | "
                f"video={video_name}"
            )
            summary_lines.append(summary_line)
            print(summary_line, flush=True)

        SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"Saved replay summary to {SUMMARY_PATH}", flush=True)
    finally:
        try:
            if base_env is not None:
                base_env.close()
        except Exception:
            pass
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
