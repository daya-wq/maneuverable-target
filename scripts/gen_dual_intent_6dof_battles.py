#!/usr/bin/env python3
"""
双层意图 + 6DoF + PN 对抗轨迹生成

需求对应:
1. 每0.5s检查战术意图; 战术意图变化时立即刷新机动方式; 每1s刷新机动意图
2. 目标与追捕方最大速度、最大加速度均为3
3. 双方均为六自由度非线性模型:
   - 目标: 规则驱动的双层意图控制器
   - 追捕方: 比例导引 (PN)
   - 目标初始在左侧, 追捕方初始在右侧
   - 10组随机初始位置, 最近初始距离约8m
   - 生成10张三维轨迹图, 标注起点/终点/最终状态/意图切换时刻
"""

import os
import sys
import math
import time
import types
import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

for name, path in [
    ("omni_drones", "omni_drones"),
    ("omni_drones.learning", "omni_drones/learning"),
    ("omni_drones.envs", "omni_drones/envs"),
    ("omni_drones.envs.hide_and_seek", "omni_drones/envs/hide_and_seek"),
]:
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(PROJECT_ROOT, path)]
    mod.__package__ = name
    sys.modules[name] = mod

from omni_drones.envs.hide_and_seek.dual_intent_target import (
    DualIntentTargetController,
    MANEUVER_NAMES,
    TACTICAL_NAMES,
    point_to_cylinder_distance,
)
from omni_drones.envs.hide_and_seek.proportional_navigation import proportional_navigation_3d
from omni_drones.envs.hide_and_seek.quadrotor_6dof import Quadrotor6DOF
from omni_drones.utils.intent_labels import INTENT_LABELS


DT = 0.01
ARENA_RADIUS = 6.0
MAX_HEIGHT = 5.0
MAX_SPEED = 3.0
MAX_ACCEL = 3.0
CAPTURE_RADIUS = 0.4
GOAL_RADIUS = 0.5
GOAL_HEIGHT = 1.0
GOAL_CENTER = torch.tensor([5.0, 0.0, 2.5], dtype=torch.float32)
MAX_STEPS = 3500
PN_GAIN = 4.0
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "dual_intent_6dof_battles")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.txt")
TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectory_npz")

INTENT_COLORS = [
    "#2ecc71",
    "#3498db",
    "#e74c3c",
    "#9b59b6",
    "#e67e22",
    "#1abc9c",
    "#f39c12",
    "#2980b9",
    "#c0392b",
    "#95a5a6",
]
TACTICAL_COLORS = {
    "Far_Rush": "#16a085",
    "Close_Evade": "#d35400",
    "Goal_Priority": "#2980b9",
    "Goal_Sprint": "#8e44ad",
}
PURSUER_COLORS = ["#d62728", "#1f77b4", "#ff7f0e"]


def project_3d_to_2d(points: np.ndarray, elev: float = 26.0, azim: float = -55.0) -> np.ndarray:
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    cos_a, sin_a = np.cos(azim_rad), np.sin(azim_rad)
    cos_e, sin_e = np.cos(elev_rad), np.sin(elev_rad)

    rz = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0],
    ])
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_e, -sin_e],
        [0.0, sin_e, cos_e],
    ])
    rot = rx @ rz
    rotated = points @ rot.T
    return np.column_stack([rotated[:, 0], rotated[:, 2]])


def draw_arena(ax, elev: float, azim: float):
    theta = np.linspace(0, 2 * np.pi, 100)
    floor = np.column_stack([
        ARENA_RADIUS * np.cos(theta),
        ARENA_RADIUS * np.sin(theta),
        np.zeros_like(theta),
    ])
    ceiling = floor.copy()
    ceiling[:, 2] = MAX_HEIGHT
    floor_2d = project_3d_to_2d(floor, elev, azim)
    ceiling_2d = project_3d_to_2d(ceiling, elev, azim)
    ax.plot(floor_2d[:, 0], floor_2d[:, 1], "--", color="#999", lw=0.8, alpha=0.35)
    ax.plot(ceiling_2d[:, 0], ceiling_2d[:, 1], "--", color="#bbb", lw=0.5, alpha=0.18)

    for angle in [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]:
        pillar = np.array([
            [ARENA_RADIUS * np.cos(angle), ARENA_RADIUS * np.sin(angle), 0.0],
            [ARENA_RADIUS * np.cos(angle), ARENA_RADIUS * np.sin(angle), MAX_HEIGHT],
        ])
        pillar_2d = project_3d_to_2d(pillar, elev, azim)
        ax.plot(pillar_2d[:, 0], pillar_2d[:, 1], "--", color="#bbb", lw=0.5, alpha=0.16)

    goal_theta = np.linspace(0, 2 * np.pi, 80)
    goal_bottom = np.column_stack([
        GOAL_CENTER[0].item() + GOAL_RADIUS * np.cos(goal_theta),
        GOAL_CENTER[1].item() + GOAL_RADIUS * np.sin(goal_theta),
        np.full_like(goal_theta, GOAL_CENTER[2].item() - GOAL_HEIGHT / 2.0),
    ])
    goal_top = goal_bottom.copy()
    goal_top[:, 2] = GOAL_CENTER[2].item() + GOAL_HEIGHT / 2.0
    goal_bottom_2d = project_3d_to_2d(goal_bottom, elev, azim)
    goal_top_2d = project_3d_to_2d(goal_top, elev, azim)
    ax.plot(goal_bottom_2d[:, 0], goal_bottom_2d[:, 1], color="#ff7f50", lw=2.0, alpha=0.85)
    ax.plot(goal_top_2d[:, 0], goal_top_2d[:, 1], color="#ffb347", lw=1.0, alpha=0.45)


def sample_initial_positions(rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    for _ in range(500):
        target = np.array([
            rng.uniform(-4.8, -3.8),
            rng.uniform(-1.6, 1.6),
            rng.uniform(2.0, 3.2),
        ], dtype=np.float32)

        center = np.array([
            target[0] + rng.uniform(7.7, 8.3),
            target[1] + rng.uniform(-0.9, 0.9),
            target[2] + rng.uniform(-0.5, 0.5),
        ], dtype=np.float32)
        if center[0] < 2.0:
            center[0] = 2.0 + rng.uniform(0.0, 0.8)
        if center[0] > 5.4:
            center[0] = 5.4 - rng.uniform(0.0, 0.4)
        center[1] = float(np.clip(center[1], -2.2, 2.2))
        center[2] = float(np.clip(center[2], 1.6, 3.4))

        pursuers = []
        for idx in range(3):
            angle = 2.0 * np.pi * idx / 3.0 + rng.uniform(-0.35, 0.35)
            radius = rng.uniform(0.45, 0.8)
            pos = center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                rng.uniform(-0.25, 0.25),
            ], dtype=np.float32)
            pos[0] = np.clip(pos[0], 1.5, 5.6)
            pos[1] = np.clip(pos[1], -2.8, 2.8)
            pos[2] = np.clip(pos[2], 1.3, 3.8)
            pursuers.append(pos)

        pursuers = np.asarray(pursuers, dtype=np.float32)
        dists = np.linalg.norm(pursuers - target[None, :], axis=1)
        if dists.min() < 7.4 or dists.min() > 8.6:
            continue
        if np.any(pursuers[:, 0] <= 0.5):
            continue
        return torch.from_numpy(target), torch.from_numpy(pursuers)
    raise RuntimeError("Failed to sample valid initial positions after 500 attempts.")

def run_scenario(
    seed: int,
    tactical_check_period: float = 0.5,
    maneuver_update_period: float = 1.0,
) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    target_init, pursuer_init = sample_initial_positions(rng)
    target = Quadrotor6DOF(
        num_envs=1,
        device="cpu",
        max_speed=MAX_SPEED,
        arena_radius=ARENA_RADIUS,
        max_height=MAX_HEIGHT,
        min_height=0.2,
        k_drag=0.2,
    )
    pursuers = [
        Quadrotor6DOF(
            num_envs=1,
            device="cpu",
            max_speed=MAX_SPEED,
            arena_radius=ARENA_RADIUS,
            max_height=MAX_HEIGHT,
            min_height=0.2,
            k_drag=0.2,
            k_R=10.0,
            k_omega=3.0,
        )
        for _ in range(3)
    ]

    target.reset(torch.tensor([0]), pos=target_init.unsqueeze(0), vel=torch.zeros(1, 3))
    for idx, pursuer in enumerate(pursuers):
        pursuer.reset(
            torch.tensor([0]),
            pos=pursuer_init[idx].unsqueeze(0),
            vel=torch.zeros(1, 3),
        )

    controller = DualIntentTargetController(
        num_envs=1,
        device="cpu",
        dt=DT,
        goal_center=GOAL_CENTER,
        goal_radius=GOAL_RADIUS,
        goal_height=GOAL_HEIGHT,
        far_distance=6.0,
        sprint_distance=2.0,
        tactical_check_period=tactical_check_period,
        maneuver_update_period=maneuver_update_period,
        arena_radius=ARENA_RADIUS,
        max_height=MAX_HEIGHT,
        min_height=0.2,
        max_accel=MAX_ACCEL,
    )
    controller.reset(torch.tensor([0]))

    target_pos_hist = []
    target_vel_hist = []
    pursuer_pos_hist = []
    tactical_hist = []
    maneuver_hist = []
    maneuver_mode_hist = []
    event_log = []
    min_dist_hist = []
    status = "TIMEOUT"
    end_step = MAX_STEPS

    target_pos_hist.append(target.pos[0].numpy().copy())
    target_vel_hist.append(target.vel[0].numpy().copy())
    pursuer_pos_hist.append(np.stack([p.pos[0].numpy().copy() for p in pursuers], axis=0))
    tactical_hist.append(int(controller.tactical_intent[0].item()))
    maneuver_hist.append(int(controller.maneuver_intent[0].item()))
    maneuver_mode_hist.append(int(controller.maneuver_mode[0].item()))

    for step in range(MAX_STEPS):
        target_pos = target.pos.clone()
        target_vel = target.vel.clone()
        pursuer_pos = torch.cat([p.pos.clone() for p in pursuers], dim=0).unsqueeze(0)
        pursuer_vel = torch.cat([p.vel.clone() for p in pursuers], dim=0).unsqueeze(0)

        accel_cmd, info = controller.step(target_pos, target_vel, pursuer_pos, pursuer_vel)

        target.step(accel_cmd, dt=DT)
        for idx, pursuer in enumerate(pursuers):
            pursuer_acc = proportional_navigation_3d(
                pursuer.pos,
                pursuer.vel,
                target_pos,
                target_vel,
                N=PN_GAIN,
                max_accel=MAX_ACCEL,
            )
            pursuer.step(pursuer_acc, dt=DT)

        target_pos_next = target.pos[0].clone()
        target_vel_next = target.vel[0].clone()
        pursuer_pos_next = np.stack([p.pos[0].numpy().copy() for p in pursuers], axis=0)

        target_pos_hist.append(target_pos_next.numpy().copy())
        target_vel_hist.append(target_vel_next.numpy().copy())
        pursuer_pos_hist.append(pursuer_pos_next)
        tactical_hist.append(int(info["tactical_intent"][0].item()))
        maneuver_hist.append(int(info["maneuver_intent"][0].item()))
        maneuver_mode_hist.append(int(info["maneuver_mode"][0].item()))

        pursuer_pos_next_tensor = torch.from_numpy(pursuer_pos_next)
        min_dist = torch.norm(pursuer_pos_next_tensor - target_pos_next, dim=-1).min().item()
        min_dist_hist.append(min_dist)

        if bool(info["tactical_changed"][0].item()):
            event_log.append(
                {
                    "step": step,
                    "time": step * DT,
                    "kind": "tactical",
                    "label": TACTICAL_NAMES[tactical_hist[-1]],
                }
            )
        if bool(info["maneuver_changed"][0].item()):
            event_log.append(
                {
                    "step": step,
                    "time": step * DT,
                    "kind": "mode",
                    "label": MANEUVER_NAMES[maneuver_mode_hist[-1]],
                }
            )

        if min_dist <= CAPTURE_RADIUS:
            status = "CAPTURED"
            end_step = step + 1
            break

        goal_dist = point_to_cylinder_distance(
            target_pos_next.unsqueeze(0),
            GOAL_CENTER.view(1, 3),
            GOAL_RADIUS,
            GOAL_HEIGHT,
        )[0].item()
        if goal_dist <= 1e-6:
            status = "GOAL_REACHED"
            end_step = step + 1
            break

    return {
        "seed": seed,
        "status": status,
        "end_step": end_step,
        "tactical_check_period": tactical_check_period,
        "maneuver_update_period": maneuver_update_period,
        "target_pos": np.asarray(target_pos_hist),
        "target_vel": np.asarray(target_vel_hist),
        "pursuer_pos": np.asarray(pursuer_pos_hist),
        "tactical_hist": tactical_hist,
        "maneuver_hist": maneuver_hist,
        "maneuver_mode_hist": maneuver_mode_hist,
        "event_log": event_log,
        "min_dist_hist": min_dist_hist,
        "init_target": target_init.numpy().copy(),
        "init_pursuers": pursuer_init.numpy().copy(),
    }


def plot_scenario(data: dict, scenario_id: int, save_path: str, elev: float, azim: float):
    target_pos = data["target_pos"]
    pursuer_pos = data["pursuer_pos"]
    tactical_hist = data["tactical_hist"]
    maneuver_hist = data["maneuver_hist"]
    maneuver_mode_hist = data["maneuver_mode_hist"]
    event_log = data["event_log"]
    duration = (len(target_pos) - 1) * DT

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#f8f9fb")
    fig.patch.set_facecolor("white")
    draw_arena(ax, elev=elev, azim=azim)

    target_2d = project_3d_to_2d(target_pos, elev=elev, azim=azim)
    for idx in range(len(target_pos) - 1):
        hist_idx = min(idx, len(maneuver_hist) - 1)
        color = INTENT_COLORS[maneuver_hist[hist_idx]]
        ax.plot(
            target_2d[idx:idx + 2, 0],
            target_2d[idx:idx + 2, 1],
            color=color,
            linewidth=2.8,
            alpha=0.92,
            solid_capstyle="round",
        )

    ax.plot(
        target_2d[0, 0],
        target_2d[0, 1],
        "^",
        color="black",
        ms=13,
        zorder=8,
        markeredgecolor="white",
        markeredgewidth=1.4,
    )
    ax.annotate(
        "Target Start",
        (target_2d[0, 0], target_2d[0, 1]),
        xytext=(8, -16),
        textcoords="offset points",
        fontsize=9,
        color="black",
    )
    ax.plot(
        target_2d[-1, 0],
        target_2d[-1, 1],
        "v",
        color="black",
        ms=13,
        zorder=8,
        markeredgecolor="white",
        markeredgewidth=1.4,
    )
    ax.annotate(
        "Target End",
        (target_2d[-1, 0], target_2d[-1, 1]),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=9,
        color="black",
    )

    for pursuer_idx in range(3):
        p_traj = pursuer_pos[:, pursuer_idx, :]
        p_2d = project_3d_to_2d(p_traj, elev=elev, azim=azim)
        ax.plot(
            p_2d[:, 0],
            p_2d[:, 1],
            "--",
            color=PURSUER_COLORS[pursuer_idx],
            lw=1.4,
            alpha=0.68,
        )
        ax.plot(
            p_2d[0, 0],
            p_2d[0, 1],
            "o",
            color=PURSUER_COLORS[pursuer_idx],
            ms=9,
            zorder=7,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
        ax.annotate(
            f"P{pursuer_idx + 1} Start",
            (p_2d[0, 0], p_2d[0, 1]),
            xytext=(6, -10 - 10 * pursuer_idx),
            textcoords="offset points",
            fontsize=8,
            color=PURSUER_COLORS[pursuer_idx],
        )
        ax.plot(
            p_2d[-1, 0],
            p_2d[-1, 1],
            "s",
            color=PURSUER_COLORS[pursuer_idx],
            ms=7,
            zorder=7,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        ax.annotate(
            f"P{pursuer_idx + 1} End",
            (p_2d[-1, 0], p_2d[-1, 1]),
            xytext=(6, 2 + 9 * pursuer_idx),
            textcoords="offset points",
            fontsize=8,
            color=PURSUER_COLORS[pursuer_idx],
        )

    max_events_to_draw = 14
    for event_idx, event in enumerate(event_log[:max_events_to_draw]):
        pt = target_2d[min(event["step"], len(target_2d) - 1)]
        is_tactical = event["kind"] == "tactical"
        marker = "*" if is_tactical else "o"
        edge_color = (
            TACTICAL_COLORS.get(event["label"], "#333333")
            if is_tactical
            else INTENT_COLORS[maneuver_hist[min(event["step"], len(maneuver_hist) - 1)]]
        )
        ax.plot(
            pt[0],
            pt[1],
            marker,
            color=edge_color,
            ms=9 if is_tactical else 5,
            zorder=9,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        offset_x = 8 if event_idx % 2 == 0 else -72
        offset_y = 10 + 12 * (event_idx % 3)
        prefix = "T" if is_tactical else "B"
        label = event["label"]
        ax.annotate(
            f"{prefix}@{event['time']:.1f}s\n{label}",
            (pt[0], pt[1]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.78, edgecolor="#d0d0d0"),
            color=edge_color if is_tactical else "#333333",
        )

    status_color = {
        "CAPTURED": "#d62728",
        "GOAL_REACHED": "#2ca02c",
        "TIMEOUT": "#444444",
    }[data["status"]]
    final_marker = target_2d[-1]
    ax.annotate(
        data["status"],
        (final_marker[0], final_marker[1]),
        xytext=(16, -26),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color=status_color,
        arrowprops=dict(arrowstyle="->", color=status_color, lw=1.2),
    )

    tactical_switches = sum(
        1 for idx in range(1, len(tactical_hist)) if tactical_hist[idx] != tactical_hist[idx - 1]
    )
    intent_switches = sum(
        1 for idx in range(1, len(maneuver_hist)) if maneuver_hist[idx] != maneuver_hist[idx - 1]
    )
    bias_switches = sum(
        1 for idx in range(1, len(maneuver_mode_hist)) if maneuver_mode_hist[idx] != maneuver_mode_hist[idx - 1]
    )
    title = (
        f"Scenario {scenario_id:02d} | Seed {data['seed']} | {data['status']}\n"
        f"Duration {duration:.2f}s | Init nearest dist {min(np.linalg.norm(data['init_pursuers'] - data['init_target'][None, :], axis=1)):.2f}m | "
        f"Min dist {min(data['min_dist_hist']):.2f}m | Tactical switches {tactical_switches} | Intent switches {intent_switches} | Bias switches {bias_switches}"
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    summary_text = (
        f"Target rules: tactical check {data['tactical_check_period']:.1f}s, maneuver update {data['maneuver_update_period']:.1f}s\n"
        f"v_max = {MAX_SPEED:.1f} m/s, a_max = {MAX_ACCEL:.1f} m/s²\n"
        f"Final tactical = {TACTICAL_NAMES[tactical_hist[-1]]}\n"
        f"Final intent = {INTENT_LABELS[maneuver_hist[-1]]}\n"
        f"Final bias mode = {MANEUVER_NAMES[maneuver_mode_hist[-1]]}"
    )
    ax.text(
        0.02,
        0.02,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=0.84, edgecolor="#d0d0d0"),
    )

    legend_items = [
        Line2D([0], [0], color="black", lw=2.8, label="Target trajectory"),
        Line2D([0], [0], color="black", marker="^", lw=0, ms=9, label="Target start"),
        Line2D([0], [0], color="black", marker="v", lw=0, ms=9, label="Target end"),
        Line2D([0], [0], color="#666666", marker="*", lw=0, ms=10, label="Tactical change"),
        Line2D([0], [0], color="#666666", marker="o", lw=0, ms=5, label="Bias change"),
    ]
    for idx in range(3):
        legend_items.append(
            Line2D([0], [0], color=PURSUER_COLORS[idx], lw=1.4, ls="--", label=f"Pursuer {idx + 1}")
        )
    for color, label in zip(INTENT_COLORS, INTENT_LABELS):
        legend_items.append(mpatches.Patch(color=color, alpha=0.9, label=label))
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.92, ncol=2)

    ax.set_aspect("equal")
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_scenario_npz(data: dict, scenario_id: int, traj_dir: str) -> str:
    save_path = os.path.join(traj_dir, f"battle_dual_intent_{scenario_id:02d}.npz")
    np.savez_compressed(
        save_path,
        seed=np.asarray([data["seed"]], dtype=np.int64),
        status=np.asarray([data["status"]]),
        end_step=np.asarray([data["end_step"]], dtype=np.int64),
        dt=np.asarray([DT], dtype=np.float32),
        tactical_check_period=np.asarray([data["tactical_check_period"]], dtype=np.float32),
        maneuver_update_period=np.asarray([data["maneuver_update_period"]], dtype=np.float32),
        arena_radius=np.asarray([ARENA_RADIUS], dtype=np.float32),
        max_height=np.asarray([MAX_HEIGHT], dtype=np.float32),
        goal_center=GOAL_CENTER.numpy().astype(np.float32),
        goal_radius=np.asarray([GOAL_RADIUS], dtype=np.float32),
        goal_height=np.asarray([GOAL_HEIGHT], dtype=np.float32),
        target_pos=data["target_pos"].astype(np.float32),
        target_vel=data["target_vel"].astype(np.float32),
        pursuer_pos=data["pursuer_pos"].astype(np.float32),
        tactical_hist=np.asarray(data["tactical_hist"], dtype=np.int64),
        maneuver_hist=np.asarray(data["maneuver_hist"], dtype=np.int64),
        maneuver_mode_hist=np.asarray(data["maneuver_mode_hist"], dtype=np.int64),
        min_dist_hist=np.asarray(data["min_dist_hist"], dtype=np.float32),
        init_target=data["init_target"].astype(np.float32),
        init_pursuers=data["init_pursuers"].astype(np.float32),
    )
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dual-intent 6DoF battle trajectories.")
    parser.add_argument("--tactical-check-period", type=float, default=0.5)
    parser.add_argument("--maneuver-update-period", type=float, default=1.0)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--num-scenarios", type=int, default=50,
                        help="Number of battle scenarios to generate.")
    parser.add_argument("--seed-base", type=int, default=0,
                        help="Master seed. 0 = use current timestamp (different each run).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating plots (faster for large batches).")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    summary_path = os.path.join(results_dir, "summary.txt")
    traj_dir = os.path.join(results_dir, "trajectory_npz")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    # Generate random seeds
    seed_base = args.seed_base if args.seed_base != 0 else int(time.time())
    rng_master = np.random.default_rng(seed_base)
    seeds = rng_master.integers(0, 1_000_000, size=args.num_scenarios).tolist()

    summary_lines = []
    print("=" * 78)
    print(f"  Dual-Intent 6DoF Battles | target rules + PN pursuers | {args.num_scenarios} scenarios")
    print(f"  seed_base={seed_base}")
    print("=" * 78)
    print(
        f"  tactical_check_period={args.tactical_check_period:.2f}s | "
        f"maneuver_update_period={args.maneuver_update_period:.2f}s"
    )

    for scenario_idx, seed in enumerate(seeds, start=1):
        print(f"[{scenario_idx}/{args.num_scenarios}] seed={seed} ...")
        data = run_scenario(
            seed,
            tactical_check_period=args.tactical_check_period,
            maneuver_update_period=args.maneuver_update_period,
        )
        if not args.no_plot:
            save_path = os.path.join(results_dir, f"battle_dual_intent_{scenario_idx:02d}.png")
            plot_scenario(
                data,
                scenario_id=scenario_idx,
                save_path=save_path,
                elev=24.0 + (scenario_idx % 3) * 3.0,
                azim=-62.0 + scenario_idx * 6.0,
            )
        npz_path = save_scenario_npz(data, scenario_idx, traj_dir)
        tactical_switches = sum(
            1
            for idx in range(1, len(data["tactical_hist"]))
            if data["tactical_hist"][idx] != data["tactical_hist"][idx - 1]
        )
        maneuver_switches = sum(
            1
            for idx in range(1, len(data["maneuver_hist"]))
            if data["maneuver_hist"][idx] != data["maneuver_hist"][idx - 1]
        )
        bias_switches = sum(
            1
            for idx in range(1, len(data["maneuver_mode_hist"]))
            if data["maneuver_mode_hist"][idx] != data["maneuver_mode_hist"][idx - 1]
        )
        line = (
            f"{scenario_idx:02d} | seed={seed} | status={data['status']:<12} | "
            f"duration={(len(data['target_pos']) - 1) * DT:6.2f}s | "
            f"init_nearest={min(np.linalg.norm(data['init_pursuers'] - data['init_target'][None, :], axis=1)):5.2f}m | "
            f"min_dist={min(data['min_dist_hist']):5.2f}m | "
            f"tactical_switches={tactical_switches:2d} | intent_switches={maneuver_switches:3d} | bias_switches={bias_switches:2d} | "
            f"traj={os.path.basename(npz_path)}"
        )
        summary_lines.append(line)
        print("  " + line)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"seed_base={seed_base}\n")
        f.write("\n".join(summary_lines) + "\n")

    print("-" * 78)
    print(f"Saved {args.num_scenarios} trajectories to {traj_dir}")
    print(f"Saved summary to {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
