#!/usr/bin/env python3
"""
OT-guided 3-pursuer vs. optimal-maneuver target: 3D trajectory visualization

生成 10 张三维对抗拦截轨迹图，使用:
  - 目标: 4类战略意图 + 9候选最优机动 + 6DoF 非线性控制链
  - 追捕方: OT-guided 基准控制 + 6DoF 非线性控制链
  - 双方参数与 cfg/task/HideAndSeek.yaml 完全一致

用法:
  conda activate sim
  python scripts/gen_ot_vs_target_3d_trajectories.py
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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── project imports ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

for name, path in [
    ("omni_drones", "omni_drones"),
    ("omni_drones.learning", "omni_drones/learning"),
    ("omni_drones.envs", "omni_drones/envs"),
    ("omni_drones.envs.hide_and_seek", "omni_drones/envs/hide_and_seek"),
    ("omni_drones.utils", "omni_drones/utils"),
]:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [os.path.join(PROJECT_ROOT, path)]
        mod.__package__ = name
        sys.modules[name] = mod

from omni_drones.envs.hide_and_seek.quadrotor_6dof import Quadrotor6DOF
from omni_drones.envs.hide_and_seek.ot_guided_intercept import (
    OTGuidedInterceptor,
    OTGuidanceConfig,
    build_kinematic_gmm,
    STRATEGIC_FAR_RUSH,
    STRATEGIC_CLOSE_EVADE,
    STRATEGIC_GOAL_PRIORITY,
    STRATEGIC_GOAL_SPRINT,
)

# ── constants (matching HideAndSeek.yaml) ────────────────────────────
DT = 0.01                   # simulation dt
ARENA_RADIUS = 6.0
MAX_HEIGHT = 5.0
MAX_SPEED = 3.0
TARGET_ACCEL_LIMIT = 3.5
PURSUER_ACCEL_LIMIT = 8.0
CAPTURE_RADIUS = 0.4
GOAL_CENTER = torch.tensor([4.0, 0.0, 2.5], dtype=torch.float32)
GOAL_RADIUS = 0.5
GOAL_HEIGHT = 1.0
MAX_STEPS = 1200             # same as env max_episode_length

# Target parameters from yaml
TARGET_TACTICAL_FAR_DIST = 10.5
TARGET_GOAL_SPRINT_DIST = 1.0
TARGET_FAR_GOAL_GAIN = 1.0
TARGET_NEAR_GOAL_GAIN = 1.0
TARGET_GOAL_SPRINT_GAIN = 1.25
TARGET_GOAL_SPRINT_REP_SCALE = 0.70
TARGET_GOAL_PRIORITY_GAIN = 1.8
TARGET_GOAL_PRIORITY_REP_SCALE = 0.80
TARGET_NEAREST_REP_COEF = 2.4
TARGET_OTHER_REP_COEF = 1.3
TARGET_MANEUVER_FORCE_COEF = 2.2
TARGET_MANEUVER_FORWARD_SCALE = 0.15
TARGET_MANEUVER_VERTICAL_SCALE = 0.65
TARGET_MANEUVER_SPEED_THRESH = 0.3
TARGET_OPT_EVADE_W = 2.0
TARGET_OPT_GOAL_W = 3.0
TARGET_OPT_LATERAL_W = 0.55
TARGET_OPT_TURN_W = 0.35
TARGET_OPT_HEIGHT_W = 4.0
TARGET_OPT_BOUNDARY_W = 3.0
TARGET_LOOKAHEAD_STEPS = 10
TARGET_LOOKAHEAD_DT = 0.05

# OT parameters from yaml
OT_KP = 5.0
OT_KV = 2.0
OT_BASE_ACCEL_LIMIT = 6.0
OT_SINKHORN_ITERS = 24
OT_SINKHORN_EPS = 2.0

# Prediction stride: OT updates every dt_pred seconds
DT_PRED = 0.05
OT_STRIDE = max(1, int(round(DT_PRED / DT)))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "ot_vs_target_3d_trajectories")

TACTICAL_NAMES = ["Far_Rush", "Close_Evade", "Goal_Priority", "Goal_Sprint"]
MANEUVER_NAMES = [
    "forward", "left", "right", "up", "down",
    "left_up", "left_down", "right_up", "right_down",
]

TACTICAL_COLORS = {
    "Far_Rush": "#16a085",
    "Close_Evade": "#d35400",
    "Goal_Priority": "#2980b9",
    "Goal_Sprint": "#8e44ad",
}
TACTICAL_COLLIST = ["#16a085", "#d35400", "#2980b9", "#8e44ad"]
PURSUER_COLORS = ["#e74c3c", "#3498db", "#f39c12"]


# ═══════════════════════════════════════════════════════════════════════
#  Target controller (optimal maneuver + strategic intent)
# ═══════════════════════════════════════════════════════════════════════

def safe_normalize(v, eps=1e-6):
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def point_to_cylinder_distance(pos, center, radius, height):
    radial = torch.norm(pos[..., :2] - center[..., :2], dim=-1)
    radial_delta = torch.relu(radial - radius)
    vert_delta = torch.relu(torch.abs(pos[..., 2] - center[..., 2]) - height / 2.0)
    return torch.sqrt(radial_delta ** 2 + vert_delta ** 2)


def compute_strategic_intent(target_pos, pursuer_pos, goal_dist, nearest_dist):
    """Determine strategic intent (matching hideandseek.py logic)."""
    if goal_dist <= TARGET_GOAL_SPRINT_DIST:
        return STRATEGIC_GOAL_SPRINT
    if nearest_dist > TARGET_TACTICAL_FAR_DIST:
        return STRATEGIC_FAR_RUSH
    if goal_dist > nearest_dist:
        return STRATEGIC_CLOSE_EVADE
    return STRATEGIC_GOAL_PRIORITY


def boundary_force(pos):
    """Arena boundary repulsion."""
    force = torch.zeros(3)
    xy_dist = torch.norm(pos[:2])
    if xy_dist > ARENA_RADIUS * 0.82:
        margin = max(ARENA_RADIUS - xy_dist, 0.05)
        radial_dir = -pos[:2] / (xy_dist + 1e-5)
        force[:2] = 3.0 * radial_dir / margin
    if pos[2] > MAX_HEIGHT * 0.9:
        force[2] -= 3.0
    if pos[2] < 0.2 + 0.25:
        force[2] += 3.0
    return force


def goal_force_and_dist(pos):
    """Attraction toward goal region."""
    goal = GOAL_CENTER.clone()
    offset = goal - pos
    dist = torch.norm(offset)
    direction = offset / (dist + 1e-5)
    goal_boundary = point_to_cylinder_distance(
        pos.unsqueeze(0), goal.unsqueeze(0), GOAL_RADIUS, GOAL_HEIGHT
    ).item()
    return direction, goal_boundary


def repulsion_forces(target_pos, pursuer_pos_3x3):
    """Compute nearest and other repulsion forces."""
    offsets = target_pos.unsqueeze(0) - pursuer_pos_3x3  # [3, 3]
    dists = torch.norm(offsets, dim=-1)  # [3]
    nearest_idx = dists.argmin().item()
    rep_dirs = offsets / (dists.unsqueeze(-1) + 1e-5)
    rep_mags = rep_dirs / (dists.unsqueeze(-1) + 1e-5)
    nearest_rep = rep_mags[nearest_idx]
    other_rep = rep_mags.sum(dim=0) - nearest_rep
    return nearest_rep, other_rep, dists.min().item()


def build_maneuver_candidates(vel, goal_dir):
    """Build 9 candidate maneuver force vectors."""
    speed = torch.norm(vel)
    if speed < TARGET_MANEUVER_SPEED_THRESH:
        forward = goal_dir.clone()
    else:
        forward = vel / speed
    forward = safe_normalize(forward.unsqueeze(0)).squeeze(0)
    up = torch.tensor([0.0, 0.0, 1.0])
    lateral = torch.cross(up, forward)
    if torch.norm(lateral) < 1e-5:
        lateral = torch.tensor([1.0, 0.0, 0.0])
    lateral = safe_normalize(lateral.unsqueeze(0)).squeeze(0)

    f_comp = TARGET_MANEUVER_FORCE_COEF * TARGET_MANEUVER_FORWARD_SCALE * forward
    l_comp = TARGET_MANEUVER_FORCE_COEF * lateral
    v_comp = TARGET_MANEUVER_FORCE_COEF * TARGET_MANEUVER_VERTICAL_SCALE * up

    candidates = torch.stack([
        f_comp,                 # forward
        l_comp,                 # left
        -l_comp,                # right
        v_comp,                 # up
        -v_comp,                # down
        l_comp + v_comp,        # left_up
        l_comp - v_comp,        # left_down
        -l_comp + v_comp,       # right_up
        -l_comp - v_comp,       # right_down
    ], dim=0)
    return candidates


def evaluate_maneuver_candidates(target_pos, target_vel, pursuer_pos_3x3, goal_dir, candidates):
    """Score each candidate with lookahead simulation."""
    best_score = -1e10
    best_idx = 0
    goal = GOAL_CENTER.clone()

    for i in range(candidates.shape[0]):
        pos_sim = target_pos.clone()
        vel_sim = target_vel.clone()
        force_extra = candidates[i]

        for t in range(TARGET_LOOKAHEAD_STEPS):
            acc = safe_normalize(force_extra.unsqueeze(0)).squeeze(0) * TARGET_ACCEL_LIMIT
            vel_sim = vel_sim + acc * TARGET_LOOKAHEAD_DT
            speed = torch.norm(vel_sim)
            if speed > MAX_SPEED:
                vel_sim = vel_sim / speed * MAX_SPEED
            pos_sim = pos_sim + vel_sim * TARGET_LOOKAHEAD_DT

        # Scores
        p_dists = torch.norm(pursuer_pos_3x3 - pos_sim.unsqueeze(0), dim=-1)
        min_p_dist = p_dists.min().item()
        goal_progress = -(torch.norm(pos_sim - goal) - torch.norm(target_pos - goal)).item()
        lateral_disp = torch.norm(pos_sim[:2] - target_pos[:2]).item()
        vel_before = target_vel / (torch.norm(target_vel) + 1e-5)
        vel_after = vel_sim / (torch.norm(vel_sim) + 1e-5)
        turn_score = (1.0 - torch.dot(vel_before, vel_after).item()) * 0.5
        height_pen = max(0, 0.3 - pos_sim[2].item()) * 5.0 + max(0, pos_sim[2].item() - MAX_HEIGHT + 0.3) * 5.0
        boundary_pen = max(0, torch.norm(pos_sim[:2]).item() - ARENA_RADIUS + 0.5) * 3.0

        score = (
            TARGET_OPT_EVADE_W * min_p_dist
            + TARGET_OPT_GOAL_W * goal_progress
            + TARGET_OPT_LATERAL_W * lateral_disp
            + TARGET_OPT_TURN_W * turn_score
            - TARGET_OPT_HEIGHT_W * height_pen
            - TARGET_OPT_BOUNDARY_W * boundary_pen
        )
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def target_step(target_pos, target_vel, pursuer_pos_3x3):
    """One step of target control: compute desired acceleration."""
    g_dir, g_dist = goal_force_and_dist(target_pos)
    b_force = boundary_force(target_pos)
    n_rep, o_rep, nearest_dist = repulsion_forces(target_pos, pursuer_pos_3x3)

    intent = compute_strategic_intent(target_pos, pursuer_pos_3x3, g_dist, nearest_dist)

    total = b_force.clone()
    maneuver_id = -1

    if intent == STRATEGIC_FAR_RUSH:
        total += TARGET_FAR_GOAL_GAIN * g_dir
    elif intent == STRATEGIC_CLOSE_EVADE:
        # Build and evaluate maneuver candidates
        candidates = build_maneuver_candidates(target_vel, g_dir)
        maneuver_id = evaluate_maneuver_candidates(
            target_pos, target_vel, pursuer_pos_3x3, g_dir, candidates
        )
        total += (
            TARGET_NEAR_GOAL_GAIN * g_dir
            + TARGET_NEAREST_REP_COEF * n_rep
            + TARGET_OTHER_REP_COEF * o_rep
            + candidates[maneuver_id]
        )
    elif intent == STRATEGIC_GOAL_PRIORITY:
        candidates = build_maneuver_candidates(target_vel, g_dir)
        maneuver_id = evaluate_maneuver_candidates(
            target_pos, target_vel, pursuer_pos_3x3, g_dir, candidates
        )
        total += (
            TARGET_GOAL_PRIORITY_GAIN * g_dir
            + TARGET_GOAL_PRIORITY_REP_SCALE * (
                TARGET_NEAREST_REP_COEF * n_rep + TARGET_OTHER_REP_COEF * o_rep
            )
            + candidates[maneuver_id]
        )
    elif intent == STRATEGIC_GOAL_SPRINT:
        candidates = build_maneuver_candidates(target_vel, g_dir)
        maneuver_id = evaluate_maneuver_candidates(
            target_pos, target_vel, pursuer_pos_3x3, g_dir, candidates
        )
        total += (
            TARGET_GOAL_SPRINT_GAIN * g_dir
            + TARGET_GOAL_SPRINT_REP_SCALE * (
                TARGET_NEAREST_REP_COEF * n_rep + TARGET_OTHER_REP_COEF * o_rep
            )
            + candidates[maneuver_id]
        )

    norm = torch.norm(total)
    if norm > 1e-6:
        accel = TARGET_ACCEL_LIMIT * total / norm
    else:
        accel = torch.zeros(3)

    return accel, intent, maneuver_id


# ═══════════════════════════════════════════════════════════════════════
#  OT-guided pursuer control
# ═══════════════════════════════════════════════════════════════════════

def make_ot_interceptor():
    cfg = OTGuidanceConfig(
        sinkhorn_iters=OT_SINKHORN_ITERS,
        sinkhorn_epsilon=OT_SINKHORN_EPS,
        block_distance=0.45,
        goal_distance=0.55,
        evade_forward=0.35,
        evade_side=0.55,
        kp=OT_KP,
        kv=OT_KV,
        base_accel_limit=OT_BASE_ACCEL_LIMIT,
        min_ref_z=0.90,
        hard_floor_z=0.30,
        low_target_ref_margin=0.10,
        traj_mass_base=0.35,
        traj_mass_far=0.15,
        traj_mass_close=0.45,
        channel_forward_fraction=0.78,
        channel_min_forward=2.20,
        channel_lateral_spread=0.65,
        channel_mass_base=0.60,
        channel_mass_far=2.40,
        channel_mass_goal_priority=2.20,
        channel_mass_goal_sprint=3.00,
        ref_forward_floor=1.80,
        ref_lateral_spread=0.65,
        ref_channel_blend=0.70,
        ref_lateral_blend=0.90,
        ref_min_separation=0.85,
        ref_separation_gain=0.70,
        ref_separation_iters=3,
        terminal_forward_min=0.12,
        dt=DT,
    )
    return OTGuidedInterceptor(num_envs=1, num_agents=3, device="cpu", cfg=cfg)


# ═══════════════════════════════════════════════════════════════════════
#  Initial position sampling (matching yaml: front_box)
# ═══════════════════════════════════════════════════════════════════════

def sample_initial_positions(rng):
    """Sample initial positions matching HideAndSeek front_box reset."""
    for _ in range(500):
        # Target: left side of arena
        tx = rng.uniform(-5.4, -4.8)
        ty = rng.uniform(-0.8, 0.8)
        tz = rng.uniform(2.0, 3.0) + rng.uniform(-0.15, 0.15)
        target = np.array([tx, ty, tz], dtype=np.float32)

        # Pursuers: right side of arena (near goal)
        pursuers = []
        for _ in range(3):
            px = rng.uniform(4.3, 5.3)
            py = rng.uniform(-0.8, 0.8)
            pz = tz + rng.uniform(-0.6, 0.6)
            pz = np.clip(pz, 1.5, 4.0)
            pursuers.append(np.array([px, py, pz], dtype=np.float32))

        pursuers = np.stack(pursuers, axis=0)
        dists = np.linalg.norm(pursuers - target[None, :], axis=1)

        # Check min separation between pursuers
        ok = True
        for i in range(3):
            for j in range(i + 1, 3):
                if np.linalg.norm(pursuers[i] - pursuers[j]) < 1.0:
                    ok = False
        if not ok:
            continue

        # Distance should be around 9-10.5m
        if dists.min() < 9.0 or dists.min() > 10.5:
            continue

        return torch.from_numpy(target), torch.from_numpy(pursuers)

    raise RuntimeError("Failed to sample valid initial positions.")


# ═══════════════════════════════════════════════════════════════════════
#  Run one scenario
# ═══════════════════════════════════════════════════════════════════════

def run_scenario(seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    target_init, pursuer_init = sample_initial_positions(rng)

    # Create 6DOF models
    target_quad = Quadrotor6DOF(num_envs=1, device="cpu", max_speed=MAX_SPEED,
                                arena_radius=ARENA_RADIUS, max_height=MAX_HEIGHT,
                                min_height=0.2, k_drag=0.2)
    pursuer_quads = [
        Quadrotor6DOF(num_envs=1, device="cpu", max_speed=MAX_SPEED,
                      arena_radius=ARENA_RADIUS, max_height=MAX_HEIGHT,
                      min_height=0.2, k_drag=0.2, k_R=10.0, k_omega=3.0)
        for _ in range(3)
    ]

    target_quad.reset(torch.tensor([0]), pos=target_init.unsqueeze(0), vel=torch.zeros(1, 3))
    for i, pq in enumerate(pursuer_quads):
        pq.reset(torch.tensor([0]), pos=pursuer_init[i].unsqueeze(0), vel=torch.zeros(1, 3))

    ot_interceptor = make_ot_interceptor()
    ot_interceptor.reset(torch.tensor([0]))

    # History containers
    target_pos_hist = [target_init.numpy().copy()]
    pursuer_pos_hist = [pursuer_init.numpy().copy()]
    tactical_hist = []
    maneuver_hist = []
    min_dist_hist = []
    status = "TIMEOUT"

    # Cached OT outputs
    cached_a_ot = torch.zeros(1, 3, 3)

    for step in range(MAX_STEPS):
        t_pos = target_quad.pos[0].clone()
        t_vel = target_quad.vel[0].clone()
        p_pos = torch.stack([pq.pos[0].clone() for pq in pursuer_quads], dim=0)
        p_vel = torch.stack([pq.vel[0].clone() for pq in pursuer_quads], dim=0)

        # ── Target step ──
        t_accel, intent, man_id = target_step(t_pos, t_vel, p_pos)
        target_quad.step(t_accel.unsqueeze(0), dt=DT)

        tactical_hist.append(intent)
        maneuver_hist.append(man_id)

        # ── OT pursuer step ──
        if step % OT_STRIDE == 0:
            # Build strategic_probs from true intent (simulating perfect prediction)
            s_probs = torch.zeros(1, 4)
            s_probs[0, intent] = 1.0

            # Use kinematic GMM fallback (no trained predictor)
            gmm_w, gmm_m, gmm_v = build_kinematic_gmm(
                t_pos.unsqueeze(0), t_vel.unsqueeze(0),
                GOAL_CENTER.unsqueeze(0), s_probs,
                horizon=10, dt_pred=DT_PRED, max_speed=MAX_SPEED,
            )

            ot_out = ot_interceptor.compute(
                pursuer_pos=p_pos.unsqueeze(0),
                pursuer_vel=p_vel.unsqueeze(0),
                target_pos=t_pos.unsqueeze(0),
                target_vel=t_vel.unsqueeze(0),
                goal_pos=GOAL_CENTER,
                strategic_probs=s_probs,
                gmm_weights=gmm_w,
                gmm_means_world=gmm_m,
                gmm_vars=gmm_v,
            )
            cached_a_ot = ot_out["a_ot"]  # [1, 3, 3]

        for i, pq in enumerate(pursuer_quads):
            pq.step(cached_a_ot[0, i].unsqueeze(0), dt=DT)

        # ── Record ──
        t_pos_new = target_quad.pos[0].numpy().copy()
        p_pos_new = np.stack([pq.pos[0].numpy().copy() for pq in pursuer_quads], axis=0)
        target_pos_hist.append(t_pos_new)
        pursuer_pos_hist.append(p_pos_new)

        dists = np.linalg.norm(p_pos_new - t_pos_new[None, :], axis=1)
        min_dist_hist.append(dists.min())

        # Check termination
        if dists.min() <= CAPTURE_RADIUS:
            status = "CAPTURED"
            break

        g_dist = point_to_cylinder_distance(
            torch.from_numpy(t_pos_new).unsqueeze(0),
            GOAL_CENTER.unsqueeze(0), GOAL_RADIUS, GOAL_HEIGHT,
        ).item()
        if g_dist <= 1e-6:
            status = "GOAL_REACHED"
            break

    return {
        "seed": seed,
        "status": status,
        "target_pos": np.array(target_pos_hist),
        "pursuer_pos": np.array(pursuer_pos_hist),
        "tactical_hist": tactical_hist,
        "maneuver_hist": maneuver_hist,
        "min_dist_hist": min_dist_hist,
    }


# ═══════════════════════════════════════════════════════════════════════
#  3D Plotting
# ═══════════════════════════════════════════════════════════════════════

def draw_cylinder_wireframe(ax, center, radius, height, color, alpha=0.3, n_pts=40):
    """Draw a transparent cylinder wireframe."""
    theta = np.linspace(0, 2 * np.pi, n_pts)
    z_lo = center[2] - height / 2.0
    z_hi = center[2] + height / 2.0
    x_circle = center[0] + radius * np.cos(theta)
    y_circle = center[1] + radius * np.sin(theta)

    ax.plot(x_circle, y_circle, np.full_like(theta, z_lo), color=color, alpha=alpha, lw=1.5)
    ax.plot(x_circle, y_circle, np.full_like(theta, z_hi), color=color, alpha=alpha, lw=1.5)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        ax.plot([x, x], [y, y], [z_lo, z_hi], color=color, alpha=alpha * 0.6, lw=0.8)


def draw_arena_boundary(ax, radius, max_height, n_pts=80):
    """Draw arena boundary circle on floor."""
    theta = np.linspace(0, 2 * np.pi, n_pts)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y, np.zeros_like(theta), '--', color='#999', alpha=0.25, lw=0.8)


def plot_scenario_3d(data, scenario_id, save_path):
    """Generate a dual-panel plot: left=3D view, right=2D top-down (XY)."""
    target_pos = data["target_pos"]
    pursuer_pos = data["pursuer_pos"]
    tactical_hist = data["tactical_hist"]
    maneuver_hist = data["maneuver_hist"]

    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor('#fafbfd')

    # ══════════════════════════════════════════════════════════════════
    #  LEFT: 3D view
    # ══════════════════════════════════════════════════════════════════
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.set_facecolor('#f0f2f5')

    # Arena & goal
    draw_arena_boundary(ax3d, ARENA_RADIUS, MAX_HEIGHT)
    gc = [GOAL_CENTER[0].item(), GOAL_CENTER[1].item(), GOAL_CENTER[2].item()]
    draw_cylinder_wireframe(ax3d, gc, GOAL_RADIUS, GOAL_HEIGHT, color='#ff6b35', alpha=0.5)
    ax3d.text(gc[0], gc[1], gc[2] + GOAL_HEIGHT / 2.0 + 0.2,
              f"GOAL\n(x={gc[0]:.0f})", fontsize=9, color='#ff6b35', ha='center', fontweight='bold')

    n_steps = len(target_pos)
    downsample = max(1, n_steps // 600)

    for i in range(0, n_steps - 1, downsample):
        idx = min(i, len(tactical_hist) - 1)
        intent = tactical_hist[idx] if idx >= 0 else STRATEGIC_FAR_RUSH
        color = TACTICAL_COLLIST[intent]
        seg_end = min(i + downsample + 1, n_steps)
        ax3d.plot(target_pos[i:seg_end, 0], target_pos[i:seg_end, 1], target_pos[i:seg_end, 2],
                  color=color, linewidth=2.8, alpha=0.9, solid_capstyle='round')

    ax3d.scatter(*target_pos[0], marker='^', s=120, c='black', edgecolors='white', linewidths=1.2, zorder=10)
    ax3d.scatter(*target_pos[-1], marker='v', s=120, c='black', edgecolors='white', linewidths=1.2, zorder=10)

    for pi in range(3):
        p_traj = pursuer_pos[:, pi, :]
        ax3d.plot(p_traj[::downsample, 0], p_traj[::downsample, 1], p_traj[::downsample, 2],
                  '--', color=PURSUER_COLORS[pi], lw=1.6, alpha=0.7)
        ax3d.scatter(*p_traj[0], marker='o', s=80, c=PURSUER_COLORS[pi], edgecolors='white', linewidths=1.0, zorder=9)
        ax3d.scatter(*p_traj[-1], marker='s', s=60, c=PURSUER_COLORS[pi], edgecolors='white', linewidths=1.0, zorder=9)
        ax3d.text(p_traj[0, 0], p_traj[0, 1], p_traj[0, 2] + 0.2,
                  f"P{pi+1}", fontsize=8, color=PURSUER_COLORS[pi], ha='center')

    # Tactical change markers
    change_count = 0
    for i in range(1, len(tactical_hist)):
        if tactical_hist[i] != tactical_hist[i - 1]:
            t_idx = min(i, n_steps - 1)
            ax3d.scatter(target_pos[t_idx, 0], target_pos[t_idx, 1], target_pos[t_idx, 2],
                         marker='*', s=150, c=TACTICAL_COLLIST[tactical_hist[i]],
                         edgecolors='white', linewidths=0.8, zorder=11)
            change_count += 1
            if change_count <= 6:
                ax3d.text(target_pos[t_idx, 0] + 0.15, target_pos[t_idx, 1] + 0.15,
                          target_pos[t_idx, 2] + 0.25,
                          f"{TACTICAL_NAMES[tactical_hist[i]]}\n@{i*DT:.1f}s",
                          fontsize=7, color=TACTICAL_COLLIST[tactical_hist[i]])

    ax3d.set_xlabel('X (m)', fontsize=10)
    ax3d.set_ylabel('Y (m)', fontsize=10)
    ax3d.set_zlabel('Z (m)', fontsize=10)
    ax3d.set_title('3D View', fontsize=11, fontweight='bold')
    ax3d.view_init(elev=28, azim=-60)

    all_pos = np.concatenate([target_pos, pursuer_pos.reshape(-1, 3)], axis=0)
    margin = 1.0
    ax3d.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ax3d.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    ax3d.set_zlim(max(0, all_pos[:, 2].min() - 0.5), min(MAX_HEIGHT + 0.5, all_pos[:, 2].max() + 1.0))

    # ══════════════════════════════════════════════════════════════════
    #  RIGHT: 2D top-down XY view (no perspective distortion)
    # ══════════════════════════════════════════════════════════════════
    ax2d = fig.add_subplot(122)
    ax2d.set_facecolor('#f5f6f8')
    ax2d.set_aspect('equal')

    # Arena boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    ax2d.plot(ARENA_RADIUS * np.cos(theta), ARENA_RADIUS * np.sin(theta),
              '--', color='#aaa', lw=1.0, alpha=0.5)

    # Goal region (circle in top-down)
    goal_circle = plt.Circle((gc[0], gc[1]), GOAL_RADIUS, fill=True,
                              facecolor='#ff6b3520', edgecolor='#ff6b35', lw=2.0, zorder=5)
    ax2d.add_patch(goal_circle)
    ax2d.annotate(f'GOAL\n(x={gc[0]:.0f}, y={gc[1]:.0f})',
                  xy=(gc[0], gc[1]), fontsize=9, color='#ff6b35',
                  ha='center', va='center', fontweight='bold', zorder=6)

    # Target trajectory with tactical coloring
    for i in range(0, n_steps - 1, downsample):
        idx = min(i, len(tactical_hist) - 1)
        intent = tactical_hist[idx] if idx >= 0 else STRATEGIC_FAR_RUSH
        color = TACTICAL_COLLIST[intent]
        seg_end = min(i + downsample + 1, n_steps)
        ax2d.plot(target_pos[i:seg_end, 0], target_pos[i:seg_end, 1],
                  color=color, linewidth=2.5, alpha=0.9, solid_capstyle='round')

    # Target start/end
    ax2d.scatter(target_pos[0, 0], target_pos[0, 1], marker='^', s=150, c='black',
                 edgecolors='white', linewidths=1.5, zorder=10)
    ax2d.scatter(target_pos[-1, 0], target_pos[-1, 1], marker='v', s=150, c='black',
                 edgecolors='white', linewidths=1.5, zorder=10)
    ax2d.annotate(f'T_start\n({target_pos[0,0]:.1f},{target_pos[0,1]:.1f})',
                  xy=(target_pos[0, 0], target_pos[0, 1]),
                  xytext=(target_pos[0, 0] - 0.5, target_pos[0, 1] + 0.5),
                  fontsize=7, color='black',
                  arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # Pursuer trajectories
    for pi in range(3):
        p_traj = pursuer_pos[:, pi, :]
        ax2d.plot(p_traj[::downsample, 0], p_traj[::downsample, 1],
                  '--', color=PURSUER_COLORS[pi], lw=1.5, alpha=0.7)
        ax2d.scatter(p_traj[0, 0], p_traj[0, 1], marker='o', s=100,
                     c=PURSUER_COLORS[pi], edgecolors='white', linewidths=1.2, zorder=9)
        ax2d.scatter(p_traj[-1, 0], p_traj[-1, 1], marker='s', s=70,
                     c=PURSUER_COLORS[pi], edgecolors='white', linewidths=1.0, zorder=9)
        ax2d.annotate(f'P{pi+1}\n({p_traj[0,0]:.1f},{p_traj[0,1]:.1f})',
                      xy=(p_traj[0, 0], p_traj[0, 1]),
                      xytext=(p_traj[0, 0] + 0.3, p_traj[0, 1] + 0.4),
                      fontsize=7, color=PURSUER_COLORS[pi])

    # Tactical change markers in 2D
    change_count = 0
    for i in range(1, len(tactical_hist)):
        if tactical_hist[i] != tactical_hist[i - 1]:
            t_idx = min(i, n_steps - 1)
            ax2d.scatter(target_pos[t_idx, 0], target_pos[t_idx, 1],
                         marker='*', s=180, c=TACTICAL_COLLIST[tactical_hist[i]],
                         edgecolors='white', linewidths=0.8, zorder=11)
            change_count += 1

    ax2d.set_xlabel('X (m)', fontsize=11)
    ax2d.set_ylabel('Y (m)', fontsize=11)
    ax2d.set_title('Top-Down XY View (no perspective distortion)', fontsize=11, fontweight='bold')
    ax2d.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Set 2D limits
    xy_margin = 1.5
    ax2d.set_xlim(all_pos[:, 0].min() - xy_margin, all_pos[:, 0].max() + xy_margin)
    ax2d.set_ylim(all_pos[:, 1].min() - xy_margin, all_pos[:, 1].max() + xy_margin)

    # ══════════════════════════════════════════════════════════════════
    #  Overall title, status, stats, legend
    # ══════════════════════════════════════════════════════════════════
    status_colors = {"CAPTURED": "#e74c3c", "GOAL_REACHED": "#2ecc71", "TIMEOUT": "#7f8c8d"}
    fig.suptitle(
        f"OT-Guided Pursuit vs. Optimal-Maneuver Target — Scenario {scenario_id:02d}  [{data['status']}]",
        fontsize=14, fontweight='bold',
        color=status_colors.get(data["status"], "#333"),
        y=0.98,
    )

    duration = (n_steps - 1) * DT
    min_d = min(data["min_dist_hist"]) if data["min_dist_hist"] else float('inf')
    tactical_counts = {n: 0 for n in TACTICAL_NAMES}
    for t in tactical_hist:
        tactical_counts[TACTICAL_NAMES[t]] += 1
    total_t = max(len(tactical_hist), 1)
    tac_str = " | ".join(f"{k}: {v/total_t*100:.0f}%" for k, v in tactical_counts.items())

    stats_text = (
        f"Seed {data['seed']} | Duration {duration:.1f}s | "
        f"Min dist: {min_d:.3f}m | Capture radius: {CAPTURE_RADIUS}m\n"
        f"Strategic: {tac_str}\n"
        f"Goal: ({gc[0]:.1f}, {gc[1]:.1f}, {gc[2]:.1f}) | "
        f"Target init: ({target_pos[0,0]:.1f}, {target_pos[0,1]:.1f}, {target_pos[0,2]:.1f})"
    )
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#ccc'),
             family='monospace')

    # Shared legend
    legend_items = []
    for i, name in enumerate(TACTICAL_NAMES):
        legend_items.append(Line2D([0], [0], color=TACTICAL_COLLIST[i], lw=3.0, label=f"Target: {name}"))
    for i in range(3):
        legend_items.append(Line2D([0], [0], color=PURSUER_COLORS[i], lw=1.6, ls='--', label=f"Pursuer {i+1} (OT)"))
    legend_items.append(Line2D([0], [0], marker='*', color='gray', lw=0, ms=12, label='Tactical change'))
    ax2d.legend(handles=legend_items, loc='lower right', fontsize=7, framealpha=0.9, ncol=2)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate OT vs. target 3D trajectory plots.")
    parser.add_argument("--num-scenarios", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed_base)
    seeds = rng.integers(0, 1_000_000, size=args.num_scenarios).tolist()

    print("=" * 78)
    print(f"  OT-Guided Pursuit vs. Optimal-Maneuver Target | {args.num_scenarios} scenarios")
    print(f"  seed_base={args.seed_base}")
    print("=" * 78)

    summary_lines = []
    for idx, seed in enumerate(seeds, start=1):
        print(f"\n[{idx}/{args.num_scenarios}] seed={seed} ...")
        data = run_scenario(seed)

        save_path = os.path.join(results_dir, f"ot_vs_target_3d_{idx:02d}.png")
        plot_scenario_3d(data, idx, save_path)

        duration = (len(data["target_pos"]) - 1) * DT
        min_d = min(data["min_dist_hist"]) if data["min_dist_hist"] else float("inf")
        tactical_switches = sum(
            1 for i in range(1, len(data["tactical_hist"]))
            if data["tactical_hist"][i] != data["tactical_hist"][i - 1]
        )
        line = (
            f"{idx:02d} | seed={seed:>6d} | {data['status']:<12s} | "
            f"duration={duration:5.1f}s | min_dist={min_d:5.3f}m | "
            f"tactical_switches={tactical_switches:2d}"
        )
        summary_lines.append(line)
        print(f"  {line}")

    # Write summary
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"seed_base={args.seed_base}\n")
        f.write(f"target: 4-strategic + 9-candidate optimal maneuver + 6DoF\n")
        f.write(f"pursuer: OT-guided kinematic GMM + 6DoF (no RL residual)\n")
        f.write(f"dt={DT}, max_steps={MAX_STEPS}\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print("\n" + "=" * 78)
    print(f"  All {args.num_scenarios} plots saved to: {results_dir}")
    print(f"  Summary: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
