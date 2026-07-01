#!/usr/bin/env python3
"""
真正的三维空间对抗轨迹图 — 10个场景

- 初始距离 7-8m
- 距离<0.4m 判定围捕成功
- 手动透视投影实现真3D (绕过mpl_toolkits版本冲突)

运行:
    cd /data/uavlab/maneuverable-target
    python3 scripts/plot_3d_battle.py
"""

import sys, os, math
import numpy as np
from pathlib import Path

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

import torch
import types

for name, path in [
    ("omni_drones", "omni_drones"),
    ("omni_drones.learning", "omni_drones/learning"),
    ("omni_drones.envs", "omni_drones/envs"),
    ("omni_drones.envs.hide_and_seek", "omni_drones/envs/hide_and_seek"),
]:
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(project_root, path)]
    mod.__package__ = name
    sys.modules[name] = mod

from omni_drones.learning.predictor.data_pipeline import label_intent_batch
from omni_drones.envs.hide_and_seek.target_fsm import TargetFSM, STRATEGY_NAMES
from omni_drones.envs.hide_and_seek.proportional_navigation import proportional_navigation_3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

RESULTS_DIR = os.path.join(project_root, "results", "3d_battles")

INTENT_COLORS = [
    "#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#e67e22",
    "#1abc9c", "#f39c12", "#2980b9", "#c0392b", "#95a5a6",
]
INTENT_LABELS = [
    "Straight", "Left-Turn", "Right-Turn", "Climb", "Dive",
    "Left-Climb", "Left-Dive", "Right-Climb", "Right-Dive", "Hover",
]
PURSUER_COLORS = ["#e74c3c", "#3498db", "#f39c12"]


class Config:
    arena_radius = 5.0
    max_height = 5.0
    max_speed = 3.0
    max_accel = 3.0
    n_pursuers = 3
    dt = 0.01
    max_steps = 2000  # 20秒
    goal_center = torch.tensor([4.0, 0.0, 2.5], dtype=torch.float32)
    capture_radius = 0.4


# ==========================================================================
# 3D透视投影
# ==========================================================================

def project_3d_to_2d(points, elev=25, azim=-60, dist=1.0):
    """
    将3D点投影到2D画布 (正交投影+视角旋转)
    
    points: [N, 3] numpy array (x, y, z)
    elev: 仰角 (度)
    azim: 方位角 (度)
    
    返回: [N, 2] numpy array (screen_x, screen_y)
    """
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    
    # 旋转矩阵: 先绕Z轴旋转azim, 再绕X轴旋转elev
    cos_a, sin_a = np.cos(azim_rad), np.sin(azim_rad)
    cos_e, sin_e = np.cos(elev_rad), np.sin(elev_rad)
    
    # 绕Z轴
    Rz = np.array([[cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]])
    # 绕X轴
    Rx = np.array([[1, 0, 0],
                    [0, cos_e, -sin_e],
                    [0, sin_e, cos_e]])
    
    R = Rx @ Rz
    
    rotated = points @ R.T
    
    # 投影到 screen_x = rotated[:, 0], screen_y = rotated[:, 2]
    screen_x = rotated[:, 0]
    screen_y = rotated[:, 2]
    
    return np.column_stack([screen_x, screen_y])


def draw_3d_axes(ax, origin, length, elev, azim):
    """画3D坐标轴"""
    axes_3d = np.array([[length, 0, 0],
                         [0, length, 0],
                         [0, 0, length]])
    origin_2d = project_3d_to_2d(np.array([origin]), elev, azim)[0]
    
    for i, (label, color) in enumerate(zip(["X", "Y", "Z"], ["#e74c3c", "#2ecc71", "#3498db"])):
        end = origin.copy()
        end[i] += length
        end_2d = project_3d_to_2d(np.array([end]), elev, azim)[0]
        ax.annotate("", xy=end_2d, xytext=origin_2d,
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        ax.text(end_2d[0] + 0.15, end_2d[1] + 0.15, label,
                fontsize=10, color=color, fontweight="bold")


def draw_arena_floor(ax, radius, height_min, elev, azim, n_pts=72):
    """画场地地面圆形边界"""
    theta = np.linspace(0, 2 * np.pi, n_pts)
    floor = np.column_stack([radius * np.cos(theta),
                              radius * np.sin(theta),
                              np.full(n_pts, height_min)])
    floor_2d = project_3d_to_2d(floor, elev, azim)
    ax.plot(floor_2d[:, 0], floor_2d[:, 1], "k--", alpha=0.15, linewidth=0.8)
    
    # 天花板
    ceiling = floor.copy()
    ceiling[:, 2] = 5.0
    ceiling_2d = project_3d_to_2d(ceiling, elev, azim)
    ax.plot(ceiling_2d[:, 0], ceiling_2d[:, 1], "k--", alpha=0.08, linewidth=0.5)
    
    # 四根柱子
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        pillar = np.array([[radius*np.cos(angle), radius*np.sin(angle), height_min],
                           [radius*np.cos(angle), radius*np.sin(angle), 5.0]])
        p2d = project_3d_to_2d(pillar, elev, azim)
        ax.plot(p2d[:, 0], p2d[:, 1], "k--", alpha=0.08, linewidth=0.5)


# ==========================================================================
# 仿真
# ==========================================================================

def simulate_battle(cfg, fsm, scenario_id, init_dist=7.5):
    """模拟对抗, 初始距离约7-8m"""
    device = "cpu"
    
    # 目标初始位置: 场地中心偏移
    angle_t = np.random.uniform(0, 2 * np.pi)
    r_t = np.random.uniform(1.0, 2.5)
    target_pos = torch.tensor([
        r_t * np.cos(angle_t),
        r_t * np.sin(angle_t),
        np.random.uniform(1.5, 3.5),
    ], dtype=torch.float32)
    target_vel = torch.zeros(3, dtype=torch.float32)
    # 初始速度朝随机方向
    v_angle = np.random.uniform(0, 2 * np.pi)
    target_vel[0] = np.random.uniform(1.0, 2.5) * np.cos(v_angle)
    target_vel[1] = np.random.uniform(1.0, 2.5) * np.sin(v_angle)
    target_vel[2] = np.random.uniform(-0.5, 0.5)
    
    # 追捕方: 围绕目标, 距离约 init_dist
    pursuer_pos = torch.zeros(cfg.n_pursuers, 3, dtype=torch.float32)
    pursuer_vel = torch.zeros(cfg.n_pursuers, 3, dtype=torch.float32)
    for p in range(cfg.n_pursuers):
        p_angle = angle_t + np.pi + np.random.uniform(-0.8, 0.8)  # 大致在目标对面
        p_angle += p * (2 * np.pi / cfg.n_pursuers)  # 均匀分布
        p_r = init_dist * np.random.uniform(0.85, 1.15)
        pursuer_pos[p] = torch.tensor([
            target_pos[0].item() + p_r * np.cos(p_angle),
            target_pos[1].item() + p_r * np.sin(p_angle),
            np.random.uniform(1.0, 4.0),
        ], dtype=torch.float32)
        # 裁剪到场地
        xy = torch.norm(pursuer_pos[p, :2])
        if xy > cfg.arena_radius:
            pursuer_pos[p, :2] = pursuer_pos[p, :2] / xy * (cfg.arena_radius - 0.1)
        pursuer_pos[p, 2] = pursuer_pos[p, 2].clamp(0.1, cfg.max_height)
    
    fsm.reset(torch.tensor([0]))
    velocity_damping = 0.3
    
    all_target_pos, all_target_vel = [], []
    all_pursuer_pos = []
    all_strategy_id = []
    all_min_dist = []
    captured = False
    capture_step = -1
    
    for step in range(cfg.max_steps):
        # APF
        goal_offset = cfg.goal_center - target_pos
        goal_dist = torch.norm(goal_offset).clamp(min=1e-5)
        f_goal = 1.0 * goal_offset / goal_dist
        
        f_repulsion = torch.zeros(3)
        for p in range(cfg.n_pursuers):
            diff = target_pos - pursuer_pos[p]
            d = torch.norm(diff).clamp(min=1e-5)
            f_repulsion += 1.5 * diff / (d ** 2 + 1e-5)
        
        f_boundary = torch.zeros(3)
        xy_dist = torch.norm(target_pos[:2])
        if xy_dist > cfg.arena_radius * 0.85:
            f_boundary[:2] = -target_pos[:2] / (xy_dist + 1e-5) * 3.0
        if target_pos[2] > cfg.max_height * 0.9:
            f_boundary[2] -= 3.0
        if target_pos[2] < 0.3:
            f_boundary[2] += 3.0
        
        apf_force = f_goal + f_repulsion + f_boundary
        
        fsm_force, strategy = fsm.compute_strategy_force(
            target_pos.unsqueeze(0), target_vel.unsqueeze(0),
            pursuer_pos.unsqueeze(0), goal_center=cfg.goal_center,
        )
        fsm_force = fsm_force.squeeze(0)
        strategy_id = strategy[0].item()
        
        total_force = apf_force + fsm_force
        force_norm = torch.norm(total_force).clamp(min=1e-6)
        accel_cmd = cfg.max_accel * total_force / force_norm
        
        # 记录
        all_target_pos.append(target_pos.clone())
        all_target_vel.append(target_vel.clone())
        all_pursuer_pos.append(pursuer_pos.clone())
        all_strategy_id.append(strategy_id)
        
        # 最近追捕方距离
        min_d = min(torch.norm(target_pos - pursuer_pos[p]).item() for p in range(cfg.n_pursuers))
        all_min_dist.append(min_d)
        
        # 更新目标
        target_vel = target_vel + (accel_cmd - velocity_damping * target_vel) * cfg.dt
        speed = torch.norm(target_vel)
        if speed > cfg.max_speed:
            target_vel = target_vel / speed * cfg.max_speed
        target_pos = target_pos + target_vel * cfg.dt
        xy_d = torch.norm(target_pos[:2])
        if xy_d > cfg.arena_radius:
            target_pos[:2] = target_pos[:2] / xy_d * cfg.arena_radius
        target_pos[2] = target_pos[2].clamp(0.1, cfg.max_height)
        
        # 更新追捕方
        for p in range(cfg.n_pursuers):
            a_pn = proportional_navigation_3d(
                pursuer_pos[p].unsqueeze(0), pursuer_vel[p].unsqueeze(0),
                target_pos.unsqueeze(0), target_vel.unsqueeze(0),
                N=4.0, max_accel=cfg.max_accel,
            ).squeeze(0)
            pursuer_vel[p] = pursuer_vel[p] + (a_pn - velocity_damping * pursuer_vel[p]) * cfg.dt
            p_speed = torch.norm(pursuer_vel[p])
            if p_speed > cfg.max_speed:
                pursuer_vel[p] = pursuer_vel[p] / p_speed * cfg.max_speed
            pursuer_pos[p] = pursuer_pos[p] + pursuer_vel[p] * cfg.dt
            p_xy = torch.norm(pursuer_pos[p, :2])
            if p_xy > cfg.arena_radius:
                pursuer_pos[p, :2] = pursuer_pos[p, :2] / p_xy * cfg.arena_radius
            pursuer_pos[p, 2] = pursuer_pos[p, 2].clamp(0.1, cfg.max_height)
        
        # 围捕判定
        if min_d < cfg.capture_radius:
            captured = True
            capture_step = step
            break
    
    return {
        "target_pos": torch.stack(all_target_pos).numpy(),
        "target_vel": torch.stack(all_target_vel).numpy(),
        "pursuer_pos": torch.stack(all_pursuer_pos).numpy(),
        "strategy_id": all_strategy_id,
        "min_dist": all_min_dist,
        "captured": captured,
        "capture_step": capture_step,
        "n_steps": len(all_target_pos),
        "scenario_id": scenario_id,
    }


# ==========================================================================
# 3D轨迹绘制
# ==========================================================================

def plot_3d_battle(ep, save_path, elev=25, azim=-55):
    """绘制真正的3D空间对抗轨迹图"""
    tpos = ep["target_pos"]    # [T, 3]
    ppos = ep["pursuer_pos"]   # [T, 3, 3]
    T = ep["n_steps"]
    captured = ep["captured"]
    capture_step = ep["capture_step"]
    duration = T * 0.01
    
    # 计算初始距离
    init_dists = [np.linalg.norm(tpos[0] - ppos[0, p]) for p in range(3)]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 画场地
    draw_arena_floor(ax, 5.0, 0.1, elev, azim)
    draw_3d_axes(ax, np.array([-5.5, -5.5, 0.0]), 2.0, elev, azim)
    
    # 目标轨迹 — 粗线, 按策略着色
    strategy_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#e67e22"]
    t2d = project_3d_to_2d(tpos, elev, azim)
    
    for t in range(T - 1):
        sid = ep["strategy_id"][t]
        color = strategy_colors[sid] if 0 <= sid < 5 else "#333333"
        ax.plot(t2d[t:t+2, 0], t2d[t:t+2, 1],
                color=color, linewidth=2.5, alpha=0.85, solid_capstyle="round")
    
    # 目标起点/终点
    ax.plot(t2d[0, 0], t2d[0, 1], "k^", markersize=14, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(t2d[-1, 0], t2d[-1, 1], "kv", markersize=14, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5)
    
    # 每隔一段时间画时间标记
    time_interval = max(1, T // 8)  # 大约8个标记
    for t in range(0, T, time_interval):
        ax.plot(t2d[t, 0], t2d[t, 1], "ko", markersize=4, zorder=8, alpha=0.5)
        ax.annotate(f"{t*0.01:.1f}s", xy=(t2d[t, 0], t2d[t, 1]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=6, alpha=0.6, color="#333333")
    
    # 追捕方轨迹
    for p in range(3):
        p_traj = ppos[:, p, :]
        p2d = project_3d_to_2d(p_traj, elev, azim)
        ax.plot(p2d[:, 0], p2d[:, 1], "--",
                color=PURSUER_COLORS[p], linewidth=1.2, alpha=0.6)
        # 起点
        ax.plot(p2d[0, 0], p2d[0, 1], "o", color=PURSUER_COLORS[p],
                markersize=10, zorder=9, markeredgecolor="white", markeredgewidth=1.5)
        # 终点
        ax.plot(p2d[-1, 0], p2d[-1, 1], "s", color=PURSUER_COLORS[p],
                markersize=8, zorder=9, markeredgecolor="white", markeredgewidth=1)
    
    # 围捕点标记
    if captured:
        capture_2d = project_3d_to_2d(tpos[capture_step:capture_step+1], elev, azim)[0]
        ax.plot(capture_2d[0], capture_2d[1], "*", color="#e74c3c",
                markersize=25, zorder=15, markeredgecolor="white", markeredgewidth=1)
        ax.annotate(f"CAPTURED\n{capture_step*0.01:.2f}s",
                    xy=(capture_2d[0], capture_2d[1]),
                    xytext=(15, -20), textcoords="offset points",
                    fontsize=10, color="#e74c3c", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c"))
    
    # 标题和信息
    status = "🔴 CAPTURED" if captured else "🟢 ESCAPED"
    title = (f"Scenario {ep['scenario_id']+1} — {status}\n"
             f"Duration: {duration:.2f}s ({T} steps) | "
             f"Init dist: {np.mean(init_dists):.1f}m | "
             f"Min dist: {min(ep['min_dist']):.2f}m")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    
    # 速度统计
    speeds = np.linalg.norm(ep["target_vel"], axis=-1)
    info_text = (f"Target: v_avg={np.mean(speeds):.2f} m/s, v_max={np.max(speeds):.2f} m/s\n"
                 f"Strategies: {len(set(ep['strategy_id']))} types used")
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 图例
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=2.5, label="Target trajectory"),
        Line2D([0], [0], color="black", linewidth=0, marker="^",
               markersize=10, label="Target start"),
        Line2D([0], [0], color="black", linewidth=0, marker="v",
               markersize=10, label="Target end"),
    ]
    for p in range(3):
        legend_elements.append(
            Line2D([0], [0], color=PURSUER_COLORS[p], linewidth=1.2,
                   linestyle="--", label=f"Pursuer {p+1}"))
    # 策略颜色
    for i, name in enumerate(STRATEGY_NAMES):
        legend_elements.append(
            mpatches.Patch(color=strategy_colors[i], alpha=0.85, label=f"策略: {name}"))
    
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9, ncol=2)
    
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("", fontsize=1)
    ax.set_ylabel("", fontsize=1)
    # 隐藏2D坐标刻度 (因为是投影, 刻度无意义)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
    
    # 添加深色背景
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ==========================================================================
# 主入口
# ==========================================================================

def main():
    print("=" * 60)
    print("  3D空间对抗轨迹 — 10个场景")
    print("=" * 60)
    
    cfg = Config()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    fsm = TargetFSM(num_envs=1, device="cpu", max_speed=cfg.max_speed,
                     max_accel=cfg.max_accel, arena_radius=cfg.arena_radius,
                     max_height=cfg.max_height, evasion_threshold=3.0)
    
    summary = []
    
    for i in range(10):
        torch.manual_seed(42 + i * 7)
        np.random.seed(42 + i * 7)
        
        init_dist = np.random.uniform(7.0, 8.0)
        print(f"\n[Scenario {i+1}/10] init_dist={init_dist:.1f}m ...")
        
        ep = simulate_battle(cfg, fsm, scenario_id=i, init_dist=init_dist)
        
        # 用两个不同视角画
        plot_3d_battle(ep, os.path.join(RESULTS_DIR, f"battle_3d_{i+1}.png"),
                       elev=25, azim=-55 + i * 8)
        
        init_dists = [np.linalg.norm(ep["target_pos"][0] - ep["pursuer_pos"][0, p])
                      for p in range(3)]
        speeds = np.linalg.norm(ep["target_vel"], axis=-1)
        n_strategy_switches = sum(1 for t in range(1, len(ep["strategy_id"]))
                                  if ep["strategy_id"][t] != ep["strategy_id"][t-1])
        
        info = {
            "scenario": i + 1,
            "captured": ep["captured"],
            "duration": ep["n_steps"] * 0.01,
            "init_dist_avg": np.mean(init_dists),
            "min_dist": min(ep["min_dist"]),
            "v_avg": np.mean(speeds),
            "v_max": np.max(speeds),
            "strategies_used": len(set(ep["strategy_id"])),
            "strategy_switches": n_strategy_switches,
        }
        summary.append(info)
        
        status = "CAPTURED" if ep["captured"] else "ESCAPED"
        print(f"  {status} | {info['duration']:.2f}s | "
              f"min_dist={info['min_dist']:.2f}m | "
              f"v_avg={info['v_avg']:.2f} | "
              f"switches={info['strategy_switches']}")
    
    # 汇总表
    print("\n" + "=" * 80)
    print(f"  {'#':>3} {'Status':<10} {'Duration':>10} {'InitDist':>10} {'MinDist':>10} "
          f"{'AvgSpeed':>10} {'MaxSpeed':>10} {'Strategies':>10} {'Switches':>10}")
    print("  " + "-" * 78)
    
    n_captured = 0
    for s in summary:
        status = "CAPTURED" if s["captured"] else "ESCAPED"
        if s["captured"]:
            n_captured += 1
        print(f"  {s['scenario']:>3} {status:<10} {s['duration']:>10.2f}s "
              f"{s['init_dist_avg']:>10.1f}m {s['min_dist']:>10.2f}m "
              f"{s['v_avg']:>10.2f} {s['v_max']:>10.2f} "
              f"{s['strategies_used']:>10} {s['strategy_switches']:>10}")
    
    print(f"\n  围捕成功率: {n_captured}/{len(summary)} ({n_captured/len(summary)*100:.0f}%)")
    print(f"  平均持续时间: {np.mean([s['duration'] for s in summary]):.2f}s")
    print(f"  平均目标速度: {np.mean([s['v_avg'] for s in summary]):.2f} m/s")
    print(f"  平均策略切换: {np.mean([s['strategy_switches'] for s in summary]):.1f} 次/episode")
    print(f"\n  图表保存在: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
