#!/usr/bin/env python3
"""
诊断脚本：
1. 3D轨迹可视化 (目标+追捕方, 按意图着色)
2. 目标是否高机动: 速度/加速度/意图切换频率统计
3. 观测窗口内标签一致性分析
4. MVS=0 消融对比

运行:
    cd /data/uavlab/maneuverable-target
    python3 scripts/diagnose_target.py
"""

import sys, os, math, time
import numpy as np
from pathlib import Path

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

import torch
import types

# 绕过 omni_drones/__init__.py
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

from omni_drones.learning.predictor.intent_gmm_predictor import IntentConditionedGMMPredictor, INTENT_LABELS
from omni_drones.learning.predictor.losses import IntentGMMLoss
from omni_drones.learning.predictor.mvs import compute_mvs_batch
from omni_drones.learning.predictor.data_pipeline import (
    PredictorDataset, TrajectoryRecorder, label_intent_batch,
)
from omni_drones.envs.hide_and_seek.target_fsm import TargetFSM, STRATEGY_NAMES
from omni_drones.envs.hide_and_seek.proportional_navigation import proportional_navigation_3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

RESULTS_DIR = os.path.join(project_root, "results")
CKPT_DIR = os.path.join(project_root, "checkpoints")

INTENT_COLORS = [
    "#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#e67e22",
    "#1abc9c", "#f39c12", "#2980b9", "#c0392b", "#95a5a6",
]

class Config:
    arena_radius = 5.0
    max_height = 5.0
    max_speed = 3.0
    max_accel = 3.0
    n_pursuers = 3
    dt = 0.01
    subsample = 5
    t_obs = 20
    t_pred = 10
    w_mvs = 60
    slide_step = 5
    max_steps = 1500
    goal_center = torch.tensor([4.0, 0.0, 2.5])
    batch_size = 64
    hidden_dim = 128


def simulate_full_episode(cfg, fsm):
    """模拟完整episode, 返回每步原始数据(不下采样)"""
    device = "cpu"
    target_pos = torch.tensor([-3.5 + np.random.uniform(-1, 1),
                                np.random.uniform(-2, 2),
                                np.random.uniform(1.0, 3.5)])
    target_vel = torch.zeros(3)
    target_vel[0] = np.random.uniform(0.5, 2.0)

    pursuer_pos = torch.zeros(cfg.n_pursuers, 3)
    pursuer_vel = torch.zeros(cfg.n_pursuers, 3)
    for p in range(cfg.n_pursuers):
        pursuer_pos[p] = torch.tensor([
            cfg.goal_center[0] + np.random.uniform(-1.5, 0.5),
            np.random.uniform(-2, 2),
            np.random.uniform(1.0, 3.5),
        ])

    fsm.reset(torch.tensor([0]))
    velocity_damping = 0.25

    # 记录每步数据
    all_target_pos, all_target_vel = [], []
    all_pursuer_pos = []
    all_accel_cmd, all_strategy_id = [], []

    for step in range(cfg.max_steps):
        # APF
        goal_offset = cfg.goal_center - target_pos
        goal_dist = torch.norm(goal_offset).clamp(min=1e-5)
        f_goal = 1.0 * goal_offset / goal_dist

        f_repulsion = torch.zeros(3)
        for p in range(cfg.n_pursuers):
            diff = target_pos - pursuer_pos[p]
            d = torch.norm(diff).clamp(min=1e-5)
            f_repulsion += 1.0 * diff / (d ** 2 + 1e-5)

        f_boundary = torch.zeros(3)
        xy_dist = torch.norm(target_pos[:2])
        if xy_dist > cfg.arena_radius * 0.85:
            f_boundary[:2] = -target_pos[:2] / (xy_dist + 1e-5) * 2.0
        if target_pos[2] > cfg.max_height * 0.9:
            f_boundary[2] -= 2.0
        if target_pos[2] < 0.3:
            f_boundary[2] += 2.0

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
        accel_cmd = accel_cmd.clamp(-cfg.max_accel, cfg.max_accel)

        # 记录
        all_target_pos.append(target_pos.clone())
        all_target_vel.append(target_vel.clone())
        all_pursuer_pos.append(pursuer_pos.clone())
        all_accel_cmd.append(accel_cmd.clone())
        all_strategy_id.append(strategy_id)

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

        # 捕获
        for p in range(cfg.n_pursuers):
            if torch.norm(target_pos - pursuer_pos[p]) < 0.4:
                break
        else:
            continue
        break

    return {
        "target_pos": torch.stack(all_target_pos),     # [T, 3]
        "target_vel": torch.stack(all_target_vel),     # [T, 3]
        "pursuer_pos": torch.stack(all_pursuer_pos),   # [T, 3, 3]
        "accel_cmd": torch.stack(all_accel_cmd),       # [T, 3]
        "strategy_id": all_strategy_id,                 # list[T]
        "n_steps": len(all_target_pos),
    }


# ==========================================================================
# 可视化1: 3D轨迹 (按意图着色)
# ==========================================================================

def plot_3d_trajectory(ep_data, save_path):
    """三视图: XY + XZ + YZ, 目标按意图着色"""
    tpos = ep_data["target_pos"].numpy()
    ppos = ep_data["pursuer_pos"].numpy()
    accel = ep_data["accel_cmd"]
    vel_t = ep_data["target_vel"]
    T = ep_data["n_steps"]
    intents = label_intent_batch(accel, vel_t).numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    views = [(ax1, 0, 1, "X", "Y", "Top (XY)"),
             (ax2, 0, 2, "X", "Z", "Side (XZ)"),
             (ax3, 1, 2, "Y", "Z", "Front (YZ)")]

    for ax, xi, yi, xl, yl, title in views:
        for t in range(T - 1):
            ax.plot(tpos[t:t+2, xi], tpos[t:t+2, yi],
                    color=INTENT_COLORS[intents[t]], linewidth=1.5, alpha=0.8)
        ax.plot(tpos[0, xi], tpos[0, yi], "k^", markersize=10, zorder=5)
        ax.plot(tpos[-1, xi], tpos[-1, yi], "kv", markersize=10, zorder=5)
        for p in range(3):
            ax.plot(ppos[:, p, xi], ppos[:, p, yi], "--", color="#7f8c8d",
                    linewidth=0.6, alpha=0.4)
        ax.set_xlabel(f"{xl} (m)", fontsize=10)
        ax.set_ylabel(f"{yl} (m)", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.2)
        if xi == 0 and yi == 1:
            ax.set_aspect("equal")

    legend_patches = [mpatches.Patch(color=INTENT_COLORS[i], label=INTENT_LABELS[i])
                      for i in range(10) if (intents == i).any()]
    fig.legend(handles=legend_patches, loc="lower center", fontsize=8, ncol=5,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f"Target Trajectory ({T} steps = {T*0.01:.1f}s)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_3d_trajectory_top_side(ep_data, save_path):
    """俯视+侧视双视图, 更清晰"""
    tpos = ep_data["target_pos"].numpy()
    ppos = ep_data["pursuer_pos"].numpy()
    accel = ep_data["accel_cmd"]
    vel_t = ep_data["target_vel"]
    T = ep_data["n_steps"]
    intents = label_intent_batch(accel, vel_t).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # XY 俯视
    for t in range(T - 1):
        ax1.plot(tpos[t:t+2, 0], tpos[t:t+2, 1],
                 color=INTENT_COLORS[intents[t]], linewidth=1.5, alpha=0.8)
    ax1.plot(tpos[0, 0], tpos[0, 1], "k^", markersize=12, zorder=5)
    ax1.plot(tpos[-1, 0], tpos[-1, 1], "kv", markersize=12, zorder=5)
    for p in range(3):
        ax1.plot(ppos[:, p, 0], ppos[:, p, 1], "--", color="#7f8c8d",
                 linewidth=0.6, alpha=0.4)

    # 场地边界
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(5*np.cos(theta), 5*np.sin(theta), "k--", alpha=0.3, linewidth=0.5)
    ax1.set_xlabel("X (m)", fontsize=11)
    ax1.set_ylabel("Y (m)", fontsize=11)
    ax1.set_title("Top View (XY)", fontsize=13)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.2)

    # XZ 侧视
    for t in range(T - 1):
        ax2.plot(tpos[t:t+2, 0], tpos[t:t+2, 2],
                 color=INTENT_COLORS[intents[t]], linewidth=1.5, alpha=0.8)
    ax2.plot(tpos[0, 0], tpos[0, 2], "k^", markersize=12, zorder=5)
    ax2.plot(tpos[-1, 0], tpos[-1, 2], "kv", markersize=12, zorder=5)
    for p in range(3):
        ax2.plot(ppos[:, p, 0], ppos[:, p, 2], "--", color="#7f8c8d",
                 linewidth=0.6, alpha=0.4)
    ax2.axhline(y=5.0, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    ax2.axhline(y=0.1, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    ax2.set_xlabel("X (m)", fontsize=11)
    ax2.set_ylabel("Z - Height (m)", fontsize=11)
    ax2.set_title("Side View (XZ)", fontsize=13)
    ax2.grid(True, alpha=0.2)

    # 图例
    legend_patches = [mpatches.Patch(color=INTENT_COLORS[i], label=INTENT_LABELS[i])
                      for i in range(10) if (intents == i).any()]
    fig.legend(handles=legend_patches, loc="lower center", fontsize=9, ncol=5,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f"Target Trajectory ({T} steps = {T*0.01:.1f}s)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ==========================================================================
# 可视化2: 速度/加速度/意图时间序列
# ==========================================================================

def plot_maneuver_profile(ep_data, save_path):
    """速度、加速度、意图随时间变化"""
    tvel = ep_data["target_vel"].numpy()
    accel = ep_data["accel_cmd"]
    vel_t = ep_data["target_vel"]
    T = ep_data["n_steps"]
    strategy_ids = ep_data["strategy_id"]

    intents = label_intent_batch(accel, vel_t).numpy()
    speed = np.linalg.norm(tvel, axis=-1)
    accel_mag = np.linalg.norm(accel.numpy(), axis=-1)
    time_axis = np.arange(T) * 0.01  # 秒

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

    # 速度
    axes[0].plot(time_axis, speed, color="#3498db", linewidth=1)
    axes[0].axhline(y=3.0, color="#e74c3c", linestyle="--", alpha=0.5, label="Max speed (3 m/s)")
    axes[0].set_ylabel("Speed (m/s)", fontsize=11)
    axes[0].set_title("Target Speed Profile", fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_ylim(0, 3.5)

    # 加速度
    axes[1].plot(time_axis, accel_mag, color="#e67e22", linewidth=1)
    axes[1].axhline(y=3.0, color="#e74c3c", linestyle="--", alpha=0.5, label="Max accel (3 m/s²)")
    axes[1].set_ylabel("Accel Magnitude (m/s²)", fontsize=11)
    axes[1].set_title("Target Acceleration Profile", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    # 意图标签
    for t in range(T):
        axes[2].bar(time_axis[t], 1, width=0.01, color=INTENT_COLORS[intents[t]],
                    edgecolor="none")
    axes[2].set_ylabel("Intent", fontsize=11)
    axes[2].set_title("Intent Label Over Time", fontsize=13)
    axes[2].set_yticks([])
    # 图例
    present = sorted(set(intents.tolist()))
    patches = [mpatches.Patch(color=INTENT_COLORS[i], label=INTENT_LABELS[i]) for i in present]
    axes[2].legend(handles=patches, fontsize=7, ncol=5, loc="upper right")

    # 策略ID
    strategy_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#e67e22"]
    for t in range(T):
        sid = strategy_ids[t]
        if 0 <= sid < 5:
            axes[3].bar(time_axis[t], 1, width=0.01, color=strategy_colors[sid],
                        edgecolor="none")
    axes[3].set_ylabel("FSM Strategy", fontsize=11)
    axes[3].set_title("FSM Strategy Over Time", fontsize=13)
    axes[3].set_xlabel("Time (s)", fontsize=11)
    axes[3].set_yticks([])
    s_patches = [mpatches.Patch(color=strategy_colors[i], label=STRATEGY_NAMES[i]) for i in range(5)]
    axes[3].legend(handles=s_patches, fontsize=8, ncol=5, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ==========================================================================
# 分析3: 意图切换频率统计
# ==========================================================================

def analyze_intent_switching(episodes_data, cfg):
    """统计意图切换频率"""
    all_switch_rates = []
    all_switches_per_sec = []
    all_unique_intents = []

    for ep in episodes_data:
        intents = label_intent_batch(ep["accel_cmd"], ep["target_vel"]).numpy()
        T = len(intents)
        switches = np.sum(intents[1:] != intents[:-1])
        switch_rate = switches / (T - 1)
        duration_sec = T * cfg.dt
        switches_per_sec = switches / duration_sec

        all_switch_rates.append(switch_rate)
        all_switches_per_sec.append(switches_per_sec)
        all_unique_intents.append(len(set(intents.tolist())))

    return {
        "switch_rate_mean": np.mean(all_switch_rates),
        "switch_rate_std": np.std(all_switch_rates),
        "switches_per_sec_mean": np.mean(all_switches_per_sec),
        "switches_per_sec_std": np.std(all_switches_per_sec),
        "unique_intents_mean": np.mean(all_unique_intents),
        "raw_switch_rates": all_switch_rates,
        "raw_switches_per_sec": all_switches_per_sec,
    }


# ==========================================================================
# 分析4: 观测窗口内标签一致性
# ==========================================================================

def analyze_label_consistency(episodes_data, cfg):
    """分析T_obs=20帧窗口内意图切换次数"""
    window_switch_counts = []
    window_unique_counts = []
    window_dominant_ratios = []

    subsample = cfg.subsample

    for ep in episodes_data:
        intents_raw = label_intent_batch(ep["accel_cmd"], ep["target_vel"]).numpy()
        T = len(intents_raw)

        # 下采样
        intents_sub = intents_raw[::subsample]
        L = len(intents_sub)

        for start in range(0, L - cfg.t_obs):
            window = intents_sub[start:start + cfg.t_obs]
            switches = np.sum(window[1:] != window[:-1])
            unique = len(set(window.tolist()))
            # 最多的意图占比
            counts = np.bincount(window, minlength=10)
            dominant_ratio = counts.max() / len(window)

            window_switch_counts.append(switches)
            window_unique_counts.append(unique)
            window_dominant_ratios.append(dominant_ratio)

    return {
        "switch_counts": np.array(window_switch_counts),
        "unique_counts": np.array(window_unique_counts),
        "dominant_ratios": np.array(window_dominant_ratios),
    }


def plot_label_consistency(consistency, save_path):
    """可视化观测窗口内的标签一致性"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 切换次数分布
    sc = consistency["switch_counts"]
    axes[0].hist(sc, bins=range(0, max(sc)+2), color="#3498db", alpha=0.8, edgecolor="white")
    axes[0].axvline(x=np.mean(sc), color="#e74c3c", linestyle="--",
                     label=f"Mean = {np.mean(sc):.1f}")
    axes[0].set_xlabel("# Intent Switches in T_obs=20 window")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Intent Switches per Observation Window\n"
                      f"(Mean={np.mean(sc):.1f}, Median={np.median(sc):.0f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # 不同意图数量
    uc = consistency["unique_counts"]
    axes[1].hist(uc, bins=range(1, max(uc)+2), color="#2ecc71", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("# Unique Intents in T_obs=20 window")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Unique Intent Count per Window\n"
                      f"(Mean={np.mean(uc):.1f})")
    axes[1].grid(True, alpha=0.2)

    # 主导意图占比
    dr = consistency["dominant_ratios"]
    axes[2].hist(dr, bins=20, color="#e67e22", alpha=0.8, edgecolor="white")
    axes[2].axvline(x=0.5, color="#e74c3c", linestyle="--", alpha=0.7, label="50%")
    axes[2].set_xlabel("Dominant Intent Ratio in Window")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Label Purity per Window\n"
                      f"(Mean={np.mean(dr):.2f}, <50% = {(dr<0.5).mean()*100:.1f}%)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)

    plt.suptitle("Observation Window Label Consistency Analysis (T_obs=20 frames = 1.0s)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ==========================================================================
# 分析5: MVS=0 消融
# ==========================================================================

def run_ablation_mvs0(model, dataset, cfg):
    """MVS=0 消融实验"""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=False, collate_fn=PredictorDataset.collate_fn)
    model.eval()
    loss_fn = IntentGMMLoss()

    results = {"mvs_real": {}, "mvs_zero": {}}

    for mode, use_real in [("mvs_real", True), ("mvs_zero", False)]:
        all_ade, all_fde, all_correct = [], [], []
        all_nll = []

        with torch.no_grad():
            for batch in loader:
                obs = batch["obs_history"]
                gt_future = batch["gt_future_pos"]
                gt_intent = batch["gt_intent_label"]
                B = obs.shape[0]

                if use_real:
                    mvs = compute_mvs_batch(batch["intent_history"])
                else:
                    mvs = torch.zeros(B, 1)

                output = model(obs, mvs=mvs)
                _, metrics = loss_fn(output, gt_future, gt_intent)

                mixed_mu = output["mixed_mu"]
                error = (mixed_mu - gt_future).norm(dim=-1)
                all_ade.extend(error.mean(dim=-1).tolist())
                all_fde.extend(error[:, -1].tolist())
                all_correct.extend((output["intent_logits"].argmax(dim=-1) == gt_intent).tolist())
                all_nll.append(metrics["loss_gmm_nll"])

        results[mode] = {
            "ade": np.mean(all_ade),
            "fde": np.mean(all_fde),
            "accuracy": np.mean(all_correct),
            "nll": np.mean(all_nll),
        }

    return results


def plot_ablation(ablation, save_path):
    """消融对比条形图"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    modes = ["MVS=Real", "MVS=0 (No MVS)"]
    keys = ["mvs_real", "mvs_zero"]
    colors = ["#3498db", "#95a5a6"]

    for ax, metric, title, fmt in zip(
        axes,
        ["accuracy", "ade", "fde", "nll"],
        ["Intent Accuracy ↑", "ADE ↓", "FDE ↓", "NLL ↓"],
        [".3f", ".4f", ".4f", ".2f"],
    ):
        vals = [ablation[k][metric] for k in keys]
        bars = ax.bar(modes, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Ablation: With MVS vs Without MVS (MVS=0)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ==========================================================================
# 主入口
# ==========================================================================

def main():
    print("=" * 60)
    print("  目标机动性诊断 & MVS消融")
    print("=" * 60)

    cfg = Config()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(123)
    np.random.seed(123)

    fsm = TargetFSM(num_envs=1, device="cpu", max_speed=cfg.max_speed,
                     max_accel=cfg.max_accel, arena_radius=cfg.arena_radius,
                     max_height=cfg.max_height, evasion_threshold=3.0)

    # =====================================================================
    # 1. 模拟几个完整episode并可视化
    # =====================================================================
    print("\n[1] Simulating episodes for visualization...")
    episodes = []
    for i in range(10):
        ep = simulate_full_episode(cfg, fsm)
        episodes.append(ep)
        if i < 3:
            # 为前3个episode画3D轨迹
            plot_3d_trajectory(ep, os.path.join(RESULTS_DIR, f"traj_3d_ep{i}.png"))
            plot_3d_trajectory_top_side(ep, os.path.join(RESULTS_DIR, f"traj_2view_ep{i}.png"))
            plot_maneuver_profile(ep, os.path.join(RESULTS_DIR, f"maneuver_profile_ep{i}.png"))

    # =====================================================================
    # 2. 意图切换频率分析
    # =====================================================================
    print("\n[2] Analyzing intent switching frequency...")
    switching = analyze_intent_switching(episodes, cfg)
    print(f"  意图切换率 (每步): {switching['switch_rate_mean']:.4f} ± {switching['switch_rate_std']:.4f}")
    print(f"  意图切换次数/秒:   {switching['switches_per_sec_mean']:.1f} ± {switching['switches_per_sec_std']:.1f}")
    print(f"  每个episode平均意图种类: {switching['unique_intents_mean']:.1f}")

    # 计算: 在T_obs=20帧(下采样后)=100步原始步中, 期望切换几次
    expected_switches_in_window = switching['switch_rate_mean'] * 100  # 100原始步
    print(f"\n  >> T_obs=20帧(1s)内期望切换次数: {expected_switches_in_window:.1f}")
    if expected_switches_in_window > 3:
        print(f"  >> ⚠️ 意图变化太频繁! 1s窗口内切换{expected_switches_in_window:.1f}次")
        print(f"     标签(取最后一帧)无法代表整个窗口内容")
    elif expected_switches_in_window > 1:
        print(f"  >> ⚠ 意图有一定切换频率, 标签可能不完全一致")
    else:
        print(f"  >> ✅ 意图变化较慢, 标签基本一致")

    # =====================================================================
    # 3. 观测窗口内标签一致性分析
    # =====================================================================
    print("\n[3] Analyzing label consistency in observation windows...")
    consistency = analyze_label_consistency(episodes, cfg)
    plot_label_consistency(consistency, os.path.join(RESULTS_DIR, "label_consistency.png"))

    sc = consistency["switch_counts"]
    dr = consistency["dominant_ratios"]
    print(f"  窗口内平均切换次数: {np.mean(sc):.1f}")
    print(f"  窗口内切换>3次的比例: {(sc > 3).mean()*100:.1f}%")
    print(f"  窗口内主导意图占比<50%的比例: {(dr < 0.5).mean()*100:.1f}%")
    print(f"  窗口内主导意图占比平均: {np.mean(dr):.2f}")

    # =====================================================================
    # 4. 速度/加速度统计
    # =====================================================================
    print("\n[4] Speed & acceleration statistics...")
    all_speeds, all_accels = [], []
    for ep in episodes:
        speeds = np.linalg.norm(ep["target_vel"].numpy(), axis=-1)
        accels = np.linalg.norm(ep["accel_cmd"].numpy(), axis=-1)
        all_speeds.extend(speeds.tolist())
        all_accels.extend(accels.tolist())

    all_speeds = np.array(all_speeds)
    all_accels = np.array(all_accels)
    print(f"  速度: mean={np.mean(all_speeds):.2f}, max={np.max(all_speeds):.2f}, "
          f">2m/s: {(all_speeds > 2).mean()*100:.1f}%")
    print(f"  加速度: mean={np.mean(all_accels):.2f}, max={np.max(all_accels):.2f}, "
          f">2m/s²: {(all_accels > 2).mean()*100:.1f}%")

    # =====================================================================
    # 5. MVS=0 消融
    # =====================================================================
    print("\n[5] MVS=0 ablation experiment...")
    ckpt_path = os.path.join(CKPT_DIR, "predictor_phase2.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = IntentConditionedGMMPredictor(**ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # 收集评估数据
        dataset = PredictorDataset(t_obs=cfg.t_obs, t_pred=cfg.t_pred,
                                    w_mvs=cfg.w_mvs, slide_step=cfg.slide_step)
        for ep in episodes:
            # 需要转成 recorder 格式
            T = ep["n_steps"]
            sub_ep = {
                "target_pos": ep["target_pos"][::cfg.subsample],
                "target_vel": ep["target_vel"][::cfg.subsample],
                "pursuer_pos": ep["pursuer_pos"][::cfg.subsample],
                "pursuer_vel": ep["pursuer_pos"][::cfg.subsample] * 0,  # approx
                "accel_cmd": ep["accel_cmd"][::cfg.subsample],
                "strategy_ids": torch.tensor(ep["strategy_id"])[::cfg.subsample],
            }
            dataset.add_episode(sub_ep)

        if len(dataset) > 0:
            ablation = run_ablation_mvs0(model, dataset, cfg)
            plot_ablation(ablation, os.path.join(RESULTS_DIR, "ablation_mvs0.png"))

            print(f"\n  {'':15} {'Accuracy':>10} {'ADE':>8} {'FDE':>8}")
            print(f"  {'-'*45}")
            for k, label in [("mvs_real", "MVS=Real"), ("mvs_zero", "MVS=0")]:
                r = ablation[k]
                print(f"  {label:15} {r['accuracy']:>10.4f} {r['ade']:>8.4f} {r['fde']:>8.4f}")
    else:
        print(f"  [SKIP] No checkpoint found at {ckpt_path}")

    # =====================================================================
    # 总结
    # =====================================================================
    print("\n" + "=" * 60)
    print("  诊断完成!")
    print(f"  所有图表保存在: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
