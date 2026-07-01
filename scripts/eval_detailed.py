#!/usr/bin/env python3
"""
详细评估与可视化脚本

生成内容:
  1. 意图分类混淆矩阵
  2. 每类意图的分类准确率
  3. 每类意图的预测轨迹可视化 (真实轨迹 + 3条GMM分量 + 2σ包络)
  4. 不同策略下的预测性能对比
  5. MVS分箱详细统计
  6. 意图分布饼图

运行:
    cd /data/uavlab/maneuverable-target
    python3 scripts/eval_detailed.py
"""

import sys, os, math, time
import numpy as np
from pathlib import Path

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

import torch
import types

# 绕过 omni_drones/__init__.py
omni_drones_pkg = types.ModuleType("omni_drones")
omni_drones_pkg.__path__ = [os.path.join(project_root, "omni_drones")]
omni_drones_pkg.__package__ = "omni_drones"
sys.modules["omni_drones"] = omni_drones_pkg

omni_drones_learning = types.ModuleType("omni_drones.learning")
omni_drones_learning.__path__ = [os.path.join(project_root, "omni_drones", "learning")]
omni_drones_learning.__package__ = "omni_drones.learning"
sys.modules["omni_drones.learning"] = omni_drones_learning

omni_drones_envs = types.ModuleType("omni_drones.envs")
omni_drones_envs.__path__ = [os.path.join(project_root, "omni_drones", "envs")]
omni_drones_envs.__package__ = "omni_drones.envs"
sys.modules["omni_drones.envs"] = omni_drones_envs

omni_drones_envs_has = types.ModuleType("omni_drones.envs.hide_and_seek")
omni_drones_envs_has.__path__ = [os.path.join(project_root, "omni_drones", "envs", "hide_and_seek")]
omni_drones_envs_has.__package__ = "omni_drones.envs.hide_and_seek"
sys.modules["omni_drones.envs.hide_and_seek"] = omni_drones_envs_has

from omni_drones.learning.predictor.intent_gmm_predictor import IntentConditionedGMMPredictor, INTENT_LABELS
from omni_drones.learning.predictor.losses import IntentLoss, IntentGMMLoss
from omni_drones.learning.predictor.mvs import compute_mvs_batch
from omni_drones.learning.predictor.data_pipeline import (
    PredictorDataset, TrajectoryRecorder, label_intent_batch,
)
from omni_drones.envs.hide_and_seek.target_fsm import TargetFSM, STRATEGY_NAMES
from omni_drones.envs.hide_and_seek.proportional_navigation import proportional_navigation_3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# 中文字体
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = os.path.join(project_root, "results")
CKPT_DIR = os.path.join(project_root, "checkpoints")

# 意图的颜色映射
INTENT_COLORS = [
    "#2ecc71",  # 0 Straight - green
    "#3498db",  # 1 Left-Turn - blue
    "#e74c3c",  # 2 Right-Turn - red
    "#9b59b6",  # 3 Climb - purple
    "#e67e22",  # 4 Dive - orange
    "#1abc9c",  # 5 Left-Climb - teal
    "#f39c12",  # 6 Left-Dive - yellow
    "#2980b9",  # 7 Right-Climb - dark blue
    "#c0392b",  # 8 Right-Dive - dark red
    "#95a5a6",  # 9 Hover - gray
]

STRATEGY_COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#e67e22"]


# ==========================================================================
# 复用 train_predictor.py 的模拟器
# ==========================================================================

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


def simulate_episode_with_tracking(cfg, fsm, env_id=0):
    """同 train_predictor.py 的 simulate_episode, 但额外跟踪策略ID per frame"""
    recorder = TrajectoryRecorder(subsample_rate=cfg.subsample)
    device = "cpu"

    target_pos = torch.tensor([-3.5 + np.random.uniform(-1, 1),
                                np.random.uniform(-2, 2),
                                np.random.uniform(1.0, 3.5)], device=device)
    target_vel = torch.zeros(3, device=device)
    target_vel[0] = np.random.uniform(0.5, 2.0)

    pursuer_pos = torch.zeros(cfg.n_pursuers, 3, device=device)
    pursuer_vel = torch.zeros(cfg.n_pursuers, 3, device=device)
    for p in range(cfg.n_pursuers):
        pursuer_pos[p] = torch.tensor([
            cfg.goal_center[0] + np.random.uniform(-1.5, 0.5),
            np.random.uniform(-2, 2),
            np.random.uniform(1.0, 3.5),
        ])

    fsm.reset(torch.tensor([env_id]))
    goal_attract = 1.0
    repulsion_coef = 1.0
    velocity_damping = 0.25

    for step in range(cfg.max_steps):
        goal_offset = cfg.goal_center - target_pos
        goal_dist = torch.norm(goal_offset).clamp(min=1e-5)
        f_goal = goal_attract * goal_offset / goal_dist

        f_repulsion = torch.zeros(3, device=device)
        for p in range(cfg.n_pursuers):
            diff = target_pos - pursuer_pos[p]
            d = torch.norm(diff).clamp(min=1e-5)
            f_repulsion += repulsion_coef * diff / (d ** 2 + 1e-5)

        xy_dist = torch.norm(target_pos[:2])
        f_boundary = torch.zeros(3, device=device)
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
        strategy_id = strategy[0]

        total_force = apf_force + fsm_force
        force_norm = torch.norm(total_force).clamp(min=1e-6)
        accel_cmd = cfg.max_accel * total_force / force_norm
        accel_cmd = accel_cmd.clamp(-cfg.max_accel, cfg.max_accel)

        target_vel = target_vel + (accel_cmd - velocity_damping * target_vel) * cfg.dt
        speed = torch.norm(target_vel)
        if speed > cfg.max_speed:
            target_vel = target_vel / speed * cfg.max_speed
        target_pos = target_pos + target_vel * cfg.dt

        xy_d = torch.norm(target_pos[:2])
        if xy_d > cfg.arena_radius:
            target_pos[:2] = target_pos[:2] / xy_d * cfg.arena_radius
        target_pos[2] = target_pos[2].clamp(0.1, cfg.max_height)

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

        recorder.record(target_pos, target_vel, pursuer_pos, pursuer_vel, accel_cmd, strategy_id)

        for p in range(cfg.n_pursuers):
            if torch.norm(target_pos - pursuer_pos[p]) < 0.4:
                return recorder.get_episode()

    return recorder.get_episode()


# ==========================================================================
# 评估函数
# ==========================================================================

def collect_predictions(model, dataset, cfg):
    """收集模型在整个数据集上的预测"""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=False, collate_fn=PredictorDataset.collate_fn)
    model.eval()

    all_results = []
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs_history"]
            gt_future = batch["gt_future_pos"]
            gt_intent = batch["gt_intent_label"]
            intent_hist = batch["intent_history"]
            mvs = compute_mvs_batch(intent_hist)

            output = model(obs, mvs=mvs)
            pred_intent = output["intent_logits"].argmax(dim=-1)
            mixed_mu = output["mixed_mu"]
            mixed_var = output["mixed_var"]

            error = (mixed_mu - gt_future).norm(dim=-1)  # [B, T]

            for i in range(obs.shape[0]):
                all_results.append({
                    "gt_intent": gt_intent[i].item(),
                    "pred_intent": pred_intent[i].item(),
                    "gt_future": gt_future[i],        # [T, 3]
                    "pred_mu": mixed_mu[i],           # [T, 3]
                    "pred_var": mixed_var[i],         # [T, 3]
                    "gmm_means": output["gmm_means"][i],   # [K, T, 3]
                    "gmm_weights": output["gmm_weights"][i],  # [K]
                    "gmm_vars": output["gmm_vars"][i],    # [K, T, 3]
                    "ade": error[i].mean().item(),
                    "fde": error[i, -1].item(),
                    "mvs": mvs[i].item(),
                    "intent_probs": output["intent_probs"][i],  # [10]
                })
    return all_results


# ==========================================================================
# 可视化函数
# ==========================================================================

def plot_confusion_matrix(results, save_path):
    """意图分类混淆矩阵"""
    n_classes = 10
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for r in results:
        cm[r["gt_intent"], r["pred_intent"]] += 1

    # 归一化
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    short_labels = ["Str", "LT", "RT", "Clb", "Div", "LC", "LD", "RC", "RD", "Hov"]
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title("Intent Classification Confusion Matrix", fontsize=14)

    # 标注数值
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            count = cm[i, j]
            if count > 0:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}\n({count})", ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Recall")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return cm


def plot_per_intent_accuracy(results, save_path):
    """每类意图的分类准确率 + 样本数"""
    n_classes = 10
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)

    for r in results:
        gt = r["gt_intent"]
        total[gt] += 1
        if r["pred_intent"] == gt:
            correct[gt] += 1

    accuracy = np.where(total > 0, correct / total, 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})

    # 准确率条形图
    bars = ax1.bar(range(n_classes), accuracy, color=INTENT_COLORS, alpha=0.85, edgecolor="white")
    ax1.set_xticks(range(n_classes))
    ax1.set_xticklabels(INTENT_LABELS, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Classification Accuracy", fontsize=11)
    ax1.set_title("Per-Intent Classification Accuracy", fontsize=14)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=np.mean(accuracy[total > 0]), color="#e74c3c", linestyle="--",
                alpha=0.7, label=f"Mean = {np.mean(accuracy[total > 0]):.3f}")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.2)

    for bar, acc, tot in zip(bars, accuracy, total):
        if tot > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{acc:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 样本数条形图
    ax2.bar(range(n_classes), total, color=INTENT_COLORS, alpha=0.5, edgecolor="white")
    ax2.set_xticks(range(n_classes))
    ax2.set_xticklabels(INTENT_LABELS, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Sample Count", fontsize=11)
    ax2.set_title("Sample Distribution per Intent", fontsize=12)
    ax2.grid(axis="y", alpha=0.2)

    for i, tot in enumerate(total):
        if tot > 0:
            ax2.text(i, tot + max(total) * 0.02, f"{int(tot)}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return accuracy, total


def plot_per_intent_prediction_error(results, save_path):
    """每类意图的轨迹预测误差"""
    n_classes = 10
    ade_by_class = {i: [] for i in range(n_classes)}
    fde_by_class = {i: [] for i in range(n_classes)}

    for r in results:
        gt = r["gt_intent"]
        ade_by_class[gt].append(r["ade"])
        fde_by_class[gt].append(r["fde"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ADE
    ade_means = [np.mean(ade_by_class[i]) if ade_by_class[i] else 0 for i in range(n_classes)]
    ade_stds = [np.std(ade_by_class[i]) if ade_by_class[i] else 0 for i in range(n_classes)]
    x = range(n_classes)
    bars1 = ax1.bar(x, ade_means, yerr=ade_stds, color=INTENT_COLORS, alpha=0.8,
                     edgecolor="white", capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(INTENT_LABELS, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("ADE (m)", fontsize=11)
    ax1.set_title("Average Displacement Error by Intent", fontsize=13)
    ax1.grid(axis="y", alpha=0.2)
    for bar, m in zip(bars1, ade_means):
        if m > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    # FDE
    fde_means = [np.mean(fde_by_class[i]) if fde_by_class[i] else 0 for i in range(n_classes)]
    fde_stds = [np.std(fde_by_class[i]) if fde_by_class[i] else 0 for i in range(n_classes)]
    bars2 = ax2.bar(x, fde_means, yerr=fde_stds, color=INTENT_COLORS, alpha=0.8,
                     edgecolor="white", capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(INTENT_LABELS, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("FDE (m)", fontsize=11)
    ax2.set_title("Final Displacement Error by Intent", fontsize=13)
    ax2.grid(axis="y", alpha=0.2)
    for bar, m in zip(bars2, fde_means):
        if m > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return ade_means, fde_means


def plot_per_intent_trajectories(results, save_path):
    """每类意图的典型预测轨迹可视化 (2×5网格, 每类选1个典型样本)"""
    n_classes = 10

    # 按意图分组, 每类选一个中等误差的样本 (不选最好也不选最差)
    by_class = {i: [] for i in range(n_classes)}
    for r in results:
        by_class[r["gt_intent"]].append(r)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for cls_id in range(n_classes):
        ax = axes[cls_id]
        samples = by_class[cls_id]

        if len(samples) == 0:
            ax.set_title(f"{INTENT_LABELS[cls_id]}\n(No samples)", fontsize=10)
            ax.set_facecolor("#f8f9fa")
            continue

        # 选中位数误差的样本
        samples_sorted = sorted(samples, key=lambda x: x["ade"])
        idx = len(samples_sorted) // 2
        s = samples_sorted[idx]

        gt = s["gt_future"].numpy()         # [T, 3]
        pred_mu = s["pred_mu"].numpy()      # [T, 3]
        pred_var = s["pred_var"].numpy()     # [T, 3]
        gmm_means = s["gmm_means"].numpy()  # [K, T, 3]
        gmm_weights = s["gmm_weights"].numpy()  # [K]
        gmm_vars = s["gmm_vars"].numpy()    # [K, T, 3]

        T = gt.shape[0]
        K = gmm_means.shape[0]

        # 画XY平面投影
        # GT轨迹
        ax.plot(gt[:, 0], gt[:, 1], "k-o", markersize=3, linewidth=2,
                label="GT", zorder=5)

        # 混合均值
        ax.plot(pred_mu[:, 0], pred_mu[:, 1], "--", color="#e74c3c",
                linewidth=1.5, label="Mixed Mean", zorder=4)

        # 2σ包络 (XY平面投影)
        sigma_xy = np.sqrt(pred_var[:, :2].mean(axis=1))  # [T] 平均XY标准差
        for t in range(T):
            circle = plt.Circle((pred_mu[t, 0], pred_mu[t, 1]),
                                2 * sigma_xy[t], color="#e74c3c", alpha=0.08, zorder=1)
            ax.add_patch(circle)

        # K条GMM分量轨迹
        gmm_colors = ["#3498db", "#2ecc71", "#f39c12"]
        for k in range(K):
            w = gmm_weights[k]
            ax.plot(gmm_means[k, :, 0], gmm_means[k, :, 1],
                    "-", color=gmm_colors[k], alpha=0.5 + 0.5 * w,
                    linewidth=1, label=f"K{k} (w={w:.2f})", zorder=3)

        # 起点标记
        ax.plot(gt[0, 0], gt[0, 1], "k^", markersize=8, zorder=6)

        ax.set_title(f"{INTENT_LABELS[cls_id]} (n={len(samples)})\n"
                     f"ADE={s['ade']:.3f}, MVS={s['mvs']:.2f}",
                     fontsize=10, color=INTENT_COLORS[cls_id], fontweight="bold")
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.7)

    plt.suptitle("Per-Intent Trajectory Predictions (XY plane, median ADE sample)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_intent_trajectories_3d_side(results, save_path):
    """每类意图的XZ侧视图 (高度维度)"""
    n_classes = 10
    by_class = {i: [] for i in range(n_classes)}
    for r in results:
        by_class[r["gt_intent"]].append(r)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for cls_id in range(n_classes):
        ax = axes[cls_id]
        samples = by_class[cls_id]

        if len(samples) == 0:
            ax.set_title(f"{INTENT_LABELS[cls_id]}\n(No samples)", fontsize=10)
            ax.set_facecolor("#f8f9fa")
            continue

        samples_sorted = sorted(samples, key=lambda x: x["ade"])
        idx = len(samples_sorted) // 2
        s = samples_sorted[idx]

        gt = s["gt_future"].numpy()
        pred_mu = s["pred_mu"].numpy()
        gmm_means = s["gmm_means"].numpy()
        gmm_weights = s["gmm_weights"].numpy()

        # XZ侧视图
        ax.plot(gt[:, 0], gt[:, 2], "k-o", markersize=3, linewidth=2, label="GT", zorder=5)
        ax.plot(pred_mu[:, 0], pred_mu[:, 2], "--", color="#e74c3c",
                linewidth=1.5, label="Mixed Mean", zorder=4)

        gmm_colors = ["#3498db", "#2ecc71", "#f39c12"]
        K = gmm_means.shape[0]
        for k in range(K):
            w = gmm_weights[k]
            ax.plot(gmm_means[k, :, 0], gmm_means[k, :, 2],
                    "-", color=gmm_colors[k], alpha=0.5 + 0.5 * w,
                    linewidth=1, label=f"K{k} (w={w:.2f})", zorder=3)

        ax.plot(gt[0, 0], gt[0, 2], "k^", markersize=8, zorder=6)

        ax.set_title(f"{INTENT_LABELS[cls_id]} (XZ side view)",
                     fontsize=10, color=INTENT_COLORS[cls_id], fontweight="bold")
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Z (height)", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.7)

    plt.suptitle("Per-Intent Trajectory Predictions (XZ side view, height dimension)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_strategy_comparison(results, dataset, save_path):
    """不同FSM策略下的预测性能对比 (基于MVS区间近似)"""
    # 按MVS分组 (无法直接得到策略ID, 用MVS作为proxy)
    mvs_bins = {
        "Low MVS [0, 0.2)": (0, 0.2),
        "Med-Low MVS [0.2, 0.35)": (0.2, 0.35),
        "Med MVS [0.35, 0.5)": (0.35, 0.5),
        "High MVS [0.5+)": (0.5, 1.01),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bin_names = list(mvs_bins.keys())
    bin_ade, bin_fde, bin_var, bin_count = [], [], [], []

    for name, (lo, hi) in mvs_bins.items():
        group = [r for r in results if lo <= r["mvs"] < hi]
        bin_count.append(len(group))
        if group:
            bin_ade.append(np.mean([r["ade"] for r in group]))
            bin_fde.append(np.mean([r["fde"] for r in group]))
            bin_var.append(np.mean([r["pred_var"].mean().item() for r in group]))
        else:
            bin_ade.append(0)
            bin_fde.append(0)
            bin_var.append(0)

    colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]

    bars1 = axes[0].bar(bin_names, bin_ade, color=colors, alpha=0.8, edgecolor="white")
    axes[0].set_title("ADE by MVS Region", fontsize=12)
    axes[0].set_ylabel("ADE (m)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.2)
    for b, v in zip(bars1, bin_ade):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)

    bars2 = axes[1].bar(bin_names, bin_fde, color=colors, alpha=0.8, edgecolor="white")
    axes[1].set_title("FDE by MVS Region", fontsize=12)
    axes[1].set_ylabel("FDE (m)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.2)
    for b, v in zip(bars2, bin_fde):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)

    bars3 = axes[2].bar(bin_names, bin_count, color=colors, alpha=0.8, edgecolor="white")
    axes[2].set_title("Sample Count by MVS Region", fontsize=12)
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.2)
    for b, v in zip(bars3, bin_count):
        axes[2].text(b.get_x() + b.get_width()/2, b.get_height(), f"{v}",
                     ha="center", va="bottom", fontsize=9)

    plt.suptitle("Prediction Performance by Behavioral Stability (MVS)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_intent_distribution_pie(results, save_path):
    """意图分布饼图"""
    n_classes = 10
    counts = np.zeros(n_classes)
    for r in results:
        counts[r["gt_intent"]] += 1

    # 去掉0的类
    nonzero = counts > 0
    labels = [INTENT_LABELS[i] for i in range(n_classes) if nonzero[i]]
    sizes = counts[nonzero]
    colors = [INTENT_COLORS[i] for i in range(n_classes) if nonzero[i]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # GT分布
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.85, textprops={"fontsize": 8})
    ax1.set_title("Ground Truth Intent Distribution", fontsize=12)

    # 预测分布
    pred_counts = np.zeros(n_classes)
    for r in results:
        pred_counts[r["pred_intent"]] += 1
    pred_nonzero = pred_counts > 0
    pred_labels = [INTENT_LABELS[i] for i in range(n_classes) if pred_nonzero[i]]
    pred_sizes = pred_counts[pred_nonzero]
    pred_colors = [INTENT_COLORS[i] for i in range(n_classes) if pred_nonzero[i]]

    ax2.pie(pred_sizes, labels=pred_labels, colors=pred_colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.85, textprops={"fontsize": 8})
    ax2.set_title("Predicted Intent Distribution", fontsize=12)

    plt.suptitle("Intent Distribution: Ground Truth vs Predicted", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def generate_text_report(results, cm, accuracy, total, ade_means, fde_means, report_path):
    """生成文本报告"""
    n_classes = 10
    overall_acc = sum(1 for r in results if r["gt_intent"] == r["pred_intent"]) / len(results)
    overall_ade = np.mean([r["ade"] for r in results])
    overall_fde = np.mean([r["fde"] for r in results])

    lines = []
    lines.append("=" * 70)
    lines.append("  意图识别 + GMM轨迹预测 — 详细评估报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  评估样本数: {len(results)}")
    lines.append(f"  整体分类准确率: {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    lines.append(f"  整体 ADE: {overall_ade:.4f} m")
    lines.append(f"  整体 FDE: {overall_fde:.4f} m")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  每类意图详细指标")
    lines.append("-" * 70)
    header = f"  {'Intent':<14} {'Samples':>8} {'Accuracy':>10} {'ADE':>8} {'FDE':>8}"
    lines.append(header)
    lines.append("  " + "-" * 55)

    for i in range(n_classes):
        n = int(total[i])
        acc = accuracy[i]
        ade = ade_means[i]
        fde = fde_means[i]
        if n > 0:
            lines.append(f"  {INTENT_LABELS[i]:<14} {n:>8} {acc:>10.4f} {ade:>8.4f} {fde:>8.4f}")
        else:
            lines.append(f"  {INTENT_LABELS[i]:<14} {n:>8}      N/A      N/A      N/A")

    lines.append("")
    lines.append("-" * 70)
    lines.append("  MVS调制统计")
    lines.append("-" * 70)

    mvs_vals = [r["mvs"] for r in results]
    lines.append(f"  MVS 范围: [{min(mvs_vals):.4f}, {max(mvs_vals):.4f}]")
    lines.append(f"  MVS 均值: {np.mean(mvs_vals):.4f}")
    lines.append(f"  MVS 中位数: {np.median(mvs_vals):.4f}")

    # 按MVS分箱
    for lo, hi, name in [(0, 0.2, "Low [0,0.2)"), (0.2, 0.4, "Med-L [0.2,0.4)"),
                          (0.4, 0.6, "Med-H [0.4,0.6)"), (0.6, 1.01, "High [0.6,1]")]:
        group = [r for r in results if lo <= r["mvs"] < hi]
        if group:
            g_ade = np.mean([r["ade"] for r in group])
            g_fde = np.mean([r["fde"] for r in group])
            g_var = np.mean([r["pred_var"].mean().item() for r in group])
            lines.append(f"  {name:<18} n={len(group):>5}, ADE={g_ade:.4f}, FDE={g_fde:.4f}, Var={g_var:.4f}")
        else:
            lines.append(f"  {name:<18} n=    0")

    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\n  Report saved: {report_path}")


# ==========================================================================
# 主入口
# ==========================================================================

def main():
    print("=" * 60)
    print("  详细评估与可视化")
    print("=" * 60)

    cfg = Config()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # =====================================================================
    # 1. 加载模型
    # =====================================================================
    ckpt_path = os.path.join(CKPT_DIR, "predictor_phase2.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        print("  请先运行 python3 scripts/train_predictor.py")
        return

    print(f"\n[1] Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ckpt["config"]
    model = IntentConditionedGMMPredictor(**model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # =====================================================================
    # 2. 收集评估数据
    # =====================================================================
    print(f"\n[2] Collecting evaluation data (100 episodes)...")
    fsm = TargetFSM(num_envs=1, device="cpu", max_speed=cfg.max_speed,
                     max_accel=cfg.max_accel, arena_radius=cfg.arena_radius,
                     max_height=cfg.max_height, evasion_threshold=3.0)

    dataset = PredictorDataset(
        t_obs=cfg.t_obs, t_pred=cfg.t_pred,
        w_mvs=cfg.w_mvs, slide_step=cfg.slide_step,
    )

    t_start = time.time()
    for ep in range(100):
        episode = simulate_episode_with_tracking(cfg, fsm, env_id=0)
        if episode is not None:
            dataset.add_episode(episode)
        if (ep + 1) % 25 == 0:
            print(f"  Episode {ep+1}/100, samples: {len(dataset)}")

    print(f"  Data collection: {time.time()-t_start:.1f}s, {len(dataset)} samples")
    stats = dataset.get_stats()
    print(f"  Intent distribution: {stats['intent_distribution']}")

    # =====================================================================
    # 3. 收集预测结果
    # =====================================================================
    print(f"\n[3] Running predictions...")
    results = collect_predictions(model, dataset, cfg)
    print(f"  Collected {len(results)} predictions")

    # =====================================================================
    # 4. 生成可视化
    # =====================================================================
    print(f"\n[4] Generating visualizations...")

    # 4a. 混淆矩阵
    cm = plot_confusion_matrix(results, os.path.join(RESULTS_DIR, "confusion_matrix.png"))

    # 4b. 每类准确率
    accuracy, total = plot_per_intent_accuracy(results,
                                                os.path.join(RESULTS_DIR, "per_intent_accuracy.png"))

    # 4c. 每类预测误差
    ade_means, fde_means = plot_per_intent_prediction_error(
        results, os.path.join(RESULTS_DIR, "per_intent_error.png"))

    # 4d. 意图分布饼图
    plot_intent_distribution_pie(results, os.path.join(RESULTS_DIR, "intent_distribution.png"))

    # 4e. 每类意图轨迹可视化 (XY俯视)
    plot_per_intent_trajectories(results, os.path.join(RESULTS_DIR, "per_intent_traj_xy.png"))

    # 4f. 每类意图轨迹可视化 (XZ侧视)
    plot_per_intent_trajectories_3d_side(results, os.path.join(RESULTS_DIR, "per_intent_traj_xz.png"))

    # 4g. 按MVS区间的性能对比
    plot_strategy_comparison(results, dataset,
                             os.path.join(RESULTS_DIR, "strategy_comparison.png"))

    # =====================================================================
    # 5. 文本报告
    # =====================================================================
    print(f"\n[5] Generating text report...")
    generate_text_report(results, cm, accuracy, total, ade_means, fde_means,
                         os.path.join(RESULTS_DIR, "evaluation_report.txt"))

    print("\n" + "=" * 60)
    print("  All visualizations saved to:", RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
