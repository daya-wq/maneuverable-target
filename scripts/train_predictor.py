#!/usr/bin/env python3
"""
Phase 4: 距离驱动目标 + 缩短时间窗 — 完整训练与评估

改动:
  - 目标FSM: 距离驱动策略 (每步评估, 不再1秒间隔)
  - T_obs=5帧(0.25s), W_mvs=10帧(0.5s), T_pred=5帧(0.25s)
  - 500 episodes
  - 初始距离 7-8m (场地扩大到10m以容纳)

输出:
  results/phase4_distance_driven/
    - confusion_matrix.png
    - per_intent_metrics.png
    - traj_vs_gt.png           (高权重/均值轨迹 vs 真实轨迹)
    - ablation_mvs.png
    - training_curves.png
    - battle_3d_1~10.png       (3D对抗轨迹)
    - report.txt               (文字报告)

运行:
    cd /data/uavlab/maneuverable-target
    python3 scripts/train_predictor.py
"""

import sys, os, math, time
import numpy as np
from pathlib import Path

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import types

# 模块路径注册
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

from omni_drones.learning.predictor.intent_gmm_predictor import IntentConditionedGMMPredictor
from omni_drones.learning.predictor.losses import IntentLoss, IntentGMMLoss
from omni_drones.learning.predictor.mvs import compute_mvs_batch
from omni_drones.learning.predictor.data_pipeline import (
    PredictorDataset, TrajectoryRecorder, label_intent_batch, INTENT_LABELS,
)
from omni_drones.envs.hide_and_seek.target_fsm import TargetFSM, STRATEGY_NAMES
from omni_drones.envs.hide_and_seek.proportional_navigation import proportional_navigation_3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ==========================================================================
# 配置
# ==========================================================================

class Config:
    # 场景
    arena_radius = 10.0     # 扩大场地, 容纳7-8m初始距离
    max_height = 5.0
    max_speed = 3.0
    max_accel = 3.0
    n_pursuers = 3
    dt = 0.01               # 物理步 100Hz
    subsample = 5            # 下采样 → 20Hz
    dt_ctrl = 0.05           # 网络时间步 0.05s

    # episode
    max_steps = 2000         # 20秒
    n_episodes_collect = 500
    init_dist_range = (7.0, 8.0)
    capture_radius = 0.4

    # 时间窗 (帧数, @20Hz)
    t_obs = 5                # 短时意图 0.25s
    t_pred = 5               # 预测 0.25s
    w_mvs = 10               # 意图变换 0.5s
    slide_step = 2           # 滑窗步长

    # 守区
    goal_center = torch.tensor([8.0, 0.0, 2.5], dtype=torch.float32)

    # 训练
    batch_size = 64
    lr_phase1 = 1e-3
    lr_phase2 = 5e-4
    epochs_phase1 = 30
    epochs_phase2 = 40
    grad_clip = 5.0
    hidden_dim = 128

    # 输出
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    results_dir = os.path.join(project_root, "results", "phase4_distance_driven")


# ==========================================================================
# 合成仿真器
# ==========================================================================

def simulate_episode(cfg, fsm, ep_id=0):
    """模拟一个episode, 使用距离驱动FSM"""
    device = "cpu"
    recorder = TrajectoryRecorder(subsample_rate=cfg.subsample)

    # 目标初始位置: 场地中心附近
    angle_t = np.random.uniform(0, 2 * np.pi)
    r_t = np.random.uniform(1.0, 3.0)
    target_pos = torch.tensor([
        r_t * np.cos(angle_t),
        r_t * np.sin(angle_t),
        np.random.uniform(1.5, 3.5),
    ], dtype=torch.float32)
    target_vel = torch.zeros(3, dtype=torch.float32)
    v_angle = np.random.uniform(0, 2 * np.pi)
    target_vel[0] = float(np.random.uniform(1.0, 2.5) * np.cos(v_angle))
    target_vel[1] = float(np.random.uniform(1.0, 2.5) * np.sin(v_angle))
    target_vel[2] = float(np.random.uniform(-0.5, 0.5))

    # 追捕方初始位置: 距目标 7-8m
    init_dist = np.random.uniform(*cfg.init_dist_range)
    pursuer_pos = torch.zeros(cfg.n_pursuers, 3, dtype=torch.float32)
    pursuer_vel = torch.zeros(cfg.n_pursuers, 3, dtype=torch.float32)
    for p in range(cfg.n_pursuers):
        p_angle = angle_t + np.pi + np.random.uniform(-0.5, 0.5)
        p_angle += p * (2 * np.pi / cfg.n_pursuers)
        p_r = init_dist * np.random.uniform(0.9, 1.1)
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

    # 记录初始距离
    init_dists = torch.tensor([torch.norm(target_pos - pursuer_pos[p]).item()
                                for p in range(cfg.n_pursuers)])
    d0 = init_dists.min()

    # FSM重置 + 设置初始距离
    fsm.reset(torch.tensor([0]))
    fsm.set_initial_distance(torch.tensor([0]), d0.unsqueeze(0))

    velocity_damping = 0.3

    for step in range(cfg.max_steps):
        # APF基础力
        goal_offset = cfg.goal_center - target_pos
        goal_dist = torch.norm(goal_offset).clamp(min=1e-5)
        f_goal = 1.0 * goal_offset / goal_dist

        f_repulsion = torch.zeros(3, dtype=torch.float32)
        for p in range(cfg.n_pursuers):
            diff = target_pos - pursuer_pos[p]
            d = torch.norm(diff).clamp(min=1e-5)
            f_repulsion += 1.5 * diff / (d ** 2 + 1e-5)

        f_boundary = torch.zeros(3, dtype=torch.float32)
        xy_dist = torch.norm(target_pos[:2])
        if xy_dist > cfg.arena_radius * 0.85:
            f_boundary[:2] = -target_pos[:2] / (xy_dist + 1e-5) * 3.0
        if target_pos[2] > cfg.max_height * 0.9:
            f_boundary[2] -= 3.0
        if target_pos[2] < 0.3:
            f_boundary[2] += 3.0

        apf_force = f_goal + f_repulsion + f_boundary

        # FSM策略力 (距离驱动, 每步评估)
        fsm_force, strategy = fsm.compute_strategy_force(
            target_pos.unsqueeze(0), target_vel.unsqueeze(0),
            pursuer_pos.unsqueeze(0), goal_center=cfg.goal_center,
        )
        fsm_force = fsm_force.squeeze(0)
        strategy_id = strategy[0]

        # 近距离时: 策略力主导, APF减弱
        min_d = min(torch.norm(target_pos - pursuer_pos[p]).item() for p in range(cfg.n_pursuers))
        if min_d < d0.item() * 0.4:
            # 近距: 策略力为主导, APF只保留30%
            total_force = 0.3 * apf_force + fsm_force
        else:
            total_force = apf_force + fsm_force

        force_norm = torch.norm(total_force).clamp(min=1e-6)
        accel_cmd = cfg.max_accel * total_force / force_norm

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

        # 追捕方: 比例导引
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

        # 记录
        recorder.record(target_pos, target_vel, pursuer_pos, pursuer_vel, accel_cmd, strategy_id)

        # 围捕判定
        if min_d < cfg.capture_radius:
            break

    return recorder.get_episode()


# ==========================================================================
# 训练循环
# ==========================================================================

def train_phase(model, dataset, loss_fn, optimizer, cfg, phase_name, n_epochs, use_mvs=False):
    """通用训练循环"""
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size,
        shuffle=True, collate_fn=PredictorDataset.collate_fn,
    )

    history = {"loss": [], "accuracy": [], "traj_mse": []}
    model.train()

    for epoch in range(n_epochs):
        epoch_loss, epoch_acc, epoch_mse = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in loader:
            obs = batch["obs_history"]
            gt_intent = batch["gt_intent_label"]
            optimizer.zero_grad()

            if use_mvs:
                mvs = compute_mvs_batch(batch["intent_history"])
                output = model(obs, mvs=mvs)
                gt_future = batch["gt_future_pos"]
                loss, metrics = loss_fn(output, gt_future, gt_intent)
                epoch_mse += metrics.get("traj_mse", 0)
            else:
                output = model.predict_intent_only(obs)
                loss, metrics = loss_fn(output, gt_intent)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += metrics.get("intent_accuracy", 0)
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        avg_mse = epoch_mse / n_batches if use_mvs else 0

        history["loss"].append(avg_loss)
        history["accuracy"].append(avg_acc)
        history["traj_mse"].append(avg_mse)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  [{phase_name}] Epoch {epoch:3d}/{n_epochs}: "
                  f"loss={avg_loss:.4f}, acc={avg_acc:.4f}" +
                  (f", mse={avg_mse:.4f}" if use_mvs else ""))

    return history


# ==========================================================================
# 评估: 混淆矩阵 + 每类指标 + 高权重轨迹 + MVS消融
# ==========================================================================

def full_evaluation(model, dataset, cfg):
    """完整评估, 生成所有图表"""
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size,
        shuffle=False, collate_fn=PredictorDataset.collate_fn,
    )
    model.eval()

    all_pred_intent = []
    all_gt_intent = []
    all_ade = []
    all_fde = []
    all_mvs = []
    # 存几个样本用于轨迹对比
    sample_outputs = []
    sample_gt = []
    sample_count = 0

    # MVS消融
    ablation = {"real": {"ade": [], "fde": [], "acc": [], "nll": []},
                "zero": {"ade": [], "fde": [], "acc": [], "nll": []}}

    loss_fn = IntentGMMLoss()

    with torch.no_grad():
        for batch in loader:
            obs = batch["obs_history"]
            gt_future = batch["gt_future_pos"]
            gt_intent = batch["gt_intent_label"]
            mvs_real = compute_mvs_batch(batch["intent_history"])
            B = obs.shape[0]

            # --- MVS=Real ---
            output = model(obs, mvs=mvs_real)
            pred_intent = output["intent_logits"].argmax(dim=-1)
            all_pred_intent.extend(pred_intent.tolist())
            all_gt_intent.extend(gt_intent.tolist())
            all_mvs.extend(mvs_real.squeeze(-1).tolist())

            error = (output["mixed_mu"] - gt_future).norm(dim=-1)
            ade = error.mean(dim=-1)
            fde = error[:, -1]
            all_ade.extend(ade.tolist())
            all_fde.extend(fde.tolist())

            acc_real = (pred_intent == gt_intent).float().mean().item()
            _, metrics_real = loss_fn(output, gt_future, gt_intent)
            ablation["real"]["ade"].extend(ade.tolist())
            ablation["real"]["fde"].extend(fde.tolist())
            ablation["real"]["acc"].append(acc_real)
            ablation["real"]["nll"].append(metrics_real.get("loss_gmm_nll", 0))

            # 保存几个样本做轨迹对比
            if sample_count < 8:
                for i in range(min(B, 8 - sample_count)):
                    sample_outputs.append({
                        "gmm_weights": output["gmm_weights"][i].cpu(),
                        "gmm_means": output["gmm_means"][i].cpu(),
                        "mixed_mu": output["mixed_mu"][i].cpu(),
                    })
                    sample_gt.append(gt_future[i].cpu())
                    sample_count += 1

            # --- MVS=0 ---
            output0 = model(obs, mvs=torch.zeros(B, 1))
            pred0 = output0["intent_logits"].argmax(dim=-1)
            err0 = (output0["mixed_mu"] - gt_future).norm(dim=-1)
            acc0 = (pred0 == gt_intent).float().mean().item()
            _, metrics0 = loss_fn(output0, gt_future, gt_intent)
            ablation["zero"]["ade"].extend(err0.mean(dim=-1).tolist())
            ablation["zero"]["fde"].extend(err0[:, -1].tolist())
            ablation["zero"]["acc"].append(acc0)
            ablation["zero"]["nll"].append(metrics0.get("loss_gmm_nll", 0))

    return {
        "pred_intent": all_pred_intent,
        "gt_intent": all_gt_intent,
        "ade": all_ade,
        "fde": all_fde,
        "mvs": all_mvs,
        "ablation": ablation,
        "sample_outputs": sample_outputs,
        "sample_gt": sample_gt,
    }


def plot_confusion_matrix(eval_data, save_dir):
    """混淆矩阵"""
    from collections import Counter
    n_classes = 10
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for gt, pred in zip(eval_data["gt_intent"], eval_data["pred_intent"]):
        cm[gt, pred] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    # 标注数字
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(INTENT_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(INTENT_LABELS, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Intent Classification Confusion Matrix\n"
                 f"Overall Accuracy: {np.trace(cm)/cm.sum()*100:.1f}%")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_per_intent_metrics(eval_data, save_dir):
    """每类意图准确率 + ADE/FDE"""
    n_classes = 10
    per_acc = []
    per_ade = []
    per_fde = []
    per_count = []

    for c in range(n_classes):
        mask = [i for i, gt in enumerate(eval_data["gt_intent"]) if gt == c]
        per_count.append(len(mask))
        if len(mask) == 0:
            per_acc.append(0)
            per_ade.append(0)
            per_fde.append(0)
            continue
        correct = sum(1 for i in mask if eval_data["pred_intent"][i] == c)
        per_acc.append(correct / len(mask))
        per_ade.append(np.mean([eval_data["ade"][i] for i in mask]))
        per_fde.append(np.mean([eval_data["fde"][i] for i in mask]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

    axes[0].bar(range(n_classes), per_acc, color=colors)
    axes[0].set_xticks(range(n_classes))
    axes[0].set_xticklabels(INTENT_LABELS, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Per-Intent Classification Accuracy")
    for i, v in enumerate(per_acc):
        axes[0].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)

    axes[1].bar(range(n_classes), per_ade, color=colors)
    axes[1].set_xticks(range(n_classes))
    axes[1].set_xticklabels(INTENT_LABELS, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("ADE (m)")
    axes[1].set_title("Per-Intent ADE")
    for i, v in enumerate(per_ade):
        axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)

    axes[2].bar(range(n_classes), per_count, color=colors)
    axes[2].set_xticks(range(n_classes))
    axes[2].set_xticklabels(INTENT_LABELS, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Sample Count")
    axes[2].set_title("Per-Intent Sample Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_intent_metrics.png"), dpi=150)
    plt.close()


def plot_traj_vs_gt(eval_data, save_dir):
    """高权重轨迹 + 均值轨迹 vs 真实轨迹"""
    n_samples = min(8, len(eval_data["sample_outputs"]))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx in range(n_samples):
        ax = axes[idx // 4, idx % 4]
        out = eval_data["sample_outputs"][idx]
        gt = eval_data["sample_gt"][idx].numpy()  # [T, 3]
        T = gt.shape[0]

        # 均值轨迹
        mixed_mu = out["mixed_mu"].numpy()  # [T, 3]
        ax.plot(range(T), gt[:, 0], 'k-', linewidth=2, label='GT-x', alpha=0.8)
        ax.plot(range(T), gt[:, 1], 'k--', linewidth=2, label='GT-y', alpha=0.8)
        ax.plot(range(T), gt[:, 2], 'k:', linewidth=2, label='GT-z', alpha=0.8)

        ax.plot(range(T), mixed_mu[:, 0], 'r-', linewidth=1.5, label='Mix-x', alpha=0.7)
        ax.plot(range(T), mixed_mu[:, 1], 'r--', linewidth=1.5, label='Mix-y', alpha=0.7)
        ax.plot(range(T), mixed_mu[:, 2], 'r:', linewidth=1.5, label='Mix-z', alpha=0.7)

        # 高权重分量
        weights = out["gmm_weights"].numpy()  # [K]
        means = out["gmm_means"].numpy()  # [K, T, 3]
        best_k = weights.argmax()
        best_traj = means[best_k]  # [T, 3]

        ax.plot(range(T), best_traj[:, 0], 'b-', linewidth=1, label=f'Best(K{best_k})-x', alpha=0.6)
        ax.plot(range(T), best_traj[:, 1], 'b--', linewidth=1, alpha=0.6)
        ax.plot(range(T), best_traj[:, 2], 'b:', linewidth=1, alpha=0.6)

        ade = np.linalg.norm(mixed_mu - gt, axis=-1).mean()
        ax.set_title(f"Sample {idx+1} | w={weights} | ADE={ade:.3f}m", fontsize=8)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, ncol=3)

    plt.suptitle("Trajectory Prediction vs Ground Truth\n"
                 "Black=GT, Red=Mixed Mean, Blue=Best Component", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "traj_vs_gt.png"), dpi=150)
    plt.close()


def plot_ablation_mvs(eval_data, save_dir):
    """MVS消融对比"""
    abl = eval_data["ablation"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    modes = ["MVS=Real", "MVS=0"]
    keys = ["real", "zero"]
    colors = ["#4a90d9", "#95a5a6"]

    metrics = [
        ("acc", "Intent Accuracy ↑"),
        ("ade", "ADE ↓"),
        ("fde", "FDE ↓"),
        ("nll", "NLL ↓"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        vals = []
        for k in keys:
            if metric in ["acc", "nll"]:
                vals.append(np.mean(abl[k][metric]))
            else:
                vals.append(np.mean(abl[k][metric]))

        bars = ax.bar(modes, vals, color=colors, alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Ablation: With MVS vs Without MVS (MVS=0)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_mvs.png"), dpi=150)
    plt.close()


def plot_training_curves(h1, h2, save_dir):
    """训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(h1["loss"], label="Phase 1", color="#4a90d9")
    offset = len(h1["loss"])
    axes[0].plot(range(offset, offset + len(h2["loss"])),
                 h2["loss"], label="Phase 2", color="#e67e22")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(h1["accuracy"], label="Phase 1", color="#4a90d9")
    axes[1].plot(range(offset, offset + len(h2["accuracy"])),
                 h2["accuracy"], label="Phase 2", color="#e67e22")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Intent Accuracy")
    axes[1].set_title("Intent Classification Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(h2["traj_mse"], color="#27ae60")
    axes[2].set_xlabel("Phase 2 Epoch")
    axes[2].set_ylabel("Trajectory MSE")
    axes[2].set_title("Trajectory Prediction MSE")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


# ==========================================================================
# 3D对抗轨迹图 (复用 plot_3d_battle.py 的投影方法)
# ==========================================================================

def project_3d_to_2d(points, elev=25, azim=-60):
    """3D → 2D 正交投影"""
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    cos_a, sin_a = np.cos(azim_rad), np.sin(azim_rad)
    cos_e, sin_e = np.cos(elev_rad), np.sin(elev_rad)
    Rz = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cos_e, -sin_e], [0, sin_e, cos_e]])
    R = Rx @ Rz
    rotated = points @ R.T
    return np.column_stack([rotated[:, 0], rotated[:, 2]])


def simulate_and_plot_battles(cfg, fsm, save_dir, n_scenarios=10):
    """模拟10个场景并保存3D轨迹图"""
    STRATEGY_COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#e67e22"]
    PURSUER_COLORS = ["#e74c3c", "#3498db", "#f39c12"]

    summary = []

    for sc in range(n_scenarios):
        torch.manual_seed(100 + sc * 13)
        np.random.seed(100 + sc * 13)

        # 模拟
        device = "cpu"
        angle_t = np.random.uniform(0, 2 * np.pi)
        r_t = np.random.uniform(1.0, 3.0)
        target_pos = torch.tensor([r_t * np.cos(angle_t), r_t * np.sin(angle_t),
                                    np.random.uniform(1.5, 3.5)], dtype=torch.float32)
        target_vel = torch.zeros(3, dtype=torch.float32)
        v_angle = np.random.uniform(0, 2 * np.pi)
        target_vel[0] = float(np.random.uniform(1.0, 2.5) * np.cos(v_angle))
        target_vel[1] = float(np.random.uniform(1.0, 2.5) * np.sin(v_angle))
        target_vel[2] = float(np.random.uniform(-0.5, 0.5))

        init_dist = np.random.uniform(7.0, 8.0)
        pursuer_pos = torch.zeros(3, 3, dtype=torch.float32)
        pursuer_vel = torch.zeros(3, 3, dtype=torch.float32)
        for p in range(3):
            p_angle = angle_t + np.pi + np.random.uniform(-0.5, 0.5) + p * (2*np.pi/3)
            p_r = init_dist * np.random.uniform(0.9, 1.1)
            pursuer_pos[p] = torch.tensor([
                target_pos[0].item() + p_r * np.cos(p_angle),
                target_pos[1].item() + p_r * np.sin(p_angle),
                np.random.uniform(1.0, 4.0)], dtype=torch.float32)
            xy = torch.norm(pursuer_pos[p, :2])
            if xy > cfg.arena_radius:
                pursuer_pos[p, :2] = pursuer_pos[p, :2] / xy * (cfg.arena_radius - 0.1)
            pursuer_pos[p, 2] = pursuer_pos[p, 2].clamp(0.1, cfg.max_height)

        d0 = min(torch.norm(target_pos - pursuer_pos[p]).item() for p in range(3))
        fsm.reset(torch.tensor([0]))
        fsm.set_initial_distance(torch.tensor([0]), torch.tensor([d0]))

        traj_t, traj_p, strats, min_dists = [], [], [], []
        captured, cap_step = False, -1

        for step in range(cfg.max_steps):
            traj_t.append(target_pos.clone().numpy())
            traj_p.append(pursuer_pos.clone().numpy())

            # APF
            goal_off = cfg.goal_center - target_pos
            f_goal = goal_off / torch.norm(goal_off).clamp(min=1e-5)
            f_rep = torch.zeros(3, dtype=torch.float32)
            for p in range(3):
                diff = target_pos - pursuer_pos[p]
                d = torch.norm(diff).clamp(min=1e-5)
                f_rep += 1.5 * diff / (d**2 + 1e-5)
            f_bnd = torch.zeros(3, dtype=torch.float32)
            xy_d = torch.norm(target_pos[:2])
            if xy_d > cfg.arena_radius * 0.85:
                f_bnd[:2] = -target_pos[:2] / (xy_d + 1e-5) * 3.0
            if target_pos[2] > cfg.max_height * 0.9: f_bnd[2] -= 3.0
            if target_pos[2] < 0.3: f_bnd[2] += 3.0

            fsm_f, strat = fsm.compute_strategy_force(
                target_pos.unsqueeze(0), target_vel.unsqueeze(0),
                pursuer_pos.unsqueeze(0), goal_center=cfg.goal_center)
            fsm_f = fsm_f.squeeze(0)
            strats.append(strat[0].item())

            md = min(torch.norm(target_pos - pursuer_pos[p]).item() for p in range(3))
            min_dists.append(md)
            if md < d0 * 0.4:
                total = 0.3 * (f_goal + f_rep + f_bnd) + fsm_f
            else:
                total = f_goal + f_rep + f_bnd + fsm_f
            fn = torch.norm(total).clamp(min=1e-6)
            accel = cfg.max_accel * total / fn

            target_vel = target_vel + (accel - 0.3 * target_vel) * cfg.dt
            sp = torch.norm(target_vel)
            if sp > cfg.max_speed: target_vel = target_vel / sp * cfg.max_speed
            target_pos = target_pos + target_vel * cfg.dt
            xyd = torch.norm(target_pos[:2])
            if xyd > cfg.arena_radius: target_pos[:2] = target_pos[:2] / xyd * cfg.arena_radius
            target_pos[2] = target_pos[2].clamp(0.1, cfg.max_height)

            for p in range(3):
                a_pn = proportional_navigation_3d(
                    pursuer_pos[p].unsqueeze(0), pursuer_vel[p].unsqueeze(0),
                    target_pos.unsqueeze(0), target_vel.unsqueeze(0),
                    N=4.0, max_accel=cfg.max_accel).squeeze(0)
                pursuer_vel[p] = pursuer_vel[p] + (a_pn - 0.3 * pursuer_vel[p]) * cfg.dt
                ps = torch.norm(pursuer_vel[p])
                if ps > cfg.max_speed: pursuer_vel[p] = pursuer_vel[p] / ps * cfg.max_speed
                pursuer_pos[p] = pursuer_pos[p] + pursuer_vel[p] * cfg.dt
                pxy = torch.norm(pursuer_pos[p, :2])
                if pxy > cfg.arena_radius: pursuer_pos[p, :2] = pursuer_pos[p, :2] / pxy * cfg.arena_radius
                pursuer_pos[p, 2] = pursuer_pos[p, 2].clamp(0.1, cfg.max_height)

            if md < cfg.capture_radius:
                captured = True
                cap_step = step
                break

        tpos = np.array(traj_t)
        ppos = np.array(traj_p)
        T = len(traj_t)
        duration = T * cfg.dt

        # 绘图
        elev, azim = 25, -55 + sc * 8
        fig, ax = plt.subplots(figsize=(12, 10))

        # 场地
        theta = np.linspace(0, 2*np.pi, 72)
        floor = np.column_stack([cfg.arena_radius*np.cos(theta),
                                  cfg.arena_radius*np.sin(theta), np.full(72, 0.1)])
        f2d = project_3d_to_2d(floor, elev, azim)
        ax.plot(f2d[:, 0], f2d[:, 1], "k--", alpha=0.15, linewidth=0.8)

        # XYZ轴
        origin = np.array([-cfg.arena_radius-0.5, -cfg.arena_radius-0.5, 0.0])
        o2d = project_3d_to_2d(np.array([origin]), elev, azim)[0]
        for i, (lbl, clr) in enumerate(zip(["X","Y","Z"], ["#e74c3c","#2ecc71","#3498db"])):
            end = origin.copy(); end[i] += 2.0
            e2d = project_3d_to_2d(np.array([end]), elev, azim)[0]
            ax.annotate("", xy=e2d, xytext=o2d, arrowprops=dict(arrowstyle="->", color=clr, lw=1.5))
            ax.text(e2d[0]+0.15, e2d[1]+0.15, lbl, fontsize=10, color=clr, fontweight="bold")

        # 目标轨迹
        t2d = project_3d_to_2d(tpos, elev, azim)
        for t in range(T-1):
            sid = strats[t]
            c = STRATEGY_COLORS[sid] if 0 <= sid < 5 else "#333"
            ax.plot(t2d[t:t+2, 0], t2d[t:t+2, 1], color=c, linewidth=2.5, alpha=0.85)

        ax.plot(t2d[0,0], t2d[0,1], "k^", markersize=14, zorder=10,
                markeredgecolor="white", markeredgewidth=1.5, label="Target START")
        ax.plot(t2d[-1,0], t2d[-1,1], "kv", markersize=14, zorder=10,
                markeredgecolor="white", markeredgewidth=1.5, label="Target END")

        # 时间标记
        ti = max(1, T // 8)
        for t in range(0, T, ti):
            ax.plot(t2d[t,0], t2d[t,1], "ko", markersize=3, alpha=0.5, zorder=8)
            ax.annotate(f"{t*cfg.dt:.1f}s", xy=(t2d[t,0], t2d[t,1]),
                        xytext=(5,5), textcoords="offset points", fontsize=6, alpha=0.6)

        # 追捕方
        for p in range(3):
            p2d = project_3d_to_2d(ppos[:, p, :], elev, azim)
            ax.plot(p2d[:,0], p2d[:,1], "--", color=PURSUER_COLORS[p], linewidth=1.2, alpha=0.6)
            ax.plot(p2d[0,0], p2d[0,1], "o", color=PURSUER_COLORS[p], markersize=10, zorder=9,
                    markeredgecolor="white", markeredgewidth=1.5, label=f"Pursuer {p+1} START")
            ax.plot(p2d[-1,0], p2d[-1,1], "s", color=PURSUER_COLORS[p], markersize=8, zorder=9,
                    markeredgecolor="white", markeredgewidth=1)

        if captured:
            c2d = project_3d_to_2d(tpos[cap_step:cap_step+1], elev, azim)[0]
            ax.plot(c2d[0], c2d[1], "*", color="#e74c3c", markersize=25, zorder=15,
                    markeredgecolor="white", markeredgewidth=1)
            ax.annotate(f"CAPTURED\n{cap_step*cfg.dt:.2f}s", xy=(c2d[0], c2d[1]),
                        xytext=(15,-20), textcoords="offset points", fontsize=10,
                        color="#e74c3c", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#e74c3c"))

        status_str = "CAPTURED" if captured else "ESCAPED"
        ax.set_title(f"Scenario {sc+1} - {status_str}\n"
                     f"Duration: {duration:.2f}s | Init dist: {d0:.1f}m | "
                     f"Min dist: {min(min_dists):.2f}m", fontsize=13, fontweight="bold")

        # 图例
        legend_els = []
        for i, name in enumerate(STRATEGY_NAMES):
            legend_els.append(mpatches.Patch(color=STRATEGY_COLORS[i], alpha=0.85, label=name))
        ax.legend(handles=legend_els, loc="upper right", fontsize=8, framealpha=0.9)

        ax.set_aspect("equal")
        ax.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
        ax.set_facecolor("#f8f9fa")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"battle_3d_{sc+1}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        n_switches = sum(1 for t in range(1, len(strats)) if strats[t] != strats[t-1])
        summary.append({"id": sc+1, "captured": captured, "duration": duration,
                         "d0": d0, "min_d": min(min_dists), "switches": n_switches})
        print(f"  Scene {sc+1}: {status_str} | {duration:.1f}s | d0={d0:.1f}m | switches={n_switches}")

    return summary


# ==========================================================================
# 主入口
# ==========================================================================

def main():
    print("=" * 60)
    print("  Phase 4: 距离驱动目标 + 缩短时间窗")
    print("  T_obs=5(0.25s) W_mvs=10(0.5s) T_pred=5(0.25s)")
    print("=" * 60)

    cfg = Config()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # =====================================================================
    # 1. 数据收集
    # =====================================================================
    print(f"\n[Step 1] 模拟 {cfg.n_episodes_collect} 个 episodes...")
    fsm = TargetFSM(num_envs=1, device="cpu", max_speed=cfg.max_speed,
                     max_accel=cfg.max_accel, arena_radius=cfg.arena_radius,
                     max_height=cfg.max_height)

    dataset = PredictorDataset(
        t_obs=cfg.t_obs, t_pred=cfg.t_pred,
        w_mvs=cfg.w_mvs, slide_step=cfg.slide_step,
    )

    strategy_counts = {name: 0 for name in STRATEGY_NAMES}
    t_start = time.time()
    ep_durations = []

    for ep in range(cfg.n_episodes_collect):
        episode = simulate_episode(cfg, fsm, ep_id=0)
        if episode is not None:
            dataset.add_episode(episode)
            for sid in episode["strategy_ids"].tolist():
                if 0 <= sid < len(STRATEGY_NAMES):
                    strategy_counts[STRATEGY_NAMES[sid]] += 1
            ep_durations.append(episode["target_pos"].shape[0] * cfg.dt_ctrl)

        if (ep + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  Episode {ep+1}/{cfg.n_episodes_collect} "
                  f"({elapsed:.1f}s), samples: {len(dataset)}")

    print(f"\n  Data collected:")
    stats = dataset.get_stats()
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Intent dist: {stats['intent_distribution']}")
    print(f"  Strategy dist: {strategy_counts}")
    print(f"  Avg episode duration: {np.mean(ep_durations):.2f}s")

    if len(dataset) < 100:
        print("  [ERROR] Too few samples")
        return

    # 分训练/验证集
    n_total = len(dataset)
    n_train = int(n_total * 0.8)

    class WrapDS(PredictorDataset):
        def __init__(self, subset):
            self._s = subset
        def __len__(self): return len(self._s)
        def __getitem__(self, idx): return self._s[idx]

    indices = list(range(n_total))
    np.random.shuffle(indices)
    train_ds = WrapDS(torch.utils.data.Subset(dataset, indices[:n_train]))
    val_ds = WrapDS(torch.utils.data.Subset(dataset, indices[n_train:]))
    print(f"  Train: {n_train}, Val: {n_total - n_train}")

    # =====================================================================
    # 2. Phase 1: 意图分类
    # =====================================================================
    print(f"\n[Step 2] Phase 1: Intent classification ({cfg.epochs_phase1} epochs)...")
    model = IntentConditionedGMMPredictor(
        obs_dim=24, n_intents=10, n_components=3,
        pred_horizon=cfg.t_pred, hidden_dim=cfg.hidden_dim,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt1 = optim.Adam(model.parameters(), lr=cfg.lr_phase1)
    h1 = train_phase(model, train_ds, IntentLoss(), opt1, cfg, "Phase1", cfg.epochs_phase1)

    ckpt1 = os.path.join(cfg.checkpoint_dir, "predictor_phase1.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": model.get_config()}, ckpt1)
    print(f"  Saved: {ckpt1}")

    # =====================================================================
    # 3. Phase 2: GMM轨迹预测
    # =====================================================================
    print(f"\n[Step 3] Phase 2: GMM trajectory ({cfg.epochs_phase2} epochs)...")
    opt2 = optim.Adam(model.parameters(), lr=cfg.lr_phase2)
    h2 = train_phase(model, train_ds, IntentGMMLoss(), opt2, cfg, "Phase2", cfg.epochs_phase2, use_mvs=True)

    ckpt2 = os.path.join(cfg.checkpoint_dir, "predictor_phase2.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": model.get_config()}, ckpt2)
    print(f"  Saved: {ckpt2}")

    # =====================================================================
    # 4. 完整评估
    # =====================================================================
    print(f"\n[Step 4] Full evaluation...")
    eval_data = full_evaluation(model, val_ds, cfg)

    overall_acc = np.mean([1 if p == g else 0
                           for p, g in zip(eval_data["pred_intent"], eval_data["gt_intent"])])
    overall_ade = np.mean(eval_data["ade"])
    overall_fde = np.mean(eval_data["fde"])
    print(f"  Accuracy: {overall_acc:.4f}")
    print(f"  ADE: {overall_ade:.4f}m")
    print(f"  FDE: {overall_fde:.4f}m")

    # 消融
    abl = eval_data["ablation"]
    print(f"\n  MVS Ablation:")
    print(f"    MVS=Real: acc={np.mean(abl['real']['acc']):.4f}, "
          f"ADE={np.mean(abl['real']['ade']):.4f}, FDE={np.mean(abl['real']['fde']):.4f}")
    print(f"    MVS=0:    acc={np.mean(abl['zero']['acc']):.4f}, "
          f"ADE={np.mean(abl['zero']['ade']):.4f}, FDE={np.mean(abl['zero']['fde']):.4f}")

    # =====================================================================
    # 5. 生成图表
    # =====================================================================
    print(f"\n[Step 5] Generating plots...")
    plot_confusion_matrix(eval_data, cfg.results_dir)
    print(f"  confusion_matrix.png")
    plot_per_intent_metrics(eval_data, cfg.results_dir)
    print(f"  per_intent_metrics.png")
    plot_traj_vs_gt(eval_data, cfg.results_dir)
    print(f"  traj_vs_gt.png")
    plot_ablation_mvs(eval_data, cfg.results_dir)
    print(f"  ablation_mvs.png")
    plot_training_curves(h1, h2, cfg.results_dir)
    print(f"  training_curves.png")

    # =====================================================================
    # 6. 10张3D对抗轨迹图
    # =====================================================================
    print(f"\n[Step 6] 10 battle scenarios...")
    battle_summary = simulate_and_plot_battles(cfg, fsm, cfg.results_dir, n_scenarios=10)

    # =====================================================================
    # 7. 文字报告
    # =====================================================================
    report_path = os.path.join(cfg.results_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("Phase 4 Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  T_obs = {cfg.t_obs} frames ({cfg.t_obs*cfg.dt_ctrl}s)\n")
        f.write(f"  W_mvs = {cfg.w_mvs} frames ({cfg.w_mvs*cfg.dt_ctrl}s)\n")
        f.write(f"  T_pred = {cfg.t_pred} frames ({cfg.t_pred*cfg.dt_ctrl}s)\n")
        f.write(f"  Episodes = {cfg.n_episodes_collect}\n")
        f.write(f"  Train/Val = {n_train}/{n_total-n_train}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Intent Accuracy: {overall_acc:.4f}\n")
        f.write(f"  ADE: {overall_ade:.4f}m\n")
        f.write(f"  FDE: {overall_fde:.4f}m\n\n")
        f.write(f"MVS Ablation:\n")
        f.write(f"  MVS=Real: acc={np.mean(abl['real']['acc']):.4f}, "
                f"ADE={np.mean(abl['real']['ade']):.4f}\n")
        f.write(f"  MVS=0:    acc={np.mean(abl['zero']['acc']):.4f}, "
                f"ADE={np.mean(abl['zero']['ade']):.4f}\n\n")
        f.write(f"Battle Scenarios:\n")
        n_cap = sum(1 for b in battle_summary if b["captured"])
        f.write(f"  Capture rate: {n_cap}/{len(battle_summary)}\n")
        for b in battle_summary:
            st = "CAPTURED" if b["captured"] else "ESCAPED"
            f.write(f"  Scene {b['id']}: {st} | {b['duration']:.1f}s | "
                    f"d0={b['d0']:.1f}m | switches={b['switches']}\n")

    print(f"\n  Report saved: {report_path}")
    print(f"\n{'=' * 60}")
    print(f"  Phase 4 complete!")
    print(f"  Results: {cfg.results_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
