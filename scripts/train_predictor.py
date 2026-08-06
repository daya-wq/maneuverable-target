#!/usr/bin/env python3
"""
意图驱动 GMM 轨迹预测网络 — 训练脚本

架构 (对应 docs/意图network_architecture.md):
  输入:  [B, T_obs=10, 30] 30维相对运动学特征
  编码:  BiLSTM → H_env
  解耦:  Cross-Attention → h_strategic / h_trajectory
  分类:  4类战术意图头 (Far_Rush / Close_Evade / Goal_Priority / Goal_Sprint)
  解码:  Fusion MLP → z → GRU自回归 → GMM(π, μ, σ²)  K=3分量
  输出:  strategic_probs [B,4] / gmm_weights [B,3] /
         gmm_means [B,3,T,3] / gmm_vars [B,3,T,3] / mixed_mu [B,T,3]

保存:
  checkpoints/predictor_strategic.pt
结果:
  results/train_predictor/

运行:
  cd /data/uavlab/maneuverable-target
  python3 scripts/train_predictor.py
"""

import sys
import os
import math
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ── 直接按文件路径导入，避免触发 omni_drones/learning/__init__.py
#    （该文件会导入整个 RL 框架/Isaac Sim，造成挂起）────────────────────────
import importlib.util

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
predictor_dir = os.path.join(project_root, "omni_drones", "learning", "predictor")
hide_seek_dir = os.path.join(project_root, "omni_drones", "envs", "hide_and_seek")


def _load_module(module_name: str, file_path: str):
    """从绝对路径直接加载一个 Python 文件为模块。"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_gmm = _load_module(
    "dual_intent_gmm_predictor",
    os.path.join(predictor_dir, "dual_intent_gmm_predictor.py"),
)
StrategicGMMConfig             = _gmm.StrategicGMMConfig
StrategicGMMTrajectoryPredictor = _gmm.StrategicGMMTrajectoryPredictor
StrategicGMMLoss               = _gmm.StrategicGMMLoss
STRATEGIC_LABELS               = _gmm.STRATEGIC_LABELS

_dp = _load_module(
    "data_pipeline",
    os.path.join(predictor_dir, "data_pipeline.py"),
)
PredictorDataset      = _dp.PredictorDataset
TrajectoryRecorder    = _dp.TrajectoryRecorder
STRATEGIC_LABEL_NAMES = _dp.STRATEGIC_LABEL_NAMES
N_STRATEGIC           = _dp.N_STRATEGIC

_fsm_mod = _load_module(
    "target_fsm",
    os.path.join(hide_seek_dir, "target_fsm.py"),
)
TargetFSM      = _fsm_mod.TargetFSM
STRATEGY_NAMES = _fsm_mod.STRATEGY_NAMES

_pn_mod = _load_module(
    "proportional_navigation",
    os.path.join(hide_seek_dir, "proportional_navigation.py"),
)
proportional_navigation_3d = _pn_mod.proportional_navigation_3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ==========================================================================
# 配置
# ==========================================================================

class Config:
    # 场景
    arena_radius   = 10.0
    max_height     = 5.0
    max_speed      = 3.0
    max_accel      = 3.0
    n_pursuers     = 3
    dt             = 0.01      # 物理步 100Hz
    subsample      = 5         # 下采样 → 20Hz
    dt_ctrl        = 0.05      # 网络时间步 0.05s

    # episode
    max_steps           = 2000
    n_episodes_collect  = 500
    init_dist_range     = (7.0, 8.0)
    capture_radius      = 0.4

    # 时间窗 (帧数, @20Hz)
    t_obs      = 10   # 历史观测 0.5s
    t_pred     = 10   # 预测 0.5s
    slide_step = 2    # 滑窗步长

    # 守区
    goal_center = torch.tensor([8.0, 0.0, 2.5], dtype=torch.float32)

    # 网络
    d_model       = 128
    lstm_layers   = 1
    n_heads       = 4
    n_components  = 3
    gru_layers    = 1
    dropout       = 0.1

    # 训练
    batch_size   = 64
    lr           = 5e-4
    n_epochs     = 60
    grad_clip    = 5.0

    # 损失权重
    lambda_strategic = 1.0
    lambda_gmm       = 1.0
    lambda_speed     = 0.02
    lambda_accel     = 0.02

    # 输出
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    results_dir    = os.path.join(project_root, "results", "train_predictor")


# ==========================================================================
# 合成仿真器
# ==========================================================================

def simulate_episode(cfg: Config, fsm: TargetFSM, ep_id: int = 0):
    """模拟一个 episode，使用距离驱动 FSM"""
    recorder = TrajectoryRecorder(subsample_rate=cfg.subsample)

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
        xy = torch.norm(pursuer_pos[p, :2])
        if xy > cfg.arena_radius:
            pursuer_pos[p, :2] = pursuer_pos[p, :2] / xy * (cfg.arena_radius - 0.1)
        pursuer_pos[p, 2] = pursuer_pos[p, 2].clamp(0.1, cfg.max_height)

    init_dists = torch.tensor([
        torch.norm(target_pos - pursuer_pos[p]).item() for p in range(cfg.n_pursuers)
    ])
    d0 = init_dists.min()

    fsm.reset(torch.tensor([0]))
    fsm.set_initial_distance(torch.tensor([0]), d0.unsqueeze(0))

    velocity_damping = 0.3

    for step in range(cfg.max_steps):
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

        fsm_force, strategy = fsm.compute_strategy_force(
            target_pos.unsqueeze(0), target_vel.unsqueeze(0),
            pursuer_pos.unsqueeze(0), goal_center=cfg.goal_center,
        )
        fsm_force = fsm_force.squeeze(0)
        strategy_id = strategy[0]

        min_d = min(torch.norm(target_pos - pursuer_pos[p]).item()
                    for p in range(cfg.n_pursuers))
        if min_d < d0.item() * 0.4:
            total_force = 0.3 * apf_force + fsm_force
        else:
            total_force = apf_force + fsm_force

        force_norm = torch.norm(total_force).clamp(min=1e-6)
        accel_cmd = cfg.max_accel * total_force / force_norm

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

        if min_d < cfg.capture_radius:
            break

    return recorder.get_episode()


# ==========================================================================
# 训练循环
# ==========================================================================

def train_one_epoch(model, loader, loss_fn, optimizer, cfg):
    model.train()
    total_loss = total_acc = total_minade = n_batches = 0

    for batch in loader:
        obs          = batch["obs_history"]      # [B, T_obs, 30]
        gt_future    = batch["gt_future_delta"]  # [B, T_pred, 3]
        gt_strategic = batch["gt_strategic"]     # [B]

        optimizer.zero_grad()
        pred = model(obs, gt_strategic=None)     # 推理时用预测意图（无监督Teacher Forcing）
        loss, metrics = loss_fn(pred, gt_future, gt_strategic)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()

        total_loss  += metrics["loss_total"]
        total_acc   += metrics["strategic_accuracy"]
        total_minade += metrics["minADE"]
        n_batches   += 1

    return {
        "loss":    total_loss  / n_batches,
        "acc":     total_acc   / n_batches,
        "minADE":  total_minade / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = total_acc = total_minade = total_fde = n_batches = 0

    for batch in loader:
        obs          = batch["obs_history"]
        gt_future    = batch["gt_future_delta"]
        gt_strategic = batch["gt_strategic"]

        pred = model(obs)
        _, metrics = loss_fn(pred, gt_future, gt_strategic)

        total_loss   += metrics["loss_total"]
        total_acc    += metrics["strategic_accuracy"]
        total_minade += metrics["minADE"]
        total_fde    += metrics["FDE"]
        n_batches    += 1

    return {
        "loss":   total_loss   / n_batches,
        "acc":    total_acc    / n_batches,
        "minADE": total_minade / n_batches,
        "FDE":    total_fde    / n_batches,
    }


# ==========================================================================
# 绘图
# ==========================================================================

def plot_training_curves(train_history, val_history, save_dir):
    epochs = range(1, len(train_history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, title in zip(
        axes,
        ["loss", "acc", "minADE"],
        ["Total Loss", "Strategic Accuracy", "minADE (m)"],
    ):
        ax.plot(epochs, train_history[key], label="Train", color="#4a90d9")
        ax.plot(epochs, val_history[key],   label="Val",   color="#e67e22")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print("  training_curves.png")


def plot_traj_vs_gt(model, val_loader, save_dir, n_samples=8):
    """可视化：混合均值轨迹 vs GT，并显示3条GMM分量"""
    model.eval()
    samples_obs, samples_gt, samples_pred = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            obs       = batch["obs_history"]
            gt_future = batch["gt_future_delta"]
            pred = model(obs)
            B = obs.shape[0]
            for i in range(min(B, n_samples - len(samples_obs))):
                samples_obs.append(obs[i].cpu())
                samples_gt.append(gt_future[i].cpu())
                samples_pred.append({
                    "gmm_weights": pred["gmm_weights"][i].cpu(),
                    "gmm_means":   pred["gmm_means"][i].cpu(),
                    "mixed_mu":    pred["mixed_mu"][i].cpu(),
                    "strategic_probs": pred["strategic_probs"][i].cpu(),
                })
            if len(samples_obs) >= n_samples:
                break

    n = len(samples_obs)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    COMPONENT_COLORS = ["#3498db", "#2ecc71", "#9b59b6"]

    for idx in range(n):
        ax = axes[idx]
        gt = samples_gt[idx].numpy()           # [T, 3]
        p  = samples_pred[idx]
        mixed_mu   = p["mixed_mu"].numpy()     # [T, 3]
        gmm_means  = p["gmm_means"].numpy()    # [K, T, 3]
        gmm_weights = p["gmm_weights"].numpy() # [K]
        strategic_probs = p["strategic_probs"].numpy()  # [4]
        T = gt.shape[0]
        K = gmm_means.shape[0]

        ax.plot(range(T), gt[:, 0], "k-",  lw=2, label="GT-x", alpha=0.8)
        ax.plot(range(T), gt[:, 1], "k--", lw=2, label="GT-y", alpha=0.8)
        ax.plot(range(T), gt[:, 2], "k:",  lw=2, label="GT-z", alpha=0.8)

        ax.plot(range(T), mixed_mu[:, 0], "r-",  lw=1.5, label="Mix-x", alpha=0.75)
        ax.plot(range(T), mixed_mu[:, 1], "r--", lw=1.5, label="Mix-y", alpha=0.75)
        ax.plot(range(T), mixed_mu[:, 2], "r:",  lw=1.5, label="Mix-z", alpha=0.75)

        for k in range(K):
            c = COMPONENT_COLORS[k]
            ax.plot(range(T), gmm_means[k, :, 0], "-",  color=c, lw=0.8,
                    alpha=0.5, label=f"K{k}(w={gmm_weights[k]:.2f})")
            ax.plot(range(T), gmm_means[k, :, 1], "--", color=c, lw=0.8, alpha=0.5)
            ax.plot(range(T), gmm_means[k, :, 2], ":",  color=c, lw=0.8, alpha=0.5)

        ade = np.linalg.norm(mixed_mu - gt, axis=-1).mean()
        best_intent = STRATEGIC_LABELS[int(strategic_probs.argmax())]
        ax.set_title(f"#{idx+1} | ADE={ade:.3f}m | {best_intent}", fontsize=8)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=5, ncol=3)

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("GMM Trajectory Prediction vs GT\nBlack=GT, Red=MixedMu, Color=Components", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "traj_vs_gt.png"), dpi=150)
    plt.close()
    print("  traj_vs_gt.png")


def plot_strategic_confusion(model, val_loader, save_dir):
    """战术分类混淆矩阵"""
    model.eval()
    all_pred, all_gt = [], []

    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch["obs_history"])
            all_pred.extend(pred["strategic_probs"].argmax(dim=-1).tolist())
            all_gt.extend(batch["gt_strategic"].tolist())

    cm = np.zeros((N_STRATEGIC, N_STRATEGIC), dtype=int)
    for gt, pr in zip(all_gt, all_pred):
        cm[gt, pr] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(N_STRATEGIC):
        for j in range(N_STRATEGIC):
            v = cm[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks(range(N_STRATEGIC))
    ax.set_yticks(range(N_STRATEGIC))
    ax.set_xticklabels(STRATEGIC_LABEL_NAMES, rotation=20, ha="right")
    ax.set_yticklabels(STRATEGIC_LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    overall_acc = np.trace(cm) / cm.sum() * 100
    ax.set_title(f"Strategic Classification  Acc={overall_acc:.1f}%")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "strategic_confusion.png"), dpi=150)
    plt.close()
    print("  strategic_confusion.png")


# ==========================================================================
# 主入口
# ==========================================================================

def main():
    print("=" * 60)
    print("  意图驱动 GMM 轨迹预测网络 — 训练")
    print(f"  T_obs={10}帧(0.5s)  T_pred={10}帧(0.5s)  obs_dim=30")
    print(f"  4类战术意图 + K=3 GMM分量 + GRU解码")
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
    fsm = TargetFSM(
        num_envs=1, device="cpu",
        max_speed=cfg.max_speed, max_accel=cfg.max_accel,
        arena_radius=cfg.arena_radius, max_height=cfg.max_height,
    )
    dataset = PredictorDataset(
        t_obs=cfg.t_obs, t_pred=cfg.t_pred,
        slide_step=cfg.slide_step,
        goal_center=cfg.goal_center,
    )

    t_start = time.time()
    for ep in range(cfg.n_episodes_collect):
        episode = simulate_episode(cfg, fsm, ep_id=ep)
        if episode is not None:
            dataset.add_episode(episode)
        if (ep + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  Episode {ep+1}/{cfg.n_episodes_collect} "
                  f"({elapsed:.1f}s), samples: {len(dataset)}")

    stats = dataset.get_stats()
    print(f"\n  数据收集完毕: {stats['total_samples']} 样本")
    print(f"  战术分布: {stats['strategic_distribution']}")

    if len(dataset) < 200:
        print("  [ERROR] 样本数量不足，退出")
        return

    # 划分训练/验证
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    indices = list(range(n_total))
    np.random.shuffle(indices)
    train_set = torch.utils.data.Subset(dataset, indices[:n_train])
    val_set   = torch.utils.data.Subset(dataset, indices[n_train:])
    print(f"  Train: {n_train}, Val: {n_total - n_train}")

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
                              shuffle=True,  collate_fn=PredictorDataset.collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size,
                              shuffle=False, collate_fn=PredictorDataset.collate_fn)

    # =====================================================================
    # 2. 构建模型
    # =====================================================================
    model_cfg = StrategicGMMConfig(
        obs_dim       = 30,
        d_model       = cfg.d_model,
        lstm_layers   = cfg.lstm_layers,
        n_heads       = cfg.n_heads,
        n_components  = cfg.n_components,
        pred_horizon  = cfg.t_pred,
        dropout       = cfg.dropout,
        gru_layers    = cfg.gru_layers,
        n_strategic   = 4,
        detach_intent_for_gmm = False,
    )
    model = StrategicGMMTrajectoryPredictor(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Step 2] 模型参数量: {n_params:,}")

    loss_fn = StrategicGMMLoss(
        lambda_strategic = cfg.lambda_strategic,
        lambda_gmm       = cfg.lambda_gmm,
        lambda_speed     = cfg.lambda_speed,
        lambda_accel     = cfg.lambda_accel,
        dt_pred          = cfg.dt_ctrl,
        max_speed        = cfg.max_speed,
        max_accel        = cfg.max_accel,
    )
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs, eta_min=1e-5)

    # =====================================================================
    # 3. 训练
    # =====================================================================
    print(f"\n[Step 3] 训练 {cfg.n_epochs} epochs...")
    train_history = {"loss": [], "acc": [], "minADE": []}
    val_history   = {"loss": [], "acc": [], "minADE": []}
    best_val_loss = float("inf")

    for epoch in range(cfg.n_epochs):
        t_m = train_one_epoch(model, train_loader, loss_fn, optimizer, cfg)
        v_m = evaluate(model, val_loader, loss_fn)
        scheduler.step()

        for h, m in [(train_history, t_m), (val_history, v_m)]:
            for k in ["loss", "acc", "minADE"]:
                h[k].append(m[k])

        if (epoch + 1) % 5 == 0 or epoch == cfg.n_epochs - 1:
            print(
                f"  Epoch {epoch+1:3d}/{cfg.n_epochs}  "
                f"train_loss={t_m['loss']:.4f}  val_loss={v_m['loss']:.4f}  "
                f"acc={v_m['acc']:.3f}  minADE={v_m['minADE']:.4f}m  "
                f"FDE={v_m['FDE']:.4f}m"
            )

        # 保存最优
        if v_m["loss"] < best_val_loss:
            best_val_loss = v_m["loss"]
            ckpt_path = os.path.join(cfg.checkpoint_dir, "predictor_strategic.pt")
            torch.save({
                "epoch":            epoch + 1,
                "model_state_dict": model.state_dict(),
                "model_config":     model_cfg.__dict__,
                "val_loss":         best_val_loss,
                "val_acc":          v_m["acc"],
                "val_minADE":       v_m["minADE"],
            }, ckpt_path)

    print(f"\n  最优 val_loss={best_val_loss:.4f}")
    print(f"  Checkpoint: {ckpt_path}")

    # 重新加载最优权重用于评估
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])

    # =====================================================================
    # 4. 生成图表
    # =====================================================================
    print(f"\n[Step 4] 生成图表...")
    plot_training_curves(train_history, val_history, cfg.results_dir)
    plot_traj_vs_gt(model, val_loader, cfg.results_dir)
    plot_strategic_confusion(model, val_loader, cfg.results_dir)

    # =====================================================================
    # 5. 文字报告
    # =====================================================================
    final_val = evaluate(model, val_loader, loss_fn)
    report_path = os.path.join(cfg.results_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("Strategic GMM Trajectory Predictor — Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model Architecture:\n")
        f.write(f"  obs_dim      = 30\n")
        f.write(f"  T_obs        = {cfg.t_obs} frames ({cfg.t_obs * cfg.dt_ctrl:.2f}s)\n")
        f.write(f"  T_pred       = {cfg.t_pred} frames ({cfg.t_pred * cfg.dt_ctrl:.2f}s)\n")
        f.write(f"  d_model      = {cfg.d_model}\n")
        f.write(f"  n_heads      = {cfg.n_heads}\n")
        f.write(f"  n_components = {cfg.n_components}\n")
        f.write(f"  Parameters   = {n_params:,}\n\n")
        f.write("Training Config:\n")
        f.write(f"  Episodes  = {cfg.n_episodes_collect}\n")
        f.write(f"  Train/Val = {n_train}/{n_total - n_train}\n")
        f.write(f"  Epochs    = {cfg.n_epochs}\n")
        f.write(f"  LR        = {cfg.lr}\n\n")
        f.write("Final Validation Metrics (best ckpt):\n")
        f.write(f"  Strategic Accuracy = {final_val['acc']:.4f}\n")
        f.write(f"  minADE             = {final_val['minADE']:.4f} m\n")
        f.write(f"  FDE                = {final_val['FDE']:.4f} m\n")
        f.write(f"  Val Loss           = {final_val['loss']:.4f}\n\n")
        f.write("Outputs:\n")
        f.write(f"  strategic_probs  : (B, 4)\n")
        f.write(f"  gmm_weights      : (B, 3)\n")
        f.write(f"  gmm_means        : (B, 3, {cfg.t_pred}, 3)\n")
        f.write(f"  gmm_vars         : (B, 3, {cfg.t_pred}, 3)\n")
        f.write(f"  mixed_mu         : (B, {cfg.t_pred}, 3)  ← RL控制器直接使用\n")

    print(f"  report.txt")
    print(f"\n{'=' * 60}")
    print(f"  训练完成！  results → {cfg.results_dir}/")
    print(f"  Checkpoint → {ckpt_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
