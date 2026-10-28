#!/usr/bin/env python3
"""
Prototype test for the dual-intent query GMM predictor (v2).

Changes from v1:
  - Unified 0.05s cadence for both input and output (10 frames each)
  - 4 strategic intents (Far_Rush, Close_Evade, Goal_Priority, Goal_Sprint)
  - GRU decoder for GMM trajectory generation
  - Comprehensive evaluation: confusion matrices, ADE/FDE/minADE/minFDE,
    per-class F1, per-tactical-group trajectory metrics, GMM quality metrics,
    physics compliance metrics
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJ_DIR = PROJECT_ROOT / "results" / "dual_intent_6dof_battles" / "trajectory_npz"
# RESULT_DIR is dynamically generated in main()


def load_dual_intent_module():
    path = PROJECT_ROOT / "omni_drones" / "learning" / "predictor" / "dual_intent_gmm_predictor.py"
    spec = importlib.util.spec_from_file_location("dual_intent_gmm_predictor_local", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dual_intent = load_dual_intent_module()
DualIntentGMMConfig = dual_intent.DualIntentGMMConfig
DualIntentQueryGMMPredictor = dual_intent.DualIntentQueryGMMPredictor
DualIntentGMMLoss = dual_intent.DualIntentGMMLoss
STRATEGIC_LABELS = dual_intent.STRATEGIC_LABELS
MANEUVER_LABELS = dual_intent.MANEUVER_LABELS


# =============================================================================
# Dataset — unified 0.05s cadence
# =============================================================================

class UnifiedCadenceDataset(Dataset):
    """
    Both input and output use 0.05s cadence, 0.5s window → 10 frames each.
    """

    def __init__(
        self,
        trajectory_files: List[Path],
        obs_seconds: float = 0.5,
        dt: float = 0.05,
        pred_seconds: float = 0.5,
        slide_seconds: float = 0.05,
        max_samples: int | None = 8192,
    ):
        self.obs_seconds = obs_seconds
        self.dt = dt
        self.pred_seconds = pred_seconds
        self.slide_seconds = slide_seconds
        self.samples = []

        for path in trajectory_files:
            self._add_npz(path)

        if max_samples is not None and len(self.samples) > max_samples:
            rng = np.random.default_rng(20260615)
            keep = np.sort(rng.choice(len(self.samples), size=max_samples, replace=False))
            self.samples = [self.samples[int(i)] for i in keep]

        # Compute normalization stats
        obs = torch.stack([s["obs_history"] for s in self.samples])
        self.obs_mean = obs.mean(dim=(0, 1), keepdim=True)
        self.obs_std = obs.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
        for sample in self.samples:
            sample["obs_history"] = (sample["obs_history"] - self.obs_mean.squeeze(0)) / self.obs_std.squeeze(0)

    def _add_npz(self, path: Path) -> None:
        data = np.load(path, allow_pickle=True)
        dt_raw = float(data["dt"][0])
        stride = max(1, int(round(self.dt / dt_raw)))
        slide_stride = max(1, int(round(self.slide_seconds / dt_raw)))
        obs_steps = int(round(self.obs_seconds / self.dt))
        pred_steps = int(round(self.pred_seconds / self.dt))

        target_pos = data["target_pos"].astype(np.float32)
        target_vel = data["target_vel"].astype(np.float32)
        pursuer_pos = data["pursuer_pos"].astype(np.float32)
        goal_center = data["goal_center"].astype(np.float32)
        strategic = data["tactical_hist"].astype(np.int64)
        maneuver = data["maneuver_mode_hist"].astype(np.int64)

        pursuer_vel = np.gradient(pursuer_pos, dt_raw, axis=0).astype(np.float32)
        min_anchor = (obs_steps - 1) * stride
        max_anchor = len(target_pos) - pred_steps * stride - 1

        for anchor in range(min_anchor, max_anchor + 1, slide_stride):
            obs_indices = anchor - np.arange(obs_steps - 1, -1, -1) * stride
            future_indices = anchor + np.arange(1, pred_steps + 1) * stride

            if obs_indices[0] < 0 or future_indices[-1] >= len(target_pos):
                continue

            obs = np.stack([
                build_relative_observation(
                    target_pos[i], target_vel[i],
                    pursuer_pos[i], pursuer_vel[i], goal_center,
                )
                for i in obs_indices
            ])
            current_target = target_pos[anchor]
            future_delta = target_pos[future_indices] - current_target
            self.samples.append({
                "obs_history": torch.from_numpy(obs),
                "gt_future_delta": torch.from_numpy(future_delta.astype(np.float32)),
                "gt_strategic": int(strategic[anchor]),
                "gt_maneuver": int(maneuver[anchor]),
                "current_target_pos": torch.from_numpy(current_target.astype(np.float32)),
                "source_file": path.name,
                "anchor_step": int(anchor),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
        return {
            "obs_history": torch.stack([b["obs_history"] for b in batch]),
            "gt_future_delta": torch.stack([b["gt_future_delta"] for b in batch]),
            "gt_strategic": torch.tensor([b["gt_strategic"] for b in batch], dtype=torch.long),
            "gt_maneuver": torch.tensor([b["gt_maneuver"] for b in batch], dtype=torch.long),
            "current_target_pos": torch.stack([b["current_target_pos"] for b in batch]),
            "source_file": [b["source_file"] for b in batch],
            "anchor_step": torch.tensor([b["anchor_step"] for b in batch], dtype=torch.long),
        }

    def stats(self) -> Dict:
        strategic = torch.tensor([s["gt_strategic"] for s in self.samples], dtype=torch.long)
        maneuver = torch.tensor([s["gt_maneuver"] for s in self.samples], dtype=torch.long)
        strategic_counts = torch.bincount(strategic, minlength=len(STRATEGIC_LABELS))
        maneuver_counts = torch.bincount(maneuver, minlength=len(MANEUVER_LABELS))
        return {
            "num_samples": len(self.samples),
            "obs_shape": list(self.samples[0]["obs_history"].shape) if self.samples else None,
            "future_shape": list(self.samples[0]["gt_future_delta"].shape) if self.samples else None,
            "strategic_counts": {
                STRATEGIC_LABELS[i]: int(strategic_counts[i]) for i in range(len(STRATEGIC_LABELS))
            },
            "maneuver_counts": {
                MANEUVER_LABELS[i]: int(maneuver_counts[i]) for i in range(len(MANEUVER_LABELS))
            },
        }


def build_relative_observation(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    pursuer_pos: np.ndarray,
    pursuer_vel: np.ndarray,
    goal_center: np.ndarray,
) -> np.ndarray:
    rel_target_to_pursuer_pos = (target_pos[None, :] - pursuer_pos).reshape(-1)
    rel_target_to_pursuer_vel = (target_vel[None, :] - pursuer_vel).reshape(-1)
    target_to_goal = target_pos - goal_center
    pursuer_to_goal = (pursuer_pos - goal_center[None, :]).reshape(-1)
    obs = np.concatenate([
        rel_target_to_pursuer_pos,
        rel_target_to_pursuer_vel,
        target_to_goal,
        pursuer_to_goal,
    ]).astype(np.float32)
    if obs.shape[0] != 30:
        raise ValueError(f"Expected 30-D observation, got {obs.shape[0]}")
    return obs


def split_dataset(dataset: Dataset, train_ratio: float = 0.8):
    n_train = max(1, int(len(dataset) * train_ratio))
    generator = torch.Generator().manual_seed(20260615)
    return torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train], generator=generator)


# =============================================================================
# Training and evaluation
# =============================================================================

def evaluate(model, loss_fn, loader, device):
    model.eval()
    accum = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs_history"].to(device)
            gt_future = batch["gt_future_delta"].to(device)
            gt_strategic = batch["gt_strategic"].to(device)
            gt_maneuver = batch["gt_maneuver"].to(device)
            current_z = batch["current_target_pos"][:, 2].to(device)
            
            kwargs = {}
            if getattr(model, "use_oracle", False):
                kwargs["gt_strategic"] = gt_strategic
                kwargs["gt_maneuver"] = gt_maneuver
                
            pred = model(obs, **kwargs)
            _, metrics = loss_fn(pred, gt_future, gt_strategic, gt_maneuver, current_z)
            batch_size = obs.shape[0]
            for key, value in metrics.items():
                if not math.isnan(value):
                    accum[key] = accum.get(key, 0.0) + value * batch_size
            count += batch_size
    return {key: value / max(1, count) for key, value in accum.items()}


def collect_predictions(model, loader, device):
    """Collect all predictions and ground truths for comprehensive evaluation."""
    model.eval()
    all_preds = {
        "strategic_logits": [], "maneuver_logits": [],
        "gmm_weights": [], "gmm_means": [], "gmm_vars": [],
        "mixed_mu": [], "mixed_var": [],
    }
    all_gt = {
        "gt_future_delta": [], "gt_strategic": [], "gt_maneuver": [],
        "current_target_pos": [],
    }
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs_history"].to(device)
            
            kwargs = {}
            if getattr(model, "use_oracle", False):
                kwargs["gt_strategic"] = batch["gt_strategic"].to(device)
                kwargs["gt_maneuver"] = batch["gt_maneuver"].to(device)
                
            pred = model(obs, **kwargs)
            for key in all_preds:
                if key in pred:
                    all_preds[key].append(pred[key].cpu())
            all_gt["gt_future_delta"].append(batch["gt_future_delta"])
            all_gt["gt_strategic"].append(batch["gt_strategic"])
            all_gt["gt_maneuver"].append(batch["gt_maneuver"])
            all_gt["current_target_pos"].append(batch["current_target_pos"])

    for key in all_preds:
        all_preds[key] = torch.cat(all_preds[key], dim=0)
    for key in all_gt:
        all_gt[key] = torch.cat(all_gt[key], dim=0)
    return all_preds, all_gt


# =============================================================================
# Comprehensive evaluation metrics
# =============================================================================

def compute_confusion_matrix(pred_labels: torch.Tensor, gt_labels: torch.Tensor, n_classes: int):
    """Manual confusion matrix computation (no sklearn dependency)."""
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for gt_cls in range(n_classes):
        for pred_cls in range(n_classes):
            cm[gt_cls, pred_cls] = ((gt_labels == gt_cls) & (pred_labels == pred_cls)).sum()
    return cm


def compute_per_class_metrics(cm: torch.Tensor, labels: List[str]) -> Dict:
    """Compute precision, recall, F1 from confusion matrix."""
    n = cm.shape[0]
    metrics = {}
    for i in range(n):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        metrics[labels[i]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": cm[i, :].sum().item(),
        }
    # Macro and weighted F1
    f1s = [metrics[l]["f1"] for l in labels]
    supports = [metrics[l]["support"] for l in labels]
    total = sum(supports)
    metrics["macro_f1"] = round(sum(f1s) / max(len(f1s), 1), 4)
    metrics["weighted_f1"] = round(sum(f * s for f, s in zip(f1s, supports)) / max(total, 1), 4)
    return metrics


def compute_trajectory_metrics(preds, gt, dt=0.05) -> Dict:
    """Compute ADE, FDE, minADE, minFDE, per-step ADE, physics metrics."""
    mixed_mu = preds["mixed_mu"]        # (N, T, 3)
    gmm_means = preds["gmm_means"]      # (N, K, T, 3)
    gt_future = gt["gt_future_delta"]    # (N, T, 3)

    # ADE / FDE (mixed mean)
    displacement = torch.norm(mixed_mu - gt_future, dim=-1)  # (N, T)
    ade = displacement.mean().item()
    fde = displacement[:, -1].mean().item()

    # Per-step ADE
    per_step_ade = displacement.mean(dim=0).tolist()  # [T]

    # Per-axis ADE
    per_axis_error = (mixed_mu - gt_future).abs().mean(dim=(0, 1))  # (3,)
    ade_x, ade_y, ade_z = per_axis_error[0].item(), per_axis_error[1].item(), per_axis_error[2].item()

    # minADE / minFDE (best GMM component)
    gt_expanded = gt_future.unsqueeze(1)  # (N, 1, T, 3)
    per_comp_disp = torch.norm(gmm_means - gt_expanded, dim=-1)  # (N, K, T)
    per_comp_ade = per_comp_disp.mean(dim=-1)  # (N, K)
    min_ade = per_comp_ade.min(dim=-1).values.mean().item()
    per_comp_fde = per_comp_disp[:, :, -1]  # (N, K)
    min_fde = per_comp_fde.min(dim=-1).values.mean().item()

    # Physics metrics from mixed_mu
    vel = torch.diff(mixed_mu, dim=1) / dt  # (N, T-1, 3)
    speed = torch.norm(vel, dim=-1)          # (N, T-1)
    speed_exceed_ratio = (speed > 3.0).float().mean().item()
    mean_speed = speed.mean().item()

    if vel.shape[1] > 1:
        acc = torch.diff(vel, dim=1) / dt
        accel_mag = torch.norm(acc, dim=-1)
        accel_exceed_ratio = (accel_mag > 3.0).float().mean().item()
        mean_accel = accel_mag.mean().item()
        accel_std = accel_mag.std().item()
    else:
        accel_exceed_ratio = 0.0
        mean_accel = 0.0
        accel_std = 0.0

    return {
        "ade": round(ade, 6),
        "fde": round(fde, 6),
        "min_ade": round(min_ade, 6),
        "min_fde": round(min_fde, 6),
        "ade_x": round(ade_x, 6),
        "ade_y": round(ade_y, 6),
        "ade_z": round(ade_z, 6),
        "per_step_ade": [round(v, 6) for v in per_step_ade],
        "speed_exceed_ratio": round(speed_exceed_ratio, 6),
        "mean_speed": round(mean_speed, 4),
        "accel_exceed_ratio": round(accel_exceed_ratio, 6),
        "mean_accel": round(mean_accel, 4),
        "accel_smoothness_std": round(accel_std, 4),
    }


def compute_gmm_quality(preds) -> Dict:
    """GMM component quality metrics."""
    weights = preds["gmm_weights"]  # (N, K)
    means = preds["gmm_means"]      # (N, K, T, 3)

    # Weight entropy
    log_w = torch.log(weights.clamp_min(1e-8))
    entropy = -(weights * log_w).sum(dim=-1).mean().item()

    # Inter-component distance
    K = means.shape[1]
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            d = torch.norm(means[:, i] - means[:, j], dim=-1).mean(dim=-1)  # (N,)
            dists.append(d.mean().item())
    mean_inter_dist = sum(dists) / max(len(dists), 1)

    return {
        "weight_entropy": round(entropy, 4),
        "mean_inter_component_distance": round(mean_inter_dist, 4),
    }


def compute_tactical_grouped_metrics(preds, gt, dt=0.05) -> Dict:
    """ADE/FDE grouped by tactical intent."""
    mixed_mu = preds["mixed_mu"]
    gt_future = gt["gt_future_delta"]
    gt_strategic = gt["gt_strategic"]
    displacement = torch.norm(mixed_mu - gt_future, dim=-1)  # (N, T)

    grouped = {}
    for idx, label in enumerate(STRATEGIC_LABELS):
        mask = gt_strategic == idx
        if mask.sum() == 0:
            grouped[label] = {"count": 0, "ade": None, "fde": None}
            continue
        ade = displacement[mask].mean().item()
        fde = displacement[mask, -1].mean().item()
        grouped[label] = {
            "count": int(mask.sum().item()),
            "ade": round(ade, 6),
            "fde": round(fde, 6),
        }
    return grouped


# =============================================================================
# Visualization
# =============================================================================

def plot_confusion_matrix(cm: torch.Tensor, labels: List[str], title: str, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(5, len(labels))))
    cm_np = cm.numpy().astype(float)
    # Normalize rows
    row_sums = cm_np.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_np, row_sums, where=row_sums > 0, out=np.zeros_like(cm_np))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=12, fontweight="bold")

    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(cm_np[i, j])
            pct = cm_norm[i, j]
            color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.0%})", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_training_curves(metrics_history: List[Dict], save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [m["epoch"] for m in metrics_history]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Loss total
    axes[0, 0].plot(epochs, [m["train"]["loss_total"] for m in metrics_history], label="train", lw=2)
    axes[0, 0].plot(epochs, [m["val"]["loss_total"] for m in metrics_history], label="val", lw=2)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # GMM NLL
    axes[0, 1].plot(epochs, [m["train"]["loss_gmm_nll"] for m in metrics_history], label="train", lw=2)
    axes[0, 1].plot(epochs, [m["val"]["loss_gmm_nll"] for m in metrics_history], label="val", lw=2)
    axes[0, 1].set_title("GMM NLL")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Strategic / maneuver loss
    axes[0, 2].plot(epochs, [m["val"]["loss_strategic"] for m in metrics_history], label="strategic", lw=2)
    axes[0, 2].plot(epochs, [m["val"]["loss_maneuver"] for m in metrics_history], label="maneuver", lw=2)
    axes[0, 2].set_title("Classification Losses (val)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Accuracies
    axes[1, 0].plot(epochs, [m["val"]["strategic_accuracy"] for m in metrics_history], label="strategic", lw=2)
    axes[1, 0].plot(epochs, [m["val"].get("maneuver_accuracy_close_evade", float("nan")) for m in metrics_history],
                    label="maneuver (CE)", lw=2)
    axes[1, 0].set_title("Intent Accuracy (val)")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ADE / FDE
    axes[1, 1].plot(epochs, [m["val"].get("ade", float("nan")) for m in metrics_history], label="ADE", lw=2)
    axes[1, 1].plot(epochs, [m["val"].get("fde", float("nan")) for m in metrics_history], label="FDE", lw=2)
    axes[1, 1].plot(epochs, [m["val"].get("min_ade", float("nan")) for m in metrics_history], label="minADE", lw=2, ls="--")
    axes[1, 1].set_title("Trajectory Metrics (val)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Physics penalties
    axes[1, 2].plot(epochs, [m["val"].get("loss_accel_penalty", 0) for m in metrics_history], label="accel penalty", lw=2)
    axes[1, 2].plot(epochs, [m["val"].get("loss_speed_penalty", 0) for m in metrics_history], label="speed penalty", lw=2)
    axes[1, 2].set_title("Physics Penalties (val)")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("epoch")

    fig.suptitle("Training Curves (v2)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_per_step_ade(per_step_ade: List[float], dt: float, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_steps = [(i + 1) * dt for i in range(len(per_step_ade))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_steps, per_step_ade, "o-", lw=2, color="#2ca02c", markersize=5)
    ax.set_xlabel("Future time (s)")
    ax.set_ylabel("ADE (m)")
    ax.set_title("Per-step ADE over prediction horizon")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_tactical_ade_bar(grouped_metrics: Dict, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [l for l in grouped_metrics if grouped_metrics[l]["ade"] is not None]
    ades = [grouped_metrics[l]["ade"] for l in labels]
    fdes = [grouped_metrics[l]["fde"] for l in labels]
    counts = [grouped_metrics[l]["count"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars1 = ax1.bar(x - width / 2, ades, width, label="ADE", color="#3498db", alpha=0.85)
    bars2 = ax1.bar(x + width / 2, fdes, width, label="FDE", color="#e74c3c", alpha=0.85)
    ax1.set_xlabel("Tactical Intent")
    ax1.set_ylabel("Error (m)")
    ax1.set_title("ADE/FDE by Tactical Intent")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(labels, counts)])
    ax1.legend()
    ax1.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_prediction_sample(
    model, sample: Dict, save_path: Path, device: str, title: str = "",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = sample["obs_history"].unsqueeze(0).to(device)
    current = sample["current_target_pos"].numpy()
    gt_abs = current[None, :] + sample["gt_future_delta"].numpy()
    model.eval()
    with torch.no_grad():
        pred = model(obs)
    pred_abs = current[None, :] + pred["mixed_mu"][0].cpu().numpy()
    K = pred["gmm_means"].shape[1]
    comp_abs = []
    for k in range(K):
        comp_abs.append(current[None, :] + pred["gmm_means"][0, k].cpu().numpy())
    weights = pred["gmm_weights"][0].cpu().numpy()
    top_k = int(weights.argmax())

    dt = 0.05
    time = np.arange(gt_abs.shape[0]) * dt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # XY trajectory
    axes[0].plot(gt_abs[:, 0], gt_abs[:, 1], label="GT", color="#1f77b4", lw=2.5)
    axes[0].plot(pred_abs[:, 0], pred_abs[:, 1], label="Mixed mean", color="#d62728", lw=2)
    colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
    for k in range(K):
        axes[0].plot(comp_abs[k][:, 0], comp_abs[k][:, 1],
                     label=f"Comp {k} (π={weights[k]:.2f})", color=colors[k % 3], lw=1.5, ls="--")
    axes[0].scatter([current[0]], [current[1]], color="black", s=40, zorder=5, label="current")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("XY Trajectory")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    # Height profile
    axes[1].plot(time, gt_abs[:, 2], label="GT z", color="#1f77b4", lw=2.5)
    axes[1].plot(time, pred_abs[:, 2], label="Pred z", color="#d62728", lw=2)
    for k in range(K):
        axes[1].plot(time, comp_abs[k][:, 2],
                     label=f"Comp {k} z", color=colors[k % 3], lw=1.5, ls="--")
    axes[1].axhline(current[2], color="black", lw=1, ls=":", alpha=0.5)
    axes[1].set_xlabel("Future time (s)")
    axes[1].set_ylabel("z (m)")
    axes[1].set_title("Height Profile")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    # Error over time
    err = np.linalg.norm(pred_abs - gt_abs, axis=-1)
    axes[2].plot(time, err, "o-", color="#e74c3c", lw=2, markersize=4)
    axes[2].set_xlabel("Future time (s)")
    axes[2].set_ylabel("Displacement error (m)")
    axes[2].set_title("Per-step Error")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def find_samples_by_ade(model, dataset, device, max_checks=512):
    """Find best, worst, and median samples by ADE."""
    model.eval()
    ade_list = []
    max_checks = min(max_checks, len(dataset))
    with torch.no_grad():
        for idx in range(max_checks):
            sample = dataset[idx]
            obs = sample["obs_history"].unsqueeze(0).to(device)
            gt = sample["gt_future_delta"].unsqueeze(0).to(device)
            pred = model(obs)
            ade = torch.norm(pred["mixed_mu"] - gt, dim=-1).mean().item()
            ade_list.append((ade, idx, sample))

    ade_list.sort(key=lambda x: x[0])
    best = ade_list[0]
    worst = ade_list[-1]
    median = ade_list[len(ade_list) // 2]
    return best, median, worst


def plot_gmm_weight_dist(preds, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights = preds["gmm_weights"].numpy()
    K = weights.shape[1]
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(K):
        ax.hist(weights[:, k], bins=30, alpha=0.6, label=f"Component {k}")
    ax.set_xlabel("Weight (π)")
    ax.set_ylabel("Count")
    ax.set_title("GMM Weight Distribution")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=Path, default=TRAJ_DIR)
    parser.add_argument("--max-samples", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pred-seconds", type=float, default=0.5)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--no-maneuver", action="store_true",
                        help="Ablation: disable maneuver intent (feed uniform 1/9 to decoder)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Dynamic results dir
    if args.no_maneuver:
        mode_str = "no_maneuver"
    elif args.oracle:
        mode_str = "oracle"
    else:
        mode_str = "baseline"
    args.results_dir = PROJECT_ROOT / f"results/dual_intent_ablation_{args.pred_seconds}s_{mode_str}"
    
    args.results_dir.mkdir(parents=True, exist_ok=True)
    traj_files = sorted(args.traj_dir.glob("battle_dual_intent_*.npz"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory npz files found in {args.traj_dir}")

    print(f"Found {len(traj_files)} trajectory files")

    # Build dataset — unified 0.05s cadence
    dataset = UnifiedCadenceDataset(traj_files, max_samples=args.max_samples, dt=0.05, pred_seconds=args.pred_seconds)
    train_ds, val_ds = split_dataset(dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=UnifiedCadenceDataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=UnifiedCadenceDataset.collate_fn)

    print(f"Dataset: {len(dataset)} samples, train={len(train_ds)}, val={len(val_ds)}")
    print(f"Input shape: {dataset.stats()['obs_shape']}, Output shape: {dataset.stats()['future_shape']}")

    # Compute Focal Loss alpha from maneuver class distribution (Close_Evade samples only)
    maneuver_labels = torch.tensor([s["gt_maneuver"] for s in dataset.samples], dtype=torch.long)
    strategic_labels = torch.tensor([s["gt_strategic"] for s in dataset.samples], dtype=torch.long)
    close_evade_maneuvers = maneuver_labels[strategic_labels == 1]
    if len(close_evade_maneuvers) > 0:
        maneuver_counts = torch.bincount(close_evade_maneuvers, minlength=len(MANEUVER_LABELS))
        focal_alpha = DualIntentGMMLoss.compute_focal_alpha(maneuver_counts)
        print(f"Focal alpha (maneuver): {focal_alpha.tolist()}")
        print(f"  Maneuver counts: {dict(zip(MANEUVER_LABELS, maneuver_counts.tolist()))}")
    else:
        focal_alpha = None
        print("Warning: No Close_Evade samples found, Focal Loss alpha not computed.")

    # Model
    device = torch.device(args.device)
    pred_horizon_steps = int(round(args.pred_seconds / 0.05))
    cfg = DualIntentGMMConfig(obs_dim=30, pred_horizon=pred_horizon_steps, n_components=3, d_model=128)
    model = DualIntentQueryGMMPredictor(cfg).to(device)
    model.use_oracle = args.oracle
    model.disable_maneuver = args.no_maneuver
    loss_fn = DualIntentGMMLoss(
        dt_pred=0.05, max_speed=3.0, max_accel=3.0, min_height_delta=0.2,
        focal_gamma=2.0, maneuver_alpha=focal_alpha, wta_fraction=1.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # =========================================================================
    # Training loop
    # =========================================================================
    metrics_history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_accum = {}
        seen = 0
        for batch in train_loader:
            obs = batch["obs_history"].to(device)
            gt_future = batch["gt_future_delta"].to(device)
            gt_strategic = batch["gt_strategic"].to(device)
            gt_maneuver = batch["gt_maneuver"].to(device)
            current_z = batch["current_target_pos"][:, 2].to(device)

            optimizer.zero_grad(set_to_none=True)
            kwargs = {}
            if getattr(model, "use_oracle", False):
                kwargs["gt_strategic"] = gt_strategic
                kwargs["gt_maneuver"] = gt_maneuver
                
            pred = model(obs, **kwargs)
            loss, metrics = loss_fn(pred, gt_future, gt_strategic, gt_maneuver, current_z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = obs.shape[0]
            for key, value in metrics.items():
                if not math.isnan(value):
                    train_accum[key] = train_accum.get(key, 0.0) + value * batch_size
            seen += batch_size

        train_metrics = {key: value / max(1, seen) for key, value in train_accum.items()}
        val_metrics = evaluate(model, loss_fn, val_loader, device)
        metrics_history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch:3d}  "
            f"train_loss={train_metrics['loss_total']:.4f}  "
            f"val_loss={val_metrics['loss_total']:.4f}  "
            f"val_strategic_acc={val_metrics['strategic_accuracy']:.3f}  "
            f"val_ade={val_metrics.get('ade', float('nan')):.4f}  "
            f"val_fde={val_metrics.get('fde', float('nan')):.4f}",
            flush=True,
        )

    # =========================================================================
    # Comprehensive Evaluation
    # =========================================================================
    print("\n" + "=" * 60)
    print("  Running comprehensive evaluation...")
    print("=" * 60)

    preds, gt = collect_predictions(model, val_loader, device)

    # A. Intent classification metrics
    pred_strategic = preds["strategic_logits"].argmax(dim=-1)
    cm_strategic = compute_confusion_matrix(pred_strategic, gt["gt_strategic"], len(STRATEGIC_LABELS))
    strategic_class_metrics = compute_per_class_metrics(cm_strategic, STRATEGIC_LABELS)

    close_mask = gt["gt_strategic"] == 1  # Close_Evade
    if close_mask.sum() > 0:
        pred_maneuver = preds["maneuver_logits"][close_mask].argmax(dim=-1)
        gt_maneuver_ce = gt["gt_maneuver"][close_mask]
        cm_maneuver = compute_confusion_matrix(pred_maneuver, gt_maneuver_ce, len(MANEUVER_LABELS))
        maneuver_class_metrics = compute_per_class_metrics(cm_maneuver, MANEUVER_LABELS)
    else:
        cm_maneuver = torch.zeros(len(MANEUVER_LABELS), len(MANEUVER_LABELS), dtype=torch.long)
        maneuver_class_metrics = {}

    # B. Trajectory metrics
    traj_metrics = compute_trajectory_metrics(preds, gt, dt=0.05)

    # C. Tactical-grouped metrics
    tactical_grouped = compute_tactical_grouped_metrics(preds, gt, dt=0.05)

    # D. GMM quality
    gmm_quality = compute_gmm_quality(preds)

    # =========================================================================
    # Save all results
    # =========================================================================
    rd = args.results_dir

    # Plots
    plot_confusion_matrix(cm_strategic, STRATEGIC_LABELS,
                          "Strategic Intent Confusion Matrix", rd / "confusion_strategic.png")
    if close_mask.sum() > 0:
        plot_confusion_matrix(cm_maneuver, MANEUVER_LABELS,
                              "Maneuver Intent Confusion Matrix (Close_Evade only)",
                              rd / "confusion_maneuver.png")
    plot_training_curves(metrics_history, rd / "training_curves.png")
    plot_per_step_ade(traj_metrics["per_step_ade"], dt=0.05, save_path=rd / "per_step_ade.png")
    plot_tactical_ade_bar(tactical_grouped, rd / "ade_by_tactical.png")
    plot_gmm_weight_dist(preds, rd / "gmm_weight_dist.png")

    # Prediction samples (best / median / worst)
    best, median, worst = find_samples_by_ade(model, val_ds, str(device))
    plot_prediction_sample(model, best[2], rd / "pred_best.png", str(device),
                           f"Best sample (ADE={best[0]:.3f}m)")
    plot_prediction_sample(model, median[2], rd / "pred_median.png", str(device),
                           f"Median sample (ADE={median[0]:.3f}m)")
    plot_prediction_sample(model, worst[2], rd / "pred_worst.png", str(device),
                           f"Worst sample (ADE={worst[0]:.3f}m)")

    # Model checkpoint
    model_path = rd / "prototype_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, model_path)

    # Metrics JSON
    summary = {
        "dataset": dataset.stats(),
        "config": asdict(cfg),
        "cadence": {
            "input_dt": 0.05, "input_frames": 10,
            "output_dt": 0.05, "output_frames": 10,
            "history_seconds": 0.5, "prediction_seconds": 0.5,
        },
        "param_count": param_count,
        "final_val_metrics": metrics_history[-1]["val"],
        "strategic_classification": strategic_class_metrics,
        "maneuver_classification": maneuver_class_metrics,
        "trajectory_metrics": traj_metrics,
        "tactical_grouped_trajectory": tactical_grouped,
        "gmm_quality": gmm_quality,
        "metrics_history": metrics_history,
    }
    (rd / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Text report
    report_lines = [
        "=" * 70,
        "双层意图 Query-GMM 轨迹预测网络 v2 评估报告",
        "=" * 70,
        "",
        "1. 基本信息",
        f"   轨迹文件: {len(traj_files)} files",
        f"   总样本数: {len(dataset)}, 训练: {len(train_ds)}, 验证: {len(val_ds)}",
        f"   输入: (B, 10, 30) @ 0.05s, 0.5s history",
        f"   输出: (B, 10, 3) @ 0.05s, 0.5s prediction",
        f"   模型参数: {param_count:,}",
        f"   训练 epochs: {args.epochs}",
        "",
        "2. 战术意图分类 (4 类)",
        f"   Accuracy: {metrics_history[-1]['val']['strategic_accuracy']:.4f}",
        f"   Macro F1: {strategic_class_metrics.get('macro_f1', 'N/A')}",
        f"   Weighted F1: {strategic_class_metrics.get('weighted_f1', 'N/A')}",
    ]
    for label in STRATEGIC_LABELS:
        if label in strategic_class_metrics:
            m = strategic_class_metrics[label]
            report_lines.append(f"   {label:16s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}")
    report_lines += [
        "",
        "3. 机动意图分类 (9 类, Close_Evade only)",
        f"   Accuracy: {metrics_history[-1]['val'].get('maneuver_accuracy_close_evade', 'N/A')}",
        f"   Macro F1: {maneuver_class_metrics.get('macro_f1', 'N/A')}",
    ]
    for label in MANEUVER_LABELS:
        if label in maneuver_class_metrics:
            m = maneuver_class_metrics[label]
            report_lines.append(f"   {label:16s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}")
    report_lines += [
        "",
        "4. 轨迹预测指标",
        f"   ADE:    {traj_metrics['ade']:.6f} m",
        f"   FDE:    {traj_metrics['fde']:.6f} m",
        f"   minADE: {traj_metrics['min_ade']:.6f} m",
        f"   minFDE: {traj_metrics['min_fde']:.6f} m",
        f"   ADE_x:  {traj_metrics['ade_x']:.6f} m",
        f"   ADE_y:  {traj_metrics['ade_y']:.6f} m",
        f"   ADE_z:  {traj_metrics['ade_z']:.6f} m",
        "",
        "5. 物理合理性",
        f"   速度超限比例:   {traj_metrics['speed_exceed_ratio']:.4f}",
        f"   加速度超限比例: {traj_metrics['accel_exceed_ratio']:.4f}",
        f"   平均速度:       {traj_metrics['mean_speed']:.4f} m/s",
        f"   平均加速度:     {traj_metrics['mean_accel']:.4f} m/s²",
        f"   加速度平滑度:   {traj_metrics['accel_smoothness_std']:.4f}",
        "",
        "6. 按战术分组的轨迹误差",
    ]
    for label in STRATEGIC_LABELS:
        g = tactical_grouped.get(label, {})
        if g.get("ade") is not None:
            report_lines.append(f"   {label:16s}  ADE={g['ade']:.4f}  FDE={g['fde']:.4f}  n={g['count']}")
        else:
            report_lines.append(f"   {label:16s}  (no samples)")
    report_lines += [
        "",
        "7. GMM 分量质量",
        f"   权重熵:           {gmm_quality['weight_entropy']:.4f}",
        f"   分量间平均距离:   {gmm_quality['mean_inter_component_distance']:.4f} m",
        "",
        "8. 输出文件",
    ]
    for f in sorted(rd.iterdir()):
        if f.is_file():
            report_lines.append(f"   {f.name}")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    (rd / "report.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nSaved results to {rd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
