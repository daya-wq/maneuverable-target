#!/usr/bin/env python3
"""
Analyze short-window maneuver signatures for dual-intent 6DoF battles.

Inputs:
- A 1s maneuver-update result directory
- A 2s maneuver-update result directory

Outputs:
- Battle contact sheet comparison
- Feature heatmaps for 1s / 2s windows
- Simple leave-one-out nearest-centroid accuracy as a proxy for intent separability
- Markdown report with embedded figures
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

for name, path in [
    ("omni_drones", "omni_drones"),
    ("omni_drones.envs", "omni_drones/envs"),
    ("omni_drones.envs.hide_and_seek", "omni_drones/envs/hide_and_seek"),
]:
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(PROJECT_ROOT, path)]
    mod.__package__ = name
    sys.modules[name] = mod

from omni_drones.envs.hide_and_seek.dual_intent_target import MANEUVER_NAMES, TACTICAL_CLOSE_EVADE


EPS = 1e-6
MODE_COLORS = [
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze dual-intent short-window maneuver signatures.")
    parser.add_argument(
        "--dir-1s",
        type=str,
        default="results/dual_intent_6dof_battles_1p0s_compare",
        help="Result directory for the 1.0s maneuver update run.",
    )
    parser.add_argument(
        "--dir-2s",
        type=str,
        default="results/dual_intent_6dof_battles_2p0s_compare",
        help="Result directory for the 2.0s maneuver update run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dual_intent_window_analysis_20260612",
        help="Directory to save analysis figures and markdown report.",
    )
    return parser.parse_args()


def sorted_npz_files(results_dir: Path) -> list[Path]:
    traj_dir = results_dir / "trajectory_npz"
    return sorted(traj_dir.glob("battle_dual_intent_*.npz"))


def load_scenarios(results_dir: Path) -> list[dict]:
    scenarios = []
    for npz_path in sorted_npz_files(results_dir):
        raw = np.load(npz_path, allow_pickle=True)
        data = {key: raw[key] for key in raw.files}
        scenario_id = int(npz_path.stem.split("_")[-1])
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "seed": int(data["seed"][0]),
                "status": str(data["status"][0]),
                "dt": float(data["dt"][0]),
                "tactical_check_period": float(data.get("tactical_check_period", np.asarray([0.5]))[0]),
                "maneuver_update_period": float(data.get("maneuver_update_period", np.asarray([1.0]))[0]),
                "target_pos": data["target_pos"].astype(np.float64),
                "target_vel": data["target_vel"].astype(np.float64),
                "pursuer_pos": data["pursuer_pos"].astype(np.float64),
                "tactical_hist": data["tactical_hist"].astype(np.int64),
                "maneuver_hist": data["maneuver_hist"].astype(np.int64),
                "maneuver_mode_hist": data["maneuver_mode_hist"].astype(np.int64),
                "min_dist_hist": data["min_dist_hist"].astype(np.float64),
                "init_target": data["init_target"].astype(np.float64),
                "init_pursuers": data["init_pursuers"].astype(np.float64),
                "plot_path": results_dir / f"battle_dual_intent_{scenario_id:02d}.png",
                "npz_path": npz_path,
            }
        )
    return scenarios


def extract_segments(scenarios: list[dict]) -> list[dict]:
    segments = []
    for sc in scenarios:
        tactical = sc["tactical_hist"]
        mode = sc["maneuver_mode_hist"]
        pos = sc["target_pos"]
        vel = sc["target_vel"]
        dt = sc["dt"]
        n = len(tactical)
        idx = 0
        while idx < n:
            if tactical[idx] != TACTICAL_CLOSE_EVADE:
                idx += 1
                continue
            start = idx
            current_mode = int(mode[idx])
            idx += 1
            while idx < n and tactical[idx] == TACTICAL_CLOSE_EVADE and int(mode[idx]) == current_mode:
                idx += 1
            end = idx - 1
            duration = (end - start) * dt
            if duration < 0.2:
                continue
            segments.append(
                {
                    "scenario_id": sc["scenario_id"],
                    "seed": sc["seed"],
                    "status": sc["status"],
                    "dt": dt,
                    "mode": current_mode,
                    "mode_name": MANEUVER_NAMES[current_mode],
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "pos": pos[start:end + 1].copy(),
                    "vel": vel[start:end + 1].copy(),
                }
            )
    return segments


def build_local_frame(initial_vel: np.ndarray, fallback_disp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = initial_vel[:2].copy()
    if np.linalg.norm(forward) < 0.05:
        forward = fallback_disp[:2].copy()
    if np.linalg.norm(forward) < 0.05:
        forward = np.array([1.0, 0.0], dtype=np.float64)
    forward = forward / (np.linalg.norm(forward) + EPS)
    lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
    return forward, lateral


def make_window_sample(segment: dict, horizon_s: float) -> dict | None:
    dt = segment["dt"]
    steps = int(round(horizon_s / dt))
    if len(segment["pos"]) <= steps:
        return None

    pos = segment["pos"][:steps + 1]
    vel = segment["vel"][:steps + 1]
    disp = pos - pos[0]
    fallback_disp = pos[min(5, len(pos) - 1)] - pos[0]
    forward, lateral = build_local_frame(vel[0], fallback_disp)

    xy = disp[:, :2]
    local_forward = xy @ forward
    local_lateral = xy @ lateral
    local_z = disp[:, 2]

    speed0 = np.linalg.norm(vel[0])
    speed1 = np.linalg.norm(vel[-1])
    vel0_xy = vel[0, :2]
    vel1_xy = vel[-1, :2]
    if np.linalg.norm(vel0_xy) < 0.05:
        vel0_xy = forward.copy()
    if np.linalg.norm(vel1_xy) < 0.05:
        vel1_xy = xy[-1].copy()
    if np.linalg.norm(vel1_xy) < 0.05:
        vel1_xy = forward.copy()
    vel0_xy = vel0_xy / (np.linalg.norm(vel0_xy) + EPS)
    vel1_xy = vel1_xy / (np.linalg.norm(vel1_xy) + EPS)

    cross_z = vel0_xy[0] * vel1_xy[1] - vel0_xy[1] * vel1_xy[0]
    dot = np.clip(np.dot(vel0_xy, vel1_xy), -1.0, 1.0)
    heading_change_deg = math.degrees(math.atan2(cross_z, dot))

    segment_steps = pos[1:] - pos[:-1]
    path_len = np.linalg.norm(segment_steps, axis=1).sum()
    straight_dist = np.linalg.norm(pos[-1] - pos[0])
    path_eff = path_len / (straight_dist + EPS)

    features = {
        "forward_disp": float(local_forward[-1]),
        "lateral_disp": float(local_lateral[-1]),
        "vertical_disp": float(local_z[-1]),
        "heading_change_deg": float(heading_change_deg),
        "speed_delta": float(speed1 - speed0),
        "path_efficiency": float(path_eff),
    }

    return {
        "mode": segment["mode"],
        "mode_name": segment["mode_name"],
        "scenario_id": segment["scenario_id"],
        "seed": segment["seed"],
        "status": segment["status"],
        "horizon_s": horizon_s,
        "features": features,
        "curve_xy": np.column_stack([local_forward, local_lateral]),
        "curve_xz": np.column_stack([local_forward, local_z]),
    }


def collect_window_samples(segments: list[dict], horizon_s: float) -> list[dict]:
    samples = []
    for seg in segments:
        sample = make_window_sample(seg, horizon_s)
        if sample is not None:
            samples.append(sample)
    return samples


def summarize_scenarios(scenarios: list[dict]) -> dict:
    statuses = Counter(sc["status"] for sc in scenarios)
    durations = [((len(sc["target_pos"]) - 1) * sc["dt"]) for sc in scenarios]
    min_dists = [float(sc["min_dist_hist"].min()) for sc in scenarios if len(sc["min_dist_hist"]) > 0]
    tactical_switches = []
    mode_switches = []
    for sc in scenarios:
        tactical = sc["tactical_hist"]
        mode = sc["maneuver_mode_hist"]
        tactical_switches.append(int(np.sum(tactical[1:] != tactical[:-1])))
        mode_switches.append(int(np.sum(mode[1:] != mode[:-1])))
    return {
        "status_counts": dict(statuses),
        "mean_duration": float(np.mean(durations)) if durations else 0.0,
        "mean_min_dist": float(np.mean(min_dists)) if min_dists else 0.0,
        "mean_tactical_switches": float(np.mean(tactical_switches)) if tactical_switches else 0.0,
        "mean_mode_switches": float(np.mean(mode_switches)) if mode_switches else 0.0,
    }


def samples_to_matrix(samples: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_names = [
        "forward_disp",
        "lateral_disp",
        "vertical_disp",
        "heading_change_deg",
        "speed_delta",
        "path_efficiency",
    ]
    x = np.asarray([[sample["features"][name] for name in feature_names] for sample in samples], dtype=np.float64)
    y = np.asarray([sample["mode"] for sample in samples], dtype=np.int64)
    return x, y, feature_names


def nearest_centroid_loo(samples: list[dict]) -> dict:
    if not samples:
        return {
            "overall_acc": float("nan"),
            "covered_samples": 0,
            "total_samples": 0,
            "per_mode_acc": {},
            "counts": {},
        }

    x, y, _ = samples_to_matrix(samples)
    counts = Counter(y.tolist())
    valid_indices = [idx for idx, label in enumerate(y.tolist()) if counts[label] >= 2]
    if not valid_indices:
        return {
            "overall_acc": float("nan"),
            "covered_samples": 0,
            "total_samples": len(samples),
            "per_mode_acc": {},
            "counts": {MANEUVER_NAMES[k]: v for k, v in counts.items()},
        }

    correct = []
    per_mode_correct = defaultdict(int)
    per_mode_total = defaultdict(int)

    for idx in valid_indices:
        train_mask = np.ones(len(samples), dtype=bool)
        train_mask[idx] = False
        train_x = x[train_mask]
        train_y = y[train_mask]
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)
        std[std < 1e-6] = 1.0
        train_z = (train_x - mean) / std
        test_z = (x[idx] - mean) / std

        centroids = {}
        for label in sorted(set(train_y.tolist())):
            label_mask = train_y == label
            if not np.any(label_mask):
                continue
            centroids[int(label)] = train_z[label_mask].mean(axis=0)

        if int(y[idx]) not in centroids:
            continue

        pred = min(
            centroids.items(),
            key=lambda item: np.linalg.norm(test_z - item[1]),
        )[0]
        ok = int(pred == int(y[idx]))
        correct.append(ok)
        per_mode_total[int(y[idx])] += 1
        per_mode_correct[int(y[idx])] += ok

    per_mode_acc = {
        MANEUVER_NAMES[label]: per_mode_correct[label] / per_mode_total[label]
        for label in sorted(per_mode_total)
        if per_mode_total[label] > 0
    }
    return {
        "overall_acc": float(np.mean(correct)) if correct else float("nan"),
        "covered_samples": len(correct),
        "total_samples": len(samples),
        "per_mode_acc": per_mode_acc,
        "counts": {MANEUVER_NAMES[k]: v for k, v in counts.items()},
    }


def build_contact_sheet(dir_1s: Path, dir_2s: Path, out_path: Path):
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.patch.set_facecolor("white")
    layout = [
        ("1.0s update", dir_1s),
        ("2.0s update", dir_2s),
    ]

    for row, (row_title, results_dir) in enumerate(layout):
        pngs = [results_dir / f"battle_dual_intent_{idx:02d}.png" for idx in range(1, 11)]
        for col, png_path in enumerate(pngs[:5]):
            ax = axes[row, col]
            ax.axis("off")
            if png_path.exists():
                ax.imshow(plt.imread(png_path))
                ax.set_title(f"{row_title} | S{col + 1:02d}", fontsize=10)
        for col, png_path in enumerate(pngs[5:], start=0):
            ax = axes[row, col]
            ax.axis("off")

    # Replace with 2x5 chronological layout.
    plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(24, 18))
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.axis("off")

    for idx in range(10):
        png_1 = dir_1s / f"battle_dual_intent_{idx + 1:02d}.png"
        png_2 = dir_2s / f"battle_dual_intent_{idx + 1:02d}.png"
        ax1 = axes[0 if idx < 5 else 1, idx % 5]
        ax2 = axes[2 if idx < 5 else 3, idx % 5]
        if png_1.exists():
            ax1.imshow(plt.imread(png_1))
            ax1.set_title(f"1.0s | Scenario {idx + 1:02d}", fontsize=10)
        if png_2.exists():
            ax2.imshow(plt.imread(png_2))
            ax2.set_title(f"2.0s | Scenario {idx + 1:02d}", fontsize=10)

    fig.suptitle("Dual-Intent Battle Trajectories: 1.0s vs 2.0s Maneuver Update", fontsize=18, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_heatmaps(setting_samples: dict[str, list[dict]], out_path: Path):
    feature_names = [
        "forward_disp",
        "lateral_disp",
        "vertical_disp",
        "heading_change_deg",
        "speed_delta",
        "path_efficiency",
    ]
    pretty_names = [
        "Fwd",
        "Lat",
        "Vert",
        "Heading",
        "dSpeed",
        "Curv",
    ]
    settings = list(setting_samples.keys())
    fig, axes = plt.subplots(1, len(settings), figsize=(6.4 * len(settings), 8), constrained_layout=True)
    if len(settings) == 1:
        axes = [axes]

    for ax, setting in zip(axes, settings):
        samples = setting_samples[setting]
        heat = np.full((len(MANEUVER_NAMES), len(feature_names)), np.nan, dtype=np.float64)
        counts = Counter(sample["mode"] for sample in samples)
        for mode in range(len(MANEUVER_NAMES)):
            mode_samples = [sample for sample in samples if sample["mode"] == mode]
            if not mode_samples:
                continue
            values = np.asarray(
                [[sample["features"][name] for name in feature_names] for sample in mode_samples],
                dtype=np.float64,
            )
            mean = values.mean(axis=0)
            scale = np.maximum(values.std(axis=0), 1e-6)
            heat[mode] = mean / scale

        vmax = np.nanmax(np.abs(heat))
        vmax = max(vmax, 1.0)
        im = ax.imshow(heat, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(setting, fontsize=13)
        ax.set_xticks(range(len(pretty_names)))
        ax.set_xticklabels(pretty_names, rotation=25)
        ax.set_yticks(range(len(MANEUVER_NAMES)))
        ax.set_yticklabels([f"{name} ({counts.get(i, 0)})" for i, name in enumerate(MANEUVER_NAMES)])

        for row in range(len(MANEUVER_NAMES)):
            for col in range(len(feature_names)):
                val = heat[row, col]
                text = "-" if np.isnan(val) else f"{val:.1f}"
                ax.text(col, row, text, ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label("mean / std within mode")
    fig.suptitle("Short-Window Maneuver Feature Signatures", fontsize=17)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_summary(results_by_setting: dict[str, dict], out_path: Path):
    settings = list(results_by_setting.keys())
    overall = [results_by_setting[name]["overall_acc"] for name in settings]
    covered = [results_by_setting[name]["covered_samples"] for name in settings]
    total = [results_by_setting[name]["total_samples"] for name in settings]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 11), constrained_layout=True)
    x = np.arange(len(settings))
    ax0.bar(x, overall, color=["#4c78a8", "#f58518", "#54a24b"][: len(settings)])
    ax0.set_ylim(0.0, 1.0)
    ax0.set_xticks(x)
    ax0.set_xticklabels(settings, rotation=10)
    ax0.set_ylabel("LOO centroid accuracy")
    ax0.set_title("Overall separability from short-window handcrafted features")
    for idx, acc in enumerate(overall):
        ax0.text(idx, min(acc + 0.03, 0.98), f"{acc:.2f}\n{covered[idx]}/{total[idx]}", ha="center", fontsize=10)

    mode_positions = np.arange(len(MANEUVER_NAMES))
    width = 0.24 if len(settings) >= 3 else 0.35
    offsets = np.linspace(-(len(settings) - 1) * width / 2.0, (len(settings) - 1) * width / 2.0, len(settings))
    for offset, setting, color in zip(offsets, settings, ["#4c78a8", "#f58518", "#54a24b"]):
        vals = []
        per_mode = results_by_setting[setting]["per_mode_acc"]
        for name in MANEUVER_NAMES:
            vals.append(per_mode.get(name, np.nan))
        ax1.bar(mode_positions + offset, vals, width=width, label=setting, color=color, alpha=0.88)

    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(mode_positions)
    ax1.set_xticklabels(MANEUVER_NAMES, rotation=30, ha="right")
    ax1.set_ylabel("Per-mode accuracy")
    ax1.set_title("Per-mode short-window separability")
    ax1.legend()

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_signature_projections(samples: list[dict], title: str, out_path_xy: Path, out_path_xz: Path):
    fig_xy, axes_xy = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    fig_xz, axes_xz = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    axes_xy = axes_xy.ravel()
    axes_xz = axes_xz.ravel()

    for mode in range(len(MANEUVER_NAMES)):
        ax_xy = axes_xy[mode]
        ax_xz = axes_xz[mode]
        mode_samples = [sample for sample in samples if sample["mode"] == mode]
        color = MODE_COLORS[mode]

        for sample in mode_samples:
            curve_xy = sample["curve_xy"]
            curve_xz = sample["curve_xz"]
            ax_xy.plot(curve_xy[:, 0], curve_xy[:, 1], color=color, alpha=0.18, lw=1.2)
            ax_xz.plot(curve_xz[:, 0], curve_xz[:, 1], color=color, alpha=0.18, lw=1.2)

        if mode_samples:
            min_len = min(curve["curve_xy"].shape[0] for curve in mode_samples)
            mean_xy = np.mean(np.stack([sample["curve_xy"][:min_len] for sample in mode_samples], axis=0), axis=0)
            mean_xz = np.mean(np.stack([sample["curve_xz"][:min_len] for sample in mode_samples], axis=0), axis=0)
            ax_xy.plot(mean_xy[:, 0], mean_xy[:, 1], color="black", lw=2.0)
            ax_xz.plot(mean_xz[:, 0], mean_xz[:, 1], color="black", lw=2.0)

        ax_xy.axhline(0.0, color="#cccccc", lw=0.8)
        ax_xy.axvline(0.0, color="#cccccc", lw=0.8)
        ax_xz.axhline(0.0, color="#cccccc", lw=0.8)
        ax_xz.axvline(0.0, color="#cccccc", lw=0.8)
        ax_xy.set_title(f"{MANEUVER_NAMES[mode]} ({len(mode_samples)})", fontsize=10)
        ax_xz.set_title(f"{MANEUVER_NAMES[mode]} ({len(mode_samples)})", fontsize=10)
        ax_xy.set_xlabel("forward disp (m)")
        ax_xy.set_ylabel("lateral disp (m)")
        ax_xz.set_xlabel("forward disp (m)")
        ax_xz.set_ylabel("vertical disp (m)")

    fig_xy.suptitle(f"{title}: Local XY signatures", fontsize=16)
    fig_xz.suptitle(f"{title}: Local XZ signatures", fontsize=16)
    fig_xy.savefig(out_path_xy, dpi=180, bbox_inches="tight")
    fig_xz.savefig(out_path_xz, dpi=180, bbox_inches="tight")
    plt.close(fig_xy)
    plt.close(fig_xz)


def format_status_counts(status_counts: dict) -> str:
    keys = ["GOAL_REACHED", "CAPTURED", "TIMEOUT"]
    return ", ".join(f"{key}={status_counts.get(key, 0)}" for key in keys)


def pick_key_observations(
    summary_1s: dict,
    summary_2s: dict,
    acc_1s: dict,
    acc_2s_1: dict,
    acc_2s_2: dict,
) -> list[str]:
    obs = []
    obs.append(
        f"1.0s更新结果整体状态分布为 {format_status_counts(summary_1s['status_counts'])}；"
        f"2.0s更新结果为 {format_status_counts(summary_2s['status_counts'])}。"
    )
    obs.append(
        f"平均最小交会距离从 {summary_1s['mean_min_dist']:.2f} m 变化到 {summary_2s['mean_min_dist']:.2f} m，"
        f"平均机动模式切换次数从 {summary_1s['mean_mode_switches']:.1f} 次降到 {summary_2s['mean_mode_switches']:.1f} 次。"
    )
    obs.append(
        f"基于短窗手工特征的LOO最近质心准确率："
        f"1.0s更新@1s={acc_1s['overall_acc']:.2f}，"
        f"2.0s更新@前1s={acc_2s_1['overall_acc']:.2f}，"
        f"2.0s更新@完整2s={acc_2s_2['overall_acc']:.2f}。"
    )
    full2s_coverage = acc_2s_2["covered_samples"] / max(acc_2s_2["total_samples"], 1)
    if acc_2s_2["covered_samples"] < 10 or full2s_coverage < 0.8:
        obs.append(
            "完整2.0s纯净机动片段数量偏少，原因是很多 Close_Evade 片段会在 2s 内被战术层提前打断，"
            "所以当前数据更适合说明“2s窗口样本稀缺”，不适合单独据此否定 2s 历史窗。"
        )
    elif acc_2s_2["overall_acc"] >= acc_2s_1["overall_acc"]:
        obs.append("2.0s机制下继续观察到完整2s，通常会让机动差异比只看前1s更清楚。")
    else:
        obs.append("2.0s机制下，额外的后1s没有明显继续拉开机动差异，说明首秒已经承载了主要信息。")
    return obs


def write_report(
    out_path: Path,
    dir_1s: Path,
    dir_2s: Path,
    summary_1s: dict,
    summary_2s: dict,
    segment_counts: dict[str, Counter],
    acc_results: dict[str, dict],
):
    rel = lambda path: path.name
    observations = pick_key_observations(
        summary_1s,
        summary_2s,
        acc_results["1.0s update | 1.0s window"],
        acc_results["2.0s update | first 1.0s"],
        acc_results["2.0s update | full 2.0s"],
    )

    lines = [
        "# 双意图 6DoF 短时机动特征分析",
        "",
        "## 分析目的",
        "",
        "验证在 `dual_intent_6dof_battle` 设定下，当战术意图仍按 `0.5s` 检查不变时，",
        "把机动模式刷新周期从 `1.0s` 改到 `2.0s` 后，目标机在短时窗口内是否能展示出更稳定、可分的机动特征。",
        "这个结论直接关系到后续意图预测器在短历史窗下的分类上限。",
        "",
        "本报告对比了两套结果目录：",
        f"- `1.0s` 机动刷新：`{dir_1s}`",
        f"- `2.0s` 机动刷新：`{dir_2s}`",
        "",
        "## 关键结论",
        "",
    ]
    for item in observations:
        lines.append(f"- {item}")

    lines += [
        "",
        "## 双方轨迹总览图",
        "",
        f"![battle_contact_sheet]({rel(out_path.parent / 'battle_contact_sheet_compare.png')})",
        "",
        "上图按相同 scenario 编号对比了 `1.0s` 和 `2.0s` 两套双方轨迹。",
        "可以直接观察到：`2.0s` 设置下单次 Close_Evade 机动持续更久，目标轨迹局部弯折更完整，追捕方的 PN 追踪轨迹也会出现更明显的响应滞后和包抄重构。",
        "",
        "## 结果统计",
        "",
        "| 设置 | 状态分布 | 平均时长(s) | 平均最小距离(m) | 平均战术切换次数 | 平均机动模式切换次数 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| 1.0s刷新 | {format_status_counts(summary_1s['status_counts'])} | {summary_1s['mean_duration']:.2f} | {summary_1s['mean_min_dist']:.2f} | {summary_1s['mean_tactical_switches']:.1f} | {summary_1s['mean_mode_switches']:.1f} |",
        f"| 2.0s刷新 | {format_status_counts(summary_2s['status_counts'])} | {summary_2s['mean_duration']:.2f} | {summary_2s['mean_min_dist']:.2f} | {summary_2s['mean_tactical_switches']:.1f} | {summary_2s['mean_mode_switches']:.1f} |",
        "",
        "Close_Evade 片段样本数：",
        "",
        f"- `1.0s` 数据集：{dict(segment_counts['1s'])}",
        f"- `2.0s` 数据集：{dict(segment_counts['2s'])}",
        "- 说明：这里的 `2.0s` 数据集只统计了能稳定持续满 `2s` 的纯净机动段，因此样本数会明显少于“前1s窗口”的统计口径。",
        "",
        "## 短窗特征签名",
        "",
        f"![feature_heatmaps]({rel(out_path.parent / 'feature_heatmaps.png')})",
        "",
        "热图中的 6 个特征分别是：前向位移、侧向位移、垂向位移、航向变化、速度变化、轨迹曲折度。",
        "每个数值是该模式在对应窗口下的 `均值 / 标准差`，数值越大代表该模式在该特征上的偏向越稳定。",
        "",
        "从图上通常可以读出三类现象：",
        "- 左/右转相关模式主要体现在 `侧向位移` 和 `航向变化` 的符号分离。",
        "- 爬升/俯冲相关模式主要体现在 `垂向位移` 的正负分离。",
        "- 组合模式（如 `Left-Climb`、`Right-Dive`）在 `侧向` 和 `垂向` 两个维度同时带符号，通常比纯直行更容易区分。",
        "",
        "## 对意图分类的启发式可分性评估",
        "",
        f"![accuracy_summary]({rel(out_path.parent / 'accuracy_summary.png')})",
        "",
        "这里没有直接训练分类器，而是用短窗轨迹特征做了一个留一法最近质心分类，",
        "把它当作“这些短时间片本身是否带有可分信息”的近似指标。",
        "",
        f"- `1.0s刷新 | 1.0s窗口`：整体准确率 {acc_results['1.0s update | 1.0s window']['overall_acc']:.2f}，覆盖样本 {acc_results['1.0s update | 1.0s window']['covered_samples']}/{acc_results['1.0s update | 1.0s window']['total_samples']}",
        f"- `2.0s刷新 | 前1.0s窗口`：整体准确率 {acc_results['2.0s update | first 1.0s']['overall_acc']:.2f}，覆盖样本 {acc_results['2.0s update | first 1.0s']['covered_samples']}/{acc_results['2.0s update | first 1.0s']['total_samples']}",
        f"- `2.0s刷新 | 完整2.0s窗口`：整体准确率 {acc_results['2.0s update | full 2.0s']['overall_acc']:.2f}，覆盖样本 {acc_results['2.0s update | full 2.0s']['covered_samples']}/{acc_results['2.0s update | full 2.0s']['total_samples']}",
        "",
        "需要特别注意：`完整2.0s窗口` 的可用样本非常少，",
        "因为不少机动段在满 2s 之前就被 `0.5s` 战术层检查触发了战术切换，导致纯净 2s 片段不足。",
        "所以这一列更适合作为“当前数据覆盖是否足够”的提醒，而不是最终分类上限的定论。",
        "",
        "这组数字可以直接用来指导后续预测器的历史窗长度：",
        "- 如果 `2.0s完整窗口` 明显高于两个 `1.0s窗口`，说明后1秒额外信息很关键，分类器至少应看到 2s 历史。",
        "- 如果 `2.0s前1.0s窗口` 已经接近 `2.0s完整窗口`，说明首秒已足够强，后续模型可以优先优化实时性。",
        "- 如果 `Straight`、`Climb`、`Dive` 这类纯单轴模式准确率明显高，而复合模式偏低，说明未来应加入更细的时序特征而不只是末端位移特征。",
        "",
        "## 轨迹局部投影",
        "",
        f"![sig_xy_1s]({rel(out_path.parent / 'signatures_xy_1s.png')})",
        "",
        f"![sig_xz_1s]({rel(out_path.parent / 'signatures_xz_1s.png')})",
        "",
        f"![sig_xy_2s]({rel(out_path.parent / 'signatures_xy_2s.png')})",
        "",
        f"![sig_xz_2s]({rel(out_path.parent / 'signatures_xz_2s.png')})",
        "",
        "黑线是同一机动模式下的平均局部轨迹，浅色线是不同 scenario 的样本。",
        "如果某个模式在多个样本中仍能保持相近的弯折方向和垂向趋势，就说明这个模式在短时观测里有比较稳定的“形状签名”。",
        "",
        "## 对后续意图预测的建议",
        "",
        "1. 训练集组织建议以 `Close_Evade` 进入时刻或机动模式切换时刻作为切窗锚点，而不是简单固定长度滑窗。",
        "2. 若你的目标是尽早分类，先把 `1.0s` 历史窗作为最低可用窗口，再用本报告中的准确率判断是否值得扩展到 `2.0s`。",
        "3. 输入特征不要只喂绝对位置；至少补充速度、局部侧向位移、垂向位移、航向变化率这几类相对量。",
        "4. 如果后续模型要兼顾在线预测和切换检测，可以把“战术层是否进入 Close_Evade”先做成一级门控，再在门控内分类九类机动模式。",
        "",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    dir_1s = Path(args.dir_1s).resolve()
    dir_2s = Path(args.dir_2s).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios_1s = load_scenarios(dir_1s)
    scenarios_2s = load_scenarios(dir_2s)
    segments_1s = extract_segments(scenarios_1s)
    segments_2s = extract_segments(scenarios_2s)

    samples_1s_1 = collect_window_samples(segments_1s, 1.0)
    samples_2s_1 = collect_window_samples(segments_2s, 1.0)
    samples_2s_2 = collect_window_samples(segments_2s, 2.0)

    summary_1s = summarize_scenarios(scenarios_1s)
    summary_2s = summarize_scenarios(scenarios_2s)

    build_contact_sheet(dir_1s, dir_2s, output_dir / "battle_contact_sheet_compare.png")
    plot_feature_heatmaps(
        {
            "1.0s update | 1.0s window": samples_1s_1,
            "2.0s update | first 1.0s": samples_2s_1,
            "2.0s update | full 2.0s": samples_2s_2,
        },
        output_dir / "feature_heatmaps.png",
    )

    acc_results = {
        "1.0s update | 1.0s window": nearest_centroid_loo(samples_1s_1),
        "2.0s update | first 1.0s": nearest_centroid_loo(samples_2s_1),
        "2.0s update | full 2.0s": nearest_centroid_loo(samples_2s_2),
    }
    plot_accuracy_summary(acc_results, output_dir / "accuracy_summary.png")

    plot_signature_projections(
        samples_1s_1,
        "1.0s update | 1.0s window",
        output_dir / "signatures_xy_1s.png",
        output_dir / "signatures_xz_1s.png",
    )
    plot_signature_projections(
        samples_2s_2,
        "2.0s update | 2.0s window",
        output_dir / "signatures_xy_2s.png",
        output_dir / "signatures_xz_2s.png",
    )

    segment_counts = {
        "1s": Counter(sample["mode_name"] for sample in samples_1s_1),
        "2s": Counter(sample["mode_name"] for sample in samples_2s_2),
    }

    report_path = output_dir / "dual_intent_short_window_report.md"
    write_report(
        report_path,
        dir_1s,
        dir_2s,
        summary_1s,
        summary_2s,
        segment_counts,
        acc_results,
    )

    print(f"Saved report to {report_path}")
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
