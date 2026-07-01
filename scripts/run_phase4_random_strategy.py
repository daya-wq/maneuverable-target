#!/usr/bin/env python3
"""
Run Phase 4 training/evaluation for the current per-step random close-range FSM.

Outputs are written to a timestamped folder under results/ so old runs are kept.
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from scripts.train_predictor import (  # noqa: E402
    Config,
    PredictorDataset,
    STRATEGY_NAMES,
    IntentConditionedGMMPredictor,
    IntentLoss,
    IntentGMMLoss,
    TargetFSM,
    full_evaluation,
    plot_ablation_mvs,
    plot_confusion_matrix,
    plot_per_intent_metrics,
    plot_traj_vs_gt,
    plot_training_curves,
    simulate_and_plot_battles,
    simulate_episode,
    train_phase,
)


EXPERIMENT_PREFIX = "phase4_closezone_random_per_step"


def build_run_dirs():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{EXPERIMENT_PREFIX}_{timestamp}"
    results_dir = os.path.join(project_root, "results", run_name)
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return run_name, results_dir, checkpoint_dir


def main():
    print("=" * 72)
    print("  Phase 4 rerun: current per-step random close-range FSM")
    print("=" * 72)

    cfg = Config()
    run_name, results_dir, checkpoint_dir = build_run_dirs()
    cfg.results_dir = results_dir
    cfg.checkpoint_dir = checkpoint_dir

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"  Run name: {run_name}")
    print(f"  Results dir: {cfg.results_dir}")

    print(f"\n[Step 1] Simulating {cfg.n_episodes_collect} episodes...")
    fsm = TargetFSM(
        num_envs=1,
        device="cpu",
        max_speed=cfg.max_speed,
        max_accel=cfg.max_accel,
        arena_radius=cfg.arena_radius,
        max_height=cfg.max_height,
    )

    dataset = PredictorDataset(
        t_obs=cfg.t_obs,
        t_pred=cfg.t_pred,
        w_mvs=cfg.w_mvs,
        slide_step=cfg.slide_step,
    )

    strategy_counts = {name: 0 for name in STRATEGY_NAMES}
    ep_durations = []
    strategy_switches = []
    t_start = time.time()

    for ep in range(cfg.n_episodes_collect):
        episode = simulate_episode(cfg, fsm, ep_id=ep)
        if episode is None:
            continue

        dataset.add_episode(episode)
        strategy_ids = episode["strategy_ids"].tolist()
        for sid in strategy_ids:
            if 0 <= sid < len(STRATEGY_NAMES):
                strategy_counts[STRATEGY_NAMES[sid]] += 1
        ep_durations.append(episode["target_pos"].shape[0] * cfg.dt_ctrl)
        strategy_switches.append(
            sum(1 for i in range(1, len(strategy_ids)) if strategy_ids[i] != strategy_ids[i - 1])
        )

        if (ep + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  Episode {ep + 1}/{cfg.n_episodes_collect} ({elapsed:.1f}s), samples: {len(dataset)}")

    stats = dataset.get_stats()
    total_strategy_steps = sum(strategy_counts.values())
    strategy_fractions = {
        name: (count / total_strategy_steps if total_strategy_steps else 0.0)
        for name, count in strategy_counts.items()
    }

    print("\n  Data collected:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Avg episode duration: {np.mean(ep_durations):.2f}s")
    print(f"  Avg strategy switches / episode: {np.mean(strategy_switches):.2f}")
    print(f"  Strategy dist: {strategy_counts}")

    if len(dataset) < 100:
        print("  [ERROR] Too few samples")
        return 1

    n_total = len(dataset)
    n_train = int(n_total * 0.8)

    class WrapDS(PredictorDataset):
        def __init__(self, subset):
            self._s = subset

        def __len__(self):
            return len(self._s)

        def __getitem__(self, idx):
            return self._s[idx]

    indices = list(range(n_total))
    np.random.shuffle(indices)
    train_ds = WrapDS(torch.utils.data.Subset(dataset, indices[:n_train]))
    val_ds = WrapDS(torch.utils.data.Subset(dataset, indices[n_train:]))
    print(f"  Train: {n_train}, Val: {n_total - n_train}")

    print(f"\n[Step 2] Phase 1: Intent classification ({cfg.epochs_phase1} epochs)...")
    model = IntentConditionedGMMPredictor(
        obs_dim=24,
        n_intents=10,
        n_components=3,
        pred_horizon=cfg.t_pred,
        hidden_dim=cfg.hidden_dim,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt1 = optim.Adam(model.parameters(), lr=cfg.lr_phase1)
    h1 = train_phase(model, train_ds, IntentLoss(), opt1, cfg, "Phase1", cfg.epochs_phase1)

    ckpt1 = os.path.join(cfg.checkpoint_dir, "predictor_phase1.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": model.get_config()}, ckpt1)
    print(f"  Saved: {ckpt1}")

    print(f"\n[Step 3] Phase 2: GMM trajectory ({cfg.epochs_phase2} epochs)...")
    opt2 = optim.Adam(model.parameters(), lr=cfg.lr_phase2)
    h2 = train_phase(
        model,
        train_ds,
        IntentGMMLoss(),
        opt2,
        cfg,
        "Phase2",
        cfg.epochs_phase2,
        use_mvs=True,
    )

    ckpt2 = os.path.join(cfg.checkpoint_dir, "predictor_phase2.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": model.get_config()}, ckpt2)
    print(f"  Saved: {ckpt2}")

    print("\n[Step 4] Full evaluation...")
    eval_data = full_evaluation(model, val_ds, cfg)
    overall_acc = np.mean([1 if p == g else 0 for p, g in zip(eval_data["pred_intent"], eval_data["gt_intent"])])
    overall_ade = np.mean(eval_data["ade"])
    overall_fde = np.mean(eval_data["fde"])
    abl = eval_data["ablation"]

    print(f"  Accuracy: {overall_acc:.4f}")
    print(f"  ADE: {overall_ade:.4f}m")
    print(f"  FDE: {overall_fde:.4f}m")
    print("  MVS Ablation:")
    print(
        f"    MVS=Real: acc={np.mean(abl['real']['acc']):.4f}, "
        f"ADE={np.mean(abl['real']['ade']):.4f}, FDE={np.mean(abl['real']['fde']):.4f}"
    )
    print(
        f"    MVS=0:    acc={np.mean(abl['zero']['acc']):.4f}, "
        f"ADE={np.mean(abl['zero']['ade']):.4f}, FDE={np.mean(abl['zero']['fde']):.4f}"
    )

    print("\n[Step 5] Generating plots...")
    plot_confusion_matrix(eval_data, cfg.results_dir)
    plot_per_intent_metrics(eval_data, cfg.results_dir)
    plot_traj_vs_gt(eval_data, cfg.results_dir)
    plot_ablation_mvs(eval_data, cfg.results_dir)
    plot_training_curves(h1, h2, cfg.results_dir)
    print("  Saved: confusion_matrix.png, per_intent_metrics.png, traj_vs_gt.png, ablation_mvs.png, training_curves.png")

    print("\n[Step 6] 10 battle scenarios...")
    battle_summary = simulate_and_plot_battles(cfg, fsm, cfg.results_dir, n_scenarios=10)

    report_path = os.path.join(cfg.results_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Phase 4 Evaluation Report\n")
        f.write("=" * 72 + "\n\n")
        f.write("Experiment:\n")
        f.write(f"  Name = {run_name}\n")
        f.write("  FSM = distance-driven, close zone randomly samples {Spiral_Climb, Sharp_Turn, Dive_Rush} every step\n")
        f.write("  Seed = 42\n\n")
        f.write("Configuration:\n")
        f.write(f"  T_obs = {cfg.t_obs} frames ({cfg.t_obs * cfg.dt_ctrl}s)\n")
        f.write(f"  W_mvs = {cfg.w_mvs} frames ({cfg.w_mvs * cfg.dt_ctrl}s)\n")
        f.write(f"  T_pred = {cfg.t_pred} frames ({cfg.t_pred * cfg.dt_ctrl}s)\n")
        f.write(f"  Episodes = {cfg.n_episodes_collect}\n")
        f.write(f"  Arena radius = {cfg.arena_radius}m\n")
        f.write(f"  Train/Val = {n_train}/{n_total - n_train}\n\n")
        f.write("Dataset Summary:\n")
        f.write(f"  Total samples = {stats['total_samples']}\n")
        f.write(f"  Avg episode duration = {np.mean(ep_durations):.2f}s\n")
        f.write(f"  Avg strategy switches / episode = {np.mean(strategy_switches):.2f}\n")
        f.write(f"  Strategy counts = {strategy_counts}\n")
        f.write(f"  Strategy fractions = {strategy_fractions}\n")
        f.write(f"  Intent distribution = {stats['intent_distribution']}\n\n")
        f.write("Overall Metrics:\n")
        f.write(f"  Intent Accuracy: {overall_acc:.4f}\n")
        f.write(f"  ADE: {overall_ade:.4f}m\n")
        f.write(f"  FDE: {overall_fde:.4f}m\n\n")
        f.write("MVS Ablation:\n")
        f.write(
            f"  MVS=Real: acc={np.mean(abl['real']['acc']):.4f}, "
            f"ADE={np.mean(abl['real']['ade']):.4f}, FDE={np.mean(abl['real']['fde']):.4f}\n"
        )
        f.write(
            f"  MVS=0:    acc={np.mean(abl['zero']['acc']):.4f}, "
            f"ADE={np.mean(abl['zero']['ade']):.4f}, FDE={np.mean(abl['zero']['fde']):.4f}\n\n"
        )
        f.write("Battle Scenarios:\n")
        n_cap = sum(1 for b in battle_summary if b["captured"])
        f.write(f"  Capture rate: {n_cap}/{len(battle_summary)}\n")
        for b in battle_summary:
            status = "CAPTURED" if b["captured"] else "ESCAPED"
            f.write(
                f"  Scene {b['id']}: {status} | {b['duration']:.1f}s | "
                f"d0={b['d0']:.1f}m | min_d={b['min_d']:.2f}m | switches={b['switches']}\n"
            )

    print(f"\n[Done] Report saved to: {report_path}")
    print(f"[Done] Results folder: {cfg.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
