"""Read-only progress audit; episode fractions are weighted by completed episodes."""
import argparse
import json
import math
from pathlib import Path


def summarize(rows):
    episodes = sum(r.get("train/episodes_completed", 0) for r in rows)
    keys = set().union(*(r.keys() for r in rows))
    terms = sorted(k for k in keys if k.startswith("train/stats.termination_"))
    metrics = [
        "drone/approx_kl", "drone/ratio_mean", "drone/ratio_std",
        "drone/ratio_clip_fraction", "drone/policy_loss", "drone/value_loss",
        "drone/value_abs_error", "drone/explained_var", "drone/advantages_mean",
        "drone/advantages_std", "drone/predictor_minADE", "drone/predictor_FDE",
        "drone/predictor_loss_total", "drone/predictor_strategic_accuracy",
    ]
    invalid = [(r.get("train/update"), k) for r in rows for k, v in r.items()
               if isinstance(v, (float, int)) and not math.isfinite(v)]
    weighted = lambda k: sum(r.get(k, 0) * r.get("train/episodes_completed", 0)
                             for r in rows) / max(episodes, 1)
    return {
        "first_update": rows[0]["train/update"],
        "last_update": rows[-1]["train/update"],
        "last_env_frames": rows[-1]["env_frames"],
        "updates": len(rows), "completed_episodes": episodes,
        "termination_fraction": {k[len("train/stats.termination_"):]: weighted(k)
                                 for k in terms},
        "episode_duration_s": weighted("train/stats.episode_duration_s"),
        "episode_return": weighted("train/stats.return"),
        "reward_means": {k[len("train/stats.m3_"):]: weighted(k)
                         for k in sorted(keys) if k.startswith("train/stats.m3_")},
        "metric_means": {k: sum(r[k] for r in rows) / len(rows) for k in metrics},
        "actor_skips": sum(r.get("drone/actor_update_skipped", 0) for r in rows),
        "nonfinite_fields": invalid,
        "missing_termination_updates": [r['train/update'] for r in rows
                                        if r.get('train/episodes_completed', 0) > 0
                                        and any(k not in r for k in terms)],
        "max_actual_thrust": max(max(r.get("safety/pursuer_motor_thrust_fraction_max", 0),
                                      r.get("safety/target_motor_thrust_fraction_max", 0))
                                 for r in rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("progress", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    args = parser.parse_args()
    rows = []
    for line in args.progress.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A writer may still be completing the final line.
    if not rows:
        raise SystemExit("No complete progress records")
    if args.plot_dir:
        from overnight_runner import plot_curves
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        plot_curves(rows, args.plot_dir)
    print(json.dumps({"all": summarize(rows), "first_250": summarize(rows[:250]),
                      "recent_50": summarize(rows[-50:])}, indent=2))
