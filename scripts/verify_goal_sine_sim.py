"""Finite target-only closed-loop check; no Actor optimisation or capture claims.

Uses the physical UAV controller at 100 Hz. The trigger is forced near/far in
memory to exercise both branches; saved task configurations are not changed.
"""
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_pv")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    torch.manual_seed(20261028)
    app = init_simulation_app(cfg)
    try:
        from omni_drones.envs.hide_and_seek.hideandseek import HideAndSeek
        env = HideAndSeek(cfg, headless=True)
        env.reset()
        sine = env.target_sine_policy
        assert sine is not None
        dp, _ = env.get_env_poses(env.drone.get_world_poses())
        tp, _ = env.get_env_poses(env.target.get_world_poses())
        nearest = (dp-tp).norm(dim=-1).min(-1).values
        expected = nearest <= sine.trigger_distance
        env._compute_target_desired_acc()
        assert torch.equal(sine.active, expected)

        # Actual partial reset must not alter another environment's oscillator.
        sine.trigger_distance = 200.0
        for _ in range(3):
            env._compute_target_desired_acc()
        snapshot = {key: getattr(sine, key).clone() for key in (
            "amplitude", "period", "phase", "active", "origin", "lateral")}
        env._reset_idx(torch.tensor([0], device=env.device))
        for key, value in snapshot.items():
            assert torch.equal(getattr(sine, key)[1:], value[1:]), key
        assert not bool(sine.active[0]) and float(sine.phase[0]) == 0
        env.reset()

        records = {key: [] for key in ("position", "velocity", "reference_offset",
                                       "phase", "active", "acceleration_command", "motor_fraction")}
        for step in range(480):
            sine.trigger_distance = 200.0 if step < 360 else 1e-6
            env._step_uav_target()
            env.sim.step(False)
            position, _ = env.get_env_poses(env.target.get_world_poses())
            velocity = env.target.get_velocities()[..., :3]
            assert bool(sine.active.all()) if step < 360 else not bool(sine.active.any())
            assert torch.isfinite(position).all() and torch.isfinite(velocity).all()
            assert (env.target_vel_cmd.norm(dim=-1, keepdim=True) <= env.target_max_speed+1e-4).all()
            assert (env.target_acc_cmd.norm(dim=-1, keepdim=True) <= env.target_max_accel+1e-4).all()
            values = dict(position=position[:, 0], velocity=velocity[:, 0],
                          reference_offset=sine.reference_offset, phase=sine.phase,
                          active=sine.active, acceleration_command=env.target_acc_cmd[:, 0],
                          motor_fraction=env.target.throttle.square().max(-1).values[:, 0])
            for key, value in values.items():
                records[key].append(value.detach().cpu().numpy())
        records = {key: np.stack(value) for key, value in records.items()}
        output = Path(__file__).resolve().parents[1] / "results/target_sine_20261028"
        np.savez_compressed(output / "target_physics.npz", **records)
        summary = dict(
            physics_steps=480, duration_s=4.8, dt_s=float(env.dt),
            near_duration_s=3.6, far_duration_s=1.2,
            partial_reset_isolation=True, trigger_matches_real_distances=True,
            amplitude_m=sine.amplitude.cpu().tolist(), period_s=sine.period.cpu().tolist(),
            actual_speed_max_mps=float(np.linalg.norm(records["velocity"], axis=-1).max()),
            acceleration_command_max_mps2=float(np.linalg.norm(records["acceleration_command"], axis=-1).max()),
            actual_motor_fraction_max=float(records["motor_fraction"].max()),
            note="Target-only physics; near/far forced in memory; pursuers not controlled; not capture evaluation")
        (output / "target_physics_summary.json").write_text(json.dumps(summary, indent=2))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(480) * float(env.dt)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for index in range(env.num_envs):
            xy = records["position"][:, index, :2]
            axes[0].plot(xy[:, 0], xy[:, 1], label="target %d" % index)
        axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Physical target trajectories")
        axes[0].legend()
        axes[1].plot(t, records["reference_offset"][:, 0], label="lateral reference, env 0")
        axes[1].axvline(3.6, color="gray", linestyle="--", label="switch to goal-only")
        axes[1].set(xlabel="time (s)", ylabel="reference offset (m)", title="Reference, not actual displacement")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output / "target_physics.png", dpi=160)
        plt.close(fig)
        print("GOAL_SINE_PHYSICS_OK " + json.dumps(summary), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
