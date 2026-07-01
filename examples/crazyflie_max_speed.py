import argparse
import csv
import math
import os
from typing import Dict, List

import torch
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.aero import as_body_linear_drag_coeff


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Measure Crazyflie's speed in a max-thrust, constant-altitude dash."
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument("--device", default="cuda:0", help="Torch/physics device, e.g. cuda:0 or cpu.")
    parser.add_argument("--duration", type=float, default=8.0, help="Total simulation time in seconds.")
    parser.add_argument("--warmup", type=float, default=1.5, help="Hover stabilization time before the dash.")
    parser.add_argument("--altitude", type=float, default=1.5, help="Target altitude in meters.")
    parser.add_argument(
        "--pitch-limit-deg",
        type=float,
        default=80.0,
        help="Clamp the commanded nose-down pitch during the dash.",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=0.5,
        help="Telemetry print interval in seconds during the dash.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional CSV output path for per-step telemetry.",
    )
    return parser.parse_args()


def _build_launch_cfg(device: str, headless: bool):
    sim_cfg = OmegaConf.load(os.path.join(CONFIG_PATH, "base", "sim_base.yaml"))
    cfg = OmegaConf.create({"headless": headless, "sim": sim_cfg.sim})
    cfg.sim.device = device
    if str(device).startswith("cpu"):
        cfg.sim.use_gpu = False
        cfg.sim.use_gpu_pipeline = False
        cfg.sim.active_gpu = None
        cfg.sim.physics_gpu = None
    return cfg


def _compute_theory(uav_params: Dict) -> Dict[str, float]:
    mass = float(uav_params["mass"])
    max_thrust_ratio = float(uav_params["max_thrust_ratio"])
    force_constant = float(uav_params["rotor_configuration"]["force_constants"][0])
    max_rot_vel = float(uav_params["rotor_configuration"]["max_rotation_velocities"][0])

    thrust_per_rotor_full = force_constant * max_rot_vel**2
    thrust_total_full = 4.0 * thrust_per_rotor_full
    thrust_total_limited = thrust_total_full * max_thrust_ratio
    weight = mass * 9.81
    hover_ratio = weight / thrust_total_full

    if thrust_total_limited <= weight:
        raise RuntimeError("The configured max thrust ratio is not enough to maintain altitude.")

    level_pitch_rad = math.acos(weight / thrust_total_limited)
    horizontal_acc = math.sqrt(thrust_total_limited**2 - weight**2) / mass
    drag_coeff_body = as_body_linear_drag_coeff(uav_params["drag_coef"]).tolist()
    vertical_aero_coeff = float(uav_params.get("vertical_aero_coef", 0.0))
    terminal_speed_x = math.inf
    if drag_coeff_body[0] > 0.0:
        terminal_speed_x = horizontal_acc / drag_coeff_body[0]

    return {
        "mass": mass,
        "weight": weight,
        "hover_ratio": hover_ratio,
        "max_thrust_ratio": max_thrust_ratio,
        "thrust_total_full": thrust_total_full,
        "thrust_total_limited": thrust_total_limited,
        "level_pitch_rad": level_pitch_rad,
        "level_pitch_deg": math.degrees(level_pitch_rad),
        "horizontal_acc": horizontal_acc,
        "drag_coeff_body_x": drag_coeff_body[0],
        "drag_coeff_body_y": drag_coeff_body[1],
        "drag_coeff_body_z": drag_coeff_body[2],
        "vertical_aero_coeff": vertical_aero_coeff,
        "terminal_speed_x": terminal_speed_x,
    }


def _tensor_to_float(x: torch.Tensor) -> float:
    return float(x.reshape(-1)[0].detach().item())


def main():
    args = _parse_args()
    cfg = _build_launch_cfg(args.device, args.headless)
    simulation_app = init_simulation_app(cfg)

    try:
        import omni_drones.utils.scene as scene_utils
        from omni.isaac.core.simulation_context import SimulationContext
        from omni_drones.controllers import PIDRateController
        from omni_drones.robots.drone import MultirotorBase
        from omni_drones.utils.torch import euler_to_quaternion, quaternion_to_euler

        sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=cfg.sim.dt,
            rendering_dt=cfg.sim.dt,
            sim_params=cfg.sim,
            backend="torch",
            device=cfg.sim.device,
        )

        drone_cls = MultirotorBase.REGISTRY["Crazyflie"]
        drone = drone_cls()

        translations = torch.tensor([[0.0, 0.0, args.altitude]], device=sim.device)
        drone.spawn(translations=translations)
        scene_utils.design_scene()

        sim.reset()
        drone.initialize()

        controller = PIDRateController(dt=cfg.sim.dt, g=9.81, uav_params=drone.params).to(sim.device)
        theory = _compute_theory(drone.params)

        print("=== Crazyflie max-thrust level-flight theory ===")
        print(f"mass: {theory['mass']:.6f} kg")
        print(f"weight: {theory['weight']:.6f} N")
        print(f"hover thrust ratio: {theory['hover_ratio']:.6f}")
        print(f"limited max thrust ratio: {theory['max_thrust_ratio']:.6f}")
        print(f"limited total thrust: {theory['thrust_total_limited']:.6f} N")
        print(f"level-flight pitch at max thrust: {theory['level_pitch_deg']:.3f} deg")
        print(f"horizontal acceleration at max thrust: {theory['horizontal_acc']:.6f} m/s^2")
        print(
            "body drag coefficients [x, y, z]: "
            f"[{theory['drag_coeff_body_x']:.4f}, {theory['drag_coeff_body_y']:.4f}, {theory['drag_coeff_body_z']:.4f}] 1/s"
        )
        print(f"world-vertical aero coefficient: {theory['vertical_aero_coeff']:.4f} 1/s")
        if theory["vertical_aero_coeff"] > 0.0:
            print(
                "note: the Crazyflie literature vertical term is modeled separately from body drag, "
                "so high-pitch forward flight no longer turns horizontal speed into a body-z force."
            )
        if math.isfinite(theory["terminal_speed_x"]):
            print(f"predicted forward terminal speed: {theory['terminal_speed_x']:.6f} m/s")
        else:
            print("predicted forward terminal speed: inf (no x-axis drag configured)")

        init_pos = torch.tensor([[0.0, 0.0, args.altitude]], device=sim.device)
        init_rpy = torch.zeros(1, 3, device=sim.device)
        init_rot = euler_to_quaternion(init_rpy)
        init_vel = torch.zeros(1, 6, device=sim.device)

        def reset():
            drone._reset_idx(torch.tensor([0]))
            drone.set_world_poses(init_pos, init_rot)
            drone.set_velocities(init_vel)
            sim._physics_sim_view.flush()

        reset()

        total_steps = int(args.duration / cfg.sim.dt)
        warmup_steps = int(args.warmup / cfg.sim.dt)
        report_every = max(1, int(args.report_interval / cfg.sim.dt))
        reset_pid = torch.ones(1, 1, dtype=torch.bool, device=sim.device)

        hover_ratio = theory["hover_ratio"]
        dash_ratio = theory["max_thrust_ratio"]
        base_pitch_deg = theory["level_pitch_deg"]
        rate_limit_deg_s = 180.0 * float(controller.target_clip)

        max_hspeed = 0.0
        max_abs_altitude_error = 0.0
        peak_row = None
        rows: List[Dict[str, float]] = []

        for step in range(total_steps):
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.render()
                continue

            state = drone.get_state()[..., :13]
            pos = state[..., 0:3]
            quat = state[..., 3:7]
            vel = state[..., 7:10]
            euler = quaternion_to_euler(quat.reshape(-1, 4)).reshape(*quat.shape[:-1], 3)

            roll_deg = torch.rad2deg(euler[..., 0])
            pitch_deg = torch.rad2deg(euler[..., 1])
            yaw_deg = torch.rad2deg(euler[..., 2])
            altitude = pos[..., 2]
            vz = vel[..., 2]

            in_dash = step >= warmup_steps
            thrust_ratio = dash_ratio if in_dash else hover_ratio

            # Keep total thrust saturated during the dash and use pitch to trade
            # vertical thrust for horizontal thrust while holding altitude.
            pitch_target_deg = 0.0
            if in_dash:
                z_error = args.altitude - _tensor_to_float(altitude)
                vz_value = _tensor_to_float(vz)
                pitch_target_deg = -base_pitch_deg + 12.0 * z_error - 4.0 * vz_value
                pitch_target_deg = max(-args.pitch_limit_deg, min(5.0, pitch_target_deg))

            target_rate = torch.zeros(1, 1, 3, device=sim.device)
            target_rate[..., 0] = -6.0 * roll_deg
            target_rate[..., 1] = 6.0 * (pitch_target_deg - pitch_deg)
            target_rate[..., 2] = -2.0 * yaw_deg
            target_rate = target_rate.clamp(-rate_limit_deg_s, rate_limit_deg_s)

            target_thrust = torch.full(
                (1, 1, 1), thrust_ratio * 2**16, device=sim.device, dtype=state.dtype
            )
            cmds, _ = controller(
                state,
                target_rate=target_rate,
                target_thrust=target_thrust,
                reset_pid=reset_pid,
            )
            reset_pid.fill_(False)
            torch.nan_to_num_(cmds, 0.0)
            drone.apply_action(cmds.clamp(-1.0, 1.0))
            sim.step(render=not args.headless)

            state_after = drone.get_state()[..., :13]
            pos_after = state_after[..., 0:3]
            quat_after = state_after[..., 3:7]
            vel_after = state_after[..., 7:10]
            vel_body_after = drone.vel_b[..., :3]
            euler_after = quaternion_to_euler(quat_after.reshape(-1, 4)).reshape(*quat_after.shape[:-1], 3)
            speed = _tensor_to_float(torch.linalg.norm(vel_after, dim=-1))
            hspeed = _tensor_to_float(torch.linalg.norm(vel_after[..., :2], dim=-1))
            z_after = _tensor_to_float(pos_after[..., 2])
            pitch_after_deg = _tensor_to_float(torch.rad2deg(euler_after[..., 1]))

            row = {
                "time": (step + 1) * cfg.sim.dt,
                "phase": 1.0 if in_dash else 0.0,
                "x": _tensor_to_float(pos_after[..., 0]),
                "y": _tensor_to_float(pos_after[..., 1]),
                "z": z_after,
                "vx": _tensor_to_float(vel_after[..., 0]),
                "vy": _tensor_to_float(vel_after[..., 1]),
                "vz": _tensor_to_float(vel_after[..., 2]),
                "body_vx": _tensor_to_float(vel_body_after[..., 0]),
                "body_vy": _tensor_to_float(vel_body_after[..., 1]),
                "body_vz": _tensor_to_float(vel_body_after[..., 2]),
                "speed": speed,
                "horizontal_speed": hspeed,
                "pitch_deg": pitch_after_deg,
                "target_pitch_deg": pitch_target_deg,
                "target_thrust_ratio": thrust_ratio,
            }
            rows.append(row)
            max_abs_altitude_error = max(max_abs_altitude_error, abs(z_after - args.altitude))

            if in_dash and hspeed >= max_hspeed:
                max_hspeed = hspeed
                peak_row = row.copy()

            if in_dash and (step - warmup_steps) % report_every == 0:
                print(
                    f"t={row['time']:.2f}s, z={row['z']:.3f}m, "
                    f"vx={row['vx']:.3f}m/s, vy={row['vy']:.3f}m/s, vz={row['vz']:.3f}m/s, "
                    f"|v|={row['speed']:.3f}m/s, |v_xy|={row['horizontal_speed']:.3f}m/s, "
                    f"pitch={row['pitch_deg']:.2f}deg"
                )

        if args.csv:
            csv_path = os.path.abspath(args.csv)
            csv_dir = os.path.dirname(csv_path)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"saved telemetry to {csv_path}")

        if peak_row is None:
            raise RuntimeError("Dash phase did not run. Increase --duration or reduce --warmup.")

        dash_time = max(args.duration - args.warmup, 0.0)
        print("=== Measured result over finite dash horizon ===")
        print(f"dash duration: {dash_time:.3f} s")
        print(f"peak horizontal speed: {peak_row['horizontal_speed']:.6f} m/s")
        print(f"peak total speed: {peak_row['speed']:.6f} m/s")
        print(
            f"at t={peak_row['time']:.3f}s, position=({peak_row['x']:.3f}, "
            f"{peak_row['y']:.3f}, {peak_row['z']:.3f}) m"
        )
        print(
            f"final altitude error: {rows[-1]['z'] - args.altitude:+.6f} m, "
            f"final horizontal speed: {rows[-1]['horizontal_speed']:.6f} m/s"
        )
        print(
            f"peak pitch: {peak_row['pitch_deg']:.3f} deg, "
            f"peak body velocity: ({peak_row['body_vx']:.3f}, {peak_row['body_vy']:.3f}, "
            f"{peak_row['body_vz']:.3f}) m/s"
        )
        if max_abs_altitude_error > 0.25:
            print(
                "warning: altitude drift exceeded 0.25 m, so this run is not a clean "
                "constant-altitude speed measurement."
            )
        if abs(peak_row["body_vz"]) > abs(peak_row["body_vx"]):
            print(
                "warning: body-z speed dominated at peak speed; treat this as a sign that the "
                "run is no longer measuring a clean forward drag-limited terminal speed."
            )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
