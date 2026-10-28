import math
import os
import sys
import importlib.util

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AERO_PATH = os.path.join(REPO_ROOT, "omni_drones", "utils", "aero.py")
spec = importlib.util.spec_from_file_location("omni_drones.utils.aero", AERO_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load aero helpers from {AERO_PATH}")
aero = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aero)
as_body_linear_drag_coeff = aero.as_body_linear_drag_coeff
as_world_vertical_aero_coeff = aero.as_world_vertical_aero_coeff
compute_body_linear_drag_force = aero.compute_body_linear_drag_force
compute_world_vertical_aero_force = aero.compute_world_vertical_aero_force

CRAZYFLIE_YAML = os.path.join(
    REPO_ROOT,
    "omni_drones",
    "robots",
    "assets",
    "usd",
    "crazyflie.yaml",
)


def main():
    with open(CRAZYFLIE_YAML, "r") as f:
        cfg = yaml.safe_load(f)

    mass = float(cfg["mass"])
    drag_coeff = as_body_linear_drag_coeff(cfg["drag_coef"], dtype=torch.float64)
    vertical_aero_coeff = as_world_vertical_aero_coeff(
        cfg.get("vertical_aero_coef", 0.0),
        dtype=torch.float64,
    )

    vel_body = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=torch.float64,
    )
    drag_force = compute_body_linear_drag_force(
        vel_body,
        drag_coeff,
        torch.tensor([[mass]], dtype=torch.float64),
    )

    assert drag_force[0, 0] < 0.0, "Positive forward speed must produce backward drag."
    assert drag_force[1, 0] > 0.0, "Negative forward speed must produce forward drag."
    assert drag_force[2, 1] < 0.0, "Positive lateral speed must produce opposite drag."
    assert drag_force[3, 2] < 0.0, "Positive body-z speed must produce opposing body-z drag."

    vel_world = torch.tensor(
        [
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -3.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    vertical_force = compute_world_vertical_aero_force(
        vel_world,
        vertical_aero_coeff,
        torch.tensor([[mass]], dtype=torch.float64),
    )
    assert abs(float(vertical_force[0, 2])) < 1e-12, "Vertical aero correction is disabled in the default config."
    assert abs(float(vertical_force[1, 2])) < 1e-12, "Vertical aero correction is disabled in the default config."
    assert abs(float(vertical_force[2, 2])) < 1e-12, "Pure horizontal motion must not trigger vertical aero force."

    force_constant = float(cfg["rotor_configuration"]["force_constants"][0])
    max_rot_vel = float(cfg["rotor_configuration"]["max_rotation_velocities"][0])
    max_thrust_ratio = float(cfg["max_thrust_ratio"])

    thrust_total = 4.0 * force_constant * max_rot_vel**2 * max_thrust_ratio
    weight = mass * 9.81
    horizontal_force = math.sqrt(thrust_total**2 - weight**2)
    horizontal_acc = horizontal_force / mass

    kx = float(drag_coeff[0].item())
    predicted_terminal_speed = horizontal_acc / kx

    dt = 0.01
    duration = 20.0
    steps = int(duration / dt)
    v = 0.0
    for _ in range(steps):
        v += (horizontal_acc - kx * v) * dt

    closed_form = predicted_terminal_speed * (1.0 - math.exp(-kx * duration))
    assert abs(v - closed_form) < 1e-3, "Discrete simulation should match the first-order drag model."

    print("Crazyflie drag model check passed.")
    print(f"mass: {mass:.6f} kg")
    print(
        "body drag coeff [x, y, z]: "
        f"[{float(drag_coeff[0]):.4f}, {float(drag_coeff[1]):.4f}, {float(drag_coeff[2]):.4f}] 1/s"
    )
    print(f"world-vertical aero coeff: {float(vertical_aero_coeff):.4f} 1/s")
    print(f"max-thrust level-flight horizontal acceleration: {horizontal_acc:.6f} m/s^2")
    print(f"predicted forward terminal speed: {predicted_terminal_speed:.6f} m/s")
    print(f"simulated speed at {duration:.1f}s: {v:.6f} m/s")


if __name__ == "__main__":
    main()
