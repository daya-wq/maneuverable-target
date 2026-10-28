"""Explicit physical assumptions for the 70%-thrust experimental aircraft."""
import math
import torch
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse, quat_axis


def configure_aircraft(robot, thrust_to_weight):
    """Scale only this robot instance; retain the source asset and mass."""
    if thrust_to_weight is None:
        return
    rotor = robot.params['rotor_configuration']
    old_thrust = sum(k * w * w for k, w in zip(
        rotor['force_constants'], rotor['max_rotation_velocities']))
    scale = float(thrust_to_weight) * float(robot.mass.item()) * 9.81 / old_thrust
    for key in ('force_constants', 'moment_constants'):
        rotor[key] = [float(x) * scale for x in rotor[key]]
    # Multirotor.initialize uses these tunable tensors to rebuild rotor params.
    robot.force_constants.mul_(scale)
    robot.moment_constants.mul_(scale)


def acceleration_budget(hover_ratio, fraction, reserve=0.9):
    thrust_accel = 9.81 * float(fraction) / float(hover_ratio)
    if thrust_accel <= 9.81:
        raise ValueError('Configured thrust budget cannot sustain hover')
    return {
        'horizontal': reserve * math.sqrt(thrust_accel ** 2 - 9.81 ** 2),
        'vertical': reserve * (thrust_accel - 9.81),
        'thrust_accel': thrust_accel,
    }


def bounded_motor_commands(cmds, fraction):
    """Motor action is linear in thrust fraction: cmd=2*f-1."""
    return cmds.clamp(-1.0, 2.0 * float(fraction) - 1.0)


def segment_min_distance(start_relative, end_relative):
    delta = end_relative - start_relative
    alpha = -(start_relative * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-12)
    closest = start_relative + alpha.clamp(0, 1).unsqueeze(-1) * delta
    return closest.norm(dim=-1)


def body_axis_thrust_ratio(desired_specific_force, quaternion, hover_ratio):
    """SO3 scalar thrust: project desired force onto the current rotor axis.

    Using the norm assumes instantaneous attitude tracking. With stochastic
    lateral commands it produces a positive vertical bias before the vehicle
    can tilt. This is the same projection used by LeePositionController.
    """
    current_axis = quat_axis(quaternion, axis=2)
    projection = (desired_specific_force * current_axis).sum(-1, keepdim=True)
    return float(hover_ratio) * projection.clamp_min(0.0) / 9.81


def floor_braking_prediction(height, vertical_speed, upward_acceleration, reaction_time, reserve=0.65):
    """Predict the lowest height after reaction delay and a conservative brake.

    A fixed lookahead alone misses the v^2/(2a) stopping distance at high speed.
    This changes acceleration requests, never teleports or overwrites velocity.
    """
    down_speed = (-vertical_speed).clamp_min(0.0)
    braking_accel = max(float(upward_acceleration) * float(reserve), 1e-6)
    return height + vertical_speed * float(reaction_time) - down_speed.square() / (2.0 * braking_accel)


def specific_force(acceleration, quaternion, velocity, robot, fraction, max_tilt_deg):
    """Gravity/known-drag compensation followed by a physical thrust cone."""
    body_velocity = quat_rotate_inverse(quaternion, velocity)
    drag = quat_rotate(quaternion, body_velocity * robot.drag_coef)
    drag = drag + float(robot.cfg.rigid_props.linear_damping or 0.0) * velocity
    force = acceleration + drag
    force = force.clone()
    force[..., 2] += 9.81
    force[..., 2].clamp_(min=0.1)
    xy = force[..., :2]
    max_xy = force[..., 2:3] * math.tan(math.radians(max_tilt_deg))
    force[..., :2] = xy * (max_xy / xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
    max_specific_force = float(fraction) * robot.KF.sum(-1, keepdim=True) / (robot.gravity / 9.81)
    force = force * (max_specific_force / force.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
    return force
