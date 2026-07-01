"""
Shared maneuver-intent labels based on total acceleration in the velocity frame.
"""

from __future__ import annotations

import torch


INTENT_STRAIGHT = 0
INTENT_LEFT_TURN = 1
INTENT_RIGHT_TURN = 2
INTENT_CLIMB = 3
INTENT_DIVE = 4
INTENT_LEFT_CLIMB = 5
INTENT_LEFT_DIVE = 6
INTENT_RIGHT_CLIMB = 7
INTENT_RIGHT_DIVE = 8
INTENT_HOVER = 9

INTENT_LABELS = [
    "Straight",
    "Left-Turn",
    "Right-Turn",
    "Climb",
    "Dive",
    "Left-Climb",
    "Left-Dive",
    "Right-Climb",
    "Right-Dive",
    "Hover",
]


def label_intent_single(
    accel_cmd: torch.Tensor,
    velocity: torch.Tensor,
    speed_threshold: float = 0.3,
    accel_threshold: float = 0.5,
) -> int:
    speed = torch.norm(velocity).item()
    if speed < speed_threshold:
        return INTENT_HOVER

    forward = velocity / speed
    z_up = torch.tensor([0.0, 0.0, 1.0], device=velocity.device, dtype=velocity.dtype)

    lateral = torch.cross(z_up, forward, dim=0)
    lat_norm = torch.norm(lateral)
    if lat_norm < 1e-6:
        lateral = torch.tensor([1.0, 0.0, 0.0], device=velocity.device, dtype=velocity.dtype)
    else:
        lateral = lateral / lat_norm

    normal = torch.cross(forward, lateral, dim=0)

    a_lat = torch.dot(accel_cmd, lateral).item()
    a_lon = torch.dot(accel_cmd, normal).item()

    left = a_lat > accel_threshold
    right = a_lat < -accel_threshold
    climb = a_lon > accel_threshold
    dive = a_lon < -accel_threshold

    if left and climb:
        return INTENT_LEFT_CLIMB
    if left and dive:
        return INTENT_LEFT_DIVE
    if right and climb:
        return INTENT_RIGHT_CLIMB
    if right and dive:
        return INTENT_RIGHT_DIVE
    if left:
        return INTENT_LEFT_TURN
    if right:
        return INTENT_RIGHT_TURN
    if climb:
        return INTENT_CLIMB
    if dive:
        return INTENT_DIVE
    return INTENT_STRAIGHT


def label_intent_batch(
    accel_cmds: torch.Tensor,
    velocities: torch.Tensor,
    speed_threshold: float = 0.3,
    accel_threshold: float = 0.5,
) -> torch.Tensor:
    labels = torch.zeros(accel_cmds.shape[0], dtype=torch.long, device=accel_cmds.device)

    speed = torch.norm(velocities, dim=-1)
    is_hover = speed < speed_threshold
    labels[is_hover] = INTENT_HOVER

    mask = ~is_hover
    if not mask.any():
        return labels

    vel = velocities[mask]
    acc = accel_cmds[mask]
    spd = speed[mask].unsqueeze(-1)

    forward = vel / spd
    z_up = torch.tensor([0.0, 0.0, 1.0], device=vel.device, dtype=vel.dtype).expand_as(forward)

    lateral = torch.cross(z_up, forward, dim=-1)
    lat_norm = torch.norm(lateral, dim=-1, keepdim=True).clamp(min=1e-6)
    lateral = lateral / lat_norm

    normal = torch.cross(forward, lateral, dim=-1)

    a_lat = (acc * lateral).sum(dim=-1)
    a_lon = (acc * normal).sum(dim=-1)

    left = a_lat > accel_threshold
    right = a_lat < -accel_threshold
    climb = a_lon > accel_threshold
    dive = a_lon < -accel_threshold

    result = torch.zeros(mask.sum(), dtype=torch.long, device=vel.device)
    result[left & climb] = INTENT_LEFT_CLIMB
    result[left & dive] = INTENT_LEFT_DIVE
    result[right & climb] = INTENT_RIGHT_CLIMB
    result[right & dive] = INTENT_RIGHT_DIVE
    result[left & ~climb & ~dive] = INTENT_LEFT_TURN
    result[right & ~climb & ~dive] = INTENT_RIGHT_TURN
    result[climb & ~left & ~right] = INTENT_CLIMB
    result[dive & ~left & ~right] = INTENT_DIVE

    labels[mask] = result
    return labels
