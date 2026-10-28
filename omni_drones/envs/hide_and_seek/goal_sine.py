"""Distance-triggered, goal-directed sine reference (no Isaac dependencies).

Amplitude is a horizontal displacement reference in metres, not an acceleration
or a promise that the physical aircraft will realise that displacement exactly.
"""
import math

import torch


class GoalSinePolicy:
    def __init__(self, num_envs, device, cfg):
        self.trigger_distance = float(cfg.get("trigger_distance", 25.0))
        self.position_gain = float(cfg.get("lateral_position_gain", 1.5))
        self.amplitude_range = self._range(cfg, "amplitude_range", (1.0, 2.0), 0.0)
        self.period_range = self._range(cfg, "period_range", (3.0, 5.0), 0.0, strict=True)
        if not math.isfinite(self.trigger_distance) or self.trigger_distance <= 0.0:
            raise ValueError("target_sine.trigger_distance must be finite and positive")
        if not math.isfinite(self.position_gain) or self.position_gain < 0.0:
            raise ValueError("target_sine.lateral_position_gain must be finite and nonnegative")
        self.amplitude = torch.zeros(num_envs, device=device)
        self.period = torch.ones(num_envs, device=device)
        self.phase = torch.zeros(num_envs, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.origin = torch.zeros(num_envs, 3, device=device)
        self.lateral = torch.zeros_like(self.origin)
        self.reference_offset = torch.zeros(num_envs, device=device)

    @staticmethod
    def _range(cfg, key, default, minimum, strict=False):
        values = tuple(float(v) for v in cfg.get(key, default))
        if (len(values) != 2 or not all(math.isfinite(v) for v in values)
                or values[0] > values[1] or values[0] < minimum
                or (strict and values[0] == minimum)):
            raise ValueError("target_sine.%s has invalid [min, max]: %r" % (key, values))
        return values

    def reset(self, env_ids):
        """Resample only reset episodes; other parallel environments are untouched."""
        for buffer, limits in ((self.amplitude, self.amplitude_range),
                               (self.period, self.period_range)):
            values = torch.empty(env_ids.numel(), device=buffer.device)
            if limits[0] == limits[1]:
                values.fill_(limits[0])
            else:
                values.uniform_(*limits)
            buffer[env_ids] = values
        self.phase[env_ids] = 0.0
        self.active[env_ids] = False
        self.origin[env_ids] = 0.0
        self.lateral[env_ids] = 0.0
        self.reference_offset[env_ids] = 0.0

    def desired_velocity(self, position, goal_position, nearest_distance, max_speed, dt):
        """Advance once per physics step; all position tensors have shape [B, 3]."""
        if not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError("Sine policy dt must be finite and positive")
        near = nearest_distance <= self.trigger_distance
        entered = near & ~self.active
        goal_delta = goal_position - position
        distance = goal_delta.norm(dim=-1, keepdim=True)
        direction = goal_delta / distance.clamp_min(1e-6)

        # Freeze a horizontal cross-track axis for each uninterrupted near phase.
        # Rebuilding this axis at every step would rotate the sine reference.
        forward_xy = direction[:, :2]
        norm_xy = forward_xy.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(forward_xy)
        fallback[:, 0] = 1.0
        forward_xy = torch.where(norm_xy > 1e-6, forward_xy / norm_xy.clamp_min(1e-6), fallback)
        lateral = torch.stack((-forward_xy[:, 1], forward_xy[:, 0],
                               torch.zeros_like(forward_xy[:, 0])), dim=-1)
        self.origin[entered] = position[entered]
        self.lateral[entered] = lateral[entered]
        omega = 2.0 * math.pi / self.period
        self.phase.copy_(torch.where(
            near & ~entered, torch.remainder(self.phase + omega * dt, 2.0 * math.pi),
            torch.zeros_like(self.phase)))
        self.active.copy_(near)
        offset = self.amplitude * self.phase.sin()
        self.reference_offset.copy_(torch.where(near, offset, torch.zeros_like(offset)))

        goal_velocity = direction * max_speed.reshape(-1, 1)
        lateral_position = ((position - self.origin) * self.lateral).sum(-1)
        lateral_velocity = (self.amplitude * omega * self.phase.cos()
                            + self.position_gain * (offset - lateral_position))
        forward_direction = goal_velocity - (
            goal_velocity * self.lateral).sum(-1, keepdim=True) * self.lateral
        limit = max_speed.reshape(-1)
        lateral_velocity = torch.minimum(torch.maximum(lateral_velocity, -limit), limit)
        forward_budget = (limit.square() - lateral_velocity.square()).clamp_min(0.0).sqrt()
        forward_velocity = (forward_direction / forward_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                            * forward_budget.unsqueeze(-1))
        sine_velocity = forward_velocity + lateral_velocity.unsqueeze(-1) * self.lateral
        velocity = torch.where(near.unsqueeze(-1), sine_velocity, goal_velocity)
        # At the goal the episode terminates; avoid a residual sine request there.
        velocity = torch.where(distance > 1e-6, velocity, torch.zeros_like(velocity))
        speed = velocity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return velocity * (max_speed.reshape(-1, 1) / speed).clamp(max=1.0)
