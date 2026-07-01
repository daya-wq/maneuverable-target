from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


STRATEGIC_FAR_RUSH = 0
STRATEGIC_CLOSE_EVADE = 1
STRATEGIC_GOAL_PRIORITY = 2
STRATEGIC_GOAL_SPRINT = 3


def safe_normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def clip_vector_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.norm(v, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    return v * scale


@dataclass
class OTGuidanceConfig:
    sinkhorn_iters: int = 24
    sinkhorn_epsilon: float = 2.0
    block_distance: float = 0.45
    goal_distance: float = 0.65
    evade_forward: float = 0.35
    evade_side: float = 0.55
    kp: float = 5.0
    kv: float = 2.0
    base_accel_limit: float = 3.0
    mode_cover_scale: float = 0.8
    line_cover_scale: float = 0.45
    variance_weight_scale: float = 1.0
    variance_weight_floor: float = 0.10
    min_ref_z: float = 0.45
    hard_floor_z: float = 0.30
    low_target_ref_margin: float = 0.10
    traj_mass_base: float = 0.35
    traj_mass_far: float = 0.15
    traj_mass_close: float = 0.45
    channel_forward_fraction: float = 0.78
    channel_min_forward: float = 2.20
    channel_lateral_spread: float = 0.65
    channel_mass_base: float = 0.60
    channel_mass_far: float = 2.20
    channel_mass_goal_priority: float = 2.00
    channel_mass_goal_sprint: float = 2.80
    ref_forward_floor: float = 1.80
    ref_lateral_spread: float = 0.65
    ref_channel_blend: float = 0.65
    ref_lateral_blend: float = 0.85
    ref_min_separation: float = 0.75
    ref_separation_gain: float = 0.60
    ref_separation_iters: int = 2
    terminal_forward_min: float = 0.12
    dt: float = 0.02


class OTGuidedInterceptor:
    """Discrete Sinkhorn OT guidance for three-pursuer interception layouts."""

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        device: torch.device | str,
        cfg: OTGuidanceConfig,
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.device = torch.device(device)
        self.cfg = cfg
        self.prev_barycenters = torch.zeros(
            self.num_envs, self.num_agents, 3, device=self.device
        )
        self.prev_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.prev_barycenters[env_ids] = 0.0
        self.prev_valid[env_ids] = False

    def compute(
        self,
        pursuer_pos: torch.Tensor,
        pursuer_vel: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        goal_pos: torch.Tensor,
        strategic_probs: torch.Tensor,
        gmm_weights: torch.Tensor,
        gmm_means_world: torch.Tensor,
        gmm_vars: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if target_pos.dim() == 3:
            target_pos = target_pos.squeeze(1)
        if target_vel.dim() == 3:
            target_vel = target_vel.squeeze(1)
        if goal_pos.dim() == 1:
            goal_pos = goal_pos.view(1, 3).expand_as(target_pos)
        if env_ids is None:
            env_ids = torch.arange(
                pursuer_pos.shape[0], device=pursuer_pos.device, dtype=torch.long
            )

        particles, weights = self._build_particles(
            target_pos,
            target_vel,
            goal_pos,
            strategic_probs,
            gmm_weights,
            gmm_means_world,
            gmm_vars,
        )
        transport, cost = self._sinkhorn(pursuer_pos, particles, weights)
        row_mass = transport.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        barycenters = torch.sum(transport.unsqueeze(-1) * particles.unsqueeze(1), dim=2) / row_mass
        barycenters = self._shape_channel_barycenters(
            barycenters, target_pos, goal_pos, strategic_probs
        )
        barycenters = self._separate_barycenters(barycenters, target_pos, goal_pos)

        valid = self.prev_valid[env_ids].view(-1, 1, 1)
        prev_barycenters = self.prev_barycenters[env_ids]
        b_dot = torch.where(
            valid,
            (barycenters - prev_barycenters) / max(float(self.cfg.dt), 1e-6),
            torch.zeros_like(barycenters),
        )
        self.prev_barycenters[env_ids] = barycenters.detach()
        self.prev_valid[env_ids] = True

        a_ot = (
            self.cfg.kp * (barycenters - pursuer_pos)
            + self.cfg.kv * (b_dot - pursuer_vel)
        )
        a_ot = clip_vector_norm(a_ot, self.cfg.base_accel_limit)

        mode_coverage, weighted_mode_dist = self._mode_coverage(
            pursuer_pos, gmm_means_world, gmm_weights, gmm_vars
        )
        line_block_score, line_block_dist = self._line_block_score(
            pursuer_pos, target_pos, goal_pos, strategic_probs
        )
        w2_error = (transport * cost).sum(dim=(-1, -2), keepdim=True)
        ref_dist = torch.norm(barycenters - pursuer_pos, dim=-1)
        target_row = torch.full_like(transport.sum(dim=-1), 1.0 / max(self.num_agents, 1))
        row_marginal_error = (transport.sum(dim=-1) - target_row).abs().amax(dim=-1, keepdim=True)
        col_marginal_error = (transport.sum(dim=1) - weights).abs().amax(dim=-1, keepdim=True)

        return {
            "barycenters": barycenters,
            "b_dot": b_dot,
            "a_ot": a_ot,
            "transport": transport,
            "particles": particles,
            "particle_weights": weights,
            "w2_error": w2_error.squeeze(-1),
            "ref_dist": ref_dist,
            "mode_coverage": mode_coverage,
            "mode_dist": weighted_mode_dist,
            "line_block_score": line_block_score,
            "line_block_dist": line_block_dist,
            "row_marginal_error": row_marginal_error,
            "col_marginal_error": col_marginal_error,
        }

    def _build_particles(
        self,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        goal_pos: torch.Tensor,
        strategic_probs: torch.Tensor,
        gmm_weights: torch.Tensor,
        gmm_means_world: torch.Tensor,
        gmm_vars: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, k, horizon, _ = gmm_means_world.shape
        mode_weights = self._variance_adjusted_mode_weights(gmm_weights, gmm_vars)

        goal_dir_future = safe_normalize(goal_pos.view(batch, 1, 1, 3) - gmm_means_world)
        traj_particles = gmm_means_world + self.cfg.block_distance * goal_dir_future

        tau = torch.linspace(0.7, 1.3, horizon, device=target_pos.device, dtype=target_pos.dtype)
        tau = tau / tau.sum()
        traj_weights = mode_weights.unsqueeze(-1) * tau.view(1, 1, horizon)
        traj_mass = (
            float(self.cfg.traj_mass_base)
            + float(self.cfg.traj_mass_far)
            * strategic_probs[:, STRATEGIC_FAR_RUSH:STRATEGIC_FAR_RUSH + 1]
            + float(self.cfg.traj_mass_close)
            * strategic_probs[:, STRATEGIC_CLOSE_EVADE:STRATEGIC_CLOSE_EVADE + 1]
        )
        traj_weights = traj_weights.reshape(batch, -1) * traj_mass
        traj_particles = traj_particles.reshape(batch, -1, 3)

        goal_dir = safe_normalize(goal_pos - target_pos)
        line_len = torch.norm(goal_pos - target_pos, dim=-1, keepdim=True).clamp_min(1e-6)
        gate_margin = torch.full_like(line_len, float(self.cfg.goal_distance))
        terminal_forward = torch.full_like(line_len, float(self.cfg.terminal_forward_min))
        max_forward = torch.minimum(
            (line_len - gate_margin).clamp_min(terminal_forward),
            line_len,
        )
        min_forward = torch.minimum(
            torch.full_like(line_len, float(self.cfg.channel_min_forward)),
            max_forward,
        )
        near_forward = torch.minimum(torch.maximum(0.42 * line_len, min_forward), max_forward)
        mid_forward = torch.minimum(
            torch.maximum(float(self.cfg.channel_forward_fraction) * line_len, near_forward),
            max_forward,
        )
        gate_forward = max_forward
        channel_centers = target_pos.unsqueeze(1) + torch.stack(
            [near_forward, mid_forward, gate_forward], dim=1
        ) * goal_dir.unsqueeze(1)
        lateral = self._lateral_axis(target_vel, goal_dir)
        lateral_offsets = torch.stack(
            [
                torch.zeros_like(lateral),
                lateral * float(self.cfg.channel_lateral_spread),
                -lateral * float(self.cfg.channel_lateral_spread),
            ],
            dim=1,
        )
        channel_particles = (
            channel_centers.unsqueeze(2) + lateral_offsets.unsqueeze(1)
        ).reshape(batch, -1, 3)
        center_weights = torch.tensor(
            [0.22, 0.33, 0.45],
            device=target_pos.device,
            dtype=target_pos.dtype,
        ).view(1, 3, 1)
        lateral_weights = torch.tensor(
            [0.55, 0.225, 0.225],
            device=target_pos.device,
            dtype=target_pos.dtype,
        ).view(1, 1, 3)
        channel_template = (center_weights * lateral_weights).reshape(1, -1)
        channel_mass = (
            float(self.cfg.channel_mass_base)
            + float(self.cfg.channel_mass_far) * strategic_probs[:, STRATEGIC_FAR_RUSH]
            + float(self.cfg.channel_mass_goal_priority)
            * strategic_probs[:, STRATEGIC_GOAL_PRIORITY]
            + float(self.cfg.channel_mass_goal_sprint)
            * strategic_probs[:, STRATEGIC_GOAL_SPRINT]
        )
        channel_weights = channel_mass.unsqueeze(-1) * channel_template

        q_left = target_pos + self.cfg.evade_forward * goal_dir + self.cfg.evade_side * lateral
        q_right = target_pos + self.cfg.evade_forward * goal_dir - self.cfg.evade_side * lateral
        evade_particles = torch.stack([q_left, q_right], dim=1)
        evade_mass = 0.90 * strategic_probs[:, STRATEGIC_CLOSE_EVADE]
        evade_weights = evade_mass.unsqueeze(-1).expand(-1, 2) / 2.0

        particles = torch.cat([traj_particles, channel_particles, evade_particles], dim=1)
        # Use a nominal safety floor for normal flight, but lower it when the
        # target itself is low so the reference remains capture-compatible.
        target_compatible_floor = target_pos[:, 2:3] - float(self.cfg.low_target_ref_margin)
        nominal_floor = torch.full_like(target_compatible_floor, float(self.cfg.min_ref_z))
        dynamic_floor = torch.minimum(nominal_floor, target_compatible_floor).clamp_min(
            float(self.cfg.hard_floor_z)
        )
        particles[..., 2] = torch.maximum(particles[..., 2], dynamic_floor)
        weights = torch.cat([traj_weights, channel_weights, evade_weights], dim=1)
        weights = weights.clamp_min(1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return particles, weights

    def _shape_channel_barycenters(
        self,
        barycenters: torch.Tensor,
        target_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        strategic_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Bias OT references into a dynamic cross-channel interception gate."""
        if self.num_agents <= 0:
            return barycenters

        out = barycenters.clone()
        goal_vec = goal_pos - target_pos
        line_len = torch.norm(goal_vec, dim=-1, keepdim=True).clamp_min(1e-6)
        goal_dir = goal_vec / line_len
        lateral = self._lateral_axis(torch.zeros_like(goal_dir), goal_dir)

        gate_margin = torch.full_like(line_len, float(self.cfg.goal_distance))
        terminal_forward = torch.full_like(line_len, float(self.cfg.terminal_forward_min))
        max_forward = torch.minimum(
            (line_len - gate_margin).clamp_min(terminal_forward),
            line_len,
        )
        forward_floor = torch.minimum(
            torch.full_like(line_len, float(self.cfg.ref_forward_floor)),
            max_forward,
        )
        channel_forward = torch.minimum(
            torch.maximum(float(self.cfg.channel_forward_fraction) * line_len, forward_floor),
            max_forward,
        )
        sprint_gate = (
            0.45 * strategic_probs[:, STRATEGIC_GOAL_PRIORITY:STRATEGIC_GOAL_PRIORITY + 1]
            + 0.65 * strategic_probs[:, STRATEGIC_GOAL_SPRINT:STRATEGIC_GOAL_SPRINT + 1]
        ).clamp(0.0, 0.85)
        desired_forward = channel_forward * (1.0 - sprint_gate) + max_forward * sprint_gate
        channel_strength = (
            float(self.cfg.ref_channel_blend)
            * (
                0.35
                + strategic_probs[:, STRATEGIC_FAR_RUSH:STRATEGIC_FAR_RUSH + 1]
                + strategic_probs[:, STRATEGIC_GOAL_PRIORITY:STRATEGIC_GOAL_PRIORITY + 1]
                + strategic_probs[:, STRATEGIC_GOAL_SPRINT:STRATEGIC_GOAL_SPRINT + 1]
            )
        ).clamp(0.0, 1.0)

        rel = out - target_pos.unsqueeze(1)
        forward = (rel * goal_dir.unsqueeze(1)).sum(dim=-1)
        forward = torch.minimum(torch.maximum(forward, forward_floor), max_forward)
        forward = forward * (1.0 - channel_strength) + desired_forward * channel_strength

        lateral_scalar = (rel * lateral.unsqueeze(1)).sum(dim=-1)
        slots = torch.linspace(
            -1.0,
            1.0,
            self.num_agents,
            device=out.device,
            dtype=out.dtype,
        ).view(1, self.num_agents)
        slots = slots * float(self.cfg.ref_lateral_spread)
        order = torch.argsort(lateral_scalar, dim=1)
        assigned_slots = torch.empty_like(lateral_scalar)
        assigned_slots.scatter_(1, order, slots.expand(out.shape[0], -1))
        lateral_strength = float(self.cfg.ref_lateral_blend)
        lateral_scalar = lateral_scalar * (1.0 - lateral_strength) + assigned_slots * lateral_strength

        shaped = (
            target_pos.unsqueeze(1)
            + forward.unsqueeze(-1) * goal_dir.unsqueeze(1)
            + lateral_scalar.unsqueeze(-1) * lateral.unsqueeze(1)
        )
        return shaped

    def _separate_barycenters(
        self,
        barycenters: torch.Tensor,
        target_pos: torch.Tensor,
        goal_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Project OT references to a minimally separated horizontal layout."""
        min_sep = float(self.cfg.ref_min_separation)
        iters = int(self.cfg.ref_separation_iters)
        gain = float(self.cfg.ref_separation_gain)
        if min_sep <= 0.0 or iters <= 0 or self.num_agents <= 1:
            return barycenters

        out = barycenters.clone()
        goal_dir = safe_normalize(goal_pos - target_pos)
        lateral = self._lateral_axis(torch.zeros_like(goal_dir), goal_dir)
        fallback_dirs = torch.stack([lateral, -lateral, goal_dir], dim=1)
        for _ in range(iters):
            delta = out[:, :, None, :2] - out[:, None, :, :2]
            dist = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
            eye = torch.eye(self.num_agents, device=out.device, dtype=torch.bool).view(
                1, self.num_agents, self.num_agents, 1
            )
            close = (dist < min_sep) & (~eye)
            fallback = fallback_dirs[:, :, None, :2] - fallback_dirs[:, None, :, :2]
            fallback = safe_normalize(torch.where(fallback.norm(dim=-1, keepdim=True) > 1e-5, fallback, delta))
            direction = torch.where(delta.norm(dim=-1, keepdim=True) > 1e-5, delta / dist, fallback)
            push = 0.5 * gain * (min_sep - dist).clamp_min(0.0) * direction
            push = torch.where(close, push, torch.zeros_like(push))
            out[..., :2] = out[..., :2] + push.sum(dim=2)
        return out

    def _sinkhorn(
        self,
        pursuer_pos: torch.Tensor,
        particles: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, n_agents, _ = pursuer_pos.shape
        cost = torch.cdist(pursuer_pos, particles).square()
        eps = max(float(self.cfg.sinkhorn_epsilon), 1e-4)
        kernel = torch.exp(-cost / eps).clamp_min(1e-12)
        row = torch.full(
            (batch, n_agents),
            1.0 / max(n_agents, 1),
            device=pursuer_pos.device,
            dtype=pursuer_pos.dtype,
        )
        col = weights
        u = torch.ones_like(row)
        v = torch.ones_like(col)
        for _ in range(max(1, int(self.cfg.sinkhorn_iters))):
            u = row / torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            v = col / torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        transport = u.unsqueeze(-1) * kernel * v.unsqueeze(1)
        return transport, cost

    def _mode_coverage(
        self,
        pursuer_pos: torch.Tensor,
        gmm_means_world: torch.Tensor,
        gmm_weights: torch.Tensor,
        gmm_vars: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, k, horizon, _ = gmm_means_world.shape
        mode_weights = self._variance_adjusted_mode_weights(gmm_weights, gmm_vars)
        modes = gmm_means_world.reshape(batch, k * horizon, 3)
        dist = torch.cdist(pursuer_pos, modes).reshape(batch, self.num_agents, k, horizon)
        per_mode = dist.min(dim=1).values.min(dim=-1).values
        weighted_dist = (mode_weights * per_mode).sum(dim=-1, keepdim=True)
        coverage = (mode_weights * torch.exp(-per_mode / max(self.cfg.mode_cover_scale, 1e-6))).sum(
            dim=-1, keepdim=True
        )
        return coverage, weighted_dist

    def _variance_adjusted_mode_weights(
        self,
        gmm_weights: torch.Tensor,
        gmm_vars: torch.Tensor | None,
    ) -> torch.Tensor:
        if gmm_vars is None:
            return gmm_weights
        # Lower-variance modes are more actionable for interception. Keep a
        # floor so high-variance branches remain represented as uncertainty.
        mode_std = torch.sqrt(gmm_vars.clamp_min(1e-6).mean(dim=(-1, -2)))
        confidence = torch.exp(-float(self.cfg.variance_weight_scale) * mode_std)
        confidence = confidence.clamp_min(float(self.cfg.variance_weight_floor))
        mode_weights = gmm_weights * confidence
        return mode_weights / mode_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _line_block_score(
        self,
        pursuer_pos: torch.Tensor,
        target_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        strategic_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        line = goal_pos - target_pos
        line_len = torch.norm(line, dim=-1, keepdim=True).clamp_min(1e-6)
        line_dir = line / line_len
        rel = pursuer_pos - target_pos.unsqueeze(1)
        proj = (rel * line_dir.unsqueeze(1)).sum(dim=-1, keepdim=True)
        proj_clamped = torch.minimum(
            torch.maximum(proj, torch.zeros_like(proj)),
            line_len.unsqueeze(1),
        )
        closest = target_pos.unsqueeze(1) + proj_clamped * line_dir.unsqueeze(1)
        dist = torch.norm(pursuer_pos - closest, dim=-1)
        min_dist = dist.min(dim=-1, keepdim=True).values
        urgency = (
            0.25
            + 0.75 * strategic_probs[:, STRATEGIC_GOAL_PRIORITY:STRATEGIC_GOAL_PRIORITY + 1]
            + strategic_probs[:, STRATEGIC_GOAL_SPRINT:STRATEGIC_GOAL_SPRINT + 1]
        ).clamp(0.0, 1.5)
        score = urgency * torch.exp(-min_dist / max(self.cfg.line_cover_scale, 1e-6))
        return score, min_dist

    @staticmethod
    def _lateral_axis(target_vel: torch.Tensor, goal_dir: torch.Tensor) -> torch.Tensor:
        up = torch.zeros_like(goal_dir)
        up[..., 2] = 1.0
        motion = torch.where(
            target_vel.norm(dim=-1, keepdim=True) > 0.05,
            safe_normalize(target_vel),
            goal_dir,
        )
        lateral = torch.linalg.cross(up, motion, dim=-1)
        fallback = torch.linalg.cross(up, goal_dir, dim=-1)
        lateral = torch.where(lateral.norm(dim=-1, keepdim=True) > 1e-5, lateral, fallback)
        fallback_x = torch.tensor([1.0, 0.0, 0.0], device=goal_dir.device, dtype=goal_dir.dtype).view(1, 3)
        lateral = torch.where(
            lateral.norm(dim=-1, keepdim=True) > 1e-5,
            lateral,
            fallback_x.expand_as(lateral),
        )
        return safe_normalize(lateral)


def build_kinematic_gmm(
    target_pos: torch.Tensor,
    target_vel: torch.Tensor,
    goal_pos: torch.Tensor,
    strategic_probs: torch.Tensor,
    horizon: int = 10,
    dt_pred: float = 0.05,
    max_speed: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fallback 3-mode future distribution in world coordinates."""
    if target_pos.dim() == 3:
        target_pos = target_pos.squeeze(1)
    if target_vel.dim() == 3:
        target_vel = target_vel.squeeze(1)
    if goal_pos.dim() == 1:
        goal_pos = goal_pos.view(1, 3).expand_as(target_pos)

    goal_dir = safe_normalize(goal_pos - target_pos)
    speed = target_vel.norm(dim=-1, keepdim=True).clamp(0.4, max_speed)
    forward = torch.where(
        target_vel.norm(dim=-1, keepdim=True) > 0.05,
        safe_normalize(target_vel),
        goal_dir,
    )
    lateral = OTGuidedInterceptor._lateral_axis(target_vel, goal_dir)
    close = strategic_probs[:, STRATEGIC_CLOSE_EVADE:STRATEGIC_CLOSE_EVADE + 1]
    dirs = torch.stack(
        [
            safe_normalize(0.65 * forward + 0.35 * goal_dir),
            safe_normalize(0.55 * forward + 0.35 * goal_dir + 0.55 * close * lateral),
            safe_normalize(0.55 * forward + 0.35 * goal_dir - 0.55 * close * lateral),
        ],
        dim=1,
    )
    tau = torch.arange(1, horizon + 1, device=target_pos.device, dtype=target_pos.dtype)
    disp = dirs.unsqueeze(2) * speed.view(-1, 1, 1, 1) * (tau.view(1, 1, horizon, 1) * dt_pred)
    means_world = target_pos.view(-1, 1, 1, 3) + disp
    evade = strategic_probs[:, STRATEGIC_CLOSE_EVADE]
    sprint = strategic_probs[:, STRATEGIC_GOAL_SPRINT]
    weights = torch.stack(
        [
            0.70 - 0.25 * evade + 0.15 * sprint,
            0.15 + 0.125 * evade,
            0.15 + 0.125 * evade,
        ],
        dim=-1,
    ).clamp_min(1e-4)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    vars_world = torch.full_like(means_world, 0.05)
    vars_world = vars_world * (1.0 + 2.0 * evade.view(-1, 1, 1, 1))
    return weights, means_world, vars_world
