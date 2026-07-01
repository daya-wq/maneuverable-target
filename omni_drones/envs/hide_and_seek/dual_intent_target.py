"""
双层意图目标运动控制器

第1层: 战术意图 (每0.5s检查一次)
    0. Far_Rush: 最近追捕方距离 > far_distance, 直逼守区
    1. Close_Evade: 最近追捕方距离 <= far_distance 且离守区 > sprint_distance,
       规避最近追捕方、远离其他追捕方、同时靠近守区。
       若目标离守区距离小于离最近追捕方距离，则进入“守区优先规避”:
       守区吸引力占主导，且暂停1s随机机动切换
    2. Goal_Sprint: 离守区 <= sprint_distance, 只向守区冲刺

第2层: 机动偏置模式 (每1.0s切换一次)
    Close_Evade模式下叠加9类偏置; 当战术意图发生变化时立即刷新偏置方式。

对外暴露的 maneuver_intent 则按“总加速度相对当前速度方向”的关系实时标注，
与训练/诊断脚本中的语义标签保持一致。

本模块只依赖PyTorch, 便于在独立6DoF脚本和主环境之间复用。
"""

from __future__ import annotations

import torch
from typing import Dict, Tuple

from omni_drones.utils.intent_labels import INTENT_HOVER, label_intent_batch


TACTICAL_FAR_RUSH = 0
TACTICAL_CLOSE_EVADE = 1
TACTICAL_GOAL_PRIORITY = 2
TACTICAL_GOAL_SPRINT = 3

TACTICAL_NAMES = [
    "Far_Rush",
    "Close_Evade",
    "Goal_Priority",
    "Goal_Sprint",
]

MANEUVER_STRAIGHT = 0
MANEUVER_LEFT_TURN = 1
MANEUVER_RIGHT_TURN = 2
MANEUVER_CLIMB = 3
MANEUVER_DIVE = 4
MANEUVER_LEFT_CLIMB = 5
MANEUVER_LEFT_DIVE = 6
MANEUVER_RIGHT_CLIMB = 7
MANEUVER_RIGHT_DIVE = 8

MANEUVER_NAMES = [
    "Straight",
    "Left-Turn",
    "Right-Turn",
    "Climb",
    "Dive",
    "Left-Climb",
    "Left-Dive",
    "Right-Climb",
    "Right-Dive",
]


def safe_normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def point_to_cylinder_distance(
    points: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    height: float,
) -> torch.Tensor:
    radial_dist = torch.norm(points[..., :2] - center[..., :2], dim=-1)
    radial_delta = torch.relu(radial_dist - radius)
    vertical_delta = torch.relu(torch.abs(points[..., 2] - center[..., 2]) - height / 2.0)
    return torch.sqrt(radial_delta.square() + vertical_delta.square())


def closest_point_on_cylinder(
    points: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    height: float,
) -> torch.Tensor:
    offset_xy = points[..., :2] - center[..., :2]
    radial_dist = torch.norm(offset_xy, dim=-1, keepdim=True)
    safe_radial = torch.where(radial_dist > 1e-5, radial_dist, torch.ones_like(radial_dist))
    clamped_radial = torch.minimum(radial_dist, torch.full_like(radial_dist, radius))
    closest_xy = center[..., :2] + offset_xy / safe_radial * clamped_radial
    closest_z = points[..., 2:3].clamp(
        center[..., 2:3] - height / 2.0,
        center[..., 2:3] + height / 2.0,
    )
    return torch.cat([closest_xy, closest_z], dim=-1)


class DualIntentTargetController:
    def __init__(
        self,
        num_envs: int,
        device: str = "cpu",
        dt: float = 0.01,
        goal_center: torch.Tensor | None = None,
        goal_radius: float = 0.5,
        goal_height: float = 1.0,
        far_distance: float = 6.0,
        sprint_distance: float = 2.0,
        tactical_check_period: float = 0.5,
        maneuver_update_period: float = 1.0,
        arena_radius: float = 6.0,
        max_height: float = 5.0,
        min_height: float = 0.2,
        max_accel: float = 3.0,
        goal_gain_far: float = 1.0,
        goal_gain_evade: float = 0.9,
        goal_gain_sprint: float = 1.5,
        nearest_repulsion_gain: float = 1.9,
        other_repulsion_gain: float = 0.9,
        goal_priority_gain: float = 2.4,
        goal_priority_repulsion_scale: float = 0.45,
        boundary_gain: float = 3.0,
        maneuver_gain: float = 1.0,
        maneuver_forward_scale: float = 0.35,
        maneuver_vertical_scale: float = 0.9,
        maneuver_speed_threshold: float = 0.3,
        intent_accel_threshold: float = 0.5,
    ):
        self.num_envs = num_envs
        self.device = device
        self.dt = float(dt)
        self.goal_center = (
            torch.tensor([5.0, 0.0, 2.5], dtype=torch.float32, device=device)
            if goal_center is None
            else goal_center.to(device=device, dtype=torch.float32)
        )
        self.goal_radius = float(goal_radius)
        self.goal_height = float(goal_height)
        self.far_distance = float(far_distance)
        self.sprint_distance = float(sprint_distance)
        self.tactical_check_steps = max(1, int(round(tactical_check_period / self.dt)))
        self.maneuver_update_steps = max(1, int(round(maneuver_update_period / self.dt)))
        self.arena_radius = float(arena_radius)
        self.max_height = float(max_height)
        self.min_height = float(min_height)
        self.max_accel = float(max_accel)
        self.goal_gain_far = float(goal_gain_far)
        self.goal_gain_evade = float(goal_gain_evade)
        self.goal_gain_sprint = float(goal_gain_sprint)
        self.nearest_repulsion_gain = float(nearest_repulsion_gain)
        self.other_repulsion_gain = float(other_repulsion_gain)
        self.goal_priority_gain = float(goal_priority_gain)
        self.goal_priority_repulsion_scale = float(goal_priority_repulsion_scale)
        self.boundary_gain = float(boundary_gain)
        self.maneuver_gain = float(maneuver_gain)
        self.maneuver_forward_scale = float(maneuver_forward_scale)
        self.maneuver_vertical_scale = float(maneuver_vertical_scale)
        self.maneuver_speed_threshold = float(maneuver_speed_threshold)
        self.intent_accel_threshold = float(intent_accel_threshold)

        self.global_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.tactical_intent = torch.full(
            (num_envs,),
            TACTICAL_FAR_RUSH,
            dtype=torch.long,
            device=device,
        )
        self.maneuver_mode = torch.full(
            (num_envs,),
            MANEUVER_STRAIGHT,
            dtype=torch.long,
            device=device,
        )
        self.maneuver_intent = torch.full(
            (num_envs,),
            INTENT_HOVER,
            dtype=torch.long,
            device=device,
        )
        self.goal_priority_active = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.last_tactical_update_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_maneuver_update_step = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.global_step[env_ids] = 0
        self.tactical_intent[env_ids] = TACTICAL_FAR_RUSH
        self.maneuver_mode[env_ids] = MANEUVER_STRAIGHT
        self.maneuver_intent[env_ids] = INTENT_HOVER
        self.goal_priority_active[env_ids] = False
        self.last_tactical_update_step[env_ids] = 0
        self.last_maneuver_update_step[env_ids] = 0

    def _sample_maneuvers(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.maneuver_mode[env_ids] = torch.randint(
            0, len(MANEUVER_NAMES), (env_ids.numel(),), device=self.device
        )
        self.last_maneuver_update_step[env_ids] = self.global_step[env_ids]

    def _goal_force_and_distance(self, target_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        goal_center = self.goal_center.view(1, 3)
        goal_closest = closest_point_on_cylinder(
            target_pos, goal_center, self.goal_radius, self.goal_height
        )
        goal_offset = goal_closest - target_pos
        goal_dist = torch.norm(goal_offset, dim=-1, keepdim=True)
        goal_force = goal_offset / (goal_dist + 1e-5)
        goal_boundary_dist = point_to_cylinder_distance(
            target_pos, goal_center, self.goal_radius, self.goal_height
        )
        return goal_force, goal_boundary_dist

    def _boundary_force(self, target_pos: torch.Tensor) -> torch.Tensor:
        force = torch.zeros_like(target_pos)
        target_origin_dist = torch.norm(target_pos[..., :2], dim=-1)
        radial_dir = -target_pos[..., :2] / (target_origin_dist.unsqueeze(-1) + 1e-5)
        near_boundary = target_origin_dist > self.arena_radius * 0.82
        if near_boundary.any():
            margin = (self.arena_radius - target_origin_dist[near_boundary]).clamp(min=0.05)
            force[near_boundary, :2] = (
                self.boundary_gain
                * radial_dir[near_boundary]
                / margin.unsqueeze(-1)
            )
        high = target_pos[..., 2] > self.max_height * 0.9
        low = target_pos[..., 2] < self.min_height + 0.25
        force[high, 2] -= self.boundary_gain
        force[low, 2] += self.boundary_gain
        return force

    def _maneuver_force(self, target_vel: torch.Tensor, goal_force: torch.Tensor) -> torch.Tensor:
        forward = safe_normalize(target_vel)
        goal_forward = safe_normalize(goal_force)
        speed = torch.norm(target_vel, dim=-1, keepdim=True)
        forward = torch.where(speed < self.maneuver_speed_threshold, goal_forward, forward)
        forward = safe_normalize(forward)

        up = torch.zeros_like(forward)
        up[..., 2] = 1.0
        lateral = torch.cross(up, forward, dim=-1)
        lateral = safe_normalize(lateral)

        forward_component = self.maneuver_gain * self.maneuver_forward_scale * forward
        lateral_component = self.maneuver_gain * lateral
        vertical_component = self.maneuver_gain * self.maneuver_vertical_scale * up

        force = torch.zeros_like(target_vel)
        ids = self.maneuver_mode
        force[ids == MANEUVER_STRAIGHT] = forward_component[ids == MANEUVER_STRAIGHT]
        force[ids == MANEUVER_LEFT_TURN] = lateral_component[ids == MANEUVER_LEFT_TURN]
        force[ids == MANEUVER_RIGHT_TURN] = -lateral_component[ids == MANEUVER_RIGHT_TURN]
        force[ids == MANEUVER_CLIMB] = vertical_component[ids == MANEUVER_CLIMB]
        force[ids == MANEUVER_DIVE] = -vertical_component[ids == MANEUVER_DIVE]
        force[ids == MANEUVER_LEFT_CLIMB] = (
            lateral_component + vertical_component
        )[ids == MANEUVER_LEFT_CLIMB]
        force[ids == MANEUVER_LEFT_DIVE] = (
            lateral_component - vertical_component
        )[ids == MANEUVER_LEFT_DIVE]
        force[ids == MANEUVER_RIGHT_CLIMB] = (
            -lateral_component + vertical_component
        )[ids == MANEUVER_RIGHT_CLIMB]
        force[ids == MANEUVER_RIGHT_DIVE] = (
            -lateral_component - vertical_component
        )[ids == MANEUVER_RIGHT_DIVE]
        return force

    def _evaluate_tactical_intent(
        self,
        target_pos: torch.Tensor,
        pursuer_pos: torch.Tensor,
        goal_dist: torch.Tensor,
    ) -> torch.Tensor:
        pursuer_dist = torch.norm(pursuer_pos - target_pos.unsqueeze(1), dim=-1)
        min_pursuer_dist = pursuer_dist.min(dim=1).values

        tactical = torch.full_like(self.tactical_intent, TACTICAL_FAR_RUSH)
        # Close_Evade: near pursuers AND goal is farther than nearest pursuer
        close_evade = (
            (min_pursuer_dist <= self.far_distance)
            & (goal_dist > self.sprint_distance)
            & (goal_dist > min_pursuer_dist)
        )
        # Goal_Priority: near pursuers AND goal is closer than nearest pursuer
        goal_priority = (
            (min_pursuer_dist <= self.far_distance)
            & (goal_dist > self.sprint_distance)
            & (goal_dist <= min_pursuer_dist)
        )
        goal_sprint = goal_dist <= self.sprint_distance
        tactical = torch.where(
            close_evade,
            torch.full_like(tactical, TACTICAL_CLOSE_EVADE),
            tactical,
        )
        tactical = torch.where(
            goal_priority,
            torch.full_like(tactical, TACTICAL_GOAL_PRIORITY),
            tactical,
        )
        tactical = torch.where(
            goal_sprint,
            torch.full_like(tactical, TACTICAL_GOAL_SPRINT),
            tactical,
        )
        return tactical

    def step(
        self,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        pursuer_pos: torch.Tensor,
        pursuer_vel: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del pursuer_vel
        goal_force, goal_dist = self._goal_force_and_distance(target_pos)
        boundary_force = self._boundary_force(target_pos)
        pursuer_offset = target_pos.unsqueeze(1) - pursuer_pos
        pursuer_dist = torch.norm(pursuer_offset, dim=-1)
        nearest_idx = pursuer_dist.argmin(dim=1)
        nearest_pursuer_dist = pursuer_dist.min(dim=1).values
        repulsion_dir = safe_normalize(pursuer_offset)
        repulsion_force = repulsion_dir / (pursuer_dist.unsqueeze(-1) + 1e-5)

        tactical_due = (
            (self.global_step - self.last_tactical_update_step) >= self.tactical_check_steps
        ) | (self.global_step == 0)
        tactical_changed = torch.zeros_like(tactical_due)
        if tactical_due.any():
            new_tactical = self._evaluate_tactical_intent(target_pos, pursuer_pos, goal_dist)
            changed_mask = tactical_due & (new_tactical != self.tactical_intent)
            self.tactical_intent = torch.where(tactical_due, new_tactical, self.tactical_intent)
            self.last_tactical_update_step[tactical_due] = self.global_step[tactical_due]
            tactical_changed = changed_mask

        nearest_mask = torch.nn.functional.one_hot(
            nearest_idx, num_classes=pursuer_pos.shape[1]
        ).unsqueeze(-1).to(repulsion_force.dtype)
        nearest_repulsion = (repulsion_force * nearest_mask).sum(dim=1)
        other_repulsion = repulsion_force.sum(dim=1) - nearest_repulsion

        maneuver_force = self._maneuver_force(target_vel, goal_force)
        total_force = boundary_force.clone()

        far_mask = self.tactical_intent == TACTICAL_FAR_RUSH
        evade_mask = self.tactical_intent == TACTICAL_CLOSE_EVADE
        goal_priority_mask = self.tactical_intent == TACTICAL_GOAL_PRIORITY
        sprint_mask = self.tactical_intent == TACTICAL_GOAL_SPRINT

        entered_evade = tactical_changed & evade_mask
        left_evade = tactical_changed & (~evade_mask)
        entered_goal_priority = tactical_changed & goal_priority_mask
        left_goal_priority = self.goal_priority_active & (~goal_priority_mask)

        # Sample new maneuvers when entering Close_Evade
        if entered_evade.any():
            self._sample_maneuvers(entered_evade.nonzero(as_tuple=False).squeeze(-1))
        # Reset maneuver to Straight when entering Goal_Priority
        if entered_goal_priority.any():
            env_ids = entered_goal_priority.nonzero(as_tuple=False).squeeze(-1)
            self.maneuver_mode[env_ids] = MANEUVER_STRAIGHT
            self.last_maneuver_update_step[env_ids] = self.global_step[env_ids]
        # Reset maneuver when leaving evade-like states
        if left_evade.any():
            env_ids = left_evade.nonzero(as_tuple=False).squeeze(-1)
            self.maneuver_mode[env_ids] = MANEUVER_STRAIGHT
            self.last_maneuver_update_step[env_ids] = self.global_step[env_ids]
        # Re-sample maneuvers when transitioning from Goal_Priority back to Close_Evade
        if left_goal_priority.any():
            re_enter_evade = left_goal_priority & evade_mask
            if re_enter_evade.any():
                self._sample_maneuvers(re_enter_evade.nonzero(as_tuple=False).squeeze(-1))

        maneuver_due = (
            evade_mask
            & ((self.global_step - self.last_maneuver_update_step) >= self.maneuver_update_steps)
        )
        maneuver_changed = torch.zeros_like(tactical_due)
        if maneuver_due.any():
            env_ids = maneuver_due.nonzero(as_tuple=False).squeeze(-1)
            self._sample_maneuvers(env_ids)
            maneuver_changed[env_ids] = True

        if far_mask.any():
            total_force[far_mask] += self.goal_gain_far * goal_force[far_mask]
            self.maneuver_mode[far_mask] = MANEUVER_STRAIGHT
        if evade_mask.any():
            total_force[evade_mask] += (
                self.goal_gain_evade * goal_force[evade_mask]
                + self.nearest_repulsion_gain * nearest_repulsion[evade_mask]
                + self.other_repulsion_gain * other_repulsion[evade_mask]
                + maneuver_force[evade_mask]
            )
        if goal_priority_mask.any():
            total_force[goal_priority_mask] += (
                self.goal_priority_gain * goal_force[goal_priority_mask]
                + self.goal_priority_repulsion_scale
                * (
                    self.nearest_repulsion_gain * nearest_repulsion[goal_priority_mask]
                    + self.other_repulsion_gain * other_repulsion[goal_priority_mask]
                )
            )
            self.maneuver_mode[goal_priority_mask] = MANEUVER_STRAIGHT
        if sprint_mask.any():
            total_force[sprint_mask] += self.goal_gain_sprint * goal_force[sprint_mask]
            self.maneuver_mode[sprint_mask] = MANEUVER_STRAIGHT

        force_norm = torch.norm(total_force, dim=-1, keepdim=True).clamp(min=1e-6)
        accel_cmd = self.max_accel * total_force / force_norm
        self.maneuver_intent = label_intent_batch(
            accel_cmd,
            target_vel,
            speed_threshold=self.maneuver_speed_threshold,
            accel_threshold=self.intent_accel_threshold,
        )

        info = {
            "tactical_intent": self.tactical_intent.clone(),
            "maneuver_intent": self.maneuver_intent.clone(),
            "maneuver_mode": self.maneuver_mode.clone(),
            "tactical_changed": tactical_changed.clone(),
            "maneuver_changed": (
                maneuver_changed.clone()
                | entered_evade.clone()
                | entered_goal_priority.clone()
                | left_evade.clone()
                | left_goal_priority.clone()
            ),
            "maneuver_mode_changed": (
                maneuver_changed.clone()
                | entered_evade.clone()
                | entered_goal_priority.clone()
                | left_evade.clone()
                | left_goal_priority.clone()
            ),
            "goal_priority_active": goal_priority_mask.clone(),
            "goal_dist": goal_dist.clone(),
            "nearest_pursuer_dist": nearest_pursuer_dist.clone(),

        }
        self.goal_priority_active = goal_priority_mask.clone()
        self.global_step += 1
        return accel_cmd, info
