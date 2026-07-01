"""
目标运动有限状态机 (Target FSM) — 距离驱动版

根据与最近追捕方的距离自动选择策略:
  d > 0.6·d₀  → 策略0: 纯APF直奔守区
  0.4·d₀ < d ≤ 0.6·d₀ → 策略1: APF + 蛇形机动
  d ≤ 0.4·d₀  → 策略2/3/4: 急转/俯冲/爬升 (主导) + 弱APF

每步(0.01s)都重新评估距离并选择策略。
近距离时在{急转,俯冲,爬升}中选一个, 持续到脱离该距离区间。

注意: 本模块是纯PyTorch实现，不依赖Isaac Sim。
"""

import torch
import math
import numpy as np
from typing import Tuple, Optional


# 策略ID
STRATEGY_APF_RUSH = 0      # 纯APF直冲
STRATEGY_SERPENTINE = 1     # APF + 蛇形
STRATEGY_SPIRAL_CLIMB = 2  # 螺旋爬升 (主导)
STRATEGY_SHARP_TURN = 3    # 急转逃逸 (主导)
STRATEGY_DIVE_RUSH = 4     # 俯冲突防 (主导)

STRATEGY_NAMES = [
    "APF_Rush", "Serpentine", "Spiral_Climb", "Sharp_Turn", "Dive_Rush"
]

# 距离档位
ZONE_FAR = 0     # d > 0.6·d₀
ZONE_MID = 1     # 0.4·d₀ < d ≤ 0.6·d₀
ZONE_CLOSE = 2   # d ≤ 0.4·d₀


class TargetFSM:
    """
    距离驱动的目标运动有限状态机

    每步根据与最近追捕方的距离选择策略。

    用法:
        fsm = TargetFSM(num_envs=100, device='cuda')
        fsm.reset(env_ids)
        fsm.set_initial_distance(env_ids, init_dist)

        # 每步调用
        extra_force, strategy_id = fsm.compute_strategy_force(
            target_pos, target_vel, pursuer_positions
        )
        total_force = apf_force + extra_force
    """

    def __init__(
        self,
        num_envs: int,
        device: str = "cpu",
        max_speed: float = 3.0,
        max_accel: float = 3.0,
        arena_radius: float = 5.0,
        max_height: float = 5.0,
        min_height: float = 0.3,
        dt: float = 0.01,
        far_ratio: float = 0.6,
        mid_ratio: float = 0.4,
        # 以下参数保留兼容旧接口
        evasion_threshold: float = 3.0,
        reeval_interval: int = 100,
    ):
        self.num_envs = num_envs
        self.device = device
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.arena_radius = arena_radius
        self.max_height = max_height
        self.min_height = min_height
        self.dt = dt
        self.far_ratio = far_ratio
        self.mid_ratio = mid_ratio

        # 每个环境的状态
        self.current_strategy = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.global_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.initial_distance = torch.ones(num_envs, device=device) * 8.0  # 默认初始距离
        self.current_zone = torch.zeros(num_envs, dtype=torch.long, device=device)

        # 近距离时选择的具体规避策略 (2/3/4)
        self.close_strategy = torch.randint(2, 5, (num_envs,), device=device)
        # 每种策略持续的剩余步数 (0时重新选)
        self.close_strategy_timer = torch.zeros(num_envs, dtype=torch.long, device=device)
        # 策略持续时间范围 (步数) — 0.2~0.5s @ 100Hz = 20~50步
        self.close_min_steps = 20
        self.close_max_steps = 50

        self._init_strategy_params()

    def _init_strategy_params(self):
        """初始化策略参数"""
        n = self.num_envs
        dev = self.device

        # 蛇形机动参数
        self.serpentine_amplitude = torch.empty(n, device=dev).uniform_(1.5, 2.5)
        self.serpentine_freq = torch.empty(n, device=dev).uniform_(0.8, 2.0)
        self.serpentine_phase = torch.empty(n, device=dev).uniform_(0, 2 * math.pi)

        # 螺旋爬升参数
        self.spiral_radius_force = torch.empty(n, device=dev).uniform_(1.0, 2.0)
        self.spiral_omega = torch.empty(n, device=dev).uniform_(1.5, 3.0)
        self.spiral_climb_force = torch.empty(n, device=dev).uniform_(1.0, 2.0)
        self.spiral_direction = torch.where(
            torch.rand(n, device=dev) > 0.5,
            torch.ones(n, device=dev),
            -torch.ones(n, device=dev),
        )

        # 急转逃逸参数
        self.sharp_turn_strength = torch.empty(n, device=dev).uniform_(2.5, 3.0)
        self.sharp_turn_direction = torch.where(
            torch.rand(n, device=dev) > 0.5,
            torch.ones(n, device=dev),
            -torch.ones(n, device=dev),
        )
        self.sharp_turn_vertical = torch.empty(n, device=dev).uniform_(-1.0, 1.0)

        # 俯冲突防参数
        self.dive_force = torch.empty(n, device=dev).uniform_(2.0, 3.0)
        self.dive_target_height = torch.empty(n, device=dev).uniform_(0.5, 1.0)
        self.rush_boost = torch.empty(n, device=dev).uniform_(1.5, 2.5)

    def reset(self, env_ids: torch.Tensor):
        """episode重置"""
        if len(env_ids) == 0:
            return
        self.current_strategy[env_ids] = STRATEGY_APF_RUSH
        self.global_step[env_ids] = 0
        self.current_zone[env_ids] = ZONE_FAR
        self.close_strategy[env_ids] = torch.randint(2, 5, (len(env_ids),), device=self.device)
        self.close_strategy_timer[env_ids] = 0
        self._resample_params(env_ids)

    def set_initial_distance(self, env_ids: torch.Tensor, distances: torch.Tensor):
        """记录初始距离, 用于计算距离比例"""
        self.initial_distance[env_ids] = distances

    def _resample_params(self, env_ids: torch.Tensor):
        """重新采样策略参数"""
        n = len(env_ids)
        dev = self.device

        self.serpentine_amplitude[env_ids] = torch.empty(n, device=dev).uniform_(1.5, 2.5)
        self.serpentine_freq[env_ids] = torch.empty(n, device=dev).uniform_(0.8, 2.0)
        self.serpentine_phase[env_ids] = torch.empty(n, device=dev).uniform_(0, 2 * math.pi)

        self.spiral_radius_force[env_ids] = torch.empty(n, device=dev).uniform_(1.0, 2.0)
        self.spiral_omega[env_ids] = torch.empty(n, device=dev).uniform_(1.5, 3.0)
        self.spiral_climb_force[env_ids] = torch.empty(n, device=dev).uniform_(1.0, 2.0)
        self.spiral_direction[env_ids] = torch.where(
            torch.rand(n, device=dev) > 0.5,
            torch.ones(n, device=dev),
            -torch.ones(n, device=dev),
        )

        self.sharp_turn_strength[env_ids] = torch.empty(n, device=dev).uniform_(2.5, 3.0)
        self.sharp_turn_direction[env_ids] = torch.where(
            torch.rand(n, device=dev) > 0.5,
            torch.ones(n, device=dev),
            -torch.ones(n, device=dev),
        )
        self.sharp_turn_vertical[env_ids] = torch.empty(n, device=dev).uniform_(-1.0, 1.0)

        self.dive_force[env_ids] = torch.empty(n, device=dev).uniform_(2.0, 3.0)
        self.dive_target_height[env_ids] = torch.empty(n, device=dev).uniform_(0.5, 1.0)
        self.rush_boost[env_ids] = torch.empty(n, device=dev).uniform_(1.5, 2.5)

    def compute_strategy_force(
        self,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        pursuer_pos: torch.Tensor,
        goal_center: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算策略叠加力

        每步根据距离决定策略, 不再有固定间隔.

        Args:
            target_pos:  [num_envs, 3]
            target_vel:  [num_envs, 3]
            pursuer_pos: [num_envs, n_pursuers, 3]
            goal_center: [3] (可选)

        Returns:
            extra_force: [num_envs, 3]
            current_strategy: [num_envs]
        """
        # 1. 计算与最近追捕方的距离
        diff = pursuer_pos - target_pos.unsqueeze(1)  # [N, n_p, 3]
        dists = torch.norm(diff, dim=-1)               # [N, n_p]
        min_dist = dists.min(dim=1).values              # [N]

        # 2. 根据距离确定zone
        d0 = self.initial_distance
        old_zone = self.current_zone.clone()

        new_zone = torch.full_like(self.current_zone, ZONE_FAR)
        new_zone = torch.where(min_dist <= d0 * self.far_ratio, torch.full_like(new_zone, ZONE_MID), new_zone)
        new_zone = torch.where(min_dist <= d0 * self.mid_ratio, torch.full_like(new_zone, ZONE_CLOSE), new_zone)

        # 安全触发: 高度过低强制爬升
        too_low = target_pos[:, 2] < self.min_height + 0.2
        new_zone = torch.where(too_low, torch.full_like(new_zone, ZONE_CLOSE), new_zone)

        # 3. zone变化时更新策略
        zone_changed = (new_zone != old_zone)
        self.current_zone = new_zone

        # 远距 → 纯APF
        far_mask = (new_zone == ZONE_FAR)
        self.current_strategy = torch.where(far_mask, torch.full_like(self.current_strategy, STRATEGY_APF_RUSH), self.current_strategy)

        # 中距 → 蛇形
        mid_mask = (new_zone == ZONE_MID)
        self.current_strategy = torch.where(mid_mask, torch.full_like(self.current_strategy, STRATEGY_SERPENTINE), self.current_strategy)

        # 近距 → 激烈规避: 每20~50步随机切换一次策略 (急转/俯冲/爬升)
        close_mask = (new_zone == ZONE_CLOSE)
        if close_mask.any():
            # 计时器倒数, 归零时重新选策略 + 重采样参数
            self.close_strategy_timer[close_mask] -= 1
            timer_expired = close_mask & (self.close_strategy_timer <= 0)

            if timer_expired.any():
                n_exp = timer_expired.sum().item()
                # 随机选择新策略 2/3/4
                self.close_strategy[timer_expired] = torch.randint(
                    2, 5, (n_exp,), device=self.device
                )
                # 重置计时器: 均匀随机 [20, 50] 步
                self.close_strategy_timer[timer_expired] = torch.randint(
                    self.close_min_steps, self.close_max_steps + 1,
                    (n_exp,), device=self.device
                )
                # 重新采样策略参数 (方向、强度等)
                self._resample_params(timer_expired.nonzero(as_tuple=False).squeeze(-1))

            # 高度过低强制螺旋爬升
            force_climb = close_mask & too_low
            self.close_strategy[force_climb] = STRATEGY_SPIRAL_CLIMB

        self.current_strategy = torch.where(close_mask, self.close_strategy, self.current_strategy)

        # 4. 计算各策略的力
        N = self.num_envs
        force = torch.zeros(N, 3, device=self.device)
        t = self.global_step.float() * self.dt

        f_serpentine = self._compute_serpentine(target_vel, t)
        f_spiral = self._compute_spiral(target_vel, t)
        f_sharp = self._compute_sharp_turn(target_pos, target_vel, pursuer_pos)
        f_dive = self._compute_dive_rush(target_pos, target_vel, goal_center)

        s = self.current_strategy
        force = torch.where((s == 1).unsqueeze(-1), f_serpentine, force)
        force = torch.where((s == 2).unsqueeze(-1), f_spiral, force)
        force = torch.where((s == 3).unsqueeze(-1), f_sharp, force)
        force = torch.where((s == 4).unsqueeze(-1), f_dive, force)

        # 近距时策略力主导: 放大策略力
        close_scale = torch.where(close_mask, torch.tensor(2.0, device=self.device),
                                   torch.tensor(1.0, device=self.device)).unsqueeze(-1)
        force = force * close_scale

        # 限幅
        force_norm = torch.norm(force, dim=-1, keepdim=True)
        force = force * (self.max_accel / (force_norm + 1e-6)).clamp(max=1.0)

        self.global_step += 1

        return force, self.current_strategy.clone()

    def _compute_serpentine(self, target_vel: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """蛇形机动: 侧向正弦扰动"""
        speed = torch.norm(target_vel, dim=-1, keepdim=True).clamp(min=0.1)
        forward = target_vel / speed

        z_up = torch.tensor([0., 0., 1.], device=self.device).expand_as(forward)
        lateral = torch.cross(z_up, forward, dim=-1)
        lat_norm = torch.norm(lateral, dim=-1, keepdim=True).clamp(min=1e-6)
        lateral = lateral / lat_norm

        A = self.serpentine_amplitude.unsqueeze(-1)
        freq = self.serpentine_freq
        phase = self.serpentine_phase
        sin_val = torch.sin(2 * math.pi * freq * t + phase).unsqueeze(-1)

        force = A * sin_val * lateral
        return force

    def _compute_spiral(self, target_vel: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """螺旋爬升: 旋转侧向力 + 上升力"""
        speed = torch.norm(target_vel, dim=-1, keepdim=True).clamp(min=0.1)
        forward = target_vel / speed

        z_up = torch.tensor([0., 0., 1.], device=self.device).expand_as(forward)
        lateral = torch.cross(z_up, forward, dim=-1)
        lat_norm = torch.norm(lateral, dim=-1, keepdim=True).clamp(min=1e-6)
        lateral = lateral / lat_norm

        A = self.spiral_radius_force.unsqueeze(-1)
        omega = self.spiral_omega
        direction = self.spiral_direction
        f_climb = self.spiral_climb_force.unsqueeze(-1)

        angle = direction * omega * t
        cos_val = torch.cos(angle).unsqueeze(-1)

        force = A * cos_val * lateral
        force[..., 2] += f_climb.squeeze(-1)

        return force

    def _compute_sharp_turn(
        self,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        pursuer_pos: torch.Tensor,
    ) -> torch.Tensor:
        """急转逃逸: 垂直于最近追捕方方向急转"""
        diff = pursuer_pos - target_pos.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        nearest_idx = dist.argmin(dim=1)

        nearest_dir = diff[torch.arange(self.num_envs, device=self.device), nearest_idx]
        nearest_dir_norm = torch.norm(nearest_dir, dim=-1, keepdim=True).clamp(min=1e-6)
        approach_dir = nearest_dir / nearest_dir_norm

        z_up = torch.tensor([0., 0., 1.], device=self.device).expand_as(approach_dir)
        escape_dir = torch.cross(approach_dir, z_up, dim=-1)
        escape_norm = torch.norm(escape_dir, dim=-1, keepdim=True).clamp(min=1e-6)
        escape_dir = escape_dir / escape_norm

        strength = self.sharp_turn_strength.unsqueeze(-1)
        direction = self.sharp_turn_direction.unsqueeze(-1)
        vertical = self.sharp_turn_vertical.unsqueeze(-1)

        force = strength * direction * escape_dir
        force[..., 2] += vertical.squeeze(-1)

        return force

    def _compute_dive_rush(
        self,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        goal_center: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """俯冲突防: 高度>目标高度时下降，否则前冲"""
        current_height = target_pos[..., 2]
        target_h = self.dive_target_height

        is_diving = current_height > target_h + 0.3

        force = torch.zeros(self.num_envs, 3, device=self.device)

        f_dive = self.dive_force
        force[..., 2] -= is_diving.float() * f_dive

        if goal_center is not None:
            goal_dir = goal_center.expand(self.num_envs, -1) - target_pos
            goal_dist = torch.norm(goal_dir, dim=-1, keepdim=True).clamp(min=1e-6)
            goal_dir_unit = goal_dir / goal_dist
            rush = self.rush_boost.unsqueeze(-1) * (~is_diving).float().unsqueeze(-1) * goal_dir_unit
            force += rush

        return force

    def get_strategy_names(self) -> list:
        """返回当前各环境的策略名称列表"""
        return [STRATEGY_NAMES[s.item()] for s in self.current_strategy]
