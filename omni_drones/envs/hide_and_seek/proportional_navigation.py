"""
3D比例导引法 (Proportional Navigation)

用于追捕方在数据收集阶段追踪目标。

原理:
    导引加速度 = N × V_c × ω
    
    N:   导航常数 (Navigation constant, 通常3-5)
    V_c: 接近速度 (Closing velocity)  
    ω:   视线角速率 (Line-of-Sight rate)

特点:
    - 纯比例导引, 不需要目标加速度信息
    - 适用于追击匀速或低机动目标
    - 对高机动目标会有滞后, 这正好是我们想要的
      (追捕方的滞后反应产生多样性数据)

注意: 本模块是纯PyTorch实现，不依赖Isaac Sim。
"""

import torch
import math
from typing import Optional


def proportional_navigation_3d(
    pursuer_pos: torch.Tensor,
    pursuer_vel: torch.Tensor,
    target_pos: torch.Tensor,
    target_vel: torch.Tensor,
    N: float = 4.0,
    max_accel: float = 3.0,
) -> torch.Tensor:
    """
    3D比例导引加速度指令

    Args:
        pursuer_pos:  [N_envs, 3] 或 [N_envs, n_pursuers, 3]
        pursuer_vel:  同上
        target_pos:   [N_envs, 3] 或 [N_envs, 1, 3]
        target_vel:   同上
        N:            导航常数 (默认4.0)
        max_accel:    最大加速度限幅 (默认3.0)

    Returns:
        a_cmd: 与pursuer_pos同shape的加速度指令
    """
    # 确保target_pos可以广播到pursuer_pos的shape
    if target_pos.dim() < pursuer_pos.dim():
        target_pos = target_pos.unsqueeze(-2)
        target_vel = target_vel.unsqueeze(-2)

    # 视线向量 (LOS)
    r = target_pos - pursuer_pos  # [..., 3]
    r_norm = torch.norm(r, dim=-1, keepdim=True).clamp(min=1e-4)
    r_hat = r / r_norm  # LOS单位向量

    # 相对速度
    v_rel = target_vel - pursuer_vel  # [..., 3]

    # 接近速度 (closing velocity)
    # V_c = -dr/dt = -(v_rel · r_hat)
    V_c = -(v_rel * r_hat).sum(dim=-1, keepdim=True)  # [..., 1]
    # V_c为正说明在接近, 为负说明在远离

    # 视线角速率 ω = (r × v_rel) / |r|²
    omega = torch.cross(r, v_rel, dim=-1) / (r_norm ** 2 + 1e-6)  # [..., 3]

    # 导引加速度: a = N × V_c × (ω × r_hat)
    a_cmd = N * V_c * torch.cross(omega, r_hat, dim=-1)  # [..., 3]

    # 当目标在远离时(V_c<0), 直接朝目标追
    # 否则纯PN可能输出反向加速度
    is_diverging = (V_c < 0).squeeze(-1)  # [...]
    pursuit_acc = r_hat * max_accel * 0.8  # 纯追踪加速度
    a_cmd = torch.where(
        is_diverging.unsqueeze(-1).expand_as(a_cmd),
        pursuit_acc,
        a_cmd,
    )

    # 限幅
    a_norm = torch.norm(a_cmd, dim=-1, keepdim=True).clamp(min=1e-6)
    a_cmd = a_cmd * (max_accel / a_norm).clamp(max=1.0)

    return a_cmd


def pn_velocity_command(
    pursuer_pos: torch.Tensor,
    pursuer_vel: torch.Tensor,
    target_pos: torch.Tensor,
    target_vel: torch.Tensor,
    current_vel_cmd: torch.Tensor,
    N: float = 4.0,
    max_accel: float = 3.0,
    max_speed: float = 3.0,
    dt: float = 0.01,
) -> torch.Tensor:
    """
    比例导引 → 速度指令 (适合现有PIDrate控制器)

    将PN加速度积分为速度指令。

    Args:
        pursuer_pos:    [N_envs, n_pursuers, 3]
        pursuer_vel:    同上
        target_pos:     [N_envs, 1, 3] 或 [N_envs, 3]
        target_vel:     同上
        current_vel_cmd: [N_envs, n_pursuers, 3] 当前速度指令
        N, max_accel, max_speed, dt: 参数

    Returns:
        new_vel_cmd: [N_envs, n_pursuers, 3] 新的速度指令
    """
    a_cmd = proportional_navigation_3d(
        pursuer_pos, pursuer_vel, target_pos, target_vel,
        N=N, max_accel=max_accel,
    )

    # 速度指令 = 当前指令 + 加速度 × dt
    new_vel_cmd = current_vel_cmd + a_cmd * dt

    # 限速
    speed = torch.norm(new_vel_cmd, dim=-1, keepdim=True).clamp(min=1e-6)
    new_vel_cmd = new_vel_cmd * (max_speed / speed).clamp(max=1.0)

    return new_vel_cmd
