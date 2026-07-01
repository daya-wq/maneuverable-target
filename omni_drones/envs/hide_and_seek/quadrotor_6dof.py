"""
非线性六自由度四旋翼动力学 + SO(3) 几何控制器

动力学:
    ṗ = v
    mv̇ = -mg·e₃ + R·[0,0,T_actual]ᵀ - k_drag·v
    Ṙ = R·[ω]×
    J·ω̇ = -ω × J·ω + τ_actual

电机延迟 (一阶惯性环节):
    Ṫ_actual = (T_cmd - T_actual) / τ_motor
    τ̇_actual = (τ_cmd - τ_actual) / τ_motor
    τ_motor ≈ 0.03s (30ms)

控制链:
    外环 (APF/FSM/PN) → 期望加速度 a_des
    → 期望推力向量 f_des = m·(a_des + g·e₃)
    → 推力 T = ‖f_des‖
    → 期望姿态 R_des (由 f_des 方向构建)
    → SO(3) 姿态误差 → 角速度指令 → 力矩 τ
    → 积分更新 R, ω

无倾角限制。

注意: 纯PyTorch实现, 不依赖Isaac Sim。
"""

import torch
import math
from typing import Tuple, Optional


def hat(v: torch.Tensor) -> torch.Tensor:
    """向量 → 反对称矩阵 [ω]×
    v: [..., 3] → [..., 3, 3]
    """
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z,         -v[..., 2],  v[..., 1]], dim=-1),
        torch.stack([v[..., 2],  z,          -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0],  z],          dim=-1),
    ], dim=-2)


def vee(M: torch.Tensor) -> torch.Tensor:
    """反对称矩阵 → 向量 (hat的逆)
    M: [..., 3, 3] → [..., 3]
    """
    return torch.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], dim=-1)


class Quadrotor6DOF:
    """
    非线性六自由度四旋翼模型

    状态: pos(3), vel(3), R(3×3), omega(3)
    控制输入: 期望加速度 a_des → 内部SO(3)控制器计算推力+力矩

    用法:
        quad = Quadrotor6DOF(num_envs=10, device='cpu')
        quad.reset(env_ids)

        # 每步:
        # 1. 外环算期望加速度
        a_des = apf_force / mass + strategy_force / mass
        # 2. 6DOF积分 (内部SO3控制)
        quad.step(a_des, dt=0.01)
        # 3. 读取状态
        pos = quad.pos   # [N, 3]
        vel = quad.vel   # [N, 3]
    """

    def __init__(
        self,
        num_envs: int,
        device: str = "cpu",
        mass: float = 1.0,            # kg
        J_diag: Tuple[float, float, float] = (0.01, 0.01, 0.02),  # kg·m²
        gravity: float = 9.81,        # m/s²
        max_thrust: float = 25.0,     # N (约2.5g)
        max_torque: float = 2.0,      # N·m
        tau_motor: float = 0.03,      # 电机时间常数 (s), 典型30ms
        k_drag: float = 0.2,          # 线性阻尼系数
        # SO(3) 控制器增益
        k_R: float = 8.0,             # 姿态比例增益
        k_omega: float = 2.5,         # 角速度阻尼增益
        # 约束
        max_speed: float = 3.0,
        arena_radius: float = 10.0,
        max_height: float = 5.0,
        min_height: float = 0.1,
    ):
        self.num_envs = num_envs
        self.device = device
        self.mass = mass
        self.gravity = gravity
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        self.tau_motor = tau_motor
        self.k_drag = k_drag
        self.k_R = k_R
        self.k_omega = k_omega
        self.max_speed = max_speed
        self.arena_radius = arena_radius
        self.max_height = max_height
        self.min_height = min_height

        # 惯性矩阵 (对角)
        self.J = torch.diag(torch.tensor(J_diag, dtype=torch.float32, device=device))  # [3, 3]
        self.J_inv = torch.diag(1.0 / torch.tensor(J_diag, dtype=torch.float32, device=device))

        # 重力向量
        self.e3 = torch.tensor([0., 0., 1.], dtype=torch.float32, device=device)
        self.g_vec = torch.tensor([0., 0., -gravity], dtype=torch.float32, device=device)

        # 状态
        self.pos = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.vel = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.R = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(num_envs, -1, -1).clone()
        self.omega = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)

        # 电机实际输出 (经过一阶延迟)
        self.T_actual = torch.full((num_envs, 1), mass * gravity,
                                    dtype=torch.float32, device=device)  # 初始悬停推力
        self.tau_actual = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)

        # 期望偏航 (用于构建R_des的参考方向)
        self.yaw_ref = torch.zeros(num_envs, dtype=torch.float32, device=device)

    def reset(self, env_ids: torch.Tensor, pos: Optional[torch.Tensor] = None,
              vel: Optional[torch.Tensor] = None):
        """重置指定环境"""
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        if pos is not None:
            self.pos[env_ids] = pos
        else:
            self.pos[env_ids] = 0.0
        if vel is not None:
            self.vel[env_ids] = vel
        else:
            self.vel[env_ids] = 0.0
        self.R[env_ids] = torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1, -1)
        self.omega[env_ids] = 0.0
        self.T_actual[env_ids] = self.mass * self.gravity  # 悬停推力
        self.tau_actual[env_ids] = 0.0

        # 如果有初速度, 设置初始偏航指向速度方向
        if vel is not None:
            sp = torch.norm(vel[:, :2], dim=-1)
            valid = sp > 0.1
            if valid.any():
                self.yaw_ref[env_ids[valid]] = torch.atan2(vel[valid, 1], vel[valid, 0])

    def _build_R_des(self, f_des: torch.Tensor) -> torch.Tensor:
        """
        从期望推力向量构建期望旋转矩阵 R_des

        b3_des = f_des / |f_des|   (机体z轴 = 推力方向)
        b1_ref = [cos(yaw), sin(yaw), 0]   (参考前进方向)
        b2_des = b3_des × b1_ref / |...|   (机体y轴)
        b1_des = b2_des × b3_des           (机体x轴)

        Args:
            f_des: [N, 3] 期望推力向量 (惯性系)
        Returns:
            R_des: [N, 3, 3]
        """
        N = f_des.shape[0]

        f_norm = torch.norm(f_des, dim=-1, keepdim=True).clamp(min=1e-6)
        b3_des = f_des / f_norm  # [N, 3]

        # 参考前进方向 (由yaw_ref决定)
        cos_y = torch.cos(self.yaw_ref[:N])
        sin_y = torch.sin(self.yaw_ref[:N])
        b1_ref = torch.stack([cos_y, sin_y, torch.zeros_like(cos_y)], dim=-1)  # [N, 3]

        # b2 = b3 × b1 (归一化)
        b2_des = torch.cross(b3_des, b1_ref, dim=-1)
        b2_norm = torch.norm(b2_des, dim=-1, keepdim=True).clamp(min=1e-6)
        b2_des = b2_des / b2_norm

        # b1 = b2 × b3
        b1_des = torch.cross(b2_des, b3_des, dim=-1)

        # 组装 R_des = [b1 | b2 | b3]
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)  # [N, 3, 3]

        return R_des

    def step(self, a_des: torch.Tensor, dt: float = 0.01):
        """
        给定期望加速度, 执行一步6-DOF仿真

        控制链:
            a_des → f_des = m·(a_des + g·e₃)
            → T = b₃ᵀ·f_des   (当前推力)
            → R_des (从f_des方向构建)
            → e_R, e_ω (SO3误差)
            → τ = -k_R·e_R - k_ω·e_ω + ω×Jω
            → 动力学积分

        Args:
            a_des: [N, 3] 期望加速度 (惯性系)
            dt: 时间步
        """
        N = self.num_envs

        # ===== 1. 外环: 期望推力向量 =====
        # f_des = m·(a_des + g·e₃)  = m·a_des + [0,0,mg]
        f_des = self.mass * (a_des + self.gravity * self.e3.unsqueeze(0))  # [N, 3]

        # ===== 2. 推力指令 =====
        # 当前机体z轴
        b3 = self.R[:, :, 2]  # [N, 3] — R的第三列
        T_cmd = (f_des * b3).sum(dim=-1, keepdim=True)  # [N, 1] 投影到当前推力方向
        T_cmd = T_cmd.clamp(0.0, self.max_thrust)  # 推力不能为负

        # 电机一阶延迟: Ṫ = (T_cmd - T_actual) / τ_motor
        self.T_actual = self.T_actual + (T_cmd - self.T_actual) * (dt / self.tau_motor)
        self.T_actual = self.T_actual.clamp(0.0, self.max_thrust)
        T = self.T_actual

        # ===== 3. 期望姿态 R_des =====
        R_des = self._build_R_des(f_des)  # [N, 3, 3]

        # 更新偏航参考 (跟踪当前速度方向)
        sp = torch.norm(self.vel[:, :2], dim=-1)
        valid_yaw = sp > 0.3
        if valid_yaw.any():
            self.yaw_ref[valid_yaw] = torch.atan2(
                self.vel[valid_yaw, 1], self.vel[valid_yaw, 0])

        # ===== 4. SO(3) 姿态误差 =====
        # e_R = 0.5 * vee(R_des^T R - R^T R_des)
        RdT_R = torch.bmm(R_des.transpose(-2, -1), self.R)   # [N, 3, 3]
        RT_Rd = torch.bmm(self.R.transpose(-2, -1), R_des)   # [N, 3, 3]
        e_R = 0.5 * vee(RdT_R - RT_Rd)  # [N, 3]

        # e_ω = ω (简化: 不跟踪期望角速度, 直接阻尼)
        e_omega = self.omega  # [N, 3]

        # ===== 5. 力矩指令 =====
        # τ_cmd = -k_R·e_R - k_ω·e_ω + ω × J·ω
        Jw = torch.matmul(self.J.unsqueeze(0), self.omega.unsqueeze(-1)).squeeze(-1)  # [N, 3]
        tau_cmd = -self.k_R * e_R - self.k_omega * e_omega + torch.cross(self.omega, Jw, dim=-1)
        # 限幅
        tau_norm = torch.norm(tau_cmd, dim=-1, keepdim=True).clamp(min=1e-6)
        tau_cmd = tau_cmd * (self.max_torque / tau_norm).clamp(max=1.0)

        # 电机一阶延迟: τ̇ = (τ_cmd - τ_actual) / τ_motor
        self.tau_actual = self.tau_actual + (tau_cmd - self.tau_actual) * (dt / self.tau_motor)
        tau_actual_norm = torch.norm(self.tau_actual, dim=-1, keepdim=True).clamp(min=1e-6)
        self.tau_actual = self.tau_actual * (self.max_torque / tau_actual_norm).clamp(max=1.0)
        tau = self.tau_actual

        # ===== 6. 动力学积分 =====
        # 平动: mv̇ = -mg·e₃ + R·[0,0,T]ᵀ - k_drag·v
        thrust_world = b3 * T  # [N, 3]
        acc = self.g_vec.unsqueeze(0) + thrust_world / self.mass - self.k_drag * self.vel  # [N, 3]
        self.vel = self.vel + acc * dt
        # 限速
        speed = torch.norm(self.vel, dim=-1, keepdim=True).clamp(min=1e-6)
        self.vel = self.vel * (self.max_speed / speed).clamp(max=1.0)
        self.pos = self.pos + self.vel * dt

        # 转动: Ṙ = R·[ω]×, J·ω̇ = -ω×Jω + τ
        omega_hat = hat(self.omega)  # [N, 3, 3]
        R_dot = torch.bmm(self.R, omega_hat)  # [N, 3, 3]
        self.R = self.R + R_dot * dt

        # 重正交化 R (防止数值漂移)
        self.R = self._reorthogonalize(self.R)

        alpha = torch.matmul(self.J_inv.unsqueeze(0),
                             (-torch.cross(self.omega, Jw, dim=-1) + tau).unsqueeze(-1)).squeeze(-1)
        self.omega = self.omega + alpha * dt

        # ===== 7. 位置约束 =====
        xy_dist = torch.norm(self.pos[:, :2], dim=-1)
        out_arena = xy_dist > self.arena_radius
        if out_arena.any():
            self.pos[out_arena, :2] = (
                self.pos[out_arena, :2] /
                xy_dist[out_arena].unsqueeze(-1) * self.arena_radius
            )
            # 反弹: 消除径向速度
            r_hat_2d = self.pos[out_arena, :2] / xy_dist[out_arena].unsqueeze(-1).clamp(min=1e-6)
            v_radial = (self.vel[out_arena, :2] * r_hat_2d).sum(dim=-1, keepdim=True)
            self.vel[out_arena, :2] -= v_radial.clamp(min=0) * r_hat_2d

        self.pos[:, 2] = self.pos[:, 2].clamp(self.min_height, self.max_height)
        at_floor = self.pos[:, 2] <= self.min_height + 0.01
        self.vel[at_floor, 2] = self.vel[at_floor, 2].clamp(min=0)
        at_ceil = self.pos[:, 2] >= self.max_height - 0.01
        self.vel[at_ceil, 2] = self.vel[at_ceil, 2].clamp(max=0)

    def _reorthogonalize(self, R: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt重正交化, 保证 R ∈ SO(3)"""
        b1 = R[:, :, 0]
        b2 = R[:, :, 1]
        b3 = R[:, :, 2]

        # 从b3开始 (推力方向最重要)
        b3 = b3 / torch.norm(b3, dim=-1, keepdim=True).clamp(min=1e-8)
        b2 = b2 - (b2 * b3).sum(dim=-1, keepdim=True) * b3
        b2 = b2 / torch.norm(b2, dim=-1, keepdim=True).clamp(min=1e-8)
        b1 = torch.cross(b2, b3, dim=-1)

        return torch.stack([b1, b2, b3], dim=-1)

    def get_euler_angles(self) -> torch.Tensor:
        """提取ZYX欧拉角 (roll, pitch, yaw) [N, 3]"""
        R = self.R
        pitch = -torch.asin(R[:, 2, 0].clamp(-1, 1))
        roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        return torch.stack([roll, pitch, yaw], dim=-1)

    def get_body_rates(self) -> torch.Tensor:
        """返回机体角速度 [N, 3]"""
        return self.omega.clone()
