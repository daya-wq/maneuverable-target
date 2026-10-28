"""
Cooperative Future Token Encoder — 协同未来 Token 编码器

将三架追捕无人机的历史状态、当前团队状态、上一周期规划轨迹和
冻结的双意图、单轨迹预测结果编码为协同未来表示。

编码器输出 cooperative_future_tokens [B, N, H, D]。本文件同时提供无参数的
BoundedTrajectoryReferenceDecoder；随机轨迹 Head 由 PPO Actor 统一创建，避免在编码器
和策略网络里维护两套输出头。

设计文档: docs/网络结构.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 配置
# ============================================================================

@dataclass
class CooperativeFutureConfig:
    """协同未来编码器的超参数。"""

    num_agents: int = 3       # N: 追捕无人机数量
    num_modes: int = 1        # singleton trajectory axis retained for adapters
    num_strategic: int = 2    # Goal_Rush / Evade

    history_len: int = 10     # L: 历史窗口长度
    pred_horizon: int = 10    # H: 预测/规划时域

    history_dim: int = 9      # F_hist: 单帧历史状态维度 (相对位置3+相对速度3+加速度3)
    prev_plan_dim: int = 6    # F_plan: 上一周期规划维度 (参考位置3+参考速度3)

    d_model: int = 128        # D: 主特征维度
    num_heads: int = 4        # 注意力头数
    ff_dim: int = 256         # FFN 宽度

    history_layers: int = 2        # History Transformer 层数
    coordination_layers: int = 2   # 协同解码器层数

    dropout: float = 0.0           # PPO 策略建议 0.0
    eps: float = 1e-6

    # 功能开关
    use_agent_interaction: bool = True
    use_previous_plan: bool = True
    use_future_agent_attention: bool = True

    # 默认关闭，只用于后续消融
    use_self_history_cross_attention: bool = False


class KinematicTrajectoryProjector(nn.Module):
    """Project a bounded acceleration chunk into a feasible local trajectory.

    PPO samples normalized accelerations in ``[-1, 1]``.  This deterministic
    module applies axis-wise acceleration limits, integrates velocity and
    position, clips speed, and optionally keeps positions inside a workspace.
    Relative positions are anchored at the current vehicle position, which
    gives the policy translation-invariant plan memory.
    """

    def __init__(
        self,
        dt: float,
        max_speed: float,
        accel_limits=(5.0, 5.0, 6.5),
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.register_buffer(
            "accel_limits",
            torch.as_tensor(accel_limits, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        normalized_accel: torch.Tensor,
        initial_pos: torch.Tensor,
        initial_vel: torch.Tensor,
        position_min: Optional[torch.Tensor] = None,
        position_max: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if normalized_accel.ndim < 3 or normalized_accel.shape[-1] != 3:
            raise ValueError("normalized_accel must have shape [..., H, 3]")
        if initial_pos.shape != normalized_accel.shape[:-2] + (3,):
            raise ValueError("initial_pos shape must match normalized_accel batch dimensions")
        if initial_vel.shape != initial_pos.shape:
            raise ValueError("initial_vel shape must match initial_pos")

        accel = normalized_accel.clamp(-1.0, 1.0) * self.accel_limits.to(
            device=normalized_accel.device,
            dtype=normalized_accel.dtype,
        )
        pos = initial_pos
        vel = initial_vel
        positions = []
        velocities = []
        for step in range(normalized_accel.shape[-2]):
            next_vel = vel + accel[..., step, :] * self.dt
            speed = next_vel.norm(dim=-1, keepdim=True)
            next_vel = next_vel * (
                self.max_speed / speed.clamp_min(1e-6)
            ).clamp(max=1.0)
            next_pos = pos + 0.5 * (vel + next_vel) * self.dt
            if position_min is not None and position_max is not None:
                next_pos = torch.maximum(
                    torch.minimum(next_pos, position_max), position_min
                )
            positions.append(next_pos)
            velocities.append(next_vel)
            pos, vel = next_pos, next_vel

        positions_world = torch.stack(positions, dim=-2)
        velocities_world = torch.stack(velocities, dim=-2)
        positions_relative = positions_world - initial_pos.unsqueeze(-2)
        return {
            "accelerations": accel,
            "positions_world": positions_world,
            "positions_relative": positions_relative,
            "velocities": velocities_world,
            "plan": torch.cat([positions_relative, velocities_world], dim=-1),
        }


class BoundedTrajectoryReferenceDecoder(nn.Module):
    """Decode bounded PPO outputs into relative-position/velocity references.

    Legacy ``position_velocity``: each token produces three position residual
    controls and three velocity controls. Position references are centred on
    constant-velocity motion and their residual is bounded by the displacement
    reachable under the configured axis acceleration limits. Velocity
    references are smoothly mapped inside ``max_speed`` (no hard speed clip).
    Independent endpoint bounds do NOT bound the Hermite acceleration between
    endpoints. Keep this mode solely for reproducible older experiments.

    ``acceleration_sequence`` uses three bounded controls per future token and
    integrates consistent position/velocity knots. A six-control compatibility
    ablation averages the two 3-vectors; the native head has no redundant pair.

    ``bounded_position_velocity`` directly predicts independent position
    increments and endpoint velocities. Their vector norms are bounded by
    dt*max_speed and max_speed. Soft P/V consistency is optimized by MAPPO;
    these are reference commands, not guaranteed dynamically reachable states.

    The returned plan is translation invariant: ``[..., :3]`` is relative to
    the position at planning time and ``[..., 3:]`` is world-frame velocity.
    """

    def __init__(
        self,
        dt: float,
        max_speed: float,
        accel_limits=(5.0, 5.0, 6.5),
        reference_mode: str = "position_velocity",
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        if reference_mode not in ("position_velocity", "acceleration_sequence", "bounded_position_velocity"):
            raise ValueError("Unknown trajectory reference_mode: " + reference_mode)
        self.reference_mode = reference_mode
        self.register_buffer(
            "accel_limits",
            torch.as_tensor(accel_limits, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _smooth_norm_limit(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
        """Keep the norm bounded while leaving the low-speed region unchanged."""
        norm = vector.norm(dim=-1, keepdim=True)
        limit = max(float(max_norm), 1e-6)
        # A high-order soft limiter is almost the identity for norm << limit,
        # unlike tanh which noticeably slows a vehicle before it reaches its
        # physical speed bound. Its output norm is still strictly below limit.
        order = 16.0
        scale = (1.0 + (norm / limit).pow(order)).pow(-1.0 / order)
        return vector * scale

    def forward(
        self,
        normalized_reference: torch.Tensor,
        initial_pos: torch.Tensor,
        initial_vel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        valid_dims = (3, 6) if self.reference_mode == "acceleration_sequence" else (6,)
        if normalized_reference.ndim < 3 or normalized_reference.shape[-1] not in valid_dims:
            raise ValueError("normalized_reference must be [..., H, C] with C in %r" % (valid_dims,))
        if initial_pos.shape != normalized_reference.shape[:-2] + (3,):
            raise ValueError("initial_pos shape must match action batch dimensions")
        if initial_vel.shape != initial_pos.shape:
            raise ValueError("initial_vel shape must match initial_pos")

        # TanhNormal already supplies values in [-1, 1]. The clamp is only a
        # numerical guard for externally supplied deployment commands.
        command = normalized_reference.clamp(-1.0, 1.0)
        if self.reference_mode == "acceleration_sequence":
            return self._decode_acceleration_sequence(command, initial_pos, initial_vel)
        if self.reference_mode == "bounded_position_velocity":
            increments, velocities = self.bounded_pv_controls(command)
            relative = (increments * (self.dt * self.max_speed)).cumsum(dim=-2)
            velocities = velocities * self.max_speed
            return {
                "positions_world": initial_pos.unsqueeze(-2) + relative,
                "positions_relative": relative,
                "velocities": velocities,
                "plan": torch.cat([relative, velocities], dim=-1),
            }
        pos_command, vel_command = command.split(3, dim=-1)
        horizon = normalized_reference.shape[-2]
        tau = (
            torch.arange(1, horizon + 1, device=command.device, dtype=command.dtype)
            * self.dt
        )
        tau = tau.view(*([1] * (command.ndim - 2)), horizon, 1)
        accel_limits = self.accel_limits.to(device=command.device, dtype=command.dtype)

        nominal_relative_pos = initial_vel.unsqueeze(-2) * tau
        reachable_residual = 0.5 * accel_limits * tau.square()
        positions_relative = nominal_relative_pos + pos_command * reachable_residual
        positions_world = initial_pos.unsqueeze(-2) + positions_relative

        velocity_raw = initial_vel.unsqueeze(-2) + vel_command * accel_limits * tau
        velocities = self._smooth_norm_limit(velocity_raw, self.max_speed)
        return {
            "positions_world": positions_world,
            "positions_relative": positions_relative,
            "velocities": velocities,
            "plan": torch.cat([positions_relative, velocities], dim=-1),
        }

    @staticmethod
    def bounded_pv_controls(command):
        """Two independent 3D unit-ball references, not acceleration controls."""
        position, velocity = command.split(3, dim=-1)
        return (BoundedTrajectoryReferenceDecoder._smooth_norm_limit(position, 1.),
                BoundedTrajectoryReferenceDecoder._smooth_norm_limit(velocity, 1.))

    @staticmethod
    def pv_consistency_loss(command):
        """Dimensionless trapezoidal consistency between consecutive future knots.

        First knot is excluded because this helper needs no privileged/current
        world velocity. This is a soft objective, not dynamic feasibility.
        """
        displacement_rate, velocity = BoundedTrajectoryReferenceDecoder.bounded_pv_controls(command)
        if command.shape[-2] < 2:
            return command.sum() * 0.
        return (displacement_rate[..., 1:, :] -
                .5 * (velocity[..., :-1, :] + velocity[..., 1:, :])).square().mean()

    def _decode_acceleration_sequence(self, command, initial_pos, initial_vel):
        # The mean of two [-1,1] controls is still bounded. It removes the
        # contradictory independently-chosen position and velocity endpoints.
        proposal = command if command.shape[-1] == 3 else 0.5 * (command[..., :3] + command[..., 3:])
        limits = self.accel_limits.to(device=command.device, dtype=command.dtype)
        velocity = initial_vel
        displacement = torch.zeros_like(initial_pos)
        positions, velocities, accelerations = [], [], []
        for step in range(command.shape[-2]):
            target_velocity = self._smooth_norm_limit(
                velocity + self.dt * limits * proposal[..., step, :], self.max_speed)
            delta = target_velocity - velocity
            # Smoothly constrain dv/dt to an acceleration ellipsoid. The new
            # velocity is a convex combination of the old velocity and a point
            # inside the speed ball. Thus a valid initial speed stays bounded.
            normalized_delta = delta / (self.dt * limits)
            scale = (1.0 + normalized_delta.norm(dim=-1, keepdim=True).pow(16.0)).pow(-1.0 / 16.0)
            delta = delta * scale
            next_velocity = velocity + delta
            acceleration = delta / self.dt
            displacement = displacement + 0.5 * self.dt * (velocity + next_velocity)
            positions.append(displacement)
            velocities.append(next_velocity)
            accelerations.append(acceleration)
            velocity = next_velocity
        relative = torch.stack(positions, dim=-2)
        velocities = torch.stack(velocities, dim=-2)
        return {
            "positions_world": initial_pos.unsqueeze(-2) + relative,
            "positions_relative": relative,
            "velocities": velocities,
            "accelerations": torch.stack(accelerations, dim=-2),
            "plan": torch.cat([relative, velocities], dim=-1),
        }

    @staticmethod
    def sample_constant_acceleration(initial_pos, initial_vel, endpoint_vel, elapsed, duration):
        """Exact first-segment sample; p/v agree and acceleration is constant.

        This gives kinematic bounds, not a guarantee of exact actuator tracking
        or bounded jerk. Thrust/tilt and feedback safety remain physical checks.
        """
        duration = max(float(duration), 1e-6)
        elapsed = min(max(float(elapsed), 0.0), duration)
        acceleration = (endpoint_vel - initial_vel) / duration
        return {
            "position": initial_pos + initial_vel * elapsed + 0.5 * acceleration * elapsed ** 2,
            "velocity": initial_vel + acceleration * elapsed,
            "acceleration": acceleration,
        }

    @staticmethod
    def sample_cubic_hermite(
        initial_pos: torch.Tensor,
        initial_vel: torch.Tensor,
        endpoint_pos: torch.Tensor,
        endpoint_vel: torch.Tensor,
        elapsed: float,
        duration: float,
    ) -> Dict[str, torch.Tensor]:
        """Sample position, velocity and feed-forward acceleration on one segment."""
        duration = max(float(duration), 1e-6)
        s = min(max(float(elapsed) / duration, 0.0), 1.0)
        s2, s3 = s * s, s * s * s

        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2
        position = (
            h00 * initial_pos
            + h10 * duration * initial_vel
            + h01 * endpoint_pos
            + h11 * duration * endpoint_vel
        )

        dh00 = 6.0 * s2 - 6.0 * s
        dh10 = 3.0 * s2 - 4.0 * s + 1.0
        dh01 = -6.0 * s2 + 6.0 * s
        dh11 = 3.0 * s2 - 2.0 * s
        velocity = (
            dh00 * initial_pos
            + dh10 * duration * initial_vel
            + dh01 * endpoint_pos
            + dh11 * duration * endpoint_vel
        ) / duration

        ddh00 = 12.0 * s - 6.0
        ddh10 = 6.0 * s - 4.0
        ddh01 = -12.0 * s + 6.0
        ddh11 = 6.0 * s - 2.0
        acceleration = (
            ddh00 * initial_pos
            + ddh10 * duration * initial_vel
            + ddh01 * endpoint_pos
            + ddh11 * duration * endpoint_vel
        ) / (duration * duration)
        return {
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
        }


class StrategicIntentTokenEncoder(nn.Module):
    """Turn a strategic probability distribution into one differentiable token."""

    def __init__(self, cfg: CooperativeFutureConfig):
        super().__init__()
        self.cfg = cfg
        self.intent_embeddings = nn.Parameter(
            torch.randn(cfg.num_strategic, cfg.d_model) * 0.02
        )
        self.output_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, strategic_probs: torch.Tensor) -> torch.Tensor:
        if strategic_probs.shape[-1] != self.cfg.num_strategic:
            raise ValueError(
                f"strategic_probs last dimension must be {self.cfg.num_strategic}"
            )
        probs = strategic_probs.clamp_min(0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        return self.output_norm(probs @ self.intent_embeddings)


# ============================================================================
# 1. Agent 历史编码器
# ============================================================================

class AgentHistoryEncoder(nn.Module):
    """
    将每架无人机的历史状态序列编码为 agent_tokens 和 history_tokens。

    Input:
        agent_history:      [B, N, L, F_hist]
        agent_history_mask: [B, N, L] or None

    Output:
        agent_tokens:   [B, N, D]     每架无人机的历史摘要
        history_tokens: [B, N, L, D]  每个历史时间点的 token (保留用于诊断)
    """

    def __init__(self, cfg: CooperativeFutureConfig):
        super().__init__()
        self.cfg = cfg
        dm = cfg.d_model

        # 历史点编码: F_hist → D
        self.history_point_encoder = nn.Sequential(
            nn.Linear(cfg.history_dim, dm),
            nn.SiLU(),
            nn.Linear(dm, dm),
        )

        # 过去时间嵌入 (可学习)
        self.past_time_embedding = nn.Parameter(
            torch.randn(1, 1, cfg.history_len, dm) * 0.02
        )

        # 可学习的 [AGENT] 摘要 token (三架无人机共享)
        self.agent_token = nn.Parameter(torch.zeros(1, 1, 1, dm))
        nn.init.trunc_normal_(self.agent_token, std=0.02)

        # History Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dm,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.history_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        agent_history: torch.Tensor,
        agent_history_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, L, _ = agent_history.shape
        dm = self.cfg.d_model

        # 1. 逐点编码
        # [B, N, L, F_hist] → [B*N, L, F_hist] → [B*N, L, D]
        x = agent_history.reshape(B * N, L, self.cfg.history_dim)
        x = self.history_point_encoder(x)  # [B*N, L, D]

        # 2. 加入过去时间嵌入
        # past_time_embedding: [1, 1, L, D] → squeeze to [L, D] → add
        time_emb = self.past_time_embedding[:, :, :L, :]  # [1, 1, L, D]
        x = x + time_emb.squeeze(0).squeeze(0)  # [B*N, L, D]

        # 3. 拼接 [AGENT] token
        # agent_token: [1, 1, 1, D] → squeeze → [1, 1, D] → expand → [B*N, 1, D]
        agent_tok = self.agent_token.squeeze(1).expand(B * N, 1, dm)
        x = torch.cat([agent_tok, x], dim=1)  # [B*N, 1+L, D]

        # 4. 构造 mask
        if agent_history_mask is not None:
            # [B, N, L] → [B*N, L]
            src_mask = agent_history_mask.reshape(B * N, L)
            # 扩展: [AGENT] token 始终有效
            agent_mask = torch.ones(B * N, 1, device=x.device, dtype=src_mask.dtype)
            src_key_padding_mask = torch.cat([agent_mask, src_mask], dim=1)
            # 转为 bool mask (True = 被屏蔽)
            src_key_padding_mask = ~src_key_padding_mask.bool()
        else:
            src_key_padding_mask = None

        # 5. Transformer
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # [B*N, 1+L, D]

        # 6. 拆分
        agent_tokens = out[:, 0, :].reshape(B, N, dm)      # [B, N, D]
        history_tokens = out[:, 1:, :].reshape(B, N, L, dm)  # [B, N, L, D]

        return {
            "agent_tokens": agent_tokens,
            "history_tokens": history_tokens,
        }


# ============================================================================
# 2. 状态级三机交互 (带关系偏置的注意力)
# ============================================================================

class RelationalAgentInteraction(nn.Module):
    """
    基于当前几何关系，让 agent_tokens 之间进行交互。

    Input:
        agent_tokens: [B, N, D]
        current_pos:  [B, N, 3]
        current_vel:  [B, N, 3]

    Output:
        cooperative_agent_tokens: [B, N, D]
        attention: [B, num_heads, N, N]
    """

    def __init__(self, cfg: CooperativeFutureConfig):
        super().__init__()
        self.cfg = cfg
        dm = cfg.d_model
        nh = cfg.num_heads
        self.num_heads = nh
        self.head_dim = dm // nh

        # 注意力投影
        self.q_proj = nn.Linear(dm, dm, bias=False)
        self.k_proj = nn.Linear(dm, dm, bias=False)
        self.v_proj = nn.Linear(dm, dm, bias=False)
        self.out_proj = nn.Linear(dm, dm, bias=False)

        # 边特征编码器: r_ij (7维) → num_heads 个标量偏置
        self.edge_encoder = nn.Sequential(
            nn.Linear(7, dm),
            nn.SiLU(),
            nn.Linear(dm, nh),
        )

        self.norm = nn.LayerNorm(dm)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

    def _compute_edge_features(
        self,
        pos: torch.Tensor,  # [B, N, 3]
        vel: torch.Tensor,  # [B, N, 3]
    ) -> torch.Tensor:
        """计算边特征 r_ij = [p_j-p_i, v_j-v_i, ||p_j-p_i||_2] ∈ R^7"""
        B, N, _ = pos.shape
        pi = pos.unsqueeze(-2)   # [B, N, 1, 3]
        pj = pos.unsqueeze(-3)   # [B, 1, N, 3]
        vi = vel.unsqueeze(-2)   # [B, N, 1, 3]
        vj = vel.unsqueeze(-3)   # [B, 1, N, 3]

        dp = pj - pi              # [B, N, N, 3]
        dv = vj - vi              # [B, N, N, 3]
        dist = dp.norm(dim=-1, keepdim=True)  # [B, N, N, 1]

        return torch.cat([dp, dv, dist], dim=-1)  # [B, N, N, 7]

    def forward(
        self,
        agent_tokens: torch.Tensor,
        current_pos: torch.Tensor,
        current_vel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, N, dm = agent_tokens.shape
        nh = self.num_heads
        hd = self.head_dim

        # 计算边特征 & 关系偏置
        edge_feat = self._compute_edge_features(current_pos, current_vel)  # [B, N, N, 7]
        edge_bias = self.edge_encoder(edge_feat)  # [B, N, N, nh]

        # Q/K/V 投影并拆分多头
        q = self.q_proj(agent_tokens).reshape(B, N, nh, hd).transpose(1, 2)  # [B, nh, N, hd]
        k = self.k_proj(agent_tokens).reshape(B, N, nh, hd).transpose(1, 2)  # [B, nh, N, hd]
        v = self.v_proj(agent_tokens).reshape(B, N, nh, hd).transpose(1, 2)  # [B, nh, N, hd]

        # 注意力分数: s_ij = (QK^T)/√d_h + b_ij
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(hd)  # [B, nh, N, N]
        scores = scores + edge_bias.permute(0, 3, 1, 2)                # [B, nh, N, N]

        attn = torch.softmax(scores, dim=-1)  # [B, nh, N, N]

        # 加权聚合
        context = torch.einsum("bhqk,bhkd->bhqd", attn, v)  # [B, nh, N, hd]
        context = context.transpose(1, 2).reshape(B, N, dm)  # [B, N, D]

        out = self.out_proj(context)

        # 残差 + LayerNorm
        cooperative_tokens = self.norm(agent_tokens + out)

        return {
            "agent_tokens": cooperative_tokens,
            "attention": attn,  # [B, nh, N, N]
        }


# ============================================================================
# 3. 上一周期规划记忆编码器
# ============================================================================

class PreviousPlanEncoder(nn.Module):
    """
    将上一周期的规划轨迹编码为 token 序列。

    Input:
        prev_plan:      [B, N, H, F_plan]
        prev_plan_mask: [B, N, H]

    Output:
        prev_plan_tokens: [B, N, H, D]
    """

    def __init__(self, cfg: CooperativeFutureConfig):
        super().__init__()
        self.cfg = cfg
        dm = cfg.d_model

        self.prev_plan_encoder = nn.Sequential(
            nn.Linear(cfg.prev_plan_dim, dm),
            nn.SiLU(),
            nn.Linear(dm, dm),
        )

        # "是否有效" Embedding
        self.prev_valid_embedding = nn.Parameter(torch.zeros(1, 1, 1, dm))
        nn.init.trunc_normal_(self.prev_valid_embedding, std=0.02)

    def forward(
        self,
        prev_plan: torch.Tensor,
        prev_plan_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 编码
        tokens = self.prev_plan_encoder(prev_plan)  # [B, N, H, D]

        # 应用有效掩码
        mask_expanded = prev_plan_mask.unsqueeze(-1).to(tokens.dtype)
        tokens = tokens * mask_expanded

        # 加入有效 Embedding
        tokens = tokens + self.prev_valid_embedding * mask_expanded

        return tokens  # [B, N, H, D]


# ============================================================================
# 4. Future Query 构造器
# ============================================================================

class FutureQueryBuilder(nn.Module):
    """
    从协同 agent token 和上一周期规划记忆构造 H 个 Future Query。

    q_{i,h}^0 = W_a * ā_i + e_h^{future} + W_p * ẽ_{i,h}^{prev}

    Input:
        cooperative_agent_tokens: [B, N, D]
        prev_plan_tokens:         [B, N, H, D]

    Output:
        future_queries: [B, N, H, D]
    """

    def __init__(self, cfg: CooperativeFutureConfig):
        super().__init__()
        self.cfg = cfg
        dm = cfg.d_model
        H = cfg.pred_horizon

        self.agent_query_proj = nn.Linear(dm, dm, bias=False)
        self.prev_query_proj = nn.Linear(dm, dm, bias=False)

        # 未来时间嵌入
        self.future_time_embedding = nn.Parameter(
            torch.randn(1, 1, H, dm) * 0.02
        )

        self.query_norm = nn.LayerNorm(dm)

    def forward(
        self,
        cooperative_agent_tokens: torch.Tensor,
        prev_plan_tokens: torch.Tensor,
    ) -> torch.Tensor:
        H = self.cfg.pred_horizon

        # agent_base: [B, N, 1, D]
        agent_base = self.agent_query_proj(cooperative_agent_tokens).unsqueeze(2)

        # future_time: [1, 1, H, D]
        future_time = self.future_time_embedding[:, :, :H, :]

        # prev_context: [B, N, H, D]
        prev_context = self.prev_query_proj(prev_plan_tokens)

        queries = self.query_norm(agent_base + future_time + prev_context)

        return queries  # [B, N, H, D]


# ============================================================================
# 5. 时间对齐的 GMM Cross-Attention
# ============================================================================

class TimeAlignedTrajectoryFusion(nn.Module):
    """Fuse each agent query with the target token at exactly the same time.

    There is one target trajectory, so mode-selection Q/K projections, softmax
    and mixture-weight priors have no role. Temporal/team attention stays in
    the following coordination blocks.
    """
    def __init__(self, cfg: CooperativeFutureConfig):
        super().__init__()
        self.cfg = cfg
        self.fuse = nn.Sequential(
            nn.LayerNorm(2 * cfg.d_model),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, queries, trajectory_tokens, point_tokens):
        if trajectory_tokens.shape[1] != 1 or point_tokens.shape[1] != 1:
            raise ValueError("TimeAlignedTrajectoryFusion requires one predicted trajectory")
        target = point_tokens[:, 0] + trajectory_tokens[:, 0, None, :]
        target = target[:, None].expand(-1, queries.shape[1], -1, -1)
        fused = self.norm(queries + self.fuse(torch.cat((queries, target), dim=-1)))
        return {"queries": fused}



class FactorizedCoordinationBlock(nn.Module):
    """
    每个 Block:
        Temporal Self-Attention (over H)
        → Future Agent Self-Attention (over N)
        → FFN

    使用 Pre-Norm 结构。
    """

    def __init__(self, cfg: CooperativeFutureConfig):
        super().__init__()
        dm = cfg.d_model
        nh = cfg.num_heads
        self.num_heads = nh
        self.head_dim = dm // nh

        # Temporal Self-Attention
        self.norm_temporal = nn.LayerNorm(dm)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dm,
            num_heads=nh,
            dropout=cfg.dropout,
            batch_first=True,
        )

        # Future Agent Self-Attention
        self.norm_agent = nn.LayerNorm(dm)
        self.future_agent_attn = nn.MultiheadAttention(
            embed_dim=dm,
            num_heads=nh,
            dropout=cfg.dropout,
            batch_first=True,
        )

        # FFN
        self.norm_ffn = nn.LayerNorm(dm)
        self.ffn = nn.Sequential(
            nn.Linear(dm, cfg.ff_dim),
            nn.GELU(),
            nn.Linear(cfg.ff_dim, dm),
        )

    def forward(
        self,
        queries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Input:
            queries: [B, N, H, D]

        Output:
            tokens: [B, N, H, D]
            temporal_attention: [B, nh, N, H, H]
            future_agent_attention: [B, nh, H, N, N]
        """
        B, N, H, dm = queries.shape

        # ── Temporal Self-Attention ──
        x = queries.reshape(B * N, H, dm)  # [B*N, H, D]
        x_norm = self.norm_temporal(x)
        attn_out, attn_weights_temporal = self.temporal_attn(
            x_norm, x_norm, x_norm, average_attn_weights=False
        )
        x = x + attn_out
        queries = x.reshape(B, N, H, dm)

        # attn_weights_temporal: [B*N, nh, H, H] → [B, nh, N, H, H]
        attn_weights_temporal = attn_weights_temporal.reshape(
            B, N, self.num_heads, H, H
        ).permute(0, 2, 1, 3, 4)

        # ── Future Agent Self-Attention ──
        x = queries.permute(0, 2, 1, 3).reshape(B * H, N, dm)  # [B*H, N, D]
        x_norm = self.norm_agent(x)
        attn_out, attn_weights_agent = self.future_agent_attn(
            x_norm, x_norm, x_norm, average_attn_weights=False
        )
        x = x + attn_out
        x = x.reshape(B, H, N, dm).permute(0, 2, 1, 3)  # [B, N, H, D]
        queries = x

        # attn_weights_agent: [B*H, nh, N, N] → [B, nh, H, N, N]
        attn_weights_agent = attn_weights_agent.reshape(
            B, H, self.num_heads, N, N
        ).permute(0, 2, 1, 3, 4)

        # ── FFN ──
        x = self.norm_ffn(queries)
        x = queries + self.ffn(x)

        return {
            "tokens": x,
            "temporal_attention": attn_weights_temporal,
            "future_agent_attention": attn_weights_agent,
        }


# ============================================================================
# 7. 总模块: CooperativeFutureTokenEncoder
# ============================================================================

class CooperativeFutureTokenEncoder(nn.Module):
    """
    协同未来 Token 编码器 — 顶层模块。

    输入:
        agent_history:      [B, N, L, F_hist]  无人机历史状态
        current_pos:        [B, N, 3]           当前相对位置
        current_vel:        [B, N, 3]           当前相对速度
        prev_plan:          [B, N, H, F_plan]   上一周期规划轨迹
        prev_plan_mask:     [B, N, H]           上一规划有效掩码
        gmm_mode_tokens:    [B, K, D]           GMM 模式 token
        gmm_point_tokens:   [B, K, H, D]        GMM 点 token
        gmm_weights:        [B, K]              GMM 权重
        strategic_probs:   [B, S]              战略意图概率
        gmm_log_weights:    [B, K] or None      GMM log 权重
        agent_history_mask: [B, N, L] or None   历史掩码

    输出字典 (key):
        cooperative_future_tokens: [B, N, H, D]
        agent_tokens:              [B, N, D]
        history_tokens:            [B, N, L, D]
        cooperative_agent_tokens:  [B, N, D]
        strategic_intent_token:    [B, D]
        initial_future_queries:    [B, N, H, D]
        gmm_fused_queries:         [B, N, H, D]
        agent_state_attention:     [B, nh, N, N]
        gmm_attention:             None（旧接口诊断键；模式选择注意力已删除）
        temporal_attention:        List[Tensor]  (每层)
        future_agent_attention:    List[Tensor]  (每层)
    """

    def __init__(self, cfg: CooperativeFutureConfig):
        super().__init__()
        self.cfg = cfg

        # 子模块
        self.history_encoder = AgentHistoryEncoder(cfg)
        self.agent_interaction = RelationalAgentInteraction(cfg)
        self.prev_plan_encoder = PreviousPlanEncoder(cfg)
        self.query_builder = FutureQueryBuilder(cfg)
        if cfg.num_modes != 1 or cfg.num_strategic != 2:
            raise ValueError("This branch's Actor requires one target trajectory and two intents")
        self.trajectory_fusion = TimeAlignedTrajectoryFusion(cfg)
        self.strategic_encoder = StrategicIntentTokenEncoder(cfg)
        self.intent_agent_norm = nn.LayerNorm(cfg.d_model)
        self.intent_query_norm = nn.LayerNorm(cfg.d_model)

        # 协同层
        self.coordination_blocks = nn.ModuleList([
            FactorizedCoordinationBlock(cfg)
            for _ in range(cfg.coordination_layers)
        ])

        # 输出 LayerNorm
        self.output_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        agent_history: torch.Tensor,
        current_pos: torch.Tensor,
        current_vel: torch.Tensor,
        prev_plan: torch.Tensor,
        prev_plan_mask: torch.Tensor,
        gmm_mode_tokens: torch.Tensor,
        gmm_point_tokens: torch.Tensor,
        gmm_weights: torch.Tensor,
        strategic_probs: torch.Tensor,
        gmm_log_weights: Optional[torch.Tensor] = None,
        agent_history_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # Legacy argument names retain an axis of length one. No mode weights
        # are used by this branch; single-target time alignment is explicit.

        # 1. 历史编码
        hist_out = self.history_encoder(agent_history, agent_history_mask)
        agent_tokens = hist_out["agent_tokens"]
        history_tokens = hist_out["history_tokens"]

        # 2. 当前状态级三机交互
        if self.cfg.use_agent_interaction:
            interaction_out = self.agent_interaction(
                agent_tokens, current_pos, current_vel
            )
            cooperative_agent_tokens = interaction_out["agent_tokens"]
            agent_state_attention = interaction_out["attention"]
        else:
            cooperative_agent_tokens = agent_tokens
            agent_state_attention = torch.zeros(
                agent_tokens.shape[0], self.cfg.num_heads,
                self.cfg.num_agents, self.cfg.num_agents,
                device=agent_tokens.device
            )

        # Preserve strategic uncertainty instead of switching on argmax.
        strategic_intent_token = self.strategic_encoder(strategic_probs)
        cooperative_agent_tokens = self.intent_agent_norm(
            cooperative_agent_tokens + strategic_intent_token.unsqueeze(1)
        )

        # 3. 上一周期规划编码
        if self.cfg.use_previous_plan:
            prev_plan_tokens = self.prev_plan_encoder(prev_plan, prev_plan_mask)
        else:
            prev_plan_tokens = torch.zeros(
                *prev_plan.shape[:-1], self.cfg.d_model,
                device=prev_plan.device, dtype=prev_plan.dtype
            )

        # 4. 建立 Future Query
        initial_queries = self.query_builder(
            cooperative_agent_tokens, prev_plan_tokens
        )
        initial_queries = self.intent_query_norm(
            initial_queries + strategic_intent_token[:, None, None, :]
        )

        # 5. Same-time fusion with the single target trajectory.
        gmm_out = self.trajectory_fusion(
            initial_queries,
            gmm_mode_tokens,
            gmm_point_tokens,
        )
        future_tokens = gmm_out["queries"]
        gmm_attention = None  # no mode-selection attention in this architecture

        # 6. 时间和三机因子化协调
        temporal_attentions = []
        future_agent_attentions = []

        for block in self.coordination_blocks:
            block_out = block(future_tokens)
            future_tokens = block_out["tokens"]
            temporal_attentions.append(block_out["temporal_attention"])
            future_agent_attentions.append(block_out["future_agent_attention"])

        future_tokens = self.output_norm(future_tokens)

        return {
            "cooperative_future_tokens": future_tokens,

            "agent_tokens": agent_tokens,
            "history_tokens": history_tokens,
            "cooperative_agent_tokens": cooperative_agent_tokens,
            "strategic_intent_token": strategic_intent_token,

            "initial_future_queries": initial_queries,
            "gmm_fused_queries": gmm_out["queries"],

            "agent_state_attention": agent_state_attention,
            "gmm_attention": gmm_attention,
            "temporal_attention": temporal_attentions,
            "future_agent_attention": future_agent_attentions,
        }
