"""
GMM 轨迹编码器 — GMMTrajectoryEncoder

将冻结预测网络的输出：
    gmm_means:   [B, K, H, 3]   目标轨迹逐点均值（相对位置差）
    gmm_vars:    [B, K, H, 3]   目标轨迹逐点三维方差
    gmm_weights: [B, K]         每条模式的概率权重（和为1）

编码为：
    mode_tokens:  [B, K, d_model]    每条轨迹的全局语义 token
    point_tokens: [B, K, H, d_model] 每条轨迹每个时间点的上下文 token
    weights:      [B, K]             归一化后的权重（供 Actor 注意力先验使用）
    log_weights:  [B, K]             log(π_k)
    variance_gate:[B, K, H, d_model] 方差门控（诊断用）
    weight_gate:  scalar             兼容诊断常量0，无权重门参数

设计原则（参见 docs/gmm_encoder_design.md）：
  - 几何信息（均值 + 位移增量 + 时间）始终保留，不被权重覆盖
  - 方差通过门控机制注入，不硬乘在几何上
  - 权重仅轻度调制轨迹级 token，强影响放在 Actor 注意力先验里
  - 三条轨迹共享 geo_encoder / unc_encoder / traj_token，无固定模式ID绑定
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class GMMEncoderConfig:
    """GMMTrajectoryEncoder 的超参数。"""

    # ── 输入维度 ──────────────────────────────────────────────────────────
    num_modes:     int = 1      # single-trajectory axis, not alternative modes
    pred_horizon:  int = 10     # H: 预测步数 (T_pred)
    coord_dim:     int = 3      # 空间维度 (x, y, z)

    # ── 模型维度 ──────────────────────────────────────────────────────────
    d_model:    int = 128       # 主特征维度
    nhead:      int = 4         # Transformer 注意力头数
    num_layers: int = 2         # Trajectory Transformer 层数
    ff_dim:     int = 256       # Transformer FFN 宽度
    dropout:    float = 0.0     # Dropout（训练冻结阶段一般设 0）

    # ── 数值安全 ──────────────────────────────────────────────────────────
    var_eps:    float = 1e-6    # 方差 clamp 下限，避免 log(0)
    weight_eps: float = 1e-6   # 权重 clamp 下限，避免 log(0)

    # ── 功能开关 ──────────────────────────────────────────────────────────
    use_variance:             bool = True  # 是否使用方差编码


# ---------------------------------------------------------------------------
# 主网络
# ---------------------------------------------------------------------------

class GMMTrajectoryEncoder(nn.Module):
    """
    将 GMM 预测输出编码为 Actor 可用的 token 序列。

    参数
    ----
    cfg : GMMEncoderConfig

    使用方式（在 PPO 训练循环中）
    ------------------------------
    # 预测网络必须冻结
    with torch.no_grad():
        gmm_out = predictor_net(obs_history)

    gmm_means   = gmm_out["gmm_means"].detach()   # [B, K, H, 3]
    gmm_vars    = gmm_out["gmm_vars"].detach()     # [B, K, H, 3]
    gmm_weights = gmm_out["gmm_weights"].detach()  # [B, K]

    encoded = gmm_encoder(gmm_means, gmm_vars, gmm_weights)
    # encoded["mode_tokens"]:  [B, K, d_model]
    # encoded["point_tokens"]: [B, K, H, d_model]
    # encoded["weights"]:      [B, K]
    # encoded["log_weights"]:  [B, K]
    """

    def __init__(self, cfg: GMMEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.num_modes != 1:
            raise ValueError("This branch encodes one predicted target trajectory, not a mixture")
        K  = cfg.num_modes
        H  = cfg.pred_horizon
        D  = cfg.coord_dim
        dm = cfg.d_model

        # ── 几何编码器：[均值(3) + 位移增量(3) + 时间(1)] → d_model ────────
        self.geo_encoder = nn.Sequential(
            nn.Linear(D * 2 + 1, dm * 2),
            nn.SiLU(),
            nn.Linear(dm * 2, dm),
        )

        # ── 方差编码器：log_std(3) → d_model ─────────────────────────────
        self.unc_encoder = nn.Sequential(
            nn.Linear(D, dm // 2),
            nn.SiLU(),
            nn.Linear(dm // 2, dm),
        )
        # 方差门控：从方差特征生成 [0,1]^d 门向量
        self.unc_gate = nn.Sequential(
            nn.Linear(dm, dm),
            nn.Sigmoid(),
        )

        # ── 逐点 token 的 LayerNorm ──────────────────────────────────────
        self.point_norm = nn.LayerNorm(dm)

        # ── 单条轨迹的 [TRAJ] token ──────────────────────────────────────
        # shape: [1, 1, 1, d_model]  广播到 [B, K, 1, d_model]
        self.traj_token = nn.Parameter(torch.zeros(1, 1, 1, dm))
        nn.init.trunc_normal_(self.traj_token, std=0.02)

        # ── Trajectory Transformer（时间维度自注意力）────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dm,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )

        # No mixture-weight embedding or gate for a single trajectory.

        # ── 模式 token 的 LayerNorm ──────────────────────────────────────
        self.mode_norm = nn.LayerNorm(dm)

        # ── 时间位置编码缓存（惰性初始化，避免设备问题）────────────────────
        self._time_cache: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------------
    # 内部工具
    # -----------------------------------------------------------------------

    def _get_time_emb(self, H: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """返回 [1, 1, H, 1] 的归一化时间特征 τ_h = h/H，h=1..H。"""
        if (
            self._time_cache is None
            or self._time_cache.shape[2] != H
            or self._time_cache.device != device
        ):
            t = torch.linspace(1.0 / H, 1.0, H, device=device, dtype=dtype)
            self._time_cache = t.view(1, 1, H, 1)
        return self._time_cache

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        gmm_means:   torch.Tensor,   # [B, K, H, 3]
        gmm_vars:    torch.Tensor,   # [B, K, H, 3]
        gmm_weights: torch.Tensor,   # [B, K]
    ) -> Dict[str, torch.Tensor]:
        """
        返回
        ----
        dict with keys:
            mode_tokens   : [B, K, d_model]    轨迹全局语义 token
            point_tokens  : [B, K, H, d_model] 时间点上下文 token
            weights       : [B, K]             归一化权重
            log_weights   : [B, K]             log(π_k)
            variance_gate : [B, K, H, d_model] 方差门控值（诊断/可视化用）
            weight_gate   : scalar             权重融合强度（诊断用）
        """
        # ── 0. 形状检查 & 数值安全 ─────────────────────────────────────
        assert gmm_means.ndim == 4, f"gmm_means 应为 4D，实际 {gmm_means.ndim}D"
        B, K, H, D = gmm_means.shape
        assert gmm_vars.shape  == (B, K, H, D), \
            f"gmm_vars 形状 {gmm_vars.shape} 与 gmm_means {gmm_means.shape} 不匹配"
        assert gmm_weights.shape == (B, K), \
            f"gmm_weights 应为 [B, K]，实际 {gmm_weights.shape}"
        assert K == self.cfg.num_modes, \
            f"K={K} 与配置 num_modes={self.cfg.num_modes} 不一致"
        assert H == self.cfg.pred_horizon, \
            f"H={H} 与配置 pred_horizon={self.cfg.pred_horizon} 不一致"

        if not torch.isfinite(gmm_means).all():
            raise ValueError("gmm_means 含有 NaN 或 Inf")
        if not torch.isfinite(gmm_vars).all():
            raise ValueError("gmm_vars 含有 NaN 或 Inf")
        if not torch.isfinite(gmm_weights).all():
            raise ValueError("gmm_weights 含有 NaN 或 Inf")

        # 权重归一化（应对数值噪声）
        weights = gmm_weights.clamp_min(self.cfg.weight_eps)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        log_weights = torch.log(weights)

        # ── 1. 轨迹位移增量 Δμ_{k,h} ──────────────────────────────────
        # delta[h=0] = μ[h=0]  (相对当前位置)
        # delta[h>0] = μ[h] - μ[h-1]
        delta_means = torch.zeros_like(gmm_means)
        delta_means[:, :, 0]  = gmm_means[:, :, 0]
        delta_means[:, :, 1:] = gmm_means[:, :, 1:] - gmm_means[:, :, :-1]

        # ── 2. 时间特征 τ_h = h/H ────────────────────────────────────
        time_emb = self._get_time_emb(H, gmm_means.device, gmm_means.dtype)
        time_emb = time_emb.expand(B, K, H, 1)

        # ── 3. 几何编码 [均值(3) + 位移(3) + 时间(1)] ────────────────
        geo_input = torch.cat([gmm_means, delta_means, time_emb], dim=-1)  # [B, K, H, 7]
        geo_feat  = self.geo_encoder(geo_input)                             # [B, K, H, dm]

        # ── 4. 方差编码 & 门控融合 ────────────────────────────────────
        if self.cfg.use_variance:
            vars_safe = gmm_vars.clamp_min(self.cfg.var_eps)
            log_std   = 0.5 * torch.log(vars_safe)                         # [B, K, H, 3]
            unc_feat  = self.unc_encoder(log_std)                          # [B, K, H, dm]
            gate      = self.unc_gate(unc_feat)                            # [B, K, H, dm]∈[0,1]
            point_tokens = self.point_norm(geo_feat + gate * unc_feat)     # [B, K, H, dm]
        else:
            gate         = torch.zeros_like(geo_feat)
            point_tokens = self.point_norm(geo_feat)

        # ── 5. 拼接 [TRAJ] token → [B, K, H+1, dm] ───────────────────
        traj_token = self.traj_token.expand(B, K, 1, self.cfg.d_model)
        trajectory_seq = torch.cat([traj_token, point_tokens], dim=2)     # [B, K, H+1, dm]

        # 合并 batch & mode 维以送入 Transformer（batch_first=True）
        BK = B * K
        trajectory_seq = trajectory_seq.reshape(BK, H + 1, self.cfg.d_model)

        # ── 6. Trajectory Transformer（时间维度自注意力）──────────────
        trajectory_out = self.trajectory_encoder(trajectory_seq)           # [BK, H+1, dm]
        trajectory_out = trajectory_out.reshape(B, K, H + 1, self.cfg.d_model)

        # 拆分：[TRAJ] token → 轨迹全局语义；其余 → 时间点上下文
        mode_geo_tokens      = trajectory_out[:, :, 0]     # [B, K, dm]
        point_context_tokens = trajectory_out[:, :, 1:]    # [B, K, H, dm]

        # ── 7. 权重编码 & 轻度调制轨迹 token ─────────────────────────
        alpha_weight = mode_geo_tokens.new_zeros(())
        mode_tokens = self.mode_norm(mode_geo_tokens)

        # ── 8. 返回 ───────────────────────────────────────────────────
        return {
            "mode_tokens":   mode_tokens,           # [B, K, dm]
            "point_tokens":  point_context_tokens,  # [B, K, H, dm]
            "weights":       weights,               # [B, K]
            "log_weights":   log_weights,           # [B, K]
            "variance_gate": gate,                  # [B, K, H, dm]
            "weight_gate":   alpha_weight,          # scalar
        }

# ---------------------------------------------------------------------------
# 单元测试（直接 python -m 运行）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = GMMEncoderConfig()
    encoder = GMMTrajectoryEncoder(cfg)
    means = torch.randn(4, 1, 10, 3)
    variances = torch.ones_like(means)
    weights = torch.ones(4, 1)
    output = encoder(means, variances, weights)
    assert output["point_tokens"].shape == (4, 1, 10, cfg.d_model)
    assert not any("weight_encoder" in name or "weight_gate" in name
                   for name, _ in encoder.named_parameters())
    modified = encoder(means, variances * 4, weights)
    assert torch.isfinite(modified["point_tokens"]).all()
    assert not torch.allclose(output["point_tokens"], modified["point_tokens"])
    output["point_tokens"][..., 0].mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    print("PASS: single trajectory geometry, variance sensitivity, gradients; no mixture weights")
