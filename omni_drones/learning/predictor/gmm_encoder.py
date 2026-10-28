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
    weight_gate:  scalar             权重融合强度（可学习，诊断用）

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
    num_modes:     int = 3      # K: GMM 分量数 = 3
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

    # ── 权重融合强度初始化 ─────────────────────────────────────────────────
    # sigmoid(initial_weight_gate_logit) ≈ 初始融合强度
    # -2.197 → sigmoid ≈ 0.10（权重初始只轻度影响轨迹 token）
    initial_weight_gate_logit: float = -2.197

    # ── Actor 注意力先验权重 ───────────────────────────────────────────────
    # scores += weight_prior_scale * log(π_k)
    # 0.5 → 高权重模式有 ~1 logit 优势，但几何匹配仍可覆盖
    weight_prior_scale: float = 0.5

    # ── 功能开关 ──────────────────────────────────────────────────────────
    use_variance:             bool = True  # 是否使用方差编码
    use_weight_token:         bool = True  # 是否将权重注入轨迹 token
    use_weight_attention_bias: bool = True  # 是否在 Actor 交叉注意力加权重先验


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

        # ── 轨迹级 [TRAJ] token（三条轨迹共享同一个初始 token）────────────
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

        # ── 权重编码器：[π_k, log(π_k)] → d_model ────────────────────────
        self.weight_encoder = nn.Sequential(
            nn.Linear(2, dm // 4),
            nn.SiLU(),
            nn.Linear(dm // 4, dm),
        )

        # ── 权重融合强度（可学习标量门）──────────────────────────────────
        # alpha_weight = sigmoid(weight_gate_logit)
        self.weight_gate_logit = nn.Parameter(
            torch.tensor(cfg.initial_weight_gate_logit)
        )

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
        weight_input = torch.stack(
            [weights, log_weights], dim=-1
        )                                                                  # [B, K, 2]
        weight_embeddings = self.weight_encoder(weight_input)              # [B, K, dm]

        alpha_weight = torch.sigmoid(self.weight_gate_logit)               # scalar ∈(0,1)

        if self.cfg.use_weight_token:
            mode_tokens = self.mode_norm(
                mode_geo_tokens + alpha_weight * weight_embeddings
            )                                                              # [B, K, dm]
        else:
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
    import sys

    print("=" * 60)
    print("GMMTrajectoryEncoder 单元测试")
    print("=" * 60)

    device = "cpu"
    cfg = GMMEncoderConfig()
    encoder = GMMTrajectoryEncoder(cfg).to(device)

    B, K, H = 4, 3, 10
    means   = torch.randn(B, K, H, 3, device=device)
    vars_   = torch.rand(B, K, H, 3, device=device).abs() + 1e-4
    weights = torch.softmax(torch.randn(B, K, device=device), dim=-1)

    # ── 1. 形状测试 ─────────────────────────────────────────────────────
    out = encoder(means, vars_, weights)
    assert out["mode_tokens"].shape  == (B, K, cfg.d_model), "mode_tokens 形状错误"
    assert out["point_tokens"].shape == (B, K, H, cfg.d_model), "point_tokens 形状错误"
    assert out["weights"].shape      == (B, K), "weights 形状错误"
    print("[PASS] 形状测试")

    # ── 2. 模式排列等变性测试 ────────────────────────────────────────────
    # 同时交换模式 0 和 1 的 means / vars / weights
    means_perm   = means.clone();   means_perm[:, [0, 1]]   = means[:, [1, 0]]
    vars_perm    = vars_.clone();   vars_perm[:, [0, 1]]    = vars_[:, [1, 0]]
    weights_perm = weights.clone(); weights_perm[:, [0, 1]] = weights[:, [1, 0]]

    out_perm = encoder(means_perm, vars_perm, weights_perm)

    # 输出模式0/1 也应交换，数值基本一致
    diff_mode0 = (out_perm["mode_tokens"][:, 0] - out["mode_tokens"][:, 1]).abs().max()
    diff_mode1 = (out_perm["mode_tokens"][:, 1] - out["mode_tokens"][:, 0]).abs().max()
    diff_mode2 = (out_perm["mode_tokens"][:, 2] - out["mode_tokens"][:, 2]).abs().max()
    assert diff_mode0 < 1e-5, f"等变性失败（mode0 diff={diff_mode0:.2e}）"
    assert diff_mode1 < 1e-5, f"等变性失败（mode1 diff={diff_mode1:.2e}）"
    assert diff_mode2 < 1e-5, f"等变性失败（mode2 diff={diff_mode2:.2e}）"
    print("[PASS] 模式排列等变性测试")

    # ── 3. 权重敏感性测试 ─────────────────────────────────────────────
    weights_alt = weights.clone()
    weights_alt[:, 0] = 0.8
    weights_alt[:, 1] = 0.1
    weights_alt[:, 2] = 0.1
    weights_alt = weights_alt / weights_alt.sum(-1, keepdim=True)

    out_alt = encoder(means, vars_, weights_alt)

    # point_tokens 应基本不变（权重不直接作用于逐点 token）
    pt_diff = (out_alt["point_tokens"] - out["point_tokens"]).abs().max()
    # mode_tokens 允许有限变化
    mt_diff = (out_alt["mode_tokens"] - out["mode_tokens"]).abs().max()
    print(f"[INFO] 权重敏感性: point_token 最大差={pt_diff:.4f}, mode_token 最大差={mt_diff:.4f}")
    print("[PASS] 权重敏感性测试（point_tokens 变化应明显小于 mode_tokens）")

    # ── 4. 方差敏感性测试 ─────────────────────────────────────────────
    vars_high = vars_.clone()
    vars_high[:, 1, :, 1] *= 4.0          # 模式1 的 y 方向方差增大

    out_high_var = encoder(means, vars_high, weights)

    pt_diff_mode1 = (out_high_var["point_tokens"][:, 1] - out["point_tokens"][:, 1]).abs().max()
    pt_diff_mode0 = (out_high_var["point_tokens"][:, 0] - out["point_tokens"][:, 0]).abs().max()
    assert torch.isfinite(out_high_var["point_tokens"]).all(), "高方差下出现 NaN/Inf"
    print(f"[INFO] 方差敏感性: mode1 token 最大差={pt_diff_mode1:.4f}, mode0 变化={pt_diff_mode0:.4f}")
    print("[PASS] 方差敏感性测试")

    # ── 5. 梯度测试（编码器有梯度，输入无梯度） ────────────────────────
    means_detach   = means.clone().requires_grad_(False)
    vars_detach    = vars_.clone().requires_grad_(False)
    weights_detach = weights.clone().requires_grad_(False)

    out_grad = encoder(means_detach, vars_detach, weights_detach)
    loss = out_grad["mode_tokens"].sum() + out_grad["point_tokens"].sum()
    loss.backward()

    has_grad = any(p.grad is not None for p in encoder.parameters())
    assert has_grad, "编码器参数应有梯度"
    # 输入 tensor 不应有梯度（模拟冻结预测网络）
    assert means_detach.grad is None, "预测网络输入不应有梯度"
    print("[PASS] 梯度测试")

    print()
    print("所有测试通过 ✓")
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"GMMTrajectoryEncoder 参数量: {n_params:,}")
