"""
IntentConditionedGMMPredictor — 意图条件化GMM轨迹预测器

整体架构：
    观测历史 [B, T_obs, obs_dim]
         │
    ┌────▼────┐
    │ BiLSTM  │  共享时序编码器
    │ Encoder │
    └────┬────┘
         │  h: [B, feat_dim]
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
┌───────────┐          ┌──────────────────┐
│ 意图分类头 │          │ GMM 预测头       │
│ → 10类    │          │ (w, μ, σ²) × K  │
└─────┬─────┘          └────────┬─────────┘
      │ P_tactical               │
      │                          │
      │  ┌──── MVS (规则) ────┐  │
      │  │                    │  │
      ▼  ▼                    ▼  ▼
  ┌──────────┐          ┌──────────────┐
  │ 上下文Z   │          │ MVS 调制     │
  │ 拼接融合  │─────────→│ 温度缩放     │
  └──────────┘          │ 方差膨胀     │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │ 混合分布输出  │
                        │ μ_mix, σ²_mix│
                        └──────────────┘

设计决策（来自讨论共识）：
  - 编码器共享BiLSTM底层
  - K=3个GMM分量
  - 训练时不使用WTA，用纯NLL
  - 多模态由MVS调制注入，不强制从数据中学习
  - 分阶段训练：Phase1 意图分类 → Phase2 GMM预测 → Phase3(可选)联合微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# 10类意图标签定义
INTENT_LABELS = [
    "Straight",      # 0: 高速直线
    "Left-Turn",     # 1: 左转
    "Right-Turn",    # 2: 右转
    "Climb",         # 3: 爬升
    "Dive",          # 4: 俯冲
    "Left-Climb",    # 5: 左转+爬升
    "Left-Dive",     # 6: 左转+俯冲
    "Right-Climb",   # 7: 右转+爬升
    "Right-Dive",    # 8: 右转+俯冲
    "Hover",         # 9: 悬停/低速
]


class IntentConditionedGMMPredictor(nn.Module):
    """
    意图条件化高斯混合模型轨迹预测器

    输入:
        obs_history: [batch, T_obs, obs_dim]
            每帧观测 = target(pos3+vel3) + 3×pursuer(pos3+vel3) = 24维
        mvs: [batch, 1]  (可选)
            机动变异度得分，由外部规则计算
            训练Phase1时不需要传入
            训练Phase2和推理时传入

    输出: dict
        intent_logits:  [batch, 10]       — 意图分类logits
        intent_probs:   [batch, 10]       — 意图概率 (softmax)
        gmm_weights:    [batch, K]        — GMM混合权重 (经MVS温度缩放)
        gmm_means:      [batch, K, T_pred, 3] — GMM均值轨迹
        gmm_vars:       [batch, K, T_pred, 3] — GMM方差 (经MVS方差膨胀)
        mixed_mu:       [batch, T_pred, 3]    — 混合均值
        mixed_var:      [batch, T_pred, 3]    — 混合方差
    """

    def __init__(
        self,
        obs_dim: int = 24,
        n_intents: int = 10,
        n_components: int = 3,
        pred_horizon: int = 10,
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        alpha: float = 2.0,
        beta: float = 1.5,
    ):
        """
        Args:
            obs_dim:       每帧观测维度 (默认24: target6 + 3*pursuer6)
            n_intents:     意图类别数 (默认10)
            n_components:  GMM分量数K (默认3)
            pred_horizon:  预测未来步数 (默认5, @20Hz=0.25s)
            hidden_dim:    LSTM每个方向的隐藏维度 (默认128, BiLSTM→256)
            lstm_layers:   LSTM层数 (默认2)
            dropout:       Dropout率 (默认0.1)
            alpha:         温度缩放系数 τ = 1 + α·MVS (默认2.0)
            beta:          方差膨胀系数 σ = σ_net·(1+β·MVS) (默认1.5)
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.n_intents = n_intents
        self.K = n_components
        self.T = pred_horizon
        self.alpha = alpha
        self.beta = beta
        self.feat_dim = hidden_dim * 2  # BiLSTM双向拼接

        # =====================================================
        # 共享编码器: BiLSTM
        # =====================================================
        self.encoder = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # =====================================================
        # 意图分类头
        # =====================================================
        self.intent_head = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_intents),
        )

        # =====================================================
        # GMM预测头
        # 输入: 编码器特征h + 意图概率P + MVS → 融合上下文Z
        # 输出: K组(权重, 均值, 对数方差)
        # =====================================================
        # 上下文融合: Z = MLP([h, P_tactical, MVS])
        context_input_dim = self.feat_dim + n_intents + 1  # 256 + 10 + 1 = 267
        self.context_mlp = nn.Sequential(
            nn.Linear(context_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # 三个输出头
        self.weight_head = nn.Linear(128, self.K)
        self.mean_head = nn.Linear(128, self.K * self.T * 3)
        self.logvar_head = nn.Linear(128, self.K * self.T * 3)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """合理的参数初始化"""
        # 均值头用小权重初始化，防止初始预测偏移太大
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.1)
        nn.init.zeros_(self.mean_head.bias)

        # 方差头偏置初始化为0 → exp(0)=1 → 初始方差为1
        nn.init.xavier_uniform_(self.logvar_head.weight, gain=0.1)
        nn.init.zeros_(self.logvar_head.bias)

        # 权重头初始化为接近均匀
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    def forward(
        self,
        obs_history: torch.Tensor,
        mvs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            obs_history: [batch, T_obs, obs_dim] 观测历史序列
            mvs: [batch, 1] 机动变异度得分 (可选, 默认为0)

        Returns:
            dict: 包含意图分类和GMM预测的所有输出
        """
        B = obs_history.shape[0]
        device = obs_history.device

        # 默认MVS=0 (无调制，等效于Rush/稳定状态)
        if mvs is None:
            mvs = torch.zeros(B, 1, device=device)

        # =====================================================
        # Step 1: 共享BiLSTM编码
        # =====================================================
        lstm_out, _ = self.encoder(obs_history)  # [B, T_obs, feat_dim]
        h = lstm_out[:, -1, :]  # 取最后一步: [B, feat_dim]

        # =====================================================
        # Step 2: 意图分类
        # =====================================================
        intent_logits = self.intent_head(h)               # [B, 10]
        intent_probs = F.softmax(intent_logits, dim=-1)   # [B, 10]

        # =====================================================
        # Step 3: 上下文融合
        # Z = MLP([h, P_tactical, MVS])
        # =====================================================
        context_input = torch.cat([h, intent_probs.detach(), mvs], dim=-1)
        # 注意: intent_probs.detach() — 意图概率不回传梯度到编码器
        # 这是分阶段训练的关键：GMM的梯度不干扰意图分类器

        z = self.context_mlp(context_input)  # [B, 128]

        # =====================================================
        # Step 4: GMM参数生成
        # =====================================================
        # 4a. 权重 (带温度缩放)
        weight_logits = self.weight_head(z)                         # [B, K]
        tau = 1.0 + self.alpha * mvs                                # [B, 1]
        gmm_weights = F.softmax(weight_logits / tau, dim=-1)        # [B, K]

        # 4b. 均值
        gmm_means = self.mean_head(z)                               # [B, K*T*3]
        gmm_means = gmm_means.reshape(B, self.K, self.T, 3)        # [B, K, T, 3]

        # 4c. 方差 (带方差膨胀)
        log_var = self.logvar_head(z)                               # [B, K*T*3]
        log_var = log_var.reshape(B, self.K, self.T, 3)             # [B, K, T, 3]
        log_var = log_var.clamp(-6.0, 2.0)  # 防止方差爆炸/消失
        gmm_vars_raw = torch.exp(log_var)                           # [B, K, T, 3]
        inflation = 1.0 + self.beta * mvs.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        gmm_vars = gmm_vars_raw * inflation                        # [B, K, T, 3]

        # =====================================================
        # Step 5: 计算混合分布的均值和方差
        # =====================================================
        mixed_mu, mixed_var = self._compute_mixture(
            gmm_weights, gmm_means, gmm_vars
        )

        return {
            "intent_logits": intent_logits,     # [B, 10]
            "intent_probs": intent_probs,       # [B, 10]
            "gmm_weights": gmm_weights,         # [B, K]
            "gmm_means": gmm_means,             # [B, K, T, 3]
            "gmm_vars": gmm_vars,               # [B, K, T, 3]
            "mixed_mu": mixed_mu,               # [B, T, 3]
            "mixed_var": mixed_var,              # [B, T, 3]
        }

    def _compute_mixture(
        self,
        weights: torch.Tensor,
        means: torch.Tensor,
        vars: torch.Tensor,
    ):
        """
        计算高斯混合分布的总均值和总方差

        总方差 = E[Var] + Var[E]  (全方差公式)
               = Σ w_k σ²_k  +  Σ w_k (μ_k - μ_mix)²

        Args:
            weights: [B, K]
            means:   [B, K, T, 3]
            vars:    [B, K, T, 3]

        Returns:
            mixed_mu:  [B, T, 3]
            mixed_var: [B, T, 3]
        """
        w = weights.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]

        # 混合均值
        mixed_mu = (w * means).sum(dim=1)  # [B, T, 3]

        # 混合方差 (全方差公式)
        mixed_var = (w * (vars + means ** 2)).sum(dim=1) - mixed_mu ** 2
        # 确保方差非负 (数值安全)
        mixed_var = mixed_var.clamp(min=1e-6)

        return mixed_mu, mixed_var

    def predict_intent_only(self, obs_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        仅执行意图分类 (Phase 1 训练/评估用)

        Args:
            obs_history: [batch, T_obs, obs_dim]

        Returns:
            dict: intent_logits, intent_probs
        """
        lstm_out, _ = self.encoder(obs_history)
        h = lstm_out[:, -1, :]
        intent_logits = self.intent_head(h)
        intent_probs = F.softmax(intent_logits, dim=-1)
        return {
            "intent_logits": intent_logits,
            "intent_probs": intent_probs,
        }

    def get_config(self) -> dict:
        """返回网络配置，方便保存和重建"""
        return {
            "obs_dim": self.obs_dim,
            "n_intents": self.n_intents,
            "n_components": self.K,
            "pred_horizon": self.T,
            "hidden_dim": self.feat_dim // 2,
            "alpha": self.alpha,
            "beta": self.beta,
        }
