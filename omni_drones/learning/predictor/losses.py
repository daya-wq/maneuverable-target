"""
损失函数模块

提供两个损失函数：
  - IntentLoss:    意图分类交叉熵损失 (Phase 1 训练)
  - IntentGMMLoss: 意图分类 + GMM负对数似然联合损失 (Phase 2 训练)

设计决策（来自讨论共识）：
  - 使用纯NLL，不使用WTA (因为数据近似确定性，WTA反而有害)
  - 不使用diversity regularization (多模态由MVS调制注入，不由loss强加)
  - 训练时GMM loss用所有分量的混合似然，不只选winner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class IntentLoss(nn.Module):
    """
    Phase 1 损失: 纯意图分类

    L = CrossEntropy(intent_logits, gt_intent_label)
    """

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_intent_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: 网络输出dict, 需要包含 'intent_logits'
            gt_intent_label: [B] 真实意图标签 (LongTensor, 0-9)

        Returns:
            loss: 标量损失
            metrics: 训练指标dict (用于日志)
        """
        intent_logits = pred["intent_logits"]  # [B, 10]
        loss = F.cross_entropy(intent_logits, gt_intent_label)

        # 计算分类准确率
        with torch.no_grad():
            pred_label = intent_logits.argmax(dim=-1)
            accuracy = (pred_label == gt_intent_label).float().mean()

        metrics = {
            "loss_intent": loss.item(),
            "intent_accuracy": accuracy.item(),
        }
        return loss, metrics


class IntentGMMLoss(nn.Module):
    """
    Phase 2 损失: 意图分类 + GMM轨迹NLL

    L = λ_intent · L_intent + λ_gmm · L_gmm

    其中:
      L_intent = CrossEntropy(intent_logits, gt_intent)
      L_gmm = -log( Σ_k  w_k · N(gt_pos | μ_k, σ²_k) )

    GMM NLL的直觉：
      让网络找到这样的(w, μ, σ²)组合，使得GT轨迹在GMM分布下的概率最大。
      如果GT轨迹落在某个分量附近，那个分量的权重和均值会被拉向GT。
    """

    def __init__(
        self,
        lambda_intent: float = 1.0,
        lambda_gmm: float = 1.0,
    ):
        """
        Args:
            lambda_intent: 意图损失权重
            lambda_gmm: GMM轨迹损失权重
        """
        super().__init__()
        self.lambda_intent = lambda_intent
        self.lambda_gmm = lambda_gmm

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_future_pos: torch.Tensor,
        gt_intent_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: 网络输出dict
            gt_future_pos: [B, T_pred, 3] 真实未来位置序列
            gt_intent_label: [B] 真实意图标签

        Returns:
            loss: 总损失标量
            metrics: 训练指标dict
        """
        # ===== 1. 意图分类损失 =====
        intent_logits = pred["intent_logits"]  # [B, 10]
        L_intent = F.cross_entropy(intent_logits, gt_intent_label)

        # ===== 2. GMM NLL 损失 =====
        L_gmm = self._gmm_nll(
            pred["gmm_weights"],   # [B, K]
            pred["gmm_means"],     # [B, K, T, 3]
            pred["gmm_vars"],      # [B, K, T, 3]
            gt_future_pos,         # [B, T, 3]
        )

        # ===== 3. 总损失 =====
        loss = self.lambda_intent * L_intent + self.lambda_gmm * L_gmm

        # ===== 指标 =====
        with torch.no_grad():
            pred_label = intent_logits.argmax(dim=-1)
            accuracy = (pred_label == gt_intent_label).float().mean()

            # 混合轨迹的MSE (用于直觉理解预测质量, 不参与训练)
            mixed_mu = pred["mixed_mu"]  # [B, T, 3]
            traj_mse = ((mixed_mu - gt_future_pos) ** 2).mean()

            # GMM权重分布的熵 (监控是否坍缩)
            w = pred["gmm_weights"]  # [B, K]
            weight_entropy = -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()

        metrics = {
            "loss_total": loss.item(),
            "loss_intent": L_intent.item(),
            "loss_gmm_nll": L_gmm.item(),
            "intent_accuracy": accuracy.item(),
            "traj_mse": traj_mse.item(),
            "gmm_weight_entropy": weight_entropy.item(),
        }
        return loss, metrics

    @staticmethod
    def _gmm_nll(
        weights: torch.Tensor,
        means: torch.Tensor,
        vars: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算高斯混合分布的负对数似然

        L = -log( Σ_k  w_k · N(gt | μ_k, σ²_k) )
          = -log( Σ_k  w_k · Π_{t,d} (2πσ²_{k,t,d})^{-1/2} exp(-||gt-μ||²/2σ²) )

        用 logsumexp 保证数值稳定性。

        Args:
            weights: [B, K]
            means:   [B, K, T, 3]
            vars:    [B, K, T, 3]
            gt:      [B, T, 3]

        Returns:
            nll: 标量, batch平均的负对数似然
        """
        gt_expanded = gt.unsqueeze(1)  # [B, 1, T, 3]

        # 每个分量在每个(时间步, 维度)上的对数概率密度
        # log N(x|μ,σ²) = -0.5 * (log(2π) + log(σ²) + (x-μ)²/σ²)
        log_prob_per_dim = -0.5 * (
            1.8378770664093453  # log(2π)
            + torch.log(vars + 1e-6)
            + (gt_expanded - means) ** 2 / (vars + 1e-6)
        )  # [B, K, T, 3]

        # 对所有时间步和维度求和 → 每个分量的总对数概率
        log_component_prob = log_prob_per_dim.sum(dim=(-1, -2))  # [B, K]

        # 加上混合权重的对数
        log_weighted = log_component_prob + torch.log(weights + 1e-8)  # [B, K]

        # logsumexp → 混合分布的对数似然
        log_likelihood = torch.logsumexp(log_weighted, dim=-1)  # [B]

        # 返回batch平均负对数似然
        nll = -log_likelihood.mean()

        return nll
