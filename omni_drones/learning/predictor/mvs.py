"""
机动变异度得分 (Maneuver Variability Score, MVS)

基于规则计算，不参与神经网络训练。
MVS = 0.6 × 意图分布熵 + 0.4 × 意图切换频率

MVS高 → 目标频繁改变机动方式 → 预测不确定性大 → GMM展开多模态
MVS低 → 目标保持稳定机动 → 预测确定性高 → GMM收敛单模态

注意: MVS不编码"逃逸还是突防"的语义，只衡量行为的统计不稳定性。
"""

import torch
import math
from typing import Union
import numpy as np


def compute_maneuver_variability_score(
    intent_history: Union[torch.Tensor, np.ndarray, list],
    n_intents: int = 10,
) -> float:
    """
    计算机动变异度得分 (MVS)

    Args:
        intent_history: 最近若干步的意图标签序列
            可以是 list[int], np.ndarray, 或 torch.Tensor
            长度至少为2
        n_intents: 意图类别总数 (默认10)

    Returns:
        mvs: float, 范围 [0, 1]
            0 = 完全稳定 (单一意图)
            1 = 最大变异 (均匀分布 + 每步都切换)
    """
    # 转换为numpy数组
    if isinstance(intent_history, torch.Tensor):
        intent_history = intent_history.cpu().numpy()
    elif isinstance(intent_history, list):
        intent_history = np.array(intent_history)

    window = len(intent_history)
    if window < 2:
        return 0.0

    # ===== 1. 意图分布熵 =====
    counts = np.bincount(intent_history.astype(int), minlength=n_intents)
    probs = counts / window
    # 排除零概率项
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    max_entropy = math.log(n_intents)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # ===== 2. 意图切换频率 =====
    switches = np.sum(intent_history[1:] != intent_history[:-1])
    switch_rate = float(switches) / (window - 1)

    # ===== 3. 综合得分 =====
    mvs = 0.6 * normalized_entropy + 0.4 * switch_rate

    return float(np.clip(mvs, 0.0, 1.0))


def compute_mvs_batch(
    intent_history_batch: torch.Tensor,
    n_intents: int = 10,
) -> torch.Tensor:
    """
    批量计算MVS (用于训练循环中)

    Args:
        intent_history_batch: [batch, window] 每个样本的意图历史序列
        n_intents: 意图类别数

    Returns:
        mvs_batch: [batch, 1] 每个样本的MVS
    """
    B = intent_history_batch.shape[0]
    device = intent_history_batch.device

    mvs_values = []
    for i in range(B):
        mvs = compute_maneuver_variability_score(
            intent_history_batch[i], n_intents
        )
        mvs_values.append(mvs)

    return torch.tensor(mvs_values, device=device, dtype=torch.float32).unsqueeze(-1)
