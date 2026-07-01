"""
数据管线：轨迹记录、自动意图标注、滑窗切分、Dataset

流程:
  仿真运行 → TrajectoryRecorder 记录每步数据
  → episode结束 → IntentLabeler 自动标注
  → SlidingWindowDataset 滑窗切分为训练样本
  → DataLoader 用于训练

独立于Isaac Sim，可用合成数据测试。
"""

import torch
import numpy as np
import math
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

from omni_drones.utils.intent_labels import INTENT_LABELS, label_intent_batch, label_intent_single


# ==========================================================================
# 意图标注器
# ==========================================================================

# ==========================================================================
# 轨迹记录器
# ==========================================================================

class TrajectoryRecorder:
    """
    记录仿真中每步的轨迹数据

    每 subsample_rate 步记录一帧。
    """

    def __init__(self, subsample_rate: int = 5):
        self.subsample_rate = subsample_rate
        self.step_count = 0
        self.buffer: List[dict] = []

    def record(
        self,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        pursuer_pos: torch.Tensor,
        pursuer_vel: torch.Tensor,
        accel_cmd: torch.Tensor,
        strategy_id: Optional[torch.Tensor] = None,
    ):
        """
        记录一步

        Args:
            target_pos:  [3] 目标位置
            target_vel:  [3] 目标速度
            pursuer_pos: [n_pursuers, 3] 追捕方位置
            pursuer_vel: [n_pursuers, 3] 追捕方速度
            accel_cmd:   [3] 目标加速度指令
            strategy_id: 标量, 目标FSM策略ID
        """
        self.step_count += 1
        if self.step_count % self.subsample_rate != 0:
            return

        self.buffer.append({
            "target_pos": target_pos.clone().cpu(),
            "target_vel": target_vel.clone().cpu(),
            "pursuer_pos": pursuer_pos.clone().cpu(),
            "pursuer_vel": pursuer_vel.clone().cpu(),
            "accel_cmd": accel_cmd.clone().cpu(),
            "strategy_id": strategy_id.item() if strategy_id is not None else -1,
        })

    def get_episode(self) -> dict:
        """返回整个episode的数据, 转为tensor"""
        if len(self.buffer) == 0:
            return None

        L = len(self.buffer)
        n_pursuers = self.buffer[0]["pursuer_pos"].shape[0]

        episode = {
            "target_pos": torch.stack([f["target_pos"] for f in self.buffer]),      # [L, 3]
            "target_vel": torch.stack([f["target_vel"] for f in self.buffer]),      # [L, 3]
            "pursuer_pos": torch.stack([f["pursuer_pos"] for f in self.buffer]),    # [L, n_p, 3]
            "pursuer_vel": torch.stack([f["pursuer_vel"] for f in self.buffer]),    # [L, n_p, 3]
            "accel_cmd": torch.stack([f["accel_cmd"] for f in self.buffer]),        # [L, 3]
            "strategy_ids": torch.tensor([f["strategy_id"] for f in self.buffer]),  # [L]
        }
        return episode

    def reset(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.step_count = 0


# ==========================================================================
# 滑窗数据集
# ==========================================================================

class PredictorDataset(Dataset):
    """
    从episode轨迹中滑窗切分训练样本

    每个样本:
        obs_history:     [T_obs, 24]
        gt_future_pos:   [T_pred, 3]
        gt_intent_label: scalar (0-9)
        intent_history:  [W_mvs] (用于MVS计算)
    """

    def __init__(
        self,
        t_obs: int = 5,
        t_pred: int = 5,
        w_mvs: int = 10,
        slide_step: int = 2,
        accel_threshold: float = 0.5,
    ):
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.w_mvs = w_mvs
        self.slide_step = slide_step
        self.accel_threshold = accel_threshold

        self.obs_list = []
        self.future_list = []
        self.intent_list = []
        self.intent_hist_list = []

    def add_episode(self, episode: dict):
        """
        将一个episode的数据滑窗切分后加入数据集

        Args:
            episode: TrajectoryRecorder.get_episode() 的输出
        """
        if episode is None:
            return

        L = episode["target_pos"].shape[0]
        n_pursuers = episode["pursuer_pos"].shape[1]

        # 先对整条轨迹做意图标注
        all_intents = label_intent_batch(
            episode["accel_cmd"],
            episode["target_vel"],
            accel_threshold=self.accel_threshold,
        )  # [L]

        # 滑窗
        min_start = max(self.w_mvs, self.t_obs)  # MVS窗口需要的历史
        for start in range(min_start, L - self.t_pred, self.slide_step):
            obs_start = start - self.t_obs
            obs_end = start
            future_end = start + self.t_pred

            if obs_start < 0 or future_end > L:
                continue

            # 构建观测: [T_obs, 24]
            obs_frames = []
            for t in range(obs_start, obs_end):
                frame = torch.cat([
                    episode["target_pos"][t],       # 3
                    episode["target_vel"][t],       # 3
                ])
                for p in range(n_pursuers):
                    frame = torch.cat([
                        frame,
                        episode["pursuer_pos"][t, p],  # 3
                        episode["pursuer_vel"][t, p],  # 3
                    ])
                obs_frames.append(frame)

            obs = torch.stack(obs_frames)  # [T_obs, 24]

            # 未来位置
            future = episode["target_pos"][start:future_end]  # [T_pred, 3]

            # 意图标签 (取观测窗口最后一帧)
            intent = all_intents[obs_end - 1].item()

            # 意图历史 (用于MVS)
            mvs_start = max(0, start - self.w_mvs)
            intent_hist = all_intents[mvs_start:start]  # [<=W_mvs]
            # 如果历史不够长, 前面填充
            if len(intent_hist) < self.w_mvs:
                pad = torch.full((self.w_mvs - len(intent_hist),), intent_hist[0].item(), dtype=torch.long)
                intent_hist = torch.cat([pad, intent_hist])

            self.obs_list.append(obs)
            self.future_list.append(future)
            self.intent_list.append(intent)
            self.intent_hist_list.append(intent_hist)

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        return {
            "obs_history": self.obs_list[idx],           # [T_obs, 24]
            "gt_future_pos": self.future_list[idx],       # [T_pred, 3]
            "gt_intent_label": self.intent_list[idx],     # scalar
            "intent_history": self.intent_hist_list[idx],  # [W_mvs]
        }

    def get_stats(self) -> dict:
        """返回数据集统计信息"""
        if len(self) == 0:
            return {"total_samples": 0}

        intents = torch.tensor(self.intent_list)
        counts = torch.bincount(intents, minlength=10)
        dist = counts.float() / counts.sum()

        return {
            "total_samples": len(self),
            "intent_counts": {INTENT_LABELS[i]: counts[i].item() for i in range(10)},
            "intent_distribution": {INTENT_LABELS[i]: f"{dist[i].item():.3f}" for i in range(10)},
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """自定义collate, 用于DataLoader"""
        return {
            "obs_history": torch.stack([b["obs_history"] for b in batch]),
            "gt_future_pos": torch.stack([b["gt_future_pos"] for b in batch]),
            "gt_intent_label": torch.tensor([b["gt_intent_label"] for b in batch], dtype=torch.long),
            "intent_history": torch.stack([b["intent_history"] for b in batch]),
        }
