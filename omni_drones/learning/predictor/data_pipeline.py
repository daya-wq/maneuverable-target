"""
数据管线：轨迹记录、战术标签生成、滑窗切分、Dataset

流程:
  仿真运行 → TrajectoryRecorder 记录每步数据
  → episode结束 → 滑窗切分
  → 构建 30维相对运动学特征 + 4类战术标签
  → DataLoader 用于训练

输入特征 (30维):
  [0:9]   target_pos - pursuer_pos  (3个追捕者 × 3D)
  [9:18]  target_vel - pursuer_vel  (3个追捕者 × 3D)
  [18:21] target_pos - goal_center  (1个目标 × 3D)
  [21:30] pursuer_pos - goal_center (3个追捕者 × 3D)

战术标签 (4类):
  0: Far_Rush      — 距离较远，全速冲刺
  1: Close_Evade   — 距离极近，启动规避机动
  2: Goal_Priority — 不顾一切冲向守区
  3: Goal_Sprint   — 最后阶段的直线冲刺

独立于 Isaac Sim，可用合成数据测试。
"""

import torch
import numpy as np
from typing import List, Optional
from torch.utils.data import Dataset

try:
    from .dual_intent_gmm_predictor import STRATEGIC_LABELS
except ImportError:
    # 通过 importlib.util 直接加载时没有包结构，从 sys.modules 取
    import sys as _sys
    STRATEGIC_LABELS = _sys.modules["dual_intent_gmm_predictor"].STRATEGIC_LABELS

# 战术标签常量
STRATEGIC_LABEL_NAMES = STRATEGIC_LABELS  # ["Far_Rush", "Close_Evade", "Goal_Priority", "Goal_Sprint"]
N_STRATEGIC = len(STRATEGIC_LABEL_NAMES)

# goal_center 默认值，可由外部覆盖
DEFAULT_GOAL_CENTER = torch.tensor([8.0, 0.0, 2.5], dtype=torch.float32)


# ==========================================================================
# 战术标签计算（基于规则）
# ==========================================================================

def label_strategic_single(
    target_pos: torch.Tensor,       # [3]
    target_vel: torch.Tensor,       # [3]
    pursuer_pos: torch.Tensor,      # [n_p, 3]
    goal_center: torch.Tensor,      # [3]
    far_rush_dist: float = 4.0,
    close_evade_dist: float = 1.5,
    goal_sprint_dist: float = 2.0,
) -> int:
    """
    根据当前态势，用规则判断4类战术标签。

    优先级: Close_Evade > Goal_Sprint > Goal_Priority > Far_Rush

    Args:
        target_pos:      目标当前位置 [3]
        target_vel:      目标当前速度 [3]
        pursuer_pos:     所有追捕者位置 [n_p, 3]
        goal_center:     守区中心 [3]
        far_rush_dist:   距追捕者平均距离 > 此值 → Far_Rush
        close_evade_dist: 最近追捕者 < 此值 → Close_Evade
        goal_sprint_dist: 距守区 < 此值 → Goal_Sprint

    Returns:
        label: int, 0~3
    """
    min_d = float((target_pos.unsqueeze(0) - pursuer_pos).norm(dim=-1).min())
    goal_d = float((target_pos - goal_center).norm())

    # 速度朝向守区的分量（cos相似度）
    vel_norm = float(target_vel.norm())
    goal_dir = (goal_center - target_pos)
    goal_dir_norm = float(goal_dir.norm())
    if vel_norm > 1e-6 and goal_dir_norm > 1e-6:
        cos_goal = float(torch.dot(target_vel, goal_dir) / (vel_norm * goal_dir_norm))
    else:
        cos_goal = 0.0

    # 优先级判断
    if min_d < close_evade_dist:
        return 1  # Close_Evade
    if goal_d < goal_sprint_dist and cos_goal > 0.5:
        return 3  # Goal_Sprint
    if cos_goal > 0.7:
        return 2  # Goal_Priority
    return 0      # Far_Rush


def label_strategic_batch(
    target_pos: torch.Tensor,   # [L, 3]
    target_vel: torch.Tensor,   # [L, 3]
    pursuer_pos: torch.Tensor,  # [L, n_p, 3]
    goal_center: torch.Tensor,  # [3]
    **kwargs,
) -> torch.Tensor:
    """批量标注战术标签，返回 [L] LongTensor"""
    L = target_pos.shape[0]
    labels = []
    for t in range(L):
        label = label_strategic_single(
            target_pos[t], target_vel[t], pursuer_pos[t], goal_center, **kwargs
        )
        labels.append(label)
    return torch.tensor(labels, dtype=torch.long)


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
            strategy_id: 标量，目标FSM策略ID（可选，仅供参考）
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
        """返回整个 episode 的数据，转为 tensor"""
        if len(self.buffer) == 0:
            return None

        episode = {
            "target_pos":  torch.stack([f["target_pos"]  for f in self.buffer]),   # [L, 3]
            "target_vel":  torch.stack([f["target_vel"]  for f in self.buffer]),   # [L, 3]
            "pursuer_pos": torch.stack([f["pursuer_pos"] for f in self.buffer]),   # [L, n_p, 3]
            "pursuer_vel": torch.stack([f["pursuer_vel"] for f in self.buffer]),   # [L, n_p, 3]
            "accel_cmd":   torch.stack([f["accel_cmd"]   for f in self.buffer]),   # [L, 3]
            "strategy_ids": torch.tensor([f["strategy_id"] for f in self.buffer]), # [L]
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
    从 episode 轨迹中滑窗切分训练样本。

    每个样本:
        obs_history:        [T_obs, 30]  — 30维相对运动学特征序列
        gt_future_delta:    [T_pred, 3]  — 未来位置相对于当前时刻的差值
        gt_strategic:       scalar (0-3) — 4类战术标签
    """

    def __init__(
        self,
        t_obs: int = 10,
        t_pred: int = 10,
        slide_step: int = 2,
        goal_center: Optional[torch.Tensor] = None,
    ):
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.slide_step = slide_step
        self.goal_center = goal_center if goal_center is not None else DEFAULT_GOAL_CENTER

        self.obs_list: List[torch.Tensor] = []
        self.future_list: List[torch.Tensor] = []
        self.strategic_list: List[int] = []

    def add_episode(self, episode: dict):
        """
        将一个 episode 的数据滑窗切分后加入数据集。

        Args:
            episode: TrajectoryRecorder.get_episode() 的输出
        """
        if episode is None:
            return

        L = episode["target_pos"].shape[0]
        n_pursuers = episode["pursuer_pos"].shape[1]
        gc = self.goal_center

        # 批量标注战术标签
        strategic_labels = label_strategic_batch(
            episode["target_pos"],
            episode["target_vel"],
            episode["pursuer_pos"],
            gc,
        )  # [L]

        for start in range(self.t_obs, L - self.t_pred, self.slide_step):
            obs_start = start - self.t_obs
            future_end = start + self.t_pred

            if obs_start < 0 or future_end > L:
                continue

            # ── 构建 30维特征序列 [T_obs, 30] ──
            obs_frames = []
            for t in range(obs_start, start):
                t_pos = episode["target_pos"][t]   # [3]
                t_vel = episode["target_vel"][t]   # [3]
                p_pos = episode["pursuer_pos"][t]   # [n_p, 3]
                p_vel = episode["pursuer_vel"][t]   # [n_p, 3]

                rel_pos = (t_pos.unsqueeze(0) - p_pos).reshape(-1)   # [n_p*3]  [0:9]
                rel_vel = (t_vel.unsqueeze(0) - p_vel).reshape(-1)   # [n_p*3]  [9:18]
                tgt_goal = t_pos - gc                                  # [3]     [18:21]
                pur_goal = (p_pos - gc.unsqueeze(0)).reshape(-1)       # [n_p*3] [21:30]

                frame = torch.cat([rel_pos, rel_vel, tgt_goal, pur_goal])  # [30]
                obs_frames.append(frame)

            obs = torch.stack(obs_frames)  # [T_obs, 30]

            # ── 未来位置差值（相对当前时刻 t=start） ──
            current_pos = episode["target_pos"][start]           # [3]
            future_abs  = episode["target_pos"][start:future_end]  # [T_pred, 3]
            future_delta = future_abs - current_pos.unsqueeze(0)   # [T_pred, 3]

            # ── 战术标签（取观测窗口末帧） ──
            label = int(strategic_labels[start - 1].item())

            self.obs_list.append(obs)
            self.future_list.append(future_delta)
            self.strategic_list.append(label)

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        return {
            "obs_history":     self.obs_list[idx],                                   # [T_obs, 30]
            "gt_future_delta": self.future_list[idx],                                # [T_pred, 3]
            "gt_strategic":    torch.tensor(self.strategic_list[idx], dtype=torch.long),  # scalar
        }

    def get_stats(self) -> dict:
        """返回数据集统计信息"""
        if len(self) == 0:
            return {"total_samples": 0}

        labels = torch.tensor(self.strategic_list)
        counts = torch.bincount(labels, minlength=N_STRATEGIC)
        dist = counts.float() / counts.sum()

        return {
            "total_samples": len(self),
            "strategic_counts": {
                STRATEGIC_LABEL_NAMES[i]: counts[i].item() for i in range(N_STRATEGIC)
            },
            "strategic_distribution": {
                STRATEGIC_LABEL_NAMES[i]: f"{dist[i].item():.3f}" for i in range(N_STRATEGIC)
            },
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """自定义 collate，用于 DataLoader"""
        return {
            "obs_history":     torch.stack([b["obs_history"]     for b in batch]),
            "gt_future_delta": torch.stack([b["gt_future_delta"] for b in batch]),
            "gt_strategic":    torch.stack([b["gt_strategic"]    for b in batch]),
        }
