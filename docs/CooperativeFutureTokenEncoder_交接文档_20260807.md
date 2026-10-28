# CooperativeFutureTokenEncoder 模块交接文档

生成时间：2026-08-07  
所属项目：高机动目标追捕 (`maneuverable-target`)  
设计依据：`docs/网络结构.md`  
代码位置：`omni_drones/learning/modules/cooperative_future.py`  
测试脚本：`scripts/test_cooperative_future.py`  
测试结果：**11/11 全部通过**

---

## 0. 一句话概括

`CooperativeFutureTokenEncoder` 是三架追捕无人机的**协同未来表示编码器**。它将每架无人机的历史、当前团队几何关系、上一周期规划记忆、以及冻结 GMM 预测器的多模态目标轨迹预测，编码为一组 `cooperative_future_tokens [B, N, H, D]`，供下游 Actor Head 做位置/速度输出。

**当前只实现到 `cooperative_future_tokens`，不包含位置/速度输出 Head 和运动学投影。**

---

## 1. 为什么需要这个模块

旧方案中 Future Query 重复查询全部三架无人机的 (3 × L) 个历史点，造成：

- 信息路径冗余（Agent token 已总结历史，队友信息已融合）
- PPO 训练和故障定位难度增加

新方案做了一个关键精简：**Future Query 不再重复查询全部历史 token**，改为只与 GMM 模式进行时间对齐的 Cross-Attention。

---

## 2. 整体架构（数据流）

```text
三架无人机历史状态 [B, N, L, F_hist=9]
        │
        ▼
┌─ AgentHistoryEncoder ─────────────────────────┐
│  共享 MLP 编码 → 过去时间嵌入 →               │
│  拼接 [AGENT] token → History Transformer     │
└───────────────────────────────────────────────┘
        │
        ├─ agent_tokens   [B, N, D]
        └─ history_tokens [B, N, L, D]  (保留用于诊断)
        │
        ▼
┌─ RelationalAgentInteraction ──────────────────┐
│  边特征 r_ij = [dp, dv, ||dp||] ∈ R^7        │
│  带关系偏置的多头注意力                        │
└───────────────────────────────────────────────┘
        │
        ▼
cooperative_agent_tokens [B, N, D]
        │
        ├─ 上一周期规划 ──► PreviousPlanEncoder → prev_plan_tokens [B, N, H, D]
        │
        ▼
┌─ FutureQueryBuilder ──────────────────────────┐
│  q_{i,h} = W_a·ā_i + e_h^future + W_p·ẽ_{i,h}^prev │
└───────────────────────────────────────────────┘
        │
        ▼
initial_future_queries [B, N, H, D]
        │
        ▼
┌─ TimeAlignedGMMCrossAttention ────────────────┐
│  同一时刻只查询 3 条 GMM 模式                 │
│  s += λ_π·log(π_k)  (权重先验)               │
└───────────────────────────────────────────────┘
        │
        ▼
gmm_fused_queries [B, N, H, D]
        │
        ▼
┌─ FactorizedCoordinationBlock × 2 ─────────────┐
│  Temporal Self-Attention (over H)              │
│  → Future Agent Self-Attention (over N)       │
│  → FFN (Pre-Norm)                              │
└───────────────────────────────────────────────┘
        │
        ▼
cooperative_future_tokens [B, N, H, D]
```

---

## 3. 模块清单（7 个类）

| 模块 | 文件位置行号 | 功能 |
|---|---|---|
| `CooperativeFutureConfig` | L32 | 配置 dataclass，所有子模块均有独立开关 |
| `AgentHistoryEncoder` | L71 | 共享历史编码器 → agent_tokens + history_tokens |
| `RelationalAgentInteraction` | L166 | 状态级三机交互（边特征偏置注意力） |
| `PreviousPlanEncoder` | L273 | 上一周期规划轨迹编码（含 mask 处理） |
| `FutureQueryBuilder` | L324 | 从 agent token / 时间嵌入 / 上一规划构造 Future Query |
| `TimeAlignedGMMCrossAttention` | L382 | 时间严格对齐的 GMM 模式查询（含权重先验） |
| `FactorizedCoordinationBlock` | L510 | Temporal SA → Future Agent SA → FFN (Pre-Norm) |
| `CooperativeFutureTokenEncoder` | L597 | 顶层封装，串联所有子模块 |

---

## 4. 输入输出接口

### 4.1 顶层 `forward()` 输入

| 参数 | Shape | 说明 |
|---|---|---|
| `agent_history` | `[B, N, L, 9]` | 相对位置(3) + 相对速度(3) + 加速度(3) |
| `current_pos` | `[B, N, 3]` | 当前相对位置（同一坐标系） |
| `current_vel` | `[B, N, 3]` | 当前相对速度 |
| `prev_plan` | `[B, N, H, 6]` | 上一周期参考位置(3) + 参考速度(3)，相对当前真实位置表达 |
| `prev_plan_mask` | `[B, N, H]` | 有效掩码，第一周期全零 |
| `gmm_mode_tokens` | `[B, K, D]` | 来自 `GMMTrajectoryEncoder` 的 mode_tokens |
| `gmm_point_tokens` | `[B, K, H, D]` | 来自 `GMMTrajectoryEncoder` 的 point_tokens |
| `gmm_weights` | `[B, K]` | GMM 模式权重 |
| `gmm_log_weights` | `[B, K]` or None | 自动从 weights 计算 |
| `agent_history_mask` | `[B, N, L]` or None | 历史有效掩码 |

> **要求**: `pred_horizon (H)` 必须与 GMM 预测点数量一致。若不一致需先时间重采样。

### 4.2 输出字典

| Key | Shape | 用途 |
|---|---|---|
| `cooperative_future_tokens` | `[B, N, H, D]` | **主输出**，下游 Actor Head 用 |
| `agent_tokens` | `[B, N, D]` | 历史摘要（诊断） |
| `history_tokens` | `[B, N, L, D]` | 逐帧历史 token（可视化/消融） |
| `cooperative_agent_tokens` | `[B, N, D]` | 团队交互后的 agent token |
| `initial_future_queries` | `[B, N, H, D]` | GMM 查询前的原始 Future Query |
| `gmm_fused_queries` | `[B, N, H, D]` | GMM Cross-Attention 输出 |
| `agent_state_attention` | `[B, nh, N, N]` | 三机交互注意力（诊断） |
| `gmm_attention` | `[B, nh, N, H, K]` | GMM 模式注意力（诊断） |
| `temporal_attention` | `List[[B, nh, N, H, H]]` | 每层时间注意力 |
| `future_agent_attention` | `List[[B, nh, H, N, N]]` | 每层三机未来注意力 |

---

## 5. 配置参数

```python
@dataclass
class CooperativeFutureConfig:
    num_agents: int = 3          # N
    num_modes: int = 3           # K (与 GMM 一致)
    history_len: int = 10        # L
    pred_horizon: int = 10       # H

    history_dim: int = 9         # F_hist: pos(3)+vel(3)+acc(3)
    prev_plan_dim: int = 6       # F_plan: ref_pos(3)+ref_vel(3)

    d_model: int = 128           # D
    num_heads: int = 4
    ff_dim: int = 256

    history_layers: int = 2
    coordination_layers: int = 2

    dropout: float = 0.0         # PPO 策略要求 0.0
    weight_prior_scale: float = 0.5
    eps: float = 1e-6

    use_agent_interaction: bool = True
    use_previous_plan: bool = True
    use_weight_prior: bool = True
    use_future_agent_attention: bool = True
    use_self_history_cross_attention: bool = False  # 默认关闭
```

---

## 6. 关键设计决策

### 6.1 默认不查询历史 Cross-Attention
`use_self_history_cross_attention = False`。Future Query 只与 GMM 模式交互，不重复查询全部历史 token。`history_tokens` 仍保留输出用于可视化/消融。

### 6.2 时间严格对齐的 GMM 查询
第 i 架无人机第 h 个 Query 只查询第 h 个时刻的 3 条 GMM 模式：
```
{ c_{0,h}, c_{1,h}, c_{2,h} }
```
而非全部 K × H 个 GMM 点。

### 6.3 无固定无人机 ID Embedding
所有模块不使用固定的无人机编号 Embedding，保证排列等变性。

### 6.4 可学习实体的共享
- `[AGENT]` token：三架无人机共享
- `past_time_embedding`：共享时间嵌入
- `history_point_encoder`：共享 MLP
- `edge_encoder`：共享边特征编码器

### 6.5 不重复查询队友历史
每架无人机的历史已压缩进 Agent token，队友信息又经过状态级三机注意力融合。额外查询会造成冗余。

### 6.6 Pre-Norm 结构
所有 Block 使用 Pre-Norm（而非 Post-Norm），更适合深度 RL 训练。

### 6.7 零初始化输出投影
`out_proj` 权重初始化为零，使残差连接初始等价于恒等映射，稳定训练。

### 6.8 Dropout = 0.0
PPO 策略网络中不使用 Dropout，避免采样阶段与更新阶段因随机 Dropout 导致 log-prob 不一致。

---

## 7. 如何在 PPO 策略中集成

### 7.1 典型使用流程

```python
from omni_drones.learning.modules.cooperative_future import (
    CooperativeFutureConfig,
    CooperativeFutureTokenEncoder,
)
from omni_drones.learning.predictor.gmm_encoder import GMMTrajectoryEncoder

# 1. 配置
future_cfg = CooperativeFutureConfig()
gmm_encoder_cfg = GMMEncoderConfig()  # 需要 d_model 一致

# 2. 初始化
gmm_encoder = GMMTrajectoryEncoder(gmm_encoder_cfg)
cooperative_encoder = CooperativeFutureTokenEncoder(future_cfg)

# 3. 冻结预测器 → GMM 编码
with torch.no_grad():
    predictor_out = predictor_net(obs_history)  # 冻结的预测器
    gmm_out = gmm_encoder(
        predictor_out["gmm_means"].detach(),
        predictor_out["gmm_vars"].detach(),
        predictor_out["gmm_weights"].detach(),
    )

# 4. 协同未来编码（有梯度）
future_out = cooperative_encoder(
    agent_history=obs["history"],
    current_pos=obs["relative_pos"],
    current_vel=obs["relative_vel"],
    prev_plan=prev_trajectory,
    prev_plan_mask=prev_mask,
    gmm_mode_tokens=gmm_out["mode_tokens"],
    gmm_point_tokens=gmm_out["point_tokens"],
    gmm_weights=gmm_out["weights"],
    agent_history_mask=obs.get("history_mask"),
)

# 5. 下游 Actor Head（待实现）
future_tokens = future_out["cooperative_future_tokens"]  # [B, N, H, D]
# positions = self.position_head(future_tokens)   # 待实现
# velocities = self.velocity_head(future_tokens)   # 待实现
```

### 7.2 prev_plan 的准备

每个 RL step 后需要维护上一周期规划：

```python
# 删除已执行的轨迹点
# 剩余轨迹相对于当前真实位置重新表达
# 补齐或重采样至长度 H
# 无效部分 prev_plan_mask = 0

# 第一周期:
prev_plan = torch.zeros(B, N, H, 6)
prev_plan_mask = torch.zeros(B, N, H)
```

---

## 8. 训练监控指标

此阶段建议至少记录以下指标以诊断训练异常：

| 指标 | 含义 |
|---|---|
| `agent_token_norm` | Agent token L2 范数 |
| `cooperative_agent_token_norm` | 交互后 token L2 范数 |
| `future_token_norm` | 最终输出 L2 范数 |
| `gmm_attention_entropy` | GMM 交叉注意力熵 |
| `gmm_attention_mean_per_mode` | 三条模式的平均注意力 |
| `agent_state_attention_entropy` | 三机交互注意力熵 |
| `future_agent_attention_entropy` | 三机未来注意力熵 |
| `future_token_agent_cosine_similarity` | 三机 Future token 余弦相似度 |
| `future_token_time_cosine_similarity` | 同一无人机不同时刻相似度 |
| `prev_plan_token_norm` | 上一规划 token L2 范数 |

### 关键异常信号

| 现象 | 可能原因 |
|---|---|
| 三机 Future token 余弦相似度 → 1.0 | 协同表示坍塌 |
| GMM 注意力熵 ≈ ln(K) (最大) | 无法区分三个模式 |
| GMM 注意力始终 ≒ 权重 | 忽略无人机几何状态 |
| `prev_plan_mask=0` 和 `1` 输出几乎相同 | 上一轨迹记忆无效 |
| 某模块梯度 → 0 | 该信息路径未被 PPO 使用 |

---

## 9. 测试结果

运行 `python3 scripts/test_cooperative_future.py`：

| # | 测试项 | 状态 | 说明 |
|---|---:|---|---|
| 1 | 形状测试 | ✅ | 10 种输出形状全部正确 |
| 2 | 无人机排列等变性 | ✅ | 交换 agent 0/1 → 输出相应交换，差异 < 2e-6 |
| 3 | GMM 模式排列等变性 | ✅ | 交换 mode 0/1 → future tokens 不变，attention 交换 |
| 4 | 上一计划 Mask 测试 | ✅ | mask=0 时改变 prev_plan 不影响输出 |
| 5 | 时间敏感性 | ✅ | 不同时刻 query 和 future token 不相同 |
| 6 | 队友敏感性 | ✅ | agent1 变化 → agent0 future tokens 变化 |
| 7 | 梯度测试 | ✅ | 全部 7 个子模块获得非零梯度 |
| 8 | 额外检查 | ✅ | 无 NaN/Inf，三种配置开关组合正常 |
| 9 | gmm_log_weights 自动计算 | ✅ | None 时自动从 weights 计算 |
| 10 | history_mask 测试 | ✅ | mask 生效，None 正常 |
| 11 | 子模块独立运行 | ✅ | 6 个子模块独立 forward 正常 |

---

## 10. 尚未实现（下一步工作）

按优先级排序：

| 优先级 | 任务 | 说明 |
|---|---:|---|
| **P0** | 位置 Head 和速度 Head | 将 `cooperative_future_tokens` 映射到具体轨迹坐标 |
| **P0** | 运动学投影层 | 保证轨迹满足速度/加速度约束 |
| **P0** | PPO 策略集成 | 在 `omni_drones/learning/_ppo.py` 中接入本模块 |
| P1 | 上一周期轨迹维护逻辑 | 在 RL step 循环中更新 prev_plan |
| P1 | 诊断指标记录 | 实现第 8 节中的监控指标（wandb log） |
| P2 | 消融实验 | 逐个关闭配置开关，验证各信息路径有效性 |
| P2 | Self History Cross-Attention | 若消融证明有用再加入 |

---

## 11. 依赖与兼容性

### 11.1 上游依赖

- `GMMTrajectoryEncoder` (`omni_drones/learning/predictor/gmm_encoder.py`) — 提供 GMM 编码
- GMM 预测器（冻结）— 提供 `gmm_means/vars/weights`
- PyTorch ≥ 2.0

### 11.2 维度约束

| 约束 | 值 |
|---|---|
| `CooperativeFutureConfig.d_model` | 必须等于 `GMMEncoderConfig.d_model`（默认 128） |
| `CooperativeFutureConfig.pred_horizon` | 必须等于 GMM 预测器的 H（默认 10） |
| `CooperativeFutureConfig.num_modes` | 必须等于 GMM 的 K（默认 3） |

### 11.3 坐标系要求

`current_pos` 和 `current_vel` 必须在同一坐标系中，且 `agent_history` 中的位置/速度应与之一致。

---

## 12. 故障排查

| 问题 | 检查项 |
|---|---|
| 输出全是 NaN | 检查 GMM 输入是否有效（`gmm_means` 无 NaN） |
| 梯度为零 | 检查 GMM 输入是否 detach（必须 detach） |
| 维度不匹配 | 确认 `pred_horizon` / `num_modes` / `d_model` 与 GMM 一致 |
| 输出不随队友变化 | 检查 `use_agent_interaction` 是否为 True |
| Transformer 警告 `enable_nested_tensor` | 已设置 `enable_nested_tensor=False`，无影响 |


实现总结
新增文件
文件	说明
cooperative_future.py	核心模块实现 (~700 行)
test_cooperative_future.py	完整单元测试 (~450 行)
实现的模块
按照 网络结构.md 的规范，实现了以下 7 个类：

类名	功能
CooperativeFutureConfig	配置 dataclass，所有模块均有独立开关
AgentHistoryEncoder	将三架无人机历史编码为 [AGENT] token + history tokens
RelationalAgentInteraction	带关系偏置的三机注意力 (边特征 7维 → 偏置)
PreviousPlanEncoder	编码上一周期规划轨迹 + 有效 mask 处理
FutureQueryBuilder	从 agent token + 时间嵌入 + 上一规划构造 Future Query
TimeAlignedGMMCrossAttention	时间严格对齐的 GMM 模式查询 (含权重先验偏置)
FactorizedCoordinationBlock	Temporal Self-Attention → Future Agent Self-Attention → FFN (Pre-Norm, 2层)
CooperativeFutureTokenEncoder	顶层模块，串联所有子模块，输出 cooperative_future_tokens
测试覆盖 (11/11 通过)
#	测试项	状态
1	形状测试 (所有输出维度)	✅
2	无人机排列等变性	✅
3	GMM 模式排列等变性	✅
4	上一计划 Mask 测试 (mask=0 隔离)	✅
5	时间敏感性 (不同时刻 query 不能相同)	✅
6	队友敏感性 (队友变化影响 agent0 future tokens)	✅
7	梯度测试 (全部 7 个模块非零梯度)	✅
8	额外检查 (NaN/Inf, 配置开关, 无 Dropout/BatchNorm/ID Embedding)	✅
9	gmm_log_weights 自动计算	✅
10	history_mask 测试	✅
11	子模块独立运行	✅
关键设计决策
默认删除 Future Query → History 的 Cross-Attention (use_self_history_cross_attention=False)
2 层 FactorizedCoordinationBlock (非 3 层)
零初始化 out_proj 权重以稳定训练
Pre-Norm 结构 (优于 Post-Norm for RL)
仅实现到 cooperative_future_tokens，无位置/速度输出 Head 和运动学投影