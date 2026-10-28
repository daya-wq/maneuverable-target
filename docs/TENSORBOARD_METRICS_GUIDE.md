# 新版精简 TensorBoard 指标说明

本文档对应 `HideAndSeek + MAPPO + minimal_v3` 主线。第一版TensorBoard只保留六组必要信息：训练进度与速度、PPO策略、Advantage与Value、预测网络、奖励与终止原因、目标机动。

完整内部诊断仍可在代码和W&B中使用，但不再全部写入TensorBoard。旧事件文件已清理，新训练不会再混入历史标签。

## 1. 统一横坐标和聚合规则

所有曲线的横坐标 `step` 都是**已完成的PPO外层更新次数**，不是物理仿真步数。

```text
一次update采样量 = 并行环境数 × rollout长度
本轮高速实验一次update = 128 × 32 = 4,096个环境帧
旧低速基线一次update = 512 × 128 = 65,536个环境帧
```

不同指标在一个update内的计算范围如下：

- PPO与Value：每个mini-batch先计算，本次update的所有PPO epoch和mini-batch再取平均。
- 预测网络：在有效未来窗口上计算，再对预测器全部mini-batch和epoch取平均。
- 奖励与终止原因：只使用本次rollout中已经结束的完整episode。
- 目标机动：使用本次rollout中全部有效目标状态；跨episode reset的差分样本会被过滤。

## 2. 训练进度、环境采样与速度

| 指标名称 | 计算过程 | 含义 |
|---|---|---|
| `train/update` | 当前完成的PPO外层更新编号 | 与TensorBoard横坐标相同 |
| `env_frames` | 从训练开始累计采集的环境帧数 | 不同并行环境数或rollout长度之间比较样本量时使用 |
| `rollout_fps` | 采样器报告的环境帧吞吐 | 只表示仿真与策略采样速度，不包含网络更新 |
| `system/rollout_wall_seconds` | 从上次update结束到本次rollout返回的实际时间 | 判断一次rollout采样花了多久 |
| `system/policy_update_seconds` | 预测器、Actor与Critic完成一次更新的墙钟时间 | 判断反向传播是否成为瓶颈 |
| `system/effective_env_frames_per_second` | `本轮环境帧数 ÷ (rollout耗时+后处理与更新耗时)` | 最重要的端到端训练速度，适合比较512与1024环境 |
| `system/torch_peak_memory_allocated_gib` | 本次update内PyTorch峰值实际分配显存÷GiB | 判断batch或环境数是否接近OOM；不含Isaac/Vulkan显存 |

环境数、rollout长度、仿真步长、规划步长等静态参数保存在TensorBoard的 `run/config` 文本中，不重复画成没有变化的水平曲线。

## 3. PPO策略指标

记：

```text
ratio = exp(log π_new(a|s) - log π_old(a|s))
```

| 指标名称 | 计算过程 | 含义 |
|---|---|---|
| `drone/approx_kl` | `mean(logπ_old-logπ_new)` | 新旧策略近似KL；突然增大表示单次策略变化过猛 |
| `drone/ratio_mean` | 对所有ratio取均值 | 应接近1，但不能单独用于判断稳定性 |
| `drone/ratio_std` | ratio的总体标准差 | 越大表示新旧策略差异越分散 |
| `drone/ratio_clip_fraction` | ratio落在 `[1-ε,1+ε]` 之外的比例 | 很高说明大量样本被PPO裁剪，学习率或优势尺度可能过大 |
| `drone/policy_loss` | `-mean(min(ratio·A, clip(ratio)·A))` | PPO裁剪代理损失；主要观察突变，不要求固定为正或负 |

`policy_loss`不包含熵、KL惩罚和锚定项，因此它不是Actor全部项相加后的总损失。

## 4. Advantage与Value指标

优势使用GAE：

```text
δ_t = r_t + γ(1-done_t)V(s_{t+1}) - V(s_t)
A_t = δ_t + γλ(1-done_t)A_{t+1}
Return_t = A_t + V(s_t)
```

rollout末尾未终止的环境会使用最后状态的Value做bootstrap；真正终止的状态通过done掩码切断bootstrap。

| 指标名称 | 计算过程 | 含义 |
|---|---|---|
| `drone/advantages_mean` | GAE归一化前所有优势的均值 | 正值表示实际结果整体高于旧Value预期，负值反之 |
| `drone/advantages_std` | GAE归一化前所有优势的标准差 | 表示策略梯度信号尺度；突增常来自奖励或Value异常 |
| `drone/value_loss` | `max(L(Return,V_new), L(Return,V_clipped))`后取均值 | Critic的PPO裁剪损失；`L`由配置选择Huber或MSE |
| `drone/explained_var` | `1-MSE(V_new,Return)/Var(Return)` | 越接近1越好；约0表示几乎没解释力；小于0表示比常数预测更差 |
| `drone/value_abs_error` | `mean(abs(V_new-Return))` | Critic典型预测误差；启用ValueNorm时处于归一化空间 |
| `drone/value_clip_fraction` | `abs(V_new-V_old)>ε`的样本比例 | 很高说明Critic单次更新变化过大 |
| `drone/returns_mean` | ValueNorm之前Return的均值 | 原始奖励尺度下的Value目标中心 |
| `drone/bootstrap_value_mean` | rollout末状态反归一化Value的均值 | 异常突变会通过GAE影响整段rollout |

## 5. 战略意图与GMM轨迹预测指标

预测器总损失：

```text
L_total = λ战略L战略 + λGMMLGMM
        + λ速度L速度 + λ加速度L加速度 + λ高度L高度
```

| 指标名称 | 计算过程 | 含义 |
|---|---|---|
| `prediction/loss_total` | 上述五项加权求和 | 预测网络实际用于反向传播的总目标 |
| `prediction/loss_strategic` | 四类战略标签的交叉熵 | 战略意图分类损失，越低越好 |
| `prediction/loss_gmm_nll` | 三个高斯轨迹分量对真实未来轨迹的负对数似然 | 同时评价模式权重、轨迹均值和方差，越低越好 |
| `prediction/strategic_accuracy` | `argmax(战略概率)==真实战略标签`的比例 | 四分类准确率，范围0到1 |
| `prediction/minADE` | 三条预测轨迹分别计算全时域平均位置误差，取最小值后跨样本平均 | 判断三种可能轨迹中是否至少一条覆盖真值，单位米 |
| `prediction/FDE` | GMM加权混合轨迹的最后一点与真实终点距离均值 | 最终位置误差，单位米 |
| `prediction/gmm_weight_entropy` | `mean(-Σ_k w_k log(w_k))` | 模式权重不确定性；接近0表示单模式，三模式均匀时约为`ln(3)` |
| `prediction/valid_windows` | 去除跨episode、历史不足等无效片段后剩余的监督窗口数 | 为0表示本次update没有可训练的预测样本 |

## 6. 奖励指标及横坐标

奖励曲线继续以 `update` 为横坐标，使同一批轨迹的奖励能够与随后产生的KL、ratio和loss对齐。

一个奖励点的计算过程：

```text
单个完整episode：某项奖励逐步累计 ÷ episode实际长度
本次update：对本轮结束的全部episode取平均
```

例如：update 20的rollout结束了300局，`reward/progress`就是这300局“每步平均进展奖励”的均值。未结束episode不进入当前点；如果本轮没有episode结束，本轮不会产生奖励与终止比例点。

| 指标名称 | 计算过程 | 含义 |
|---|---|---|
| `reward/progress` | `0.7×最近追捕机归一化接近量 + 0.3×三机平均归一化接近量`；终止步记0 | 正值表示接近目标，负值表示被拉开 |
| `reward/safety_pair` | 机间距离低于安全距离时的负二次惩罚；终止步记0 | 绝对值越大表示机间靠得过近越频繁 |
| `reward/safety_env` | `-0.35 × mean[clip((d_ref-d_nearest_boundary)/d_ref,0,1)^2]`，终止步记0；attempt_03的d_ref=0.5米，attempt_04为5米 | 衡量贴边/贴地/贴顶；不同安全距离下，绝对数值不能直接横比 |
| `reward/control` | 原始有界控制的幅值范数惩罚与相邻控制变化范数惩罚之和；终止步记0 | 不是实际电机能量；旧03/04每点6个控制，05为3个控制，跨头比较时须说明参数化改变 |
| `reward/time` | 每个非终止步固定为 `-m3_time_penalty` | 鼓励更快完成抓捕 |
| `reward/terminal` | 抓捕`+30`、目标进区`-30`、碰撞/越界/降落为`-m3_terminal_collision_penalty`、超时`-3`，非终止步为0；attempt_03为-15，attempt_04为-40 | 最终事件对episode平均奖励的贡献；比较run前须核对配置 |
| `reward/dense_raw` | 非终止步将进展、安全、控制和时间直接相加，不做裁剪；终止步记0 | 实际执行的稠密奖励，等于四项分项之和 |
| `reward/reward` | 非终止步使用稠密奖励，终止步只使用终止奖励 | 环境真正交给PPO的每步平均奖励 |
| `reward/episode_return` | 完整episode的实际奖励逐步求和，不除以episode长度 | 一整局累计回报，同时受表现和episode长度影响 |

新版统计满足：

```text
reward/reward
= reward/dense_raw + reward/terminal
```

稠密奖励不再裁剪，`m3_dense_raw` 就是进展、安全、控制和时间四项之和，因此

```text
reward/dense_raw
= reward/progress + reward/safety_pair
  + reward/safety_env + reward/control
  + reward/time
```

（终止步各项均记0，等式同样成立。）内部保留m3版本字段，TensorBoard显示为reward分组，预测器显示为prediction分组，结束原因显示为termination分组。

固定512环境、128步时，update和环境帧一一对应，使用update最直观。比较512与1024环境时，必须同时参考 `env_frames`，不能直接比较同一update位置。

## 7. Episode结束类型

每个结束episode只归入一个原因。若同一步发生多个事件，优先级为：

```text
抓捕成功 > 目标进区 > 撞地 > 降落 > 追捕机相撞 > 越界 > 超时
```

每条终止比例的计算都是：

```text
该原因结束的episode数 ÷ 本次update结束的episode总数
```

| 指标名称 | 计算过程 | 含义 |
|---|---|---|
| `train/episodes_completed` | 本次rollout内结束的episode数量 | 奖励和终止比例的样本数；太少时曲线波动较大 |
| `termination/capture` | 原因编码1的比例 | 抓捕成功率 |
| `termination/target_goal` | 原因编码2的比例 | 目标进入守区的比例 |
| `termination/ground_collision` | 原因编码3的比例 | 任一追捕机撞地的比例 |
| `termination/drone_collision` | 原因编码4的比例 | 追捕机之间相撞的比例 |
| `termination/timeout_near_miss` | 超时且整局最近距离不超过擦肩阈值的比例 | 曾经接近但没有完成最后抓捕；默认阈值1米 |
| `termination/timeout_no_close` | 超时且整局最近距离始终大于擦肩阈值的比例 | 整局都没有有效接近目标 |
| `termination/out_of_bounds` | 原因编码7的比例 | 追捕机超出场地或高度上界的比例 |
| `termination/landed` | 原因编码8的比例 | 追捕机低高度且低速度，被判为降落的比例 |
| `train/stats.episode_min_target_distance` | 每局全程最小追捕距离，再跨结束episode平均 | 区分“完全追不上”和“已经接近但没抓住” |

当 `train/episodes_completed>0` 时，八种终止比例之和应接近1。旧的原因编码直方图不再写入。

## 8. 目标机动指标

所有目标指标以 `target/` 开头，统计当前rollout。第一版只保留能描述机动强度和模式覆盖的必要统计，不保留同一物理量的全部重复分位数。

| 指标名称 | 计算过程 | 含义 |
|---|---|---|
| `target/speed_mps_mean/p95` | `||v_t||`的均值与95分位数 | 目标实际速度，单位m/s；p95比单个尖峰更适合报告 |
| `target/realized_accel_mps2_mean/p95` | `||(v_t-v_{t-1})/0.05||`的均值与95分位数，过滤跨reset样本 | 目标在20 Hz规划尺度真正实现的加速度，单位m/s² |
| `target/turn_rate_deg_s_mean/p95` | 相邻速度方向夹角`÷0.05`的均值与95分位数；任一速度低于0.25 m/s时不统计 | 目标实际转向强度，单位°/s |
| `target/command_accel_mps2_p95` | `||a_command||`的95分位数 | 目标控制器大多数高强度指令的水平，表示“想怎么机动” |
| `target/sampled_max_speed_mps_mean/max` | 本批episode采样速度上限的均值与最大值 | 检查本轮每局15–17 m/s随机化是否生效（旧低速实验为3.0–3.5） |
| `target/sampled_max_accel_mps2_mean/max` | 本批episode采样加速度上限的均值与最大值 | 检查本轮每局10–12 m/s²随机化是否生效（旧低速实验为3.0–3.5） |
| `target/sampled_initial_speed_mps_max` | 本批reset初速度大小的最大值 | 实机一致基线应为0；非零直接说明初始化错误 |
| `target/tactical_switches_per_s` | 战略编号变化次数÷有效采样总时长 | 四种战略切换频率；过高可能是阈值抖动 |
| `target/maneuver_switches_per_s` | 机动编号变化次数÷有效采样总时长 | 九种局部机动切换频率 |
| `target/tactical_0_fraction` | 战略0样本数÷有效战略样本数 | `FAR_RUSH`占比 |
| `target/tactical_1_fraction` | 战略1样本数÷有效战略样本数 | `CLOSE_EVADE`占比 |
| `target/tactical_2_fraction` | 战略2样本数÷有效战略样本数 | `GOAL_PRIORITY`占比 |
| `target/tactical_3_fraction` | 战略3样本数÷有效战略样本数 | `GOAL_SPRINT`占比 |
| `target/maneuver_0_fraction` | 机动0样本比例 | 直行占比 |
| `target/maneuver_1_fraction` | 机动1样本比例 | 左转占比 |
| `target/maneuver_2_fraction` | 机动2样本比例 | 右转占比 |
| `target/maneuver_3_fraction` | 机动3样本比例 | 爬升占比 |
| `target/maneuver_4_fraction` | 机动4样本比例 | 俯冲占比 |
| `target/maneuver_5_fraction` | 机动5样本比例 | 左转爬升占比 |
| `target/maneuver_6_fraction` | 机动6样本比例 | 左转俯冲占比 |
| `target/maneuver_7_fraction` | 机动7样本比例 | 右转爬升占比 |
| `target/maneuver_8_fraction` | 机动8样本比例 | 右转俯冲占比 |

判断目标是否达到“高机动”不能只看控制指令或单个最大尖峰。至少同时报告：`speed_mps_p95`、`realized_accel_mps2_p95`、`turn_rate_deg_s_p95`、战略/机动占比和两类切换频率。

## 9. 本轮新增的推力与episode指标

| 指标名称 | 计算过程 | 含义 |
|---|---|---|
| `safety/pursuer_motor_thrust_fraction_max` | 每个规划采样读取全部追捕机电机转速比例的平方，再对当前rollout取最大值 | 实际电机推力分数，本轮应不超过0.70（允许浮点误差） |
| `safety/target_motor_thrust_fraction_max` | 同上，对目标机电机取最大值 | 验证目标也使用同一70%推力限制 |
| `train/stats.episode_duration_s` | 每个本轮已结束episode的规划步数×0.05秒，再跨episode平均 | 真实回合持续时间；成功局抓捕时间单独保存在评测summary中 |
| `train/stats.target_maneuver_switches` | 在100 Hz控制步逐次比较目标局部机动编号，累计后跨结束episode平均 | 每局实际发生多少次机动模式改变，不是每秒切换率 |
| `train/stats.target_tactical_switches` | 在100 Hz控制步逐次比较四类战略编号，累计后跨结束episode平均 | 每局战略模式改变次数 |

当前79项精确白名单，内部统计保留完整；新标签只写入本轮新run，不修改历史事件文件。
场景、70%推力能力推导和夜间评测计划见`docs/OVERNIGHT_70PCT_TEST_PLAN.md`。

## 10. TensorBoard打开方式

远程服务器：

```bash
cd /data/uavlab/maneuverable-target
/home/uavlab/miniconda3/envs/sim/bin/tensorboard \
  --logdir runs/本次训练目录 \
  --host 127.0.0.1 \
  --port 6006
```

本机Windows PowerShell：

```powershell
ssh -C -N -L 16006:127.0.0.1:6006 uav-tailscale
```

浏览器打开：`http://127.0.0.1:16006`。

## 2026-10-22补充：直接位置/速度轨迹头

新入口`--config-name=train_pv`使用400步/20 s回合、0.05 s规划和0.01 s底层控制；rollout仍是独立参数（默认128）。场地高度现为50 m，推重比3.2和70%推力上限未改。

| 指标名称 | 计算过程 | 含义 |
|---|---|---|
| `drone/pv_consistency_loss`：参考位置/速度一致性损失 | 将Actor确定性输出分别映射为单位球内位置增量率d和速度v；对第2–10个点计算`mean((d_k-(v_(k-1)+v_k)/2)^2)`，再对本次更新mini-batch取平均 | 越小表示两种参考更一致；无量纲，不是实际米制跟踪误差，不证明动力学可达或围捕成功；乘0.01加入Actor目标 |

横坐标仍为PPO外层更新次数。使用确定性输出计算辅助损失，不改采样动作的PPO log-prob；第一段不使用目标相对速度冒充追捕机世界速度。详细限制与验证见`PV50_SCENE_AND_TIMING.md`。原夜间TensorBoard目录不会自动混入本次诊断run。
