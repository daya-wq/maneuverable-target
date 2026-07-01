# 双层意图 Query-GMM 轨迹预测网络训练报告

生成脚本：`scripts/test_dual_intent_network.py`  
结果目录：`results/dual_intent_network_test`

## 1. 实验目的

本次实验是双层意图轨迹预测网络的原型验证，不是最终论文级结果。它主要检查以下链路是否能端到端跑通：

- 构建 30 维相对运动学输入。
- 意图输入和轨迹预测使用不同频率。
- 输出 3 类战术意图概率。
- 输出 9 类机动意图概率。
- 使用 GMM 头输出目标未来轨迹概率分布。
- 将指标、曲线、预测图和模型 checkpoint 保存到 `results` 目录。

## 2. 训练数据来源

轨迹目录：`results/dual_intent_6dof_battles/trajectory_npz/`

轨迹文件：`battle_dual_intent_01.npz` 到 `battle_dual_intent_10.npz`

数据规模：

- 轨迹文件数量：`10`
- 原始轨迹步长：`0.01 s`
- 本次最大采样数：`4096`
- 训练集样本数：`3276`
- 验证集样本数：`820`
- 训练/验证划分比例：`80% / 20%`
- 随机种子：`20260615`

样本由保存好的六自由度规则对抗轨迹滑窗切片得到。原始 `.npz` 提供 `target_pos`、`target_vel`、`pursuer_pos`、`tactical_hist`、`maneuver_mode_hist`、`goal_center`、`dt`。追捕方速度由 `np.gradient(pursuer_pos, dt_raw, axis=0)` 从追捕方位置序列重建。

## 3. 输入特征定义

每个意图输入帧为 `30` 维：

- `target_pos - pursuer_pos`，目标相对 3 架追捕方的位置：`9` 维。
- `target_vel - pursuer_vel`，目标相对 3 架追捕方的速度：`9` 维。
- `target_pos - goal_center`，目标相对守区中心的位置：`3` 维。
- `pursuer_pos - goal_center`，3 架追捕方相对守区中心的位置：`9` 维。

总维度为 `9 + 9 + 3 + 9 = 30`。该输入满足当前设计要求：包含目标相对追捕方的位置、速度，也显式包含目标和追捕方相对守区的位置。输入在全部选中样本上计算数据集级 `mean/std`，并做标准化。

## 4. 频率和时间窗

意图输入：

- 历史长度：`0.5 s`
- 采样间隔：`0.05 s`
- 输入帧数：`10`
- 张量形状：`(B, 10, 30)`

轨迹预测：

- 未来长度：`0.5 s`
- 采样间隔：`0.02 s`
- 输出帧数：`25`
- 轨迹真值张量：`(B, 25, 3)`

这里刻意保留了你要求的不同频率设计：意图分支读取 `0.05 s` 间隔的低频历史序列，轨迹 GMM 头输出 `0.02 s` 间隔的高频未来相对位移序列。轨迹真值是相对当前目标位置的未来位移，不是绝对坐标。

## 5. 标签定义和分布

### 战术意图标签

战术意图标签来自 `tactical_hist`，共 3 类：

| ID | 标签 | 含义 |
|---:|---|---|
| 0 | `Far_Rush` | 目标距离追捕方较远时，主要直冲守区。 |
| 1 | `Close_Evade` | 近距规避，规避最近追捕方，远离其他追捕方，并叠加 9 类机动。 |
| 2 | `Goal_Sprint` | 目标接近守区时，只向守区冲刺。 |

本次 `4096` 个样本中的战术意图分布：

| 标签 | 样本数 | 占比 |
|---|---:|---:|
| `Far_Rush` | 438 | 10.69% |
| `Close_Evade` | 2894 | 70.65% |
| `Goal_Sprint` | 764 | 18.65% |

`Close_Evade` 占比最高，战术分类的整体准确率会受到这个主类影响。后续正式评估不能只看整体 accuracy，还需要每类 Precision、Recall、F1 和混淆矩阵。

### 机动意图标签

机动意图标签来自 `maneuver_mode_hist`，共 9 类：

| ID | 标签 | 含义 |
|---:|---|---|
| 0 | `Straight` | 直飞 |
| 1 | `Left-Turn` | 左转 |
| 2 | `Right-Turn` | 右转 |
| 3 | `Climb` | 爬升 |
| 4 | `Dive` | 俯冲 |
| 5 | `Left-Climb` | 左转爬升 |
| 6 | `Left-Dive` | 左转俯冲 |
| 7 | `Right-Climb` | 右转爬升 |
| 8 | `Right-Dive` | 右转俯冲 |

本次 `4096` 个样本中的机动意图分布：

| 标签 | 样本数 | 占比 |
|---|---:|---:|
| `Straight` | 1587 | 38.75% |
| `Left-Turn` | 394 | 9.62% |
| `Right-Turn` | 309 | 7.54% |
| `Climb` | 386 | 9.42% |
| `Dive` | 381 | 9.30% |
| `Left-Climb` | 296 | 7.23% |
| `Left-Dive` | 238 | 5.81% |
| `Right-Climb` | 291 | 7.10% |
| `Right-Dive` | 214 | 5.22% |

机动分类损失只在真实战术意图为 `Close_Evade` 的样本上计算。`Far_Rush` 和 `Goal_Sprint` 是规则定义下非随机机动主导的状态，不强行监督 9 类机动。

## 6. 网络结构

实现文件：`omni_drones/learning/predictor/dual_intent_gmm_predictor.py`

主类：`DualIntentQueryGMMPredictor`

网络配置：

- `obs_dim`：`30`
- `d_model`：`128`
- `lstm_layers`：`1`
- `n_heads`：`4`
- `n_components`：`3`
- `pred_horizon`：`25`
- `dropout`：`0.1`
- `min_var`：`0.001`
- `detach_intents_for_gmm`：`false`

信息流：

- 输入 `obs_history`：`(B, 10, 30)`
- BiLSTM 时序编码得到 `H_env`：`(B, 10, 128)`
- 战术 Query 对 `H_env` 做 Cross-Attention，得到 `h_strategic`：`(B, 128)`
- 机动 Query 对 `H_env` 做 Cross-Attention，得到 `h_maneuver`：`(B, 128)`
- `h_strategic` 进入战术分类头，输出 `strategic_logits`：`(B, 3)`
- `h_maneuver` 与 `strategic_probs` 拼接后进入机动分类头，输出 `maneuver_logits`：`(B, 9)`
- `h_strategic`、`h_maneuver`、`strategic_probs`、`maneuver_probs` 融合后进入 GMM 轨迹头

GMM 输出：

- `gmm_weights`：`(B, 3)`
- `gmm_means`：`(B, 3, 25, 3)`
- `gmm_vars`：`(B, 3, 25, 3)`
- `mixed_mu`：`(B, 25, 3)`
- `mixed_var`：`(B, 25, 3)`

## 7. 损失函数

实现类：`DualIntentGMMLoss`

总损失：

```text
L_total =
  1.0 * L_strategic
  + 0.5 * L_maneuver
  + 1.0 * L_gmm_nll
  + 0.02 * L_speed
  + 0.02 * L_accel
  + 0.02 * L_height
```

各项含义：

- `L_strategic`：3 类战术意图交叉熵，所有样本参与计算。
- `L_maneuver`：9 类机动意图交叉熵，只在 `Close_Evade` 样本上计算。
- `L_gmm_nll`：未来相对轨迹在 GMM 分布下的负对数似然。
- `L_speed`：预测速度超过 `3.0 m/s` 的软惩罚。
- `L_accel`：预测加速度超过 `3.0 m/s^2` 的软惩罚。
- `L_height`：预测绝对高度低于 `0.2 m` 的软惩罚。

报告中打印的 `speed`、`accel`、`height penalty` 是未乘权重的原始值。`GMM NLL` 可以为负，因为它是连续概率密度的负对数似然，不是离散概率的负对数。

## 8. 训练参数

训练命令：

```bash
python3 scripts/test_dual_intent_network.py --epochs 30 --max-samples 4096 --batch-size 128 --device cpu
```

优化设置：

- 优化器：`AdamW`
- 学习率：`1e-3`
- Weight decay：`1e-4`
- Batch size：`128`
- Epochs：`30`
- 梯度裁剪：`5.0`
- 运行设备：`CPU`

物理约束参数：

- 最大速度：`3.0 m/s`
- 最大加速度：`3.0 m/s^2`
- 最低高度：`0.2 m`
- 轨迹预测 dt：`0.02 s`

## 9. 最终验证集指标

第 30 个 epoch 的验证集指标：

| 指标 | 数值 |
|---|---:|
| `loss_total` | -38.355590 |
| `loss_strategic` | 0.225617 |
| `loss_maneuver` | 1.531752 |
| `loss_gmm_nll` | -50.524661 |
| `loss_speed_penalty` | 0.000000 |
| `loss_accel_penalty` | 558.878738 |
| `loss_height_penalty` | 0.000153 |
| `strategic_accuracy` | 0.912195 |
| `maneuver_accuracy_close_evade` | 0.452313 |
| `traj_mse_delta` | 0.158020 |
| `close_evade_fraction` | 0.745122 |

损失复算：

```text
L_total = 0.225617 + 0.5 * 1.531752 - 50.524661
          + 0.02 * 558.878738 + 0.02 * 0.000153
L_total = -38.355590
```

## 10. 结果分析

有效的部分：

- 双频率数据管线已经跑通，意图输入形状为 `(B, 10, 30)`，未来轨迹形状为 `(B, 25, 3)`。
- 网络同时输出意图概率和 GMM 轨迹分布，说明整体结构在工程上可执行。
- 战术意图分支表现最好，验证集准确率达到 `91.22%`。
- GMM 轨迹头可以训练，验证集 GMM NLL 在训练过程中明显下降，未来相对位移 MSE 达到 `0.158020`。

当前弱点：

- 机动意图分支还不够强，`Close_Evade` 内部 9 类机动准确率为 `45.23%`。
- 机动标签分布不均衡，`Straight` 占 `38.75%`，而 `Right-Dive` 只有 `5.22%`。
- 原始 acceleration penalty 为 `558.878738`，说明 `0.02 s` 高频预测下 GMM 均值轨迹可能仍存在局部抖动。
- 当前缺少混淆矩阵、每类 Precision/Recall/F1、ADE/FDE 和基线模型对比。

结论判断：

- 这次结果可以证明网络结构、双频率数据处理和训练流程已经跑通。
- 这次结果不能直接作为最终性能结论。
- 下一版重点应该放在机动类别均衡、轨迹平滑、分类细粒度评估和基线对比。

## 11. 输出文件

本次结果文件：

- `results/dual_intent_network_test/metrics.json`
- `results/dual_intent_network_test/report.txt`
- `results/dual_intent_network_test/training_report.md`
- `results/dual_intent_network_test/training_curves.png`
- `results/dual_intent_network_test/sample_prediction.png`
- `results/dual_intent_network_test/best_validation_prediction.png`
- `results/dual_intent_network_test/prototype_model.pt`

## 12. 下一步建议

优先补充评估：

- 增加战术意图和机动意图混淆矩阵。
- 增加每类 Precision、Recall、F1。
- 增加 ADE/FDE 轨迹预测指标。
- 按战术意图分组统计轨迹误差。

优先改进训练：

- 对机动分类使用 weighted cross entropy、focal loss 或 balanced sampler。
- 增加稀有机动类别样本。
- 加强轨迹平滑约束，重点处理 acceleration penalty 偏大的问题。
- 尝试预测速度/加速度增量，再积分得到轨迹，减少高频抖动。

优先补充对比：

- BiLSTM + MLP 单轨迹头基线。
- 单意图 GMM 基线。
- 无 Query 的双意图 GMM 消融。
- 去掉守区相对特征的消融。

## 13. 当前结论

这次结果说明双层意图 Query-GMM 网络的数据管线和训练流程已经可运行，战术意图分支具备明显学习能力，GMM 轨迹分布头也能收敛。但机动意图识别和高频轨迹平滑还不够强，下一版需要补充更完整的分类指标、轨迹指标、类别均衡策略和基线对比，之后才能把它作为正式实验结果。
