# 双战略意图、单轨迹预测分支

日期：2026-10-28。

## 1. 分支与保存范围

- 原版本：`origin/archive/pv-sine-four-intent-20261028`，提交`d44490a`。保留四类意图、三模态预测、PV输出头与新增的正弦目标模式。
- 当前开发：`feat/two-strategic-intents-20261028`，直接从上述提交建立，新增本页所述双意图/单轨迹修改。
- Git保存源代码、配置、文档、验证脚本；训练检查点和运行输出仍留在服务器，不上传数GB实验数据。
- 归档提交记录了工作区之前已有的OT代码/产物清理，并没有在本次分支操作中再次删除服务器实验数据。

## 2. 战略意图只有两个

| 编号 | 名称 | 中文 | 对应旧类别 |
|---|---|---|---|
| 0 | Goal_Rush | 直冲守区 | Far_Rush、Goal_Priority、Goal_Sprint |
| 1 | Evade | 规避 | Close_Evade |

对新正弦目标模式：最近追捕机距离<=`target_sine.trigger_distance`为规避，否则为直冲守区；不额外插入冲刺标签。

对旧adaptive目标控制：设`d_p`为最近追捕机距离，`d_g`为守区圆柱表面距离，仅当`d_p <= target_tactical_far_distance`、`d_g > target_goal_sprint_distance`且`d_g > d_p`时标为规避，否则为直冲守区。

旧控制器内部仍需要识别不同执行阶段以重置机动计时器。它们是私有控制状态，不是额外意图，不输出给预测分类头、意图token或Actor。原目标运动方式、速度/加速度能力和边界保护不因此改变。

离线`PredictorDataset`可通过`label_kwargs`传入实际场景的模式、距离阈值和守区尺寸，与在线标签保持一致。不能拿默认小场地标注阈值直接套到100 m场地数据。

## 3. 只预测一条轨迹，不是从三条中选一条

- 预测器实际只构建一个轨迹分支：均值`trajectory_mean [B,H,3]`，逐点方差`trajectory_var [B,H,3]`。
- 保留逐点方差来表示这条预测的误差范围；这是单个高斯轨迹分布，不是多模态混合。
- 删除可学习的模式权重头；不再产生三条候选、不计算混合均值/方差、不做argmax选轨迹。
- 预测损失为二分类交叉熵＋单轨迹高斯负对数似然＋原有物理正则，权重沿用原配置。
- 预热期间的运动学预测也只有一条恒速轨迹，不再偷偷保留直行/向守区/远离追捕机三个分支。

为避免同时重写所有TensorDict适配器，兼容键名暂留：`gmm_means/gmm_vars [B,1,H,3]`，`gmm_weights [B,1]`恒为1；`mixed_mu/mixed_var`是单轨迹均值/方差的直接别名。单元素轴与旧字段名不表示仍有混合预测。

H仍为10、时间间隔仍为0.05 s，即0.5 s未来预测。双意图概率为`[B,2]`，不是每个意图各输出一条轨迹。

## 4. Actor中的变化

```text
目标及团队历史
  -> 二类意图概率 + 一条目标未来轨迹（含方差）
  -> 单轨迹时间编码
  -> 同一时刻的追捕未来query与目标点token拼接、MLP融合
  -> 原时间注意力 + 原三机协同注意力
  -> 位置/速度轨迹头
  -> 原SO3/PIDRate闭环执行
```

原同一时刻对K条模式做softmax选择的`TimeAlignedGMMCrossAttention`已删除，替换成`TimeAlignedTrajectoryFusion`。单一候选softmax恒等于1，会使原Q/K选择投影没有意义，因此不保留那套参数。

轨迹编码器中的权重embedding和权重门参数也删除。保留三机交互、上一规划、时间编码、意图token，以及Actor的输出维度和物理执行方式。

## 5. 启动和检查点

在已配置Isaac Sim的项目终端中：

```bash
# 新分支代码本身已默认双意图、单轨迹；目标采用正弦距离模式。
python scripts/train.py --config-name=train_pv task=HideAndSeekPVSine

# 同一网络，保留旧adaptive目标运动控制。
python scripts/train.py --config-name=train_pv
```

这只是启动说明，本次不启动长训练。

旧四类/三模态的预测器和Actor检查点不能直接无损续训：分类头、轨迹头、意图embedding和同刻融合层均已变化。请新训或显式设计迁移实验；本次没有静默裁剪、合并权重，也没有伪装为完整加载。显式加载不兼容的预测器会报错提示返回归档分支。

旧检查点和旧TensorBoard没有删除。新日志使用`prediction/ADE`、`prediction/loss_trajectory_nll`，不再写三候选`minADE`或模式权重熵。目标战略占比只有编号0/1；局部动作编号不等于战略类别，正弦分支用动作9。

## 6. 验证证据与限制

产物：`results/two_intent_single_20261028/`。

- `unit_cpu.log`、`unit_cuda.log`：CPU和cuda:1各7项契约测试通过。检查2类概率、唯一轨迹、无权重头、负对数似然反传、在线/离线标签一致、单轨迹预热和同刻融合梯度/时间对应。
- `sine_regression.log`：15项正弦策略回归通过。
- `cooperative.log`：13组协同编码、意图条件、三机等变性、mask、梯度及有界轨迹检查通过；旧三模式排列测试替换为模式选择已删除的检查。
- `context.log`：现有物理context适配器回归通过。
- `ppo_smoke.log`：真实仿真8环境、rollout64、2 mini-batch、2次PPO更新，共1024环境帧；run为`HideAndSeek_20261028_214029`，保存最终检查点。预热之后的学习预测也保持2类/1轨迹，训练及保存正常完成。触发距离临时覆写200 m以确保规避分支被走到，未修改默认25 m配置。
- `adaptive_smoke.log`：最终代码又在旧adaptive目标模式验证8环境、rollout32、1次PPO更新，共256环境帧；run为`HideAndSeek_20261028_214623`，正常退出并保存检查点。临时将近距阈值覆写200 m以覆盖旧规避分支；日志只出现战略0/1占比，实际本批为规避占比1。正式阈值未变。
- 这是接口和执行链路验证，不是收敛、泛化或围捕成功率评测；两次更新不足以评价新方法好坏。

当前历史文档中保留的四类/三模态数值仅用于旧实验追溯；新接口以本页和代码为准。
