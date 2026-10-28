# Actor物理状态输入：候选对照，尚未替换主训

日期：2026-10-22 06:10起，北京时间。

## 完成结果（06:40核对，先读本节）

两组均真实完成128更新=524288新帧，独立128局及6张联合轨迹齐全；脚本已退出，completed.txt存在，不重复启动。

| 指标 | baseline | context_retry1 |
|---|---:|---:|
| 抓捕 | 0/128 | 0/128 |
| 目标进区 | 84 | 88 |
| 升高越界 | 44 | 40 |
| 平均episode秒数（非抓捕时间） | 9.181 | 9.027 |
| 平均最近距离/m | 8.449 | 7.981 |
| 追捕机速度p95/(m/s) | 4.808 | 4.891 |
| 实际电机最大推力分数 | 0.6999648 | 0.6995783 |
| 训练结束episode越界比例 | 32.93% | 39.73% |
| 训练KL均值 | 0.000166 | 0.000152 |
| Actor跳过/非有限数 | 0/无 | 0/无 |

配对初始种子/环境编号：19局从越界变非越界、15局反向变化，双侧精确配对二项p=0.60759（paired_comparison.json）。改善4/128、均0抓捕，而且训练越界更高，不支持稳定安全收益或围捕改善。因此**不替换06主训练**；不据此断言6维输入永远无效，也不将单种子128更新称为泛化结果。Context模型/代码保留，下一步有限LR单变量测试另见ACTOR_LR_AB.md。

context训练2844个结束episode：39.73%越界/60.27%进区/0抓捕，Value loss均值0.00994、EV0.96210、minADE/FDE0.847/1.646 m，数值有限。有效结果为context_retry1，首次0更新context.log失败仍保留。

## 历史进展（06:28）

- baseline已完整完成128更新，2800个训练结束episode中32.93%越界、67.07%进区、0抓捕，数值有限/Actor未跳过；独立128局84进区、44升高越界、0抓捕，平均9.181秒、最近距离8.449 m、实际电机最大0.6999648，结果在baseline/summary.json。
- context首次在0更新因TensorSpec设备不匹配（CPU/cuda:1）退出，context.log保留；不是CUDA核崩溃或训练发散。新增spec已显式使用环境设备，回归新增设备断言通过。
- 只重试候选：脚本PID168722，入口附加参数context_retry1，run HideAndSeek_20261022_062538；输出目录与日志改为context_retry1，失败的context目录不冒充有效结果。33更新135168帧、650结束episode：43.69%越界、56.31%进区、0抓捕，数值有限/Actor跳过0。仍待完整128更新/独立128局，不能与baseline最终比例直接比较或提前切换主训。
- 实际checkpoint兼容只缺新增context层weight/bias，无unexpected keys；两组Actor都明确重置，候选日志有效LR3e-6/5e-5。重复启动仍由文件锁与日志覆盖检查防止。原脚本162489已退出，后续看experiment.pid及completed.txt。
- 主训练06继续，未替换。后续比较baseline/summary.json与context_retry1/summary.json；后者未生成不表示应该再启动。

## 为什么测试

06的250更新独立64局：37目标进区、27升高越界、0抓捕；初始权重为44进区、20越界。KL/ratio/Value数值正常、预测误差下降，但越界逐渐上升，不是CUDA或NaN崩溃。

现有CooperativeTrajectoryEncoder消费目标相对历史/相对位置速度、上一轮规划、GMM和意图，没有读取观测里的state_self。Critic看到了更多状态，不代表Actor也看到了。本机世界速度可以间接来自上一轮规划，GMM也可能间接包含场地信息，但没有显式本机高度/场地坐标。相对几何相同、整体高度不同，安全动作可能不同；这是待验证的观测缺口，不宣称为升高唯一根因。

## 候选设计

`HideAndSeek70Context`继承Isolated，只增加actor_flight_context开关：

- 输入6维：`[x/100,y/100,z/30,vx/17,vy/17,vz/17]`，用本机场地局部坐标和可测世界速度，不用并行场景平铺全局坐标或目标能力真值。
- 一层Linear(6,128)，共896参数；输出向量广播加到该机10个未来token，再走同一有界动作头和运动学解码。不加注意力、教师或额外奖励，不改推力/任务难度，不任意截断轨迹点。
- 线性层零初始化，给定相同旧观测时初始动作保持不变；构造时保留初始化随机数状态。原Isolated没有该输入键，仍走原路径。

CPU生产适配器回归已通过：零初始化一致、梯度有限且可训练、高度输入影响对应token、机体/环境间不泄漏。首测框架遗漏注册装饰器环境，修正抽取方式后通过，不是训练失败。完整128更新与独立评测现已完成，只代表集成/筛选完成，不是学会抓捕。

## 对照设置

GPU1顺序运行baseline与context，两组同从06@250检查点`HideAndSeek_20261022_055552/checkpoint_1024000.pt`加载。

| 项目 | baseline | context |
|---|---|---|
| 任务 | HideAndSeek70Isolated | HideAndSeek70Context |
| 新采样 | 128更新=524288帧 | 相同 |
| PPO | 128环境×32步、4 mini、1 epoch | 相同 |
| Actor/Critic LR | 3e-6/5e-5 | 相同 |
| 探索 | log_std=-3 | 相同 |
| Actor优化器 | 明确重置 | 明确重置，包含新参数 |
| Critic/预测器优化器 | 恢复检查点 | 相同 |
| 最终评测 | 128环境首局，seed=20261022 | 相同 |

Actor优化器两组都重置以对称比较；不能把候选与保留旧Adam状态的主训直接视作严格单变量对照。新128计数不加旧250冒充378节点。诊断事件不会自动混入主训TB聚合，但各自完整日志与模型保留。

历史启动记录：总脚本PID162489，`scripts/test_flight_context_pair70.sh`有锁并拒绝覆盖旧日志。06:15 baseline run为HideAndSeek_20261022_061519；context首次0更新失败，重试run为062538。两个脚本均已结束，结果以上方完成节为准；主训仍06、未被替换。

## 接手与判断

目录：`results/overnight70_20261021/attempt_06/flight_context_ab/`；总日志为同级flight_context_ab.log。

1. 看experiment.pid及completed.txt，不重复启动；从baseline.log/context.log读取真实run目录。
2. 确认两组Actor优化器重置，候选checkpoint兼容日志应仅缺新增context层参数。
3. 比较baseline/summary.json与context/summary.json中的抓捕、越界、最近距离、速度、目标机动及70%推力，并结合各自KL/ratio/Value/预测误差。
4. 128更新只作筛选，单种子改善不能称为泛化；未明显改善先记录，不继续盲目扩大网络。保留06自己的1000和5000节点验证。

主训250联合轨迹在eval_00250/；已查看的392更新曲线在review_0618/training_curves.png，预测和Value拟合稳定但越界上升，不能用损失收敛替代任务成功。

06:22另修复优化器恢复旧LR覆盖新配置的问题：恢复后重设本次Actor/Critic配置LR，保留Adam历史，生产函数真实Adam回归通过。本对照两组LR原本就与检查点相同，所以不构成两组LR差异；两组Actor优化器仍按设计显式重置。baseline进程此前已加载，context后续加载新代码应在日志打印effective learning rate=3e-6/5e-5。
