# 距离触发的正弦目标模式

更新：2026-10-28。远程项目：`/data/uavlab/maneuverable-target`。

## 1. 行为定义

新增 `task.target_motion_mode=goal_sine`，旧的 `adaptive` 模式原样保留，仍为旧入口的默认值。

- 计算目标到所有追捕机的三维欧氏距离，取最近距离 `d_min`。
- `d_min <= trigger_distance`：目标一边向守区推进，一边沿水平方向左右正弦摆动。
- `d_min > trigger_distance`：只生成朝守区的速度参考，不叠加追捕机斥力或旧的机动优化。
- 新模式没有额外的“守区优先/冲刺”覆盖：近距就摆动，远距就朝守区。保留现有边界、地面保护及底层动力学，因此靠边时不保证严格直线。
- 不改变双方推重比、70%电机推力上限、场地、奖励、捕获条件、Actor输出头及正式训练超参数。

这是可调的简化逃逸基线；尚未证明其围捕难度或训练效果优于旧场景。

## 2. 参数与单位

当前50 m高度PV主线可直接选择 `HideAndSeekPVSine`：

| 配置项 | 新增PV入口默认值 | 含义 |
|---|---|---|
| `task.target_motion_mode` | `goal_sine` | 新目标控制模式 |
| `task.target_sine.trigger_distance` | `25.0` | 最近追捕机触发距离，米；等于阈值也触发 |
| `task.target_sine.amplitude_range` | `[1.0, 2.0]` | 每回合横向位移参考振幅的均匀采样范围，米；不是峰峰值 |
| `task.target_sine.period_range` | `[3.0, 5.0]` | 每回合周期的均匀采样范围，秒 |
| `task.target_sine.lateral_position_gain` | `1.5` | 横向位置误差转为速度修正的增益，单位1/s |

小场地基础任务 `HideAndSeek` 的候选参数较小：触发6 m、振幅0.3–0.5 m、周期3–5 s；但仍须显式切换为新模式。

振幅必须有限、非负；周期、触发距离必须有限且大于0；范围必须为升序二元组。不合法配置立即报错。

## 3. 怎样随机化，怎样保持连贯

每个并行环境分别在自身回合重置时独立采样A、T。回合内不再重采样，其他环境结束不会改变它们。

每次由远距进入近距时，以目标当前位置为横向零点，用当时朝守区方向的水平垂线作为摆动轴，并从相位0开始：

`y_ref(t) = A * sin(2*pi*t/T)`

连续近距阶段固定摆动轴，避免每步旋转参考系。出阈值后不再追踪该横向参考，也不强行拉回旧中线；重新入阈值时以新位置重新建立参考轴，但仍使用本回合采样的A、T。

横向速度由正弦参考的导数及少量位置误差反馈组成；剩余速度预算用于向守区推进，总速度参考不超过本回合最大速度。再通过现有加速度滤波、速度参考积分、姿态/角速度控制和真实多旋翼动力学执行。目标逻辑每个0.01 s物理步更新，Actor仍为0.05 s一次决策。

特别处理：旧模式把合力方向归一化到最大加速度。新模式不走该归一化，而是保留速度反馈产生的加速度幅值，只在超过上限时限幅；否则小幅度正弦参考会被变成始终满加速度。此改动仅作用于新模式。

## 4. 振幅不等于实际飞行幅度

A是横向位移**参考振幅**。真实飞机存在跟踪延迟、推力/加速度界和边界保护，实际轨迹不一定精确达到A，退出近距后也不会瞬间停止横向速度。

理想正弦横向参考需要：

- 最大横向速度约 `2*pi*A/T`。
- 最大横向加速度约 `A*(2*pi/T)^2`。

例如A=2 m、T=3 s，分别约4.19 m/s和8.77 m/s²。默认范围最强组合的这一理想横向加速度低于当前目标采样加速度下界10 m/s²，但加速前进、转向、姿态响应也会占用能力，所以不是全程可精确执行的保证。

不要把A改得很大、T改得极短后仍假定实际幅值相同：总速度参考会受限，朝守区推进也可能明显变慢。代码不会暗中重采样或改写你设置的A、T来掩盖该问题。

## 5. 启用方法

在已经配置好Isaac Sim与sim环境的终端、项目根目录执行：

```bash
# 新模式，保持train_pv其余正式训练参数；执行后会启动训练。
python scripts/train.py --config-name=train_pv task=HideAndSeekPVSine

# 固定振幅1.5 m、周期4 s，用于排查和对照。
python scripts/train.py --config-name=train_pv task=HideAndSeekPVSine \
  'task.target_sine.amplitude_range=[1.5,1.5]' \
  'task.target_sine.period_range=[4.0,4.0]'

# 调整每轮采样范围和触发阈值。
python scripts/train.py --config-name=train_pv task=HideAndSeekPVSine \
  task.target_sine.trigger_distance=20 \
  'task.target_sine.amplitude_range=[0.5,2.0]' \
  'task.target_sine.period_range=[3.0,6.0]'

# 原PV任务仍保持旧目标模式。
python scripts/train.py --config-name=train_pv
```

等端点范围表示固定值。上述命令是使用说明；本次没有启动长训练。

## 6. 数据与网络兼容

新增字段只放在环境`info`诊断中，不增加Actor特权输入，也不改变Actor/预测网络的输入维数：

| `info`字段 | 内容 |
|---|---|
| `target_sine_amplitude_m` | 当前回合采样A，米 |
| `target_sine_period_s` | 当前回合采样T，秒 |
| `target_sine_active` | 是否处于距离触发的摆动分支，0/1 |
| `target_sine_reference_offset_m` | 当前横向参考位移，米；非实际位移，远距为0 |

未启用新模式时，这些字段保持0。它们不会自动变成新增TensorBoard曲线。

当前开发分支已改为两维战略接口：标签0为Goal_Rush（直冲守区），标签1为Evade（规避）。正弦模式仍为远距0、近距1；原adaptive控制器的远距/守区优先/冲刺均归为0，普通近距规避归为1。新模式的`target_maneuver_id=9`表示摆动分支、0表示直行；原优化器仍只有0–8九个动作候选，它们不是战略意图类别。四意图版本保存在`archive/pv-sine-four-intent-20261028`。

`target_maneuver_switches`统计新模式0/9分支变化，不是正弦每半周期的换向次数。需要换向统计时应从实际速度方向计算，不能把这两个量混用。

## 7. 已执行验证与限制

产物目录：`results/target_sine_20261028/`。

1. `unit_cpu.log`、`unit_cuda.log`：CPU与cuda:1各15项测试通过。覆盖阈值等号、正弦振幅/周期、远距朝守区、重新进入、参数回合内固定、局部重置隔离、非法配置、极端参数有界、点质量/无人机分支及旧模式分发回归。
2. `ppo_smoke.log`：8环境、rollout64、2个mini-batch、2次PPO更新，共1024环境帧；保存`HideAndSeek_20261028_205757/checkpoint_final.pt`。诊断将触发距离临时覆写200 m，确保摆动分支被执行；没有改配置文件的25 m默认值。仅为接入短测，不是正式训练，也没有可用的捕获率结论。
3. `physics.log`、`target_physics_summary.json`、`target_physics.npz`、`target_physics.png`：4环境真实目标控制器，480个100 Hz物理步（4.8 s）。固定A=1.5 m、T=3 s；3.6 s内存强制近距、1.2 s强制远距，检查切换和有限性，并实际调用局部重置验证其他环境振荡状态不受影响。此测试不控制追捕机，不评估围捕。
4. 该目标物理短测实际最大速度16.5813 m/s，最大滤波加速度指令11.5181 m/s²，实际电机推力比例最大0.54844。这里只报告本次短测观察值，不代表新模式实际加速度的系统鉴定，也不是性能上限。
5. Isaac启动日志含已有无窗口GLInterop错误行，短测进程仍正常退出并保存结果；PPO关闭时有已有semaphore警告。本次未将它们当成训练崩溃，也未声称修复。

正式长训练尚未启动；需要后续用相同初始化和能力条件比较捕获率，才能判断简化目标是否解决旧场景过难的问题。

## 8. 文件定位与回退

- 控制器：`omni_drones/envs/hide_and_seek/goal_sine.py`。
- 接入：`hideandseek.py`的目标加速度入口、回合重置、战略标签及info诊断。
- 基础配置：`cfg/task/HideAndSeek.yaml`和`HideAndSeek70.yaml`。
- 当前PV新入口：`cfg/task/HideAndSeekPVSine.yaml`。
- 测试：`scripts/test_goal_sine.py`和`scripts/verify_goal_sine_sim.py`。
- 修改前快照：`results/target_sine_20261028/before/`。包含本次修改前已有用户改动的文件，不是Git基线。

不需要恢复旧文件就能回到原行为：选择`task=HideAndSeekPV`或设置`task.target_motion_mode=adaptive`。
