# 最终方案：协同相对航路点策略 + 约束轨迹优化 + SO3/PIDRate执行

## 1. 方案定位

论文重点是：

> 在目标多模态轨迹预测及逐点不确定性条件下，三架追捕无人机生成具有协同性的未来空间轨迹。

因此，高层Actor只负责回答：

[
\boxed{\text{未来各时刻，三架无人机分别应该到达哪里}}
]

速度、加速度、jerk、倾角和推力可行性，不再由Actor重复学习，而由确定性的轨迹优化器和低层控制器负责。

最终链路确定为：

```text
GMM目标轨迹编码
+ 三机历史状态编码
+ Future Token协同解码
        ↓
cooperative_future_tokens [B,N,H,D]
        ↓
相对航路点概率Head
        ↓
三机原始位置偏好
        ↓
约束轨迹优化器
        ↓
一致且可行的 p*, v*, a*
        ↓
分段连续轨迹插值
        ↓
SO3位置控制 + PIDRate
        ↓
总推力 T + 机体角速度 ω
```

当前系统参数保持文档设定：

[
N=3,\quad H=10,\quad \Delta t_p=0.05,\mathrm{s}
]

一次规划未来 (0.5,\mathrm{s})，执行前 (0.1,\mathrm{s}) 后重新规划；最大速度、加速度、倾角和场地约束沿用现有环境配置。

---

# 2. Actor动作定义

## 2.1 不直接输出世界坐标

Actor不直接输出：

[
\mathbf p_{i,h}^{\mathrm{world}}
]

而是输出相对于当前运动趋势的三维位置残差。

第 (i) 架无人机当前状态为：

[
\mathbf p_{i,0},\qquad \mathbf v_{i,0}
]

未来第 (h) 个节点对应时间：

[
\tau_h=h\Delta t_p
]

根据当前速度构造名义轨迹：

[
\mathbf p_{i,h}^{\mathrm{nom}}
==============================

\mathbf p_{i,0}
+
\mathbf v_{i,0}\tau_h
]

它表示无人机保持当前速度时的未来位置。

Actor只输出：

[
\Delta\mathbf p_{i,h}^{\mathrm{RL}}
]

最终原始航路点偏好为：

[
\boxed{
\hat{\mathbf p}_{i,h}
=====================

\mathbf p_{i,h}^{\mathrm{nom}}
+
\Delta\mathbf p_{i,h}^{\mathrm{RL}}
}
]

这种表达具有三个优势：

* 近端轨迹自然继承当前速度，不容易突变；
* Actor重点学习协同转向和空间覆盖，不需要重新学习惯性；
* 不依赖绝对世界坐标，方便更换实验区域。

---

# 3. 相对航路点概率Head

输入：

```python
cooperative_future_tokens: [B, N, H, D]
```

每个Future Token输出三维高斯分布均值：

[
\boldsymbol\mu_{i,h}^{p}
========================

f_{\mathrm{head}}
\left(
\mathbf q_{i,h}
\right)
\in\mathbb R^3
]

第一版采用三维共享或按时间共享的可学习标准差：

[
\log\boldsymbol\sigma^p\in\mathbb R^3
]

不建议为所有 (3\times10\times3) 个动作维度分别设置独立标准差，以降低PPO训练难度。

采样：

[
\mathbf z_{i,h}
\sim
\mathcal N
\left(
\boldsymbol\mu_{i,h}^{p},
\boldsymbol\sigma^p
\right)
]

归一化动作：

[
\mathbf u_{i,h}
===============

\tanh(\mathbf z_{i,h})
]

再做三维径向归一化：

[
\bar{\mathbf u}_{i,h}
=====================

\frac{
\mathbf u_{i,h}
}{
\max(1,|\mathbf u_{i,h}|_2)
}
]

因此：

[
|\bar{\mathbf u}_{i,h}|_2\leq1
]

不能只对三个坐标分别乘 `tanh`，否则三维位移范数仍可能超过限制。

---

# 4. 基于可达性的航路点范围

相对于匀速名义轨迹，无人机在时间 (\tau_h) 内通过加速度能够产生的最大位置偏移近似为：

[
r_h
===

c_r\frac12a_{\max}\tau_h^2
]

其中：

* (a_{\max})：最大加速度；
* (c_r)：安全缩放系数，第一版可取 (0.8\sim1.0)。

于是：

[
\boxed{
\Delta\mathbf p_{i,h}^{\mathrm{RL}}
===================================

r_h\bar{\mathbf u}_{i,h}
}
]

这意味着：

* (h)较小时，航路点不能突然偏离当前运动趋势；
* (h)较大时，Actor有更大的空间调整范围；
* 位置输出范围直接对应无人机物理可达能力。

原始航路点为：

[
\hat{\mathbf p}_{i,h}
=====================

\mathbf p_{i,0}
+
\mathbf v_{i,0}\tau_h
+
r_h\bar{\mathbf u}_{i,h}
]

Actor动作维度为：

[
3\times10\times3=90
]

相比逐点同时输出位置和速度的180维动作，维度减半，并且不存在位置—速度相互冲突。

---

# 5. PPO动作概率定义

必须统一动作定义。

推荐将：

[
\mathbf u=\tanh(\mathbf z)
]

视为PPO的有界动作，使用Squashed Gaussian计算概率：

[
\log\pi(\mathbf u\mid s)
========================

\log\mathcal N
(\mathbf z;\boldsymbol\mu,\boldsymbol\sigma)
--------------------------------------------

\sum_d
\log
\left(
1-\tanh^2(z_d)+\varepsilon
\right)
]

Rollout buffer保存：

```python
pre_tanh_action
squashed_action
old_log_prob
```

从 (\mathbf u) 到实际航路点的：

* 径向归一化；
* 可达域缩放；
* 名义轨迹叠加；
* 轨迹优化；

均视为确定性动作解码过程。

不能对轨迹优化后的航路点重新计算策略概率。

---

# 6. Actor只输出位置，但最终仍会得到速度和加速度

Actor输出的是：

[
\hat{\mathbf P}
===============

{\hat{\mathbf p}_{i,h}}
]

但轨迹优化器的变量包括：

[
\mathbf P_i
===========

{\mathbf p_{i,h}}
]

[
\mathbf V_i
===========

{\mathbf v_{i,h}}
]

[
\mathbf A_i
===========

{\mathbf a_{i,h}}
]

因此最终输出仍然包含：

```python
optimized_positions:     [B,N,H,3]
optimized_velocities:    [B,N,H,3]
optimized_accelerations: [B,N,H,3]
```

区别在于：

> 速度和加速度由同一条动力学一致轨迹产生，而不是由Actor独立猜测。

---

# 7. 轨迹优化目标

对每架无人机分别求解：

[
\min_{\mathbf P_i,\mathbf V_i,\mathbf A_i}
J_i
]

目标函数建议为：

[
\begin{aligned}
J_i={}&
\lambda_p
\sum_{h=1}^{H}
w_h
\left|
\mathbf p_{i,h}
---------------

\hat{\mathbf p}*{i,h}
\right|*2^2\
&+
\lambda_a
\sum*{h=1}^{H}
|\mathbf a*{i,h}|*2^2\
&+
\lambda_j
\sum*{h=1}^{H}
\left|
\mathbf a_{i,h}
---------------

\mathbf a_{i,h-1}
\right|*2^2\
&+
\lambda*{\mathrm{prev}}
J_i^{\mathrm{previous}}
\end{aligned}
]

## 7.1 航路点拟合项

[
\left|
\mathbf p_{i,h}
---------------

\hat{\mathbf p}_{i,h}
\right|^2
]

保证轨迹优化器尽量保留Actor给出的协同空间意图。

建议近端权重更高：

[
w_1>w_2>\cdots>w_H
]

因为只执行前 (0.1,\mathrm{s})，近端轨迹精度比远端更重要。

## 7.2 加速度项

限制过激运动：

[
\sum_h|\mathbf a_{i,h}|^2
]

## 7.3 Jerk项

[
\sum_h
|\mathbf a_{i,h}-\mathbf a_{i,h-1}|^2
]

用于抑制实机上的加速度突变和轨迹抖动。

## 7.4 上一周期连续性

对新旧轨迹的重叠时间段增加：

[
J_i^{\mathrm{previous}}
=======================

\sum_h
\left|
\mathbf p_{i,h}^{\mathrm{new}}
------------------------------

\mathbf p_{i,h}^{\mathrm{old}}
\right|^2
]

也可以同时约束速度连续性。

---

# 8. 运动学一致性约束

优化轨迹必须满足：

[
\boxed{
\mathbf p_{i,h}
---------------

# \mathbf p_{i,h-1}

\frac12
\left(
\mathbf v_{i,h-1}
+
\mathbf v_{i,h}
\right)
\Delta t_p
}
]

该式保证位置和速度一致。

速度与加速度满足：

[
\boxed{
\mathbf v_{i,h}
---------------

# \mathbf v_{i,h-1}

\frac12
\left(
\mathbf a_{i,h-1}
+
\mathbf a_{i,h}
\right)
\Delta t_p
}
]

起始条件固定为实际测量值：

[
\mathbf p_{i,0}=\mathbf p_{i}^{\mathrm{current}}
]

[
\mathbf v_{i,0}=\mathbf v_{i}^{\mathrm{current}}
]

[
\mathbf a_{i,0}=\mathbf a_{i}^{\mathrm{filtered}}
]

若实机加速度估计噪声较大，可以只将 (\mathbf a_{i,0})作为软约束。

---

# 9. 物理约束

沿用现有环境配置中的典型限制：最大速度 (3,\mathrm{m/s})、最大加速度 (6,\mathrm{m/s^2})、最大倾角 (35^\circ)、场地半径和高度边界等。

## 9.1 速度

[
|\mathbf v_{i,h}|*2\leq v*{\max}
]

## 9.2 加速度

[
|\mathbf a_{i,h}|*2\leq a*{\max}
]

## 9.3 Jerk

[
|\mathbf a_{i,h}-\mathbf a_{i,h-1}|*2
\leq
j*{\max}\Delta t_p
]

## 9.4 倾角

[
|\mathbf a_{xy,i,h}|*2
\leq
(g+a*{z,i,h})\tan\theta_{\max}
]

并保证：

[
g+a_{z,i,h}>0
]

## 9.5 场地

[
\sqrt{x_{i,h}^2+y_{i,h}^2}
\leq r_{\mathrm{arena}}
]

[
z_{\min}\leq z_{i,h}\leq z_{\max}
]

三机碰撞约束：

[
|\mathbf p_{i,h}-\mathbf p_{j,h}|
\geq d_{\mathrm{safe}}
]

属于非凸约束。第一版建议：

* RL奖励中加入轨迹碰撞软惩罚；
* 轨迹优化阶段先不进行三机联合非凸求解；
* 实机低层增加CBF-QP安全过滤。

后续若需要严格联合规划，再采用序列凸化或NMPC。

---

# 10. 轨迹优化器的定位

当前文档中的顺序 `clip/clamp` 可以保留为快速基线，但最终版本建议使用：

> 以 (\mathbf p,\mathbf v,\mathbf a) 为变量的小规模约束轨迹优化器。

由于：

[
N=3,\quad H=10
]

可以先为每架无人机独立求解动力学轨迹，再由安全过滤器解决三机碰撞。

轨迹优化器不要求参与PPO反向传播。PPO通过环境奖励更新Actor，优化器是固定的动作执行模块。

同时加入投影修改惩罚：

[
R_{\mathrm{proj}}
=================

*

\lambda_{\mathrm{proj}}
\frac1{NH}
\sum_{i,h}
\left|
\mathbf p_{i,h}^*
-----------------

\hat{\mathbf p}_{i,h}
\right|^2
]

防止Actor长期输出不可执行轨迹，并完全依赖优化器修正。

---

# 11. 连续轨迹生成

优化器输出离散节点：

[
\mathbf p_{i,h}^*,\quad
\mathbf v_{i,h}^*,\quad
\mathbf a_{i,h}^*
]

在相邻节点间构造五次多项式：

[
\mathbf p_i(\tau)
=================

\mathbf c_0+
\mathbf c_1\tau+
\mathbf c_2\tau^2+
\mathbf c_3\tau^3+
\mathbf c_4\tau^4+
\mathbf c_5\tau^5
]

六个系数由两端的：

[
(\mathbf p_h,\mathbf v_h,\mathbf a_h)
]

和：

[
(\mathbf p_{h+1},\mathbf v_{h+1},\mathbf a_{h+1})
]

确定。

因此低层控制器可在任意控制时刻获得：

[
\mathbf p_d(t),\quad
\mathbf v_d(t),\quad
\mathbf a_d(t),\quad
\mathbf j_d(t)
]

而不是只在每 (0.05,\mathrm{s}) 获得一个离散航路点。

---

# 12. 多频率滚动执行

推荐频率结构：

```text
Actor + 轨迹优化：10 Hz
轨迹节点间隔：    0.05 s（20 Hz节点）
低层控制器：      100 Hz
```

每次Actor规划：

[
H\Delta t_p=10\times0.05=0.5,\mathrm{s}
]

只执行：

[
0.1,\mathrm{s}
]

即前两个轨迹区间，然后重新：

1. 获取真实无人机状态；
2. 更新目标GMM预测；
3. 编码上一周期剩余轨迹；
4. 重新生成三机协同航路点；
5. 重新优化并执行。

不能在整个 (0.1,\mathrm{s}) 内只跟踪第一个离散点，必须跟踪连续插值轨迹。

---

# 13. 从连续轨迹到控制命令

实际状态：

[
\mathbf p,\quad\mathbf v,\quad R
]

期望轨迹：

[
\mathbf p_d,\quad\mathbf v_d,\quad\mathbf a_d
]

定义误差：

[
\mathbf e_p=\mathbf p_d-\mathbf p
]

[
\mathbf e_v=\mathbf v_d-\mathbf v
]

期望平动加速度：

[
\boxed{
\mathbf a_c
===========

\mathbf a_d
+
K_p\mathbf e_p
+
K_v\mathbf e_v
}
]

注意不能再使用：

[
K_p(\mathbf p_d-\mathbf p)/\Delta t
]

因为该项量纲不是加速度。

期望总推力向量：

[
\mathbf f_d
===========

m
\left(
\mathbf a_c+g\mathbf e_3
\right)
]

期望推力方向：

[
\mathbf b_{3d}
==============

\frac{\mathbf f_d}{|\mathbf f_d|}
]

结合期望偏航方向构造期望姿态 (R_d)。

总推力：

[
\boxed{
T=\mathbf f_d^\top R\mathbf e_3
}
]

姿态误差：

[
\mathbf e_R
===========

\frac12
\operatorname{vee}
\left(
R_d^\top R-R^\top R_d
\right)
]

机体角速度命令：

[
\boxed{
\boldsymbol\omega_{\mathrm{cmd}}
================================

## R^\top R_d\boldsymbol\omega_d

K_R\mathbf e_R
}
]

工程实现上可直接复用现有：

```python
_acceleration_to_pidrate_action
```

将 (\mathbf a_c) 转换为归一化推力和机体角速度，再交给PIDRate内环。

---

# 14. 上一周期轨迹记忆

执行前两个节点后，保留：

[
\mathbf p_{i,3:H}^*,\quad
\mathbf v_{i,3:H}^*
]

转换为相对当前真实状态：

[
\Delta\mathbf p_{i,h}^{\mathrm{prev}}
=====================================

## \mathbf p_{i,h}^{\mathrm{prev}}

\mathbf p_i^{\mathrm{current}}
]

[
\Delta\mathbf v_{i,h}^{\mathrm{prev}}
=====================================

## \mathbf v_{i,h}^{\mathrm{prev}}

\mathbf v_i^{\mathrm{current}}
]

形成：

```python
prev_plan: [B,N,H,6]
```

无效补齐部分的mask为0。补齐点可以用于张量对齐，但不能被Previous Plan Encoder视为真实轨迹。

---

# 15. 训练与实机监控指标

必须同时记录：

### Actor输出质量

[
E_{\mathrm{proj}}
=================

\frac1{NH}
\sum_{i,h}
|\mathbf p_{i,h}^*-\hat{\mathbf p}_{i,h}|^2
]

### 运动学质量

```text
max_speed
max_acceleration
max_jerk
tilt_saturation_rate
optimizer_infeasible_rate
optimizer_solve_time
```

### 控制跟踪质量

[
E_p=
|\mathbf p_d-\mathbf p|
]

[
E_v=
|\mathbf v_d-\mathbf v|
]

记录：

```text
position_tracking_rmse
velocity_tracking_rmse
body_rate_saturation_rate
thrust_saturation_rate
```

### 协同质量

```text
minimum_inter_agent_distance
trajectory_overlap_ratio
GMM mode coverage
uncertainty–team-spread correlation
capture success rate
```

Sim-to-real比较不能只比较最终捕获率，还要比较：

[
\text{仿真跟踪RMSE}
\quad\text{与}\quad
\text{实机跟踪RMSE}
]

以及轨迹优化修改量和控制饱和率。

---

# 16. 最终数据流

```text
cooperative_future_tokens [B,N,H,D]
        ↓
Position Distribution Head
        ↓
Squashed Gaussian位置残差动作 [B,N,H,3]
        ↓
当前匀速名义轨迹
+
时间相关可达域残差
        ↓
Actor原始航路点偏好 p_hat
        ↓
约束轨迹优化
变量：p,v,a
目标：位置拟合 + 加速度 + jerk + 上一计划连续
约束：运动学、速度、加速度、jerk、倾角、边界
        ↓
优化节点 p*,v*,a*
        ↓
五次多项式连续插值
        ↓
p_d(t),v_d(t),a_d(t)
        ↓
a_c = a_d + Kp e_p + Kv e_v
        ↓
SO3期望姿态和总推力
        ↓
现有PIDRate内环
        ↓
T + ω
        ↓
执行0.1s后重新规划
```

---

## 供专家重点评判的问题

1. Actor采用“当前匀速名义轨迹 + 有界位置残差”的动作参数化，是否足以表达高机动围捕轨迹。
2. (H=10,\Delta t=0.05,\mathrm{s})下，90维PPO动作是否仍应进一步压缩为5个关键航路点。
3. 每架无人机独立轨迹优化、碰撞交给奖励与CBF的分层方式，是否足以保证三机协同性不会被优化器破坏。
4. 轨迹优化器是否应采用凸近似、序列凸化，还是直接使用短时域NMPC。
5. 只输出位置、由优化器生成速度和加速度，是否比额外输出终端速度更适合当前追捕任务。
6. 当前 (0.5,\mathrm{s})规划、(0.1,\mathrm{s})执行的滚动时域是否能兼顾目标快速规避和实机求解时间。

最终推荐版本是：

[
\boxed{
\text{协同Actor只输出相对位置航路点}
+
\text{模型驱动轨迹优化生成 }p,v,a
+
\text{SO3/PIDRate闭环执行}
}
]

它将协同决策、动力学可行性和真实飞行控制清晰分层，是当前更有利于训练稳定、方法解释和sim-to-real迁移的主方案。
