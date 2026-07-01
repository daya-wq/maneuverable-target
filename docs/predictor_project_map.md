# 意图识别 + GMM轨迹预测 — 项目代码地图

> 本文档记录"机动识别与轨迹预测"模块的所有代码文件及运行方式。

## 代码文件清单

### 网络模块 `omni_drones/learning/predictor/`

| 文件 | 功能 | 依赖 |
|------|------|------|
| [__init__.py](file:///data/uavlab/maneuverable-target/omni_drones/learning/predictor/__init__.py) | 模块入口，导出三个主要组件 | — |
| [intent_gmm_predictor.py](file:///data/uavlab/maneuverable-target/omni_drones/learning/predictor/intent_gmm_predictor.py) | 核心网络：共享BiLSTM + 意图分类头 + GMM预测头 + MVS调制 | torch |
| [losses.py](file:///data/uavlab/maneuverable-target/omni_drones/learning/predictor/losses.py) | 损失函数：IntentLoss(Phase1) + IntentGMMLoss(Phase2) | torch |
| [mvs.py](file:///data/uavlab/maneuverable-target/omni_drones/learning/predictor/mvs.py) | 机动变异度得分(MVS)的规则计算 | numpy |
| [data_pipeline.py](file:///data/uavlab/maneuverable-target/omni_drones/learning/predictor/data_pipeline.py) | 意图标注 + 轨迹记录 + 滑窗数据集 | torch |

### 仿真模块 `omni_drones/envs/hide_and_seek/`

| 文件 | 功能 | 依赖 |
|------|------|------|
| [target_fsm.py](file:///data/uavlab/maneuverable-target/omni_drones/envs/hide_and_seek/target_fsm.py) | 目标运动FSM：5种策略 (APF直冲/蛇形/螺旋/急转/俯冲) + 切换逻辑 | torch |
| [proportional_navigation.py](file:///data/uavlab/maneuverable-target/omni_drones/envs/hide_and_seek/proportional_navigation.py) | 3D比例导引法 (追捕方数据收集阶段) | torch |

### 脚本

| 文件 | 功能 | 运行方式 |
|------|------|---------|
| [test_predictor.py](file:///data/uavlab/maneuverable-target/scripts/test_predictor.py) | 网络单元测试 (9项) | `python3 scripts/test_predictor.py` |
| [train_predictor.py](file:///data/uavlab/maneuverable-target/scripts/train_predictor.py) | 完整训练+MVS评估 (合成数据) | `python3 scripts/train_predictor.py` |
| [eval_detailed.py](file:///data/uavlab/maneuverable-target/scripts/eval_detailed.py) | 详细评估: 混淆矩阵、每类意图可视化、MVS分箱 | `python3 scripts/eval_detailed.py` |

### 设计文档

| 文件 | 内容 |
|------|------|
| [intent_gmm_design.md](file:///data/uavlab/maneuverable-target/docs/intent_gmm_design.md) | 完整设计文档（问题定义、网络架构、数据流、损失函数、创新点） |
| [双意图思路.md](file:///data/uavlab/maneuverable-target/docs/papers/双意图思路.md) | 早期讨论记录 |

### 输出目录

| 目录 | 内容 |
|------|------|
| `checkpoints/` | 训练好的模型 (predictor_phase1.pt, predictor_phase2.pt) |
| `results/` | 评估图表 (MVS散点图、对比条形图、分箱统计) |

---

## 运行方式

### 运行网络单元测试
```bash
cd /data/uavlab/maneuverable-target
python3 scripts/test_predictor.py
```

### 运行完整训练+评估
```bash
cd /data/uavlab/maneuverable-target
python3 scripts/train_predictor.py
```

输出：
- `checkpoints/predictor_phase1.pt` — Phase 1 意图分类模型
- `checkpoints/predictor_phase2.pt` — Phase 2 完整模型
- `results/training_curves.png` — 训练曲线
- `results/mvs_scatter.png` — MVS vs 权重熵/方差/ADE 散点图
- `results/mvs_comparison.png` — 三组对比条形图
- `results/mvs_binned.png` — MVS分箱统计

---

## 时间窗参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 物理步 dt | 0.01s | 100Hz |
| 下采样率 | 5 | 每5步取1帧 → 20Hz |
| T_obs | 20帧 | 1.0秒观测窗 |
| T_pred | 10帧 | 0.5秒预测窗 |
| W_mvs | 60帧 | 3.0秒MVS统计窗 |

## 场景参数

| 参数 | 值 |
|------|-----|
| 场地 | 半径5m, 高5m 圆柱 |
| 最大速度 | 3 m/s (指令上限) |
| 最大加速度 | 3 m/s² (指令上限) |
| 追捕方 | 3架, 比例导引 (N=4) |
| 目标 | FSM切换, 每1秒重评估 |

---

## 后续待开发

- [ ] 接入Isaac Sim环境 (替换合成模拟器)
- [ ] 与RL追捕方策略的对接
- [ ] 在真实仿真数据上微调网络
- [ ] 实验结果整理, 写入论文
