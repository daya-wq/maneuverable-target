#!/usr/bin/env python3
"""
意图识别 + GMM轨迹预测 完整测试脚本

用合成的3D点质量数据测试整条pipeline:
  1. 合成数据生成 (模拟APF目标 + 3追捕方)
  2. 自动意图标注
  3. 网络前向传播验证
  4. 损失计算验证
  5. 梯度更新验证
  6. 迷你训练循环 (验证loss下降)
  7. MVS调制效果验证

运行方式:
    cd /data/uavlab/maneuverable-target
    python scripts/test_predictor.py

预期输出: 所有测试通过, 无FAIL
"""

import sys
import os
import math
import numpy as np

# 直接导入predictor子包，绕过omni_drones顶层__init__的重依赖(tensordict等)
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

import torch
import torch.optim as optim

# 绕过 omni_drones/__init__.py (依赖tensordict/Isaac Sim)
# 通过临时替换 omni_drones 包来避免顶层导入
import types
import importlib

# 创建空的 omni_drones 命名空间
omni_drones_pkg = types.ModuleType("omni_drones")
omni_drones_pkg.__path__ = [os.path.join(project_root, "omni_drones")]
omni_drones_pkg.__package__ = "omni_drones"
sys.modules["omni_drones"] = omni_drones_pkg

omni_drones_learning = types.ModuleType("omni_drones.learning")
omni_drones_learning.__path__ = [os.path.join(project_root, "omni_drones", "learning")]
omni_drones_learning.__package__ = "omni_drones.learning"
sys.modules["omni_drones.learning"] = omni_drones_learning

# 现在可以安全导入 predictor 子包
from omni_drones.learning.predictor.intent_gmm_predictor import IntentConditionedGMMPredictor
from omni_drones.learning.predictor.losses import IntentLoss, IntentGMMLoss
from omni_drones.learning.predictor.mvs import (
    compute_maneuver_variability_score,
    compute_mvs_batch,
)


# ==========================================================================
# 合成数据生成器
# ==========================================================================

INTENT_LABELS = [
    "Straight", "Left-Turn", "Right-Turn", "Climb", "Dive",
    "Left-Climb", "Left-Dive", "Right-Climb", "Right-Dive", "Hover",
]


def generate_synthetic_trajectory(
    n_steps: int = 50,
    dt: float = 0.1,
    intent: int = 0,
    noise_std: float = 0.02,
) -> dict:
    """
    生成一条合成3D轨迹, 带自动意图标签

    模拟一个简化的点质量目标 + 3个追捕方

    Args:
        n_steps: 总步数
        dt: 时间步长
        intent: 0-9的意图标签, 决定目标运动方式
        noise_std: 观测噪声

    Returns:
        dict: target_pos, target_vel, pursuer_pos, pursuer_vel, intent_label
    """
    # 目标初始状态
    target_pos = np.zeros((n_steps, 3))
    target_vel = np.zeros((n_steps, 3))

    # 初始条件
    target_pos[0] = [0.0, 0.0, 1.5]
    target_vel[0] = [2.0, 0.0, 0.0]  # 初始朝x正方向飞

    # 根据意图设定加速度方向
    accel_map = {
        0: [0.5, 0.0, 0.0],       # Straight: 前向加速
        1: [0.2, 1.0, 0.0],       # Left-Turn: 侧向
        2: [0.2, -1.0, 0.0],      # Right-Turn
        3: [0.2, 0.0, 0.8],       # Climb
        4: [0.2, 0.0, -0.8],      # Dive
        5: [0.2, 0.8, 0.6],       # Left-Climb
        6: [0.2, 0.8, -0.6],      # Left-Dive
        7: [0.2, -0.8, 0.6],      # Right-Climb
        8: [0.2, -0.8, -0.6],     # Right-Dive
        9: [-1.5, 0.0, 0.0],      # Hover: 减速到接近停止
    }
    accel = np.array(accel_map.get(intent, [0.0, 0.0, 0.0]))

    for t in range(1, n_steps):
        # 简单的质点运动
        target_vel[t] = target_vel[t - 1] + accel * dt
        # 限速
        speed = np.linalg.norm(target_vel[t])
        if speed > 5.0:
            target_vel[t] = target_vel[t] / speed * 5.0
        # Hover特殊处理: 速度衰减
        if intent == 9:
            target_vel[t] *= 0.95
        target_pos[t] = target_pos[t - 1] + target_vel[t] * dt

    # 生成3个追捕方的轨迹 (简化: 从不同方向跟踪目标)
    pursuer_offsets = [
        [-3.0, 2.0, 0.5],   # 追捕方1: 左后上方
        [-3.0, -2.0, 0.0],  # 追捕方2: 右后方
        [-2.0, 0.0, -0.5],  # 追捕方3: 正后下方
    ]
    pursuer_pos = np.zeros((3, n_steps, 3))
    pursuer_vel = np.zeros((3, n_steps, 3))

    for p in range(3):
        offset = np.array(pursuer_offsets[p])
        for t in range(n_steps):
            # 追捕方简单跟踪目标
            pursuer_pos[p, t] = target_pos[t] + offset * (1 - 0.01 * t)
            if t > 0:
                pursuer_vel[p, t] = (pursuer_pos[p, t] - pursuer_pos[p, t - 1]) / dt

    # 添加观测噪声
    target_pos += np.random.randn(*target_pos.shape) * noise_std
    target_vel += np.random.randn(*target_vel.shape) * noise_std
    pursuer_pos += np.random.randn(*pursuer_pos.shape) * noise_std
    pursuer_vel += np.random.randn(*pursuer_vel.shape) * noise_std

    return {
        "target_pos": target_pos,         # [n_steps, 3]
        "target_vel": target_vel,         # [n_steps, 3]
        "pursuer_pos": pursuer_pos,       # [3, n_steps, 3]
        "pursuer_vel": pursuer_vel,       # [3, n_steps, 3]
        "intent_label": intent,
    }


def create_dataset(
    n_samples: int = 500,
    t_obs: int = 10,
    t_pred: int = 5,
) -> dict:
    """
    生成训练数据集

    Args:
        n_samples: 样本数
        t_obs: 观测窗口步数
        t_pred: 预测步数

    Returns:
        dict: obs_history, gt_future_pos, gt_intent_label, mvs
    """
    obs_list = []
    future_list = []
    intent_list = []
    intent_history_list = []

    for i in range(n_samples):
        intent = i % 10  # 均匀覆盖10类
        traj = generate_synthetic_trajectory(
            n_steps=t_obs + t_pred + 5,
            intent=intent,
        )

        # 构建观测: [T_obs, 24]
        # target(pos3+vel3) + 3*pursuer(pos3+vel3)
        obs_frames = []
        for t in range(t_obs):
            frame = np.concatenate([
                traj["target_pos"][t],
                traj["target_vel"][t],
                traj["pursuer_pos"][0, t],
                traj["pursuer_vel"][0, t],
                traj["pursuer_pos"][1, t],
                traj["pursuer_vel"][1, t],
                traj["pursuer_pos"][2, t],
                traj["pursuer_vel"][2, t],
            ])
            obs_frames.append(frame)

        obs = np.stack(obs_frames)  # [T_obs, 24]

        # 未来位置: [T_pred, 3]
        future = traj["target_pos"][t_obs: t_obs + t_pred]

        obs_list.append(obs)
        future_list.append(future)
        intent_list.append(intent)

        # 模拟意图历史 (用于MVS计算)
        # 稳定意图: 历史全是同一个
        if i % 3 == 0:
            intent_hist = [intent] * 30
        # 中等变异: 偶尔切换
        elif i % 3 == 1:
            intent_hist = [intent if j % 5 != 0 else (intent + 1) % 10 for j in range(30)]
        # 高变异: 频繁切换
        else:
            intent_hist = [np.random.randint(0, 10) for _ in range(30)]
        intent_history_list.append(intent_hist)

    obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32)
    future_tensor = torch.tensor(np.stack(future_list), dtype=torch.float32)
    intent_tensor = torch.tensor(intent_list, dtype=torch.long)

    # 计算MVS
    intent_hist_tensor = torch.tensor(
        np.stack(intent_history_list), dtype=torch.long
    )
    mvs_tensor = compute_mvs_batch(intent_hist_tensor)

    return {
        "obs_history": obs_tensor,         # [N, T_obs, 24]
        "gt_future_pos": future_tensor,    # [N, T_pred, 3]
        "gt_intent_label": intent_tensor,  # [N]
        "mvs": mvs_tensor,                 # [N, 1]
    }


# ==========================================================================
# 测试函数
# ==========================================================================

def test_model_construction():
    """测试1: 网络能否正确构建"""
    print("\n" + "=" * 60)
    print("测试1: 网络构建")
    print("=" * 60)

    model = IntentConditionedGMMPredictor(
        obs_dim=24, n_intents=10, n_components=3,
        pred_horizon=5, hidden_dim=128,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  网络配置: {model.get_config()}")

    assert total_params > 0, "参数量不应为0"
    assert total_params == trainable_params, "所有参数应该可训练"
    print("  ✅ PASS")
    return model


def test_forward_pass(model):
    """测试2: 前向传播输出shape"""
    print("\n" + "=" * 60)
    print("测试2: 前向传播")
    print("=" * 60)

    B, T_obs = 8, 10
    obs = torch.randn(B, T_obs, 24)
    mvs = torch.rand(B, 1)

    # 完整前向传播
    output = model(obs, mvs=mvs)

    expected_shapes = {
        "intent_logits": (B, 10),
        "intent_probs": (B, 10),
        "gmm_weights": (B, 3),
        "gmm_means": (B, 3, 5, 3),
        "gmm_vars": (B, 3, 5, 3),
        "mixed_mu": (B, 5, 3),
        "mixed_var": (B, 5, 3),
    }

    all_pass = True
    for key, expected in expected_shapes.items():
        actual = tuple(output[key].shape)
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            all_pass = False
        print(f"  {key}: expected {expected}, got {actual} {status}")

    # 验证概率约束
    probs_sum = output["intent_probs"].sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones(B), atol=1e-5), "意图概率应该求和为1"
    print(f"  意图概率和: {probs_sum.mean():.6f} ✅")

    weights_sum = output["gmm_weights"].sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones(B), atol=1e-5), "GMM权重应该求和为1"
    print(f"  GMM权重和: {weights_sum.mean():.6f} ✅")

    # 验证方差为正
    assert (output["gmm_vars"] > 0).all(), "方差应该为正"
    assert (output["mixed_var"] > 0).all(), "混合方差应该为正"
    print(f"  方差正性: ✅")

    # 验证无NaN
    for key, val in output.items():
        assert not torch.isnan(val).any(), f"{key}中存在NaN"
    print(f"  无NaN: ✅")

    assert all_pass, "存在shape不匹配"
    print("  ✅ PASS")
    return output


def test_intent_only(model):
    """测试3: 仅意图分类模式"""
    print("\n" + "=" * 60)
    print("测试3: 意图分类模式 (Phase 1)")
    print("=" * 60)

    B, T_obs = 4, 10
    obs = torch.randn(B, T_obs, 24)

    output = model.predict_intent_only(obs)

    assert "intent_logits" in output, "应该包含intent_logits"
    assert "intent_probs" in output, "应该包含intent_probs"
    assert output["intent_logits"].shape == (B, 10)
    assert "gmm_weights" not in output, "不应该包含gmm_weights"

    print(f"  intent_logits shape: {output['intent_logits'].shape} ✅")
    print(f"  intent_probs shape: {output['intent_probs'].shape} ✅")
    print("  ✅ PASS")


def test_losses():
    """测试4: 损失函数计算"""
    print("\n" + "=" * 60)
    print("测试4: 损失函数")
    print("=" * 60)

    model = IntentConditionedGMMPredictor()
    B, T_obs = 16, 10

    obs = torch.randn(B, T_obs, 24)
    mvs = torch.rand(B, 1)
    gt_future = torch.randn(B, 5, 3)
    gt_intent = torch.randint(0, 10, (B,))

    output = model(obs, mvs=mvs)

    # Phase 1 损失
    intent_loss_fn = IntentLoss()
    loss1, metrics1 = intent_loss_fn(output, gt_intent)
    assert not torch.isnan(loss1), "Phase 1 loss为NaN"
    assert loss1.item() > 0, "Phase 1 loss应该为正"
    print(f"  Phase 1 Loss: {loss1.item():.4f}, Accuracy: {metrics1['intent_accuracy']:.4f} ✅")

    # Phase 2 损失
    gmm_loss_fn = IntentGMMLoss()
    loss2, metrics2 = gmm_loss_fn(output, gt_future, gt_intent)
    assert not torch.isnan(loss2), "Phase 2 loss为NaN"
    assert math.isfinite(loss2.item()), "Phase 2 loss不是有限数"
    print(f"  Phase 2 Loss: {loss2.item():.4f}")
    print(f"    L_intent: {metrics2['loss_intent']:.4f}")
    print(f"    L_gmm_nll: {metrics2['loss_gmm_nll']:.4f}")
    print(f"    Traj MSE: {metrics2['traj_mse']:.4f}")
    print(f"    Weight Entropy: {metrics2['gmm_weight_entropy']:.4f}")
    print("  ✅ PASS")


def test_gradient_flow():
    """测试5: 梯度反向传播"""
    print("\n" + "=" * 60)
    print("测试5: 梯度流")
    print("=" * 60)

    model = IntentConditionedGMMPredictor()
    B, T_obs = 8, 10

    obs = torch.randn(B, T_obs, 24)
    mvs = torch.rand(B, 1)
    gt_future = torch.randn(B, 5, 3)
    gt_intent = torch.randint(0, 10, (B,))

    output = model(obs, mvs=mvs)
    loss_fn = IntentGMMLoss()
    loss, _ = loss_fn(output, gt_future, gt_intent)
    loss.backward()

    # 检查关键层的梯度
    layers_to_check = {
        "encoder.weight_ih_l0": model.encoder.weight_ih_l0,
        "intent_head[-1].weight": model.intent_head[-1].weight,
        "context_mlp[0].weight": model.context_mlp[0].weight,
        "weight_head.weight": model.weight_head.weight,
        "mean_head.weight": model.mean_head.weight,
        "logvar_head.weight": model.logvar_head.weight,
    }

    all_have_grad = True
    for name, param in layers_to_check.items():
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        is_finite = param.grad is not None and torch.isfinite(param.grad).all()
        status = "✅" if (has_grad and is_finite) else "❌"
        if not (has_grad and is_finite):
            all_have_grad = False
        grad_norm = param.grad.norm().item() if param.grad is not None else 0
        print(f"  {name}: grad_norm={grad_norm:.6f} {status}")

    assert all_have_grad, "存在没有梯度的层"
    print("  ✅ PASS")


def test_mvs():
    """测试6: MVS计算"""
    print("\n" + "=" * 60)
    print("测试6: 机动变异度得分 (MVS)")
    print("=" * 60)

    # 完全稳定: 全是同一个意图
    mvs_stable = compute_maneuver_variability_score([0] * 30)
    print(f"  稳定序列 [0,0,0,...]:      MVS = {mvs_stable:.4f}")
    assert mvs_stable < 0.1, f"稳定序列MVS应该接近0, 实际={mvs_stable}"

    # 完全随机
    np.random.seed(42)
    mvs_random = compute_maneuver_variability_score(
        np.random.randint(0, 10, 30)
    )
    print(f"  随机序列 [5,2,7,1,...]:    MVS = {mvs_random:.4f}")
    assert mvs_random > 0.5, f"随机序列MVS应该较高, 实际={mvs_random}"

    # 中等变异: 每5步切换一次
    medium_hist = [i // 5 % 10 for i in range(30)]
    mvs_medium = compute_maneuver_variability_score(medium_hist)
    print(f"  中等变异 [0,0,0,0,0,1,...]: MVS = {mvs_medium:.4f}")
    assert 0.1 < mvs_medium < 0.8, f"中等变异MVS应该在中间, 实际={mvs_medium}"

    # 批量计算
    batch_hist = torch.tensor([
        [0] * 30,
        list(range(10)) * 3,
        np.random.randint(0, 10, 30).tolist(),
    ], dtype=torch.long)
    mvs_batch = compute_mvs_batch(batch_hist)
    assert mvs_batch.shape == (3, 1), f"batch MVS shape应为(3,1), 实际={mvs_batch.shape}"
    print(f"  批量MVS: {mvs_batch.squeeze().tolist()}")

    print("  ✅ PASS")


def test_mvs_modulation():
    """测试7: MVS调制效果"""
    print("\n" + "=" * 60)
    print("测试7: MVS 温度缩放 + 方差膨胀效果")
    print("=" * 60)

    model = IntentConditionedGMMPredictor(alpha=2.0, beta=1.5)
    # 给weight_head设置非均匀权重, 否则零初始化下logits全等, 温度缩放不可见
    with torch.no_grad():
        model.weight_head.bias.copy_(torch.tensor([2.0, 0.5, -1.0]))
    model.eval()

    B, T_obs = 4, 10
    obs = torch.randn(B, T_obs, 24)

    # MVS=0 (稳定)
    mvs_low = torch.zeros(B, 1)
    out_low = model(obs, mvs=mvs_low)

    # MVS=1 (高变异)
    mvs_high = torch.ones(B, 1)
    out_high = model(obs, mvs=mvs_high)

    # 验证: MVS高时权重更平坦 (熵更大)
    w_low = out_low["gmm_weights"]
    w_high = out_high["gmm_weights"]
    entropy_low = -(w_low * torch.log(w_low + 1e-8)).sum(dim=-1).mean()
    entropy_high = -(w_high * torch.log(w_high + 1e-8)).sum(dim=-1).mean()
    print(f"  MVS=0 权重: {w_low[0].tolist()}")
    print(f"  MVS=1 权重: {w_high[0].tolist()}")
    print(f"  MVS=0 权重熵: {entropy_low.item():.4f}")
    print(f"  MVS=1 权重熵: {entropy_high.item():.4f}")
    assert entropy_high > entropy_low, "高MVS的权重熵应该更大"
    print(f"  温度缩放效果: 高MVS权重更平坦 ✅")

    # 验证: MVS高时方差更大
    var_low = out_low["gmm_vars"].mean()
    var_high = out_high["gmm_vars"].mean()
    print(f"  MVS=0 平均方差: {var_low.item():.4f}")
    print(f"  MVS=1 平均方差: {var_high.item():.4f}")
    assert var_high > var_low, "高MVS的方差应该更大"
    print(f"  方差膨胀效果: 高MVS方差更大 ✅")

    # 方差膨胀比例验证
    expected_ratio = 1.0 + 1.5 * 1.0  # 1 + beta * mvs = 2.5
    actual_ratio = var_high.item() / var_low.item()
    print(f"  方差膨胀比例: 期望≈{expected_ratio:.1f}, 实际={actual_ratio:.2f}")
    assert abs(actual_ratio - expected_ratio) < 0.1, "方差膨胀比例不对"

    print("  ✅ PASS")


def test_mini_training():
    """测试8: 迷你训练循环 (验证loss下降)"""
    print("\n" + "=" * 60)
    print("测试8: 迷你训练循环")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # 创建合成数据集
    print("  生成合成数据...")
    dataset = create_dataset(n_samples=200, t_obs=10, t_pred=5)

    model = IntentConditionedGMMPredictor(
        obs_dim=24, n_intents=10, n_components=3,
        pred_horizon=5, hidden_dim=64,  # 小网络加速测试
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = IntentGMMLoss()

    # 训练循环
    n_epochs = 20
    batch_size = 32
    N = dataset["obs_history"].shape[0]
    losses = []

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        # 简单的batch迭代
        indices = torch.randperm(N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = indices[start:end]

            obs = dataset["obs_history"][idx]
            gt_future = dataset["gt_future_pos"][idx]
            gt_intent = dataset["gt_intent_label"][idx]
            mvs = dataset["mvs"][idx]

            optimizer.zero_grad()
            output = model(obs, mvs=mvs)
            loss, metrics = loss_fn(output, gt_future, gt_intent)
            loss.backward()

            # 梯度裁剪 (防止爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, "
                  f"acc={metrics['intent_accuracy']:.4f}, "
                  f"traj_mse={metrics['traj_mse']:.4f}, "
                  f"weight_ent={metrics['gmm_weight_entropy']:.4f}")

    # 验证loss下降
    first_loss = np.mean(losses[:3])
    last_loss = np.mean(losses[-3:])
    print(f"\n  前3个epoch平均loss: {first_loss:.4f}")
    print(f"  后3个epoch平均loss: {last_loss:.4f}")
    assert last_loss < first_loss, f"Loss未下降! 前={first_loss:.4f}, 后={last_loss:.4f}"
    print(f"  Loss下降: {first_loss:.4f} → {last_loss:.4f} ✅")
    print("  ✅ PASS")


def test_phase1_training():
    """测试9: Phase 1 纯意图分类训练"""
    print("\n" + "=" * 60)
    print("测试9: Phase 1 — 纯意图分类训练")
    print("=" * 60)

    torch.manual_seed(123)
    np.random.seed(123)

    dataset = create_dataset(n_samples=200, t_obs=10, t_pred=5)

    model = IntentConditionedGMMPredictor(hidden_dim=64)
    # Phase 1: 只训练编码器和意图分类头
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = IntentLoss()

    model.train()
    losses, accs = [], []
    for epoch in range(30):
        indices = torch.randperm(200)
        for start in range(0, 200, 32):
            end = min(start + 32, 200)
            idx = indices[start:end]

            obs = dataset["obs_history"][idx]
            gt_intent = dataset["gt_intent_label"][idx]

            optimizer.zero_grad()
            output = model.predict_intent_only(obs)
            loss, metrics = loss_fn(output, gt_intent)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        losses.append(loss.item())
        accs.append(metrics["intent_accuracy"])

        if epoch % 10 == 0 or epoch == 29:
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, accuracy={metrics['intent_accuracy']:.4f}")

    final_acc = np.mean(accs[-5:])
    print(f"\n  最后5个epoch平均准确率: {final_acc:.4f}")
    assert final_acc > 0.3, f"准确率太低: {final_acc:.4f} (10类随机=0.1, 应该>0.3)"
    print(f"  意图分类学习有效 ✅")
    print("  ✅ PASS")


# ==========================================================================
# 主入口
# ==========================================================================

def main():
    print("=" * 60)
    print("  意图识别 + GMM 轨迹预测 — 完整测试")
    print("=" * 60)

    device = "cpu"  # 合成数据测试不需要GPU
    print(f"\n  设备: {device}")
    print(f"  PyTorch版本: {torch.__version__}")

    # 依次运行所有测试
    model = test_model_construction()
    test_forward_pass(model)
    test_intent_only(model)
    test_losses()
    test_gradient_flow()
    test_mvs()
    test_mvs_modulation()
    test_mini_training()
    test_phase1_training()

    print("\n" + "=" * 60)
    print("  🎉 所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
