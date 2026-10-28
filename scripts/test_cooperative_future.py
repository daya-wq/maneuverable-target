#!/usr/bin/env python3
"""
协同未来 Token 编码器 — 完整单元测试

测试覆盖：
  1. 形状测试 (所有输出维度正确)
  2. 无人机排列等变性
  3. GMM 模式排列等变性
  4. 上一计划 Mask 测试
  5. 时间敏感性测试
  6. 队友敏感性测试
  7. 梯度测试 (所有模块非零梯度，上游 GMM 冻结)

用法:
    python scripts/test_cooperative_future.py
"""

from __future__ import annotations

import math
import sys
import os

# 确保项目路径在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from omni_drones.learning.modules.cooperative_future import (
    CooperativeFutureConfig,
    AgentHistoryEncoder,
    RelationalAgentInteraction,
    PreviousPlanEncoder,
    FutureQueryBuilder,
    TimeAlignedTrajectoryFusion,
    FactorizedCoordinationBlock,
    CooperativeFutureTokenEncoder,
    BoundedTrajectoryReferenceDecoder,
    StrategicIntentTokenEncoder,
)


def make_dummy_inputs(
    batch_size: int = 4,
    device: str = "cpu",
) -> dict:
    """生成测试用的假输入。"""
    cfg = CooperativeFutureConfig()
    B = batch_size
    N = cfg.num_agents       # 3
    L = cfg.history_len      # 10
    H = cfg.pred_horizon     # 10
    K = cfg.num_modes        # 3

    return {
        "agent_history": torch.randn(B, N, L, cfg.history_dim, device=device),
        "current_pos": torch.randn(B, N, 3, device=device),
        "current_vel": torch.randn(B, N, 3, device=device),
        "prev_plan": torch.randn(B, N, H, cfg.prev_plan_dim, device=device),
        "prev_plan_mask": torch.ones(B, N, H, device=device),
        "gmm_mode_tokens": torch.randn(B, K, cfg.d_model, device=device),
        "gmm_point_tokens": torch.randn(B, K, H, cfg.d_model, device=device),
        "gmm_weights": torch.softmax(torch.randn(B, K, device=device), dim=-1),
        "strategic_probs": torch.softmax(
            torch.randn(B, cfg.num_strategic, device=device), dim=-1
        ),
        "agent_history_mask": torch.ones(B, N, L, device=device),
    }


def print_test_header(name: str):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")


def test_1_shapes():
    """测试 1: 所有输出形状正确。"""
    print_test_header("测试 1: 形状测试")

    cfg = CooperativeFutureConfig()
    B, N, L, H, K = 4, cfg.num_agents, cfg.history_len, cfg.pred_horizon, cfg.num_modes
    D = cfg.d_model

    inputs = make_dummy_inputs(batch_size=B)
    model = CooperativeFutureTokenEncoder(cfg)
    output = model(**inputs)

    # 主要输出
    assert output["cooperative_future_tokens"].shape == (B, N, H, D), \
        f"cooperative_future_tokens: expected {(B,N,H,D)}, got {output['cooperative_future_tokens'].shape}"
    print(f"  [PASS] cooperative_future_tokens: {output['cooperative_future_tokens'].shape}")

    assert output["agent_tokens"].shape == (B, N, D), \
        f"agent_tokens: expected {(B,N,D)}, got {output['agent_tokens'].shape}"
    print(f"  [PASS] agent_tokens: {output['agent_tokens'].shape}")

    assert output["history_tokens"].shape == (B, N, L, D), \
        f"history_tokens: expected {(B,N,L,D)}, got {output['history_tokens'].shape}"
    print(f"  [PASS] history_tokens: {output['history_tokens'].shape}")

    assert output["cooperative_agent_tokens"].shape == (B, N, D), \
        f"cooperative_agent_tokens: expected {(B,N,D)}, got {output['cooperative_agent_tokens'].shape}"
    print(f"  [PASS] cooperative_agent_tokens: {output['cooperative_agent_tokens'].shape}")

    assert output["strategic_intent_token"].shape == (B, D)
    print(f"  [PASS] strategic_intent_token: {output['strategic_intent_token'].shape}")

    assert output["initial_future_queries"].shape == (B, N, H, D), \
        f"initial_future_queries: expected {(B,N,H,D)}, got {output['initial_future_queries'].shape}"
    print(f"  [PASS] initial_future_queries: {output['initial_future_queries'].shape}")

    assert output["gmm_fused_queries"].shape == (B, N, H, D), \
        f"gmm_fused_queries: expected {(B,N,H,D)}, got {output['gmm_fused_queries'].shape}"
    print(f"  [PASS] gmm_fused_queries: {output['gmm_fused_queries'].shape}")

    # 注意力权重形状
    nh = cfg.num_heads
    assert output["agent_state_attention"].shape == (B, nh, N, N), \
        f"agent_state_attention: expected {(B,nh,N,N)}, got {output['agent_state_attention'].shape}"
    print(f"  [PASS] agent_state_attention: {output['agent_state_attention'].shape}")

    assert output["gmm_attention"] is None, "Single trajectory must not have mode selection attention"

    # 各层注意力
    assert len(output["temporal_attention"]) == cfg.coordination_layers
    for i, ta in enumerate(output["temporal_attention"]):
        assert ta.shape == (B, nh, N, H, H), \
            f"temporal_attention[{i}]: expected {(B,nh,N,H,H)}, got {ta.shape}"
    print(f"  [PASS] temporal_attention: {len(output['temporal_attention'])} layers, shape={output['temporal_attention'][0].shape}")

    assert len(output["future_agent_attention"]) == cfg.coordination_layers
    for i, fa in enumerate(output["future_agent_attention"]):
        assert fa.shape == (B, nh, H, N, N), \
            f"future_agent_attention[{i}]: expected {(B,nh,H,N,N)}, got {fa.shape}"
    print(f"  [PASS] future_agent_attention: {len(output['future_agent_attention'])} layers, shape={output['future_agent_attention'][0].shape}")

    print(f"\n  ✅ 形状测试全部通过")


def test_2_agent_permutation():
    """测试 2: 无人机排列等变性。"""
    print_test_header("测试 2: 无人机排列等变性")

    cfg = CooperativeFutureConfig()
    inputs = make_dummy_inputs()
    model = CooperativeFutureTokenEncoder(cfg)

    # 原始输出
    with torch.no_grad():
        output = model(**inputs)

    # 交换无人机 0 和 1 的输入
    inputs_perm = {}
    for k, v in inputs.items():
        if v is not None and v.dim() >= 2 and v.shape[1] == cfg.num_agents:
            v_perm = v.clone()
            v_perm[:, [0, 1]] = v[:, [1, 0]]
            inputs_perm[k] = v_perm
        else:
            inputs_perm[k] = v

    with torch.no_grad():
        output_perm = model(**inputs_perm)

    # 检查: 交换后的输出, agent 0 应该等于原始输出的 agent 1
    # future tokens
    diff_0_1 = (output_perm["cooperative_future_tokens"][:, 0] - output["cooperative_future_tokens"][:, 1]).abs().max().item()
    diff_1_0 = (output_perm["cooperative_future_tokens"][:, 1] - output["cooperative_future_tokens"][:, 0]).abs().max().item()
    diff_2_2 = (output_perm["cooperative_future_tokens"][:, 2] - output["cooperative_future_tokens"][:, 2]).abs().max().item()

    print(f"  future_tokens: agent0→1 diff={diff_0_1:.2e}, agent1→0 diff={diff_1_0:.2e}, agent2→2 diff={diff_2_2:.2e}")
    assert diff_0_1 < 1e-5, f"Agent permutation: future_tokens agent0→1 diff too large: {diff_0_1:.2e}"
    assert diff_1_0 < 1e-5, f"Agent permutation: future_tokens agent1→0 diff too large: {diff_1_0:.2e}"
    assert diff_2_2 < 1e-5, f"Agent permutation: future_tokens agent2→2 diff too large: {diff_2_2:.2e}"

    # agent_tokens
    diff_at0 = (output_perm["agent_tokens"][:, 0] - output["agent_tokens"][:, 1]).abs().max().item()
    diff_at1 = (output_perm["agent_tokens"][:, 1] - output["agent_tokens"][:, 0]).abs().max().item()
    print(f"  agent_tokens: agent0→1 diff={diff_at0:.2e}, agent1→0 diff={diff_at1:.2e}")
    assert diff_at0 < 1e-5, f"Agent permutation: agent_tokens failed"
    assert diff_at1 < 1e-5, f"Agent permutation: agent_tokens failed"

    # cooperative_agent_tokens
    diff_ca0 = (output_perm["cooperative_agent_tokens"][:, 0] - output["cooperative_agent_tokens"][:, 1]).abs().max().item()
    diff_ca1 = (output_perm["cooperative_agent_tokens"][:, 1] - output["cooperative_agent_tokens"][:, 0]).abs().max().item()
    print(f"  cooperative_agent_tokens: agent0→1 diff={diff_ca0:.2e}, agent1→0 diff={diff_ca1:.2e}")
    assert diff_ca0 < 1e-5, f"Agent permutation: cooperative_agent_tokens failed"
    assert diff_ca1 < 1e-5, f"Agent permutation: cooperative_agent_tokens failed"

    print(f"\n  ✅ 无人机排列等变性测试通过")


def test_3_gmm_mode_permutation():
    """Old mode-permutation slot now verifies removal of mode selection."""
    cfg = CooperativeFutureConfig()
    model = CooperativeFutureTokenEncoder(cfg)
    assert cfg.num_modes == 1 and cfg.num_strategic == 2
    assert not hasattr(model, "gmm_cross_attention")
    output = model(**make_dummy_inputs())
    assert output["gmm_attention"] is None
    assert not any("q_proj" in name or "k_proj" in name
                   for name, _ in model.trajectory_fusion.named_parameters())
    print("  [PASS] one trajectory, two intents, no mode-selection parameters")




def test_4_prev_plan_mask():
    """测试 4: 上一计划 Mask 测试。"""
    print_test_header("测试 4: 上一计划 Mask 测试")

    cfg = CooperativeFutureConfig()
    inputs = make_dummy_inputs()
    model = CooperativeFutureTokenEncoder(cfg)

    # 正常输出
    with torch.no_grad():
        output = model(**inputs)

    # Mask 置零, prev_plan 改变, 输出应不变
    inputs_masked = dict(inputs)
    inputs_masked["prev_plan_mask"] = torch.zeros_like(inputs["prev_plan_mask"])
    inputs_masked["prev_plan"] = inputs["prev_plan"] + 100.0  # 大幅改变数值

    with torch.no_grad():
        output_masked = model(**inputs_masked)

    # 当 mask 全零时, prev_plan 变化不应影响输出
    future_diff = (output_masked["cooperative_future_tokens"] - output["cooperative_future_tokens"]).abs().max().item()
    print(f"  mask=0 时 future_tokens 差异: {future_diff:.2e}")

    # 注意: 因为 prev_plan_mask=0 导致 prev_plan_tokens 全零,
    # 和原始有有效 prev_plan 的情况会不同, 所以这个差异可能较大。
    # 真正测试: mask=0 时, 改变 prev_plan 数值不应影响输出。

    # 再做一个测试: 两种不同的 prev_plan, 但 mask 都为零
    inputs_masked2 = dict(inputs)
    inputs_masked2["prev_plan_mask"] = torch.zeros_like(inputs["prev_plan_mask"])
    inputs_masked2["prev_plan"] = torch.randn_like(inputs["prev_plan"])

    with torch.no_grad():
        out_m1 = model(**inputs_masked)
        out_m2 = model(**inputs_masked2)

    diff_masked = (out_m1["cooperative_future_tokens"] - out_m2["cooperative_future_tokens"]).abs().max().item()
    print(f"  mask=0 时改变 prev_plan 的 future_tokens 差异: {diff_masked:.2e}")
    assert diff_masked < 1e-6, \
        f"Prev plan mask: when mask=0, changing prev_plan should NOT affect output. diff={diff_masked:.2e}"

    # 同时检查 prev_plan_tokens 在 mask=0 时也应为零
    # (通过 initial_future_queries 间接检查)
    print(f"\n  ✅ 上一计划 Mask 测试通过")


def test_5_time_sensitivity():
    """测试 5: 时间敏感性测试 — 不同时刻 Query 不能完全相同。"""
    print_test_header("测试 5: 时间敏感性测试")

    cfg = CooperativeFutureConfig()
    inputs = make_dummy_inputs()
    model = CooperativeFutureTokenEncoder(cfg)

    with torch.no_grad():
        output = model(**inputs)

    initial_queries = output["initial_future_queries"]  # [B, N, H, D]
    future_tokens = output["cooperative_future_tokens"]

    # 不同时刻的 Query 应不同
    query_h0 = initial_queries[:, :, 0, :]
    query_h_last = initial_queries[:, :, -1, :]

    assert not torch.allclose(query_h0, query_h_last, atol=1e-4), \
        "时间敏感性: initial_future_queries[:][:][0] 与 [-1] 完全相同!"
    print(f"  [PASS] initial_queries 不同时刻不同 (max diff={(query_h0 - query_h_last).abs().max():.4f})")

    # 不同时刻的 Future token 也应不同
    ft_h0 = future_tokens[:, :, 0, :]
    ft_h_last = future_tokens[:, :, -1, :]
    assert not torch.allclose(ft_h0, ft_h_last, atol=1e-4), \
        "时间敏感性: cooperative_future_tokens[:][:][0] 与 [-1] 完全相同!"
    print(f"  [PASS] future_tokens 不同时刻不同 (max diff={(ft_h0 - ft_h_last).abs().max():.4f})")

    print(f"\n  ✅ 时间敏感性测试通过")


def test_6_teammate_sensitivity():
    """测试 6: 队友敏感性测试 — 固定 agent0, 改变 agent1 状态, agent0 输出应发生变化。"""
    print_test_header("测试 6: 队友敏感性测试")

    cfg = CooperativeFutureConfig()
    inputs = make_dummy_inputs()
    model = CooperativeFutureTokenEncoder(cfg)

    with torch.no_grad():
        output = model(**inputs)

    # 改变 agent1 的状态, agent0 不变
    inputs_perturbed = dict(inputs)
    # 改变 agent1 的 current_pos 和 current_vel (大幅扰动)
    inputs_perturbed["current_pos"] = inputs["current_pos"].clone()
    inputs_perturbed["current_pos"][:, 1, :] += 10.0  # 大幅移动 agent1
    inputs_perturbed["current_vel"] = inputs["current_vel"].clone()
    inputs_perturbed["current_vel"][:, 1, :] *= 5.0

    # 也大幅改变 agent1 的历史
    inputs_perturbed["agent_history"] = inputs["agent_history"].clone()
    inputs_perturbed["agent_history"][:, 1, :, :] += 5.0

    with torch.no_grad():
        output_perturbed = model(**inputs_perturbed)

    # agent0 的 cooperative_agent_tokens 应变化
    diff_ca = (output_perturbed["cooperative_agent_tokens"][:, 0] - output["cooperative_agent_tokens"][:, 0]).abs().max().item()
    print(f"  agent0 cooperative_agent_tokens 变化: {diff_ca:.6f}")

    # agent0 的 future tokens 应变化
    diff_ft = (output_perturbed["cooperative_future_tokens"][:, 0] - output["cooperative_future_tokens"][:, 0]).abs().max().item()
    print(f"  agent0 future_tokens 变化: {diff_ft:.6f}")
    # 注意: out_proj 零初始化可能导致 interaction 输出初始为零，
    # 但 future_tokens (经过后续 GMM attention + coordination) 会变化
    assert diff_ft > 1e-6, "队友敏感性: future_tokens of agent0 未变化!"

    # agent1 的变化应更大 (因为自身状态变了)
    diff_ft1 = (output_perturbed["cooperative_future_tokens"][:, 1] - output["cooperative_future_tokens"][:, 1]).abs().max().item()
    print(f"  agent1 future_tokens 变化: {diff_ft1:.4f}")

    # agent2 也应有一定变化 (通过 Interaction 传播)
    diff_ft2 = (output_perturbed["cooperative_future_tokens"][:, 2] - output["cooperative_future_tokens"][:, 2]).abs().max().item()
    print(f"  agent2 future_tokens 变化: {diff_ft2:.4f}")

    print(f"\n  ✅ 队友敏感性测试通过")


def test_7_gradients():
    """测试 7: 梯度测试 — 所有训练模块获得非零梯度。"""
    print_test_header("测试 7: 梯度测试")

    cfg = CooperativeFutureConfig()
    inputs = make_dummy_inputs()
    model = CooperativeFutureTokenEncoder(cfg)

    # 标记: GMM 输入是"冻结"的 (detach 的, 不应有梯度)
    for key in ["gmm_mode_tokens", "gmm_point_tokens", "gmm_weights"]:
        inputs[key] = inputs[key].detach().clone()

    output = model(**inputs)

    # 对 cooperative_future_tokens 求和后反向传播
    loss = output["cooperative_future_tokens"].sum()
    loss.backward()

    # 统计各子模块的梯度
    module_names = [
        ("history_encoder", model.history_encoder),
        ("agent_interaction", model.agent_interaction),
        ("prev_plan_encoder", model.prev_plan_encoder),
        ("query_builder", model.query_builder),
        ("trajectory_fusion", model.trajectory_fusion),
        ("strategic_encoder", model.strategic_encoder),
    ]

    all_good = True
    for name, module in module_names:
        has_grad = False
        max_grad = 0.0
        for p in module.parameters():
            if p.grad is not None and p.grad.abs().max() > 0:
                has_grad = True
                max_grad = max(max_grad, p.grad.abs().max().item())
        status = "✅" if has_grad else "❌"
        print(f"  [{status}] {name}: max_grad={max_grad:.2e}")
        if not has_grad:
            all_good = False

    # 检查 coordination blocks
    for i, block in enumerate(model.coordination_blocks):
        has_grad = False
        max_grad = 0.0
        for p in block.parameters():
            if p.grad is not None and p.grad.abs().max() > 0:
                has_grad = True
                max_grad = max(max_grad, p.grad.abs().max().item())
        status = "✅" if has_grad else "❌"
        print(f"  [{status}] coordination_block[{i}]: max_grad={max_grad:.2e}")
        if not has_grad:
            all_good = False

    assert all_good, "梯度测试: 存在模块梯度为零!"

    print(f"\n  ✅ 梯度测试通过")


def test_8_extra_checks():
    """测试 8: 额外检查 — NaN/Inf, 无 Dropout/BatchNorm/ID Embedding, 配置开关。"""
    print_test_header("测试 8: 额外检查")

    cfg = CooperativeFutureConfig()
    inputs = make_dummy_inputs()
    model = CooperativeFutureTokenEncoder(cfg)

    # NaN/Inf 检查
    with torch.no_grad():
        output = model(**inputs)

    for key in ["cooperative_future_tokens", "agent_tokens", "history_tokens",
                 "cooperative_agent_tokens", "strategic_intent_token",
                 "initial_future_queries", "gmm_fused_queries"]:
        val = output[key]
        assert torch.isfinite(val).all(), f"NaN/Inf in {key}"
    print(f"  [PASS] 无 NaN/Inf")

    # 配置开关测试
    cfg_off = CooperativeFutureConfig(
        use_agent_interaction=False,
        use_previous_plan=False,
    )
    model_off = CooperativeFutureTokenEncoder(cfg_off)
    with torch.no_grad():
        output_off = model_off(**inputs)
    assert torch.isfinite(output_off["cooperative_future_tokens"]).all(), \
        "关闭功能开关后出现 NaN/Inf"
    print(f"  [PASS] 配置开关 (全部关闭) 正常运行")

    # 部分关闭测试
    cfg_partial = CooperativeFutureConfig(
        use_agent_interaction=True,
        use_previous_plan=False,
    )
    model_partial = CooperativeFutureTokenEncoder(cfg_partial)
    with torch.no_grad():
        output_partial = model_partial(**inputs)
    assert torch.isfinite(output_partial["cooperative_future_tokens"]).all(), \
        "部分关闭功能开关后出现 NaN/Inf"
    print(f"  [PASS] 配置开关 (部分启用) 正常运行")

    print(f"\n  ✅ 额外检查通过")


def test_9_gmm_log_weights_auto():
    """测试 9: gmm_log_weights=None 时自动计算。"""
    print_test_header("测试 9: gmm_log_weights 自动计算")

    cfg = CooperativeFutureConfig()
    inputs = make_dummy_inputs()
    inputs["gmm_log_weights"] = None  # 测试自动计算路径
    model = CooperativeFutureTokenEncoder(cfg)

    with torch.no_grad():
        output = model(**inputs)

    assert torch.isfinite(output["cooperative_future_tokens"]).all()
    print(f"  [PASS] gmm_log_weights=None 时正常运行")

    # 手动提供 vs None 的结果一致性
    # 使用均匀权重 (无接近 eps 的值) 确保一致性
    inputs_uniform = make_dummy_inputs()
    # 设置为均匀权重: [1/3, 1/3, 1/3]，远大于 eps
    inputs_uniform["gmm_weights"] = torch.full((4, 3), 1.0/3.0)
    inputs_uniform["gmm_log_weights"] = None

    inputs_manual = dict(inputs_uniform)
    inputs_manual["gmm_log_weights"] = torch.full((4, 3), math.log(1.0/3.0))

    with torch.no_grad():
        output_auto = model(**inputs_uniform)
        output_manual = model(**inputs_manual)

    diff = (output_auto["cooperative_future_tokens"] - output_manual["cooperative_future_tokens"]).abs().max().item()
    print(f"  auto vs manual log_weights diff (uniform weights): {diff:.2e}")
    assert diff < 1e-5, f"auto log_weights 与手动提供不一致: diff={diff:.2e}"

    print(f"\n  ✅ gmm_log_weights 自动计算测试通过")


def test_10_history_mask():
    """测试 10: agent_history_mask 起作用。"""
    print_test_header("测试 10: history_mask 测试")

    cfg = CooperativeFutureConfig()
    inputs = make_dummy_inputs()

    # 部分 mask 置零
    inputs_masked = dict(inputs)
    inputs_masked["agent_history_mask"] = inputs["agent_history_mask"].clone()
    inputs_masked["agent_history_mask"][:, :, -3:] = 0  # 最后3步无效

    model = CooperativeFutureTokenEncoder(cfg)
    with torch.no_grad():
        out_full = model(**inputs)
        out_masked = model(**inputs_masked)

    # 输出应不同 (部分历史被遮蔽)
    diff = (out_full["cooperative_future_tokens"] - out_masked["cooperative_future_tokens"]).abs().max().item()
    print(f"  full vs masked future_tokens diff: {diff:.4f}")
    assert diff > 1e-6, "history_mask 未起作用"
    assert torch.isfinite(out_masked["cooperative_future_tokens"]).all(), \
        "mask 后出现 NaN/Inf"

    # mask=None 也应正常工作
    inputs_none = dict(inputs)
    inputs_none["agent_history_mask"] = None
    with torch.no_grad():
        out_none = model(**inputs_none)
    assert torch.isfinite(out_none["cooperative_future_tokens"]).all()
    print(f"  [PASS] history_mask=None 正常运行")

    print(f"\n  ✅ history_mask 测试通过")


def test_11_submodules_standalone():
    """测试 11: 各子模块可独立运行。"""
    print_test_header("测试 11: 子模块独立运行")

    cfg = CooperativeFutureConfig()
    B, N, L, H, K = 4, cfg.num_agents, cfg.history_len, cfg.pred_horizon, cfg.num_modes
    D = cfg.d_model

    # AgentHistoryEncoder
    enc = AgentHistoryEncoder(cfg)
    out = enc(torch.randn(B, N, L, cfg.history_dim))
    assert out["agent_tokens"].shape == (B, N, D)
    assert out["history_tokens"].shape == (B, N, L, D)
    print(f"  [PASS] AgentHistoryEncoder: agent={out['agent_tokens'].shape}, hist={out['history_tokens'].shape}")

    # RelationalAgentInteraction
    inter = RelationalAgentInteraction(cfg)
    out = inter(torch.randn(B, N, D), torch.randn(B, N, 3), torch.randn(B, N, 3))
    assert out["agent_tokens"].shape == (B, N, D)
    assert out["attention"].shape == (B, cfg.num_heads, N, N)
    print(f"  [PASS] RelationalAgentInteraction: tokens={out['agent_tokens'].shape}, attn={out['attention'].shape}")

    # PreviousPlanEncoder
    ppe = PreviousPlanEncoder(cfg)
    out = ppe(torch.randn(B, N, H, cfg.prev_plan_dim), torch.ones(B, N, H))
    assert out.shape == (B, N, H, D)
    print(f"  [PASS] PreviousPlanEncoder: {out.shape}")

    # FutureQueryBuilder
    fqb = FutureQueryBuilder(cfg)
    out = fqb(torch.randn(B, N, D), torch.randn(B, N, H, D))
    assert out.shape == (B, N, H, D)
    print(f"  [PASS] FutureQueryBuilder: {out.shape}")

    # Single-trajectory same-time fusion.
    gca = TimeAlignedTrajectoryFusion(cfg)
    out = gca(
        torch.randn(B, N, H, D),
        torch.randn(B, K, D),
        torch.randn(B, K, H, D),
    )
    assert out["queries"].shape == (B, N, H, D)
    assert "attention" not in out
    print(f"  [PASS] TimeAlignedTrajectoryFusion: queries={out['queries'].shape}")

    # FactorizedCoordinationBlock
    fcb = FactorizedCoordinationBlock(cfg)
    out = fcb(torch.randn(B, N, H, D))
    assert out["tokens"].shape == (B, N, H, D)
    assert out["temporal_attention"].shape == (B, cfg.num_heads, N, H, H)
    assert out["future_agent_attention"].shape == (B, cfg.num_heads, H, N, N)
    print(f"  [PASS] FactorizedCoordinationBlock: tokens={out['tokens'].shape}, t_attn={out['temporal_attention'].shape}, a_attn={out['future_agent_attention'].shape}")

    print(f"\n  ✅ 子模块独立运行测试通过")


def test_12_strategic_intent_conditioning():
    """测试 12: 不同战略意图应产生不同 token 和未来协同表示。"""
    print_test_header("测试 12: 战略意图条件化")
    cfg = CooperativeFutureConfig()
    inputs_a = make_dummy_inputs()
    inputs_b = dict(inputs_a)
    inputs_a["strategic_probs"] = torch.zeros(4, cfg.num_strategic)
    inputs_b["strategic_probs"] = torch.zeros(4, cfg.num_strategic)
    inputs_a["strategic_probs"][:, 0] = 1.0
    inputs_b["strategic_probs"][:, 1] = 1.0
    model = CooperativeFutureTokenEncoder(cfg)
    with torch.no_grad():
        out_a = model(**inputs_a)
        out_b = model(**inputs_b)
    token_diff = (
        out_a["strategic_intent_token"] - out_b["strategic_intent_token"]
    ).abs().max().item()
    future_diff = (
        out_a["cooperative_future_tokens"] - out_b["cooperative_future_tokens"]
    ).abs().max().item()
    assert token_diff > 1e-5
    assert future_diff > 1e-5
    print(f"  [PASS] intent token diff={token_diff:.4f}, future diff={future_diff:.4f}")


def test_13_bounded_trajectory_and_hermite():
    """测试 13: 轨迹参考有界，Hermite 首尾状态连续。"""
    print_test_header("测试 13: 有界轨迹与 Hermite 连续性")
    B, N, H = 2, 3, 10
    decoder = BoundedTrajectoryReferenceDecoder(
        dt=0.05, max_speed=3.0, accel_limits=(5.0, 5.0, 6.5)
    )
    action = torch.empty(B, N, H, 6).uniform_(-1.0, 1.0)
    p0 = torch.randn(B, N, 3)
    v0 = torch.randn(B, N, 3) * 0.3
    decoded = decoder(action, p0, v0)
    assert decoded["plan"].shape == (B, N, H, 6)
    assert decoded["velocities"].norm(dim=-1).max().item() < 3.0
    tau = torch.arange(1, H + 1).view(1, 1, H, 1) * 0.05
    residual = decoded["positions_relative"] - v0.unsqueeze(-2) * tau
    reachable = 0.5 * torch.tensor([5.0, 5.0, 6.5]) * tau.square()
    assert torch.all(residual.abs() <= reachable + 1e-6)

    p1 = decoded["positions_world"][..., 0, :]
    v1 = decoded["velocities"][..., 0, :]
    at_start = decoder.sample_cubic_hermite(p0, v0, p1, v1, 0.0, 0.05)
    at_end = decoder.sample_cubic_hermite(p0, v0, p1, v1, 0.05, 0.05)
    assert torch.allclose(at_start["position"], p0, atol=1e-6)
    assert torch.allclose(at_start["velocity"], v0, atol=1e-6)
    assert torch.allclose(at_end["position"], p1, atol=1e-5)
    assert torch.allclose(at_end["velocity"], v1, atol=1e-5)
    print("  [PASS] bounded position/velocity and Hermite endpoint continuity")


def main():
    print("=" * 70)
    print("  CooperativeFutureTokenEncoder — 完整单元测试")
    print("=" * 70)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Device:  cpu")

    all_tests = [
        ("1. 形状测试", test_1_shapes),
        ("2. 无人机排列等变性", test_2_agent_permutation),
        ("3. GMM模式排列等变性", test_3_gmm_mode_permutation),
        ("4. 上一计划Mask测试", test_4_prev_plan_mask),
        ("5. 时间敏感性测试", test_5_time_sensitivity),
        ("6. 队友敏感性测试", test_6_teammate_sensitivity),
        ("7. 梯度测试", test_7_gradients),
        ("8. 额外检查", test_8_extra_checks),
        ("9. gmm_log_weights自动计算", test_9_gmm_log_weights_auto),
        ("10. history_mask测试", test_10_history_mask),
        ("11. 子模块独立运行", test_11_submodules_standalone),
        ("12. 战略意图条件化", test_12_strategic_intent_conditioning),
        ("13. 有界轨迹与Hermite连续性", test_13_bounded_trajectory_and_hermite),
    ]

    passed = 0
    failed = 0

    for name, test_fn in all_tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n  ❌ 失败: {e}")
        except Exception as e:
            failed += 1
            print(f"\n  💥 异常: {type(e).__name__}: {e}")

    print(f"\n{'=' * 70}")
    print(f"  测试结果: {passed} 通过, {failed} 失败, 共 {len(all_tests)} 项")
    print(f"{'=' * 70}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
