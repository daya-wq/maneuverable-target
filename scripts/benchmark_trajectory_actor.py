"""Parameter and latency benchmark for the cooperative trajectory actor."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from omni_drones.learning.mappo import make_critic, make_ppo_actor
from omni_drones.learning.modules.cooperative_future import (
    BoundedTrajectoryReferenceDecoder,
)
from omni_drones.learning.predictor.dual_intent_gmm_predictor import (
    StrategicGMMConfig,
    StrategicGMMTrajectoryPredictor,
)
from omni_drones.utils.torchrl.specs import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def make_specs(num_agents: int, history_len: int, horizon: int, num_modes: int):
    observation_spec = CompositeSpec(
        {
            "agent_history": UnboundedContinuousTensorSpec((num_agents, history_len, 9)),
            "current_pos": UnboundedContinuousTensorSpec((num_agents, 3)),
            "current_vel": UnboundedContinuousTensorSpec((num_agents, 3)),
            "prev_plan": UnboundedContinuousTensorSpec((num_agents, horizon, 6)),
            "prev_plan_mask": UnboundedContinuousTensorSpec((num_agents, horizon)),
            "gmm_means": UnboundedContinuousTensorSpec(
                (num_agents, num_modes, horizon, 3)
            ),
            "gmm_vars": UnboundedContinuousTensorSpec(
                (num_agents, num_modes, horizon, 3)
            ),
            "gmm_weights": UnboundedContinuousTensorSpec((num_agents, num_modes)),
            "strategic_probs": UnboundedContinuousTensorSpec((num_agents, 4)),
        }
    )
    action_spec = UnboundedContinuousTensorSpec((num_agents, horizon, 6))
    return observation_spec, action_spec


def make_observation(batch: int, num_agents: int, history_len: int, horizon: int, num_modes: int, device):
    weights = torch.softmax(torch.randn(batch, num_modes, device=device), dim=-1)
    return TensorDict(
        {
            "agent_history": torch.randn(batch, num_agents, history_len, 9, device=device),
            "current_pos": torch.randn(batch, num_agents, 3, device=device),
            "current_vel": torch.randn(batch, num_agents, 3, device=device),
            "prev_plan": torch.randn(batch, num_agents, horizon, 6, device=device),
            "prev_plan_mask": torch.ones(batch, num_agents, horizon, device=device),
            "gmm_means": torch.randn(batch, num_agents, num_modes, horizon, 3, device=device),
            "gmm_vars": torch.rand(batch, num_agents, num_modes, horizon, 3, device=device) + 0.01,
            "gmm_weights": weights[:, None].expand(-1, num_agents, -1),
            "strategic_probs": torch.softmax(
                torch.randn(batch, num_agents, 4, device=device), dim=-1
            ),
        },
        batch_size=[batch, num_agents],
    )


def benchmark(fn, warmup: int, repeats: int, device: torch.device):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)
    samples_ms.sort()
    p95_index = min(len(samples_ms) - 1, int(0.95 * len(samples_ms)))
    return statistics.median(samples_ms), samples_ms[p95_index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    num_agents, history_len, horizon, num_modes = 3, 10, 10, 3
    observation_spec, action_spec = make_specs(
        num_agents, history_len, horizon, num_modes
    )
    actor_cfg = OmegaConf.load("cfg/algo/mappo.yaml").actor
    critic_cfg = OmegaConf.load("cfg/algo/mappo.yaml").critic
    actor = make_ppo_actor(actor_cfg, observation_spec, action_spec).to(device).eval()
    # Matches HideAndSeek: state_self_dim=20, cooperation_dim=9+3*horizon.
    critic_state_spec = CompositeSpec(
        {
            "state_drones": UnboundedContinuousTensorSpec(
                (num_agents, 20 + 9 + 3 * horizon)
            )
        }
    )
    reward_spec = UnboundedContinuousTensorSpec((num_agents, 1))
    critic = make_critic(
        critic_cfg, critic_state_spec, reward_spec, centralized=True
    ).to(device).eval()
    predictor = StrategicGMMTrajectoryPredictor(
        StrategicGMMConfig(obs_dim=30, n_components=num_modes, pred_horizon=horizon)
    ).to(device).eval()
    decoder = BoundedTrajectoryReferenceDecoder(
        dt=0.05, max_speed=3.0, accel_limits=(5.0, 5.0, 6.5)
    ).to(device)
    observation = make_observation(
        args.batch, num_agents, history_len, horizon, num_modes, device
    )
    predictor_input = torch.randn(args.batch, history_len, 30, device=device)

    def actor_step():
        return actor(observation, deterministic=True)

    def predictor_step():
        return predictor(predictor_input)

    actor_median, actor_p95 = benchmark(actor_step, 10, args.repeats, device)
    predictor_median, predictor_p95 = benchmark(
        predictor_step, 10, args.repeats, device
    )

    with torch.inference_mode():
        action, log_prob, _ = actor_step()
        projected = decoder(
            action,
            initial_pos=torch.zeros(args.batch, num_agents, 3, device=device),
            initial_vel=torch.zeros(args.batch, num_agents, 3, device=device),
        )
    assert action.shape == (args.batch, num_agents, horizon, 6)
    assert log_prob.shape == (args.batch, num_agents, 1)
    assert projected["plan"].shape == (args.batch, num_agents, horizon, 6)
    assert projected["velocities"].norm(dim=-1).max() <= 3.0 + 1e-5

    actor.train()
    _, train_log_prob, _ = actor(observation, deterministic=False)
    (-train_log_prob.mean()).backward()
    trainable_grads = [
        parameter.grad
        for parameter in actor.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_grads and all(torch.isfinite(grad).all() for grad in trainable_grads)
    actor.zero_grad(set_to_none=True)
    actor.eval()

    print(f"device={device} batch={args.batch} threads={args.threads}")
    print(f"predictor_params={count_parameters(predictor):,}")
    print(f"actor_params={count_parameters(actor):,}")
    print(f"critic_params={count_parameters(critic):,}")
    print(f"total_policy_params={count_parameters(predictor) + count_parameters(actor):,}")
    print(
        "total_training_params="
        f"{count_parameters(predictor) + count_parameters(actor) + count_parameters(critic):,}"
    )
    print(f"predictor_latency_ms median={predictor_median:.3f} p95={predictor_p95:.3f}")
    print(f"actor_latency_ms median={actor_median:.3f} p95={actor_p95:.3f}")
    print(f"estimated_total_median_ms={predictor_median + actor_median:.3f}")
    print("actor_backward=OK")
    print("trajectory_projection=OK")


if __name__ == "__main__":
    main()
