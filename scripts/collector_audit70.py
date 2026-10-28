"""Audit the real training collector without changing checkpoint parameters."""
import json
from pathlib import Path
import torch


@torch.no_grad()
def audit_collector(collector, env, base_env, policy, output_dir, batches=16):
    env.train()
    base_env.train()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    n = base_env.num_envs
    finished = torch.zeros(n, dtype=torch.bool, device=base_env.device)
    reasons = torch.zeros(n, dtype=torch.long, device=base_env.device)
    lengths = torch.zeros_like(reasons)
    records = []
    before = [p.detach().clone() for p in policy.actor.parameters()]
    for index, data in enumerate(collector):
        data = data.to_tensordict()
        action = data['agents', 'action'][..., 0, :].detach()
        state = data['next', 'info', 'drone_state'].detach()
        done = data['next', 'done'].squeeze(-1).bool()
        reason = data['next', 'stats', 'termination_reason'].squeeze(-1).long()
        direct = policy(data[:, 0].clone(), deterministic=True)['agents', 'action'][..., 0, :]
        record = {
            'batch': index + 1,
            'sampled_action_mean': action.mean(dim=(0, 1, 2)).cpu().tolist(),
            'sampled_action_std': action.std(dim=(0, 1, 2)).cpu().tolist(),
            'direct_deterministic_action_mean': direct.mean(dim=(0, 1)).cpu().tolist(),
            'sampled_action_abs_max': float(action.abs().max()),
            'height_min': float(state[..., 2].min()),
            'height_max': float(state[..., 2].max()),
            'height_mean': float(state[..., 2].mean()),
            'done_counts': {str(i): int(((reason == i) & done).sum()) for i in range(9)},
        }
        records.append(record)
        for step in range(done.shape[1]):
            alive = ~finished
            lengths[alive] += 1
            newly = alive & done[:, step]
            reasons[newly] = reason[newly, step]
            finished |= newly
        if index + 1 >= batches:
            break
    unchanged = all(torch.equal(old, new.detach()) for old, new in zip(before, policy.actor.parameters()))
    result = {
        'weights_updated': False, 'actor_parameters_unchanged': unchanged,
        'collector_policy_type': str(type(collector.policy)),
        'collector_policy_repr': repr(collector.policy)[:2000],
        'batches': len(records), 'first_episodes_finished': int(finished.sum()),
        'first_episode_counts': {str(i): int(((reasons == i) & finished).sum()) for i in range(9)},
        'first_episode_mean_duration_s': float(lengths[finished].float().mean()) * base_env.trajectory_plan_dt,
        'records': records,
    }
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    return {k: v for k, v in result.items() if k != 'records'}
