"""Reward grouping and conservation must not confuse duration with capture time."""
import torch
from overnight_evaluation import summarize_reward_outcomes

reasons = torch.tensor([2, 7, 7, 0])
finished = torch.tensor([True, True, True, False])
lengths = torch.tensor([100, 80, 60, 400])
returns = torch.tensor([-10., -20., -30., -999.])
discounted = torch.tensor([-9., -18., -27., -999.])
components = dict(dense_raw=torch.tensor([-5., -5., -10., 0.]),
                  terminal=torch.tensor([-5., -15., -20., 0.]),
                  progress=torch.tensor([-1., -2., -3., 0.]))
grouped, error = summarize_reward_outcomes(reasons, finished, lengths, .05, returns, discounted, components)
assert 'capture' not in grouped and 'ongoing' not in grouped
assert grouped['target_goal']['episodes'] == 1
assert grouped['out_of_bounds']['episodes'] == 2
assert grouped['out_of_bounds']['mean_return'] == -25.
assert grouped['out_of_bounds']['mean_discounted_return'] == -22.5
assert grouped['out_of_bounds']['mean_duration_s'] == 3.5
assert grouped['out_of_bounds']['mean_progress_sum'] == -2.5
assert error == 0.
components['terminal'][1] += 1.
assert summarize_reward_outcomes(reasons, finished, lengths, .05, returns, discounted, components)[1] == 1.
print('PASS: per-reason returns, completed-only means, discounted units and reward conservation')
