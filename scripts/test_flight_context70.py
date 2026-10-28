"""Test the production adapter's optional six-value residual path on CPU."""
import ast
from pathlib import Path
import torch
from torch import nn

path = Path(__file__).resolve().parents[1] / 'omni_drones/learning/modules/networks.py'
tree = ast.parse(path.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CooperativeTrajectoryEncoder')
node.decorator_list = []
node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef) or n.name != '__init__']
module = ast.parse('')
module.body = [node]
ns = dict(torch=torch, nn=nn, Tensor=torch.Tensor, TensorDict=dict)
exec(compile(module, str(path), 'exec'), ns)

class GMM(nn.Module):
    def forward(self, means, vars_, weights):
        return dict(mode_tokens=torch.zeros(2, 3, 8), point_tokens=torch.zeros(2, 3, 10, 8),
                    weights=weights, log_weights=weights.log())

class Cooperative(nn.Module):
    def forward(self, **kwargs):
        return dict(cooperative_future_tokens=torch.ones(2, 3, 10, 8))

encoder = ns['CooperativeTrajectoryEncoder']()
encoder.num_agents, encoder.pred_horizon, encoder.prev_plan_dim = 3, 10, 6
encoder.num_modes, encoder.num_strategic, encoder.d_model = 3, 4, 8
encoder.gmm_encoder, encoder.cooperative_encoder = GMM(), Cooperative()
encoder.flight_context_encoder = None
x = dict(agent_history=torch.zeros(2, 3, 10, 9), current_pos=torch.zeros(2, 3, 3),
         current_vel=torch.zeros(2, 3, 3), prev_plan=torch.zeros(2, 3, 10, 6),
         prev_plan_mask=torch.ones(2, 3, 10), gmm_means=torch.zeros(2, 3, 3, 10, 3),
         gmm_vars=torch.ones(2, 3, 3, 10, 3), gmm_weights=torch.full((2, 3, 3), 1/3),
         strategic_probs=torch.full((2, 3, 4), .25), flight_context=torch.zeros(2, 3, 6))
old = encoder(x)
x['flight_context'][0, 0, 2] = .25
encoder.flight_context_encoder = nn.Linear(6, 8)
nn.init.zeros_(encoder.flight_context_encoder.weight)
nn.init.zeros_(encoder.flight_context_encoder.bias)
assert torch.equal(old, encoder(x)), 'zero-initialized context changed old output'
encoder(x).sum().backward()
assert torch.isfinite(encoder.flight_context_encoder.weight.grad).all()
assert encoder.flight_context_encoder.weight.grad.abs().sum() > 0
with torch.no_grad():
    encoder.flight_context_encoder.weight[0, 2] = 1.
    x['flight_context'][0, 0, 2] = .5
new = encoder(x)
assert new.shape == (2, 3, 10, 8)
assert (new[0, 0, :, 0] == 1.5).all()
assert torch.equal(new[1], old[1]), 'context leaked across independent envs'
assert torch.equal(new[0, 1:], old[0, 1:]), 'context leaked across agents'
print('PASS: production residual path, exact zero-init compatibility, height sensitivity, finite gradient and isolation')
env_path = path.parents[2] / 'envs/hide_and_seek/hideandseek.py'
env_tree = ast.parse(env_path.read_text())
assignment = next(n for n in ast.walk(env_tree) if isinstance(n, ast.Assign)
                  and isinstance(n.value, ast.Call)
                  and any(isinstance(t, ast.Subscript) and 'flight_context' in ast.dump(t)
                          for t in n.targets))
assert any(k.arg == 'device' and isinstance(k.value, ast.Attribute) and k.value.attr == 'device'
           for k in assignment.value.keywords), 'new spec must explicitly match CompositeSpec device'
print('PASS: optional context TensorSpec explicitly uses environment device')
