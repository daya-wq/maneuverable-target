"""A resumed optimizer keeps Adam history while respecting a changed config LR."""
import ast
import math
from pathlib import Path
import torch

path = Path(__file__).resolve().parents[1] / 'omni_drones/learning/mappo.py'
tree = ast.parse(path.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_restore_configured_optimizer_lr')
module = ast.parse('')
module.body = [node]
ns = dict(math=math)
exec(compile(module, str(path), 'exec'), ns)
restore_lr = ns[node.name]
p = torch.nn.Parameter(torch.tensor([2.]))
old = torch.optim.Adam([p], lr=3e-6)
(p.square().sum()).backward()
old.step()
saved = old.state_dict()
q = torch.nn.Parameter(torch.tensor([2.]))
new = torch.optim.Adam([q], lr=1.5e-6)
new.load_state_dict(saved)
assert new.param_groups[0]['lr'] == 3e-6, 'did not reproduce checkpoint override'
moment = new.state[q]['exp_avg'].clone()
step = new.state[q]['step'].clone()
restore_lr(new, 1.5e-6)
assert new.param_groups[0]['lr'] == 1.5e-6
assert torch.equal(moment, new.state[q]['exp_avg'])
assert torch.equal(step, new.state[q]['step'])
for invalid in [float('nan'), -1.]:
    try:
        restore_lr(new, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid learning rate accepted')
print('PASS: old LR override reproduced; configured half-LR restored without losing Adam moments/step')
