"""Regression: terminal telemetry uses next state, including rollout end."""
import ast
from pathlib import Path
from typing import Sequence
import torch
from tensordict import TensorDict, TensorDictBase

tree=ast.parse((Path(__file__).parent/'train.py').read_text())
class_node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EpisodeStats')
module=ast.Module(body=[class_node])
namespace=dict(torch=torch,Sequence=Sequence,TensorDictBase=TensorDictBase)
exec(compile(module,'EpisodeStats','exec'),namespace)
durations=torch.tensor([[[1.],[2.],[3.]],[[4.],[5.],[6.]]])
done=torch.tensor([[[False],[True],[True]],[[True],[False],[True]]])
data=TensorDict({'stats':{'episode_duration_s':torch.zeros(2,3,1)},
                 'next':{'done':done,'stats':{'episode_duration_s':durations}}},[2,3])
stats=namespace['EpisodeStats']([('stats','episode_duration_s')])
stats(data)
assert len(stats)==4, 'Must include both terminal transitions at last rollout step'
actual=stats.pop()['stats','episode_duration_s'].reshape(-1)
assert torch.equal(actual,torch.tensor([2.,3.,4.,6.])), actual
assert len(stats)==0
print('PASS: next-state alignment, rollout-end terminal events, batch aggregation')
