"""CPU regression of production direct-PV decoding and soft consistency."""
import ast
from pathlib import Path
from typing import Dict
import torch
from torch import nn

path = Path(__file__).resolve().parents[1]/'omni_drones/learning/modules/cooperative_future.py'
tree = ast.parse(path.read_text())
node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BoundedTrajectoryReferenceDecoder')
ns = dict(torch=torch,nn=nn,Dict=Dict)
exec(compile(ast.Module(body=[node]),str(path),'exec'),ns)
Decoder = ns['BoundedTrajectoryReferenceDecoder']
torch.manual_seed(12)
decoder = Decoder(.05,17.,reference_mode='bounded_position_velocity')
raw = torch.randn(1024,3,10,6,requires_grad=True)
action = raw.tanh()
p = torch.randn(1024,3,3)
v = torch.randn_like(p)
out = decoder(action,p,v)
previous = torch.cat([torch.zeros_like(out['positions_relative'][...,:1,:]),out['positions_relative'][...,:-1,:]],-2)
assert (out['positions_relative']-previous).norm(dim=-1).max() <= .85+1e-5
assert out['velocities'].norm(dim=-1).max() <= 17.+1e-5
assert out['positions_relative'].norm(dim=-1).max() <= 8.5+1e-5
loss=Decoder.pv_consistency_loss(action)
loss.backward()
assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0
good=torch.full((1,10,6),.1)
bad=good.clone();bad[...,3:]=-.1
assert Decoder.pv_consistency_loss(good)<1e-10
assert Decoder.pv_consistency_loss(bad)>.01
shifted=decoder(action,p+100.,v)
assert torch.allclose(out['positions_relative'],shifted['positions_relative'])
print('PASS: adjacent 0.85m / 17mps vector bounds; relative anchors; finite soft-loss gradients; contradictory P/V penalized')
