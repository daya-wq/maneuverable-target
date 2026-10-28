"""CPU regression: legacy counterexample and consistent bounded references."""
import ast
from pathlib import Path
from typing import Dict
import torch
from torch import nn

source = Path(__file__).resolve().parents[1]/'omni_drones/learning/modules/cooperative_future.py'
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BoundedTrajectoryReferenceDecoder')
namespace = dict(torch=torch, nn=nn, Dict=Dict)
exec(compile(ast.Module(body=[node]), str(source), 'exec'), namespace)
Decoder = namespace['BoundedTrajectoryReferenceDecoder']
legacy = Decoder(.05, 17., (12., 12., 6.))
command = torch.tensor([[[1., 0., 0., -1., 0., 0.]]])
zero = torch.zeros(1, 3)
plan = legacy(command, zero, zero)
bad = legacy.sample_cubic_hermite(zero, zero, plan['positions_world'][:, 0], plan['velocities'][:, 0], .05, .05)
assert abs(float(bad['acceleration'][0, 0]) + 84.) < 1e-3

torch.manual_seed(7)
decoder = Decoder(.05, 17., (12., 12., 6.), reference_mode='acceleration_sequence')
raw = torch.randn(64, 3, 10, 6, requires_grad=True)
action = raw.tanh()
p0 = torch.randn(64, 3, 3) * 30.
v0 = torch.randn_like(p0)
v0 = v0 / v0.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 16.9
plan = decoder(action, p0, v0)
assert plan['plan'].shape == (64, 3, 10, 6)
assert plan['velocities'].norm(dim=-1).max() <= 17. + 1e-5
assert (plan['accelerations'] / decoder.accel_limits).norm(dim=-1).max() <= 1. + 1e-5
previous_v, previous_p = v0, p0
for t in range(10):
    end_v = plan['velocities'][..., t, :]
    end_p = plan['positions_world'][..., t, :]
    expected_p = previous_p + .5 * .05 * (previous_v + end_v)
    assert torch.allclose(end_p, expected_p, atol=2e-5)
    for elapsed in [.0, .01, .02, .03, .04, .05]:
        sample = decoder.sample_constant_acceleration(previous_p, previous_v, end_v, elapsed, .05)
        assert sample['velocity'].norm(dim=-1).max() <= 17. + 1e-5
        assert (sample['acceleration']/decoder.accel_limits).norm(dim=-1).max() <= 1. + 1e-5
    assert torch.allclose(sample['position'], end_p, atol=2e-5)
    previous_v, previous_p = end_v, end_p
shift = torch.tensor([100., -200., 30.])
shifted = decoder(action, p0+shift, v0)
assert torch.allclose(plan['positions_relative'], shifted['positions_relative'])
assert torch.allclose(plan['velocities'], shifted['velocities'])
loss = plan['positions_relative'].square().mean() + plan['velocities'].square().mean()
loss.backward()
assert raw.grad is not None and torch.isfinite(raw.grad).all()
print('PASS: legacy 84 m/s2 counterexample; V3 p/v consistency, speed/acceleration bounds at all substeps, translation invariance, finite gradients')
native = torch.randn(64, 3, 10, 3, requires_grad=True)
native_plan = decoder(native.tanh(), p0, v0)
assert native_plan['plan'].shape == (64, 3, 10, 6)
assert native_plan['velocities'].norm(dim=-1).max() <= 17. + 1e-5
assert (native_plan['accelerations']/decoder.accel_limits).norm(dim=-1).max() <= 1. + 1e-5
native_plan['plan'].square().mean().backward()
assert native.grad is not None and torch.isfinite(native.grad).all()
print('PASS: native three-control head produces six-channel position/velocity plans without redundant action coordinates')
