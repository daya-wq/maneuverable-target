"""CPU regression using the production projection and quaternion utilities."""
import ast
import importlib.util
import math
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('quaternion_utils', root/'omni_drones/utils/torch.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
tree = ast.parse((root/'omni_drones/envs/hide_and_seek/flight_capacity.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'body_axis_thrust_ratio')
namespace = dict(quat_axis=utils.quat_axis)
exec(compile(ast.Module(body=[node]), 'flight_capacity', 'exec'), namespace)
projection = namespace['body_axis_thrust_ratio']
hover = .31639412
q = torch.tensor([1., 0., 0., 0.]).expand(4, 3, 4)
force = torch.tensor([12., 0., 9.81]).expand(4, 3, 3)
assert torch.allclose(projection(force, q, hover), torch.full((4, 3, 1), hover))
# If the aircraft is aligned with the desired force, norm and projection agree.
angle = math.pi/3
tilted = torch.tensor([math.cos(angle/2), 0., math.sin(angle/2), 0.]).expand(4, 3, 4)
aligned_force = utils.quat_axis(tilted, 2) * 18.
assert torch.allclose(projection(aligned_force, tilted, hover), torch.full((4, 3, 1), hover*18/9.81), atol=1e-6)
assert torch.equal(projection(-aligned_force, tilted, hover), torch.zeros(4, 3, 1))
assert projection(force, q, hover).shape == (4, 3, 1)
print('PASS: no lateral-force vertical boost, aligned SO3 thrust, nonnegative thrust, batched axes')
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'floor_braking_prediction')
exec(compile(ast.Module(body=[node]), 'flight_capacity', 'exec'), namespace)
brake = namespace['floor_braking_prediction']
z = torch.tensor([[6.], [6.], [6.]])
vz = torch.tensor([[0.], [-5.], [2.]])
predicted = brake(z, vz, 5.5, .3)
assert torch.allclose(predicted[0], torch.tensor([6.]))
assert torch.allclose(predicted[1], torch.tensor([6.-1.5-25./(2.*5.5*.65)]))
assert torch.allclose(predicted[2], torch.tensor([6.6]))
print('PASS: floor stopping distance, reaction delay, rising/stationary branches')
