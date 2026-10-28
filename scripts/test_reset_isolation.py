"""CPU structural regression against the production reset helper."""
import ast
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).resolve().parents[1] / 'omni_drones/envs/hide_and_seek/hideandseek.py'
tree = ast.parse(path.read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'HideAndSeek')
helper = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_advance_physics_after_reset')
reset = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_reset_idx')
assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == 'step' for node in ast.walk(reset)), 'reset must not directly advance physics'
ns = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), ns)
calls = []
fake = SimpleNamespace(advance_physics_on_reset=False,
                       sim=SimpleNamespace(step=lambda render: calls.append(render)),
                       _should_render=lambda substep: False)
ns['_advance_physics_after_reset'](fake)
assert calls == [], 'isolated reset advanced the world'
fake.advance_physics_on_reset = True
ns['_advance_physics_after_reset'](fake)
assert calls == [False], 'legacy reproduction changed'
print('PASS: isolated reset has zero global physics steps; legacy switch reproduces one step')
