"""Production history helper: one row per planner tick, not per observation read."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch

path = Path(__file__).resolve().parents[1] / 'omni_drones/envs/hide_and_seek/hideandseek.py'
tree = ast.parse(path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HideAndSeek')
helper = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_record_agent_history')
module = ast.parse('')
module.body = [helper]
ns = dict(torch=torch)
exec(compile(module, str(path), 'exec'), ns)
update = ns['_record_agent_history']
fake = SimpleNamespace(
    advance_physics_on_reset=False, history_step=10,
    agent_history_buffer=torch.zeros(2, 3, 10, 9),
    agent_history_initialized=torch.zeros(2, dtype=torch.bool),
    agent_history_last_step=torch.full((2,), -1.), progress_buf=torch.zeros(2),
)
update(fake, torch.zeros(2, 3, 9))
fake.progress_buf += 1
update(fake, torch.ones(2, 3, 9))
before = fake.agent_history_buffer.clone()
update(fake, torch.full((2, 3, 9), 9.))
assert torch.equal(before, fake.agent_history_buffer), 'observation reread duplicated history'
fake.agent_history_initialized[0] = False
fake.progress_buf[0] = 0
update(fake, torch.full((2, 3, 9), -99.))
assert torch.equal(fake.agent_history_buffer[1], before[1]), 'partial reset changed live history'
assert (fake.agent_history_buffer[0] == -99.).all()
fake.progress_buf[1] += 1
update(fake, torch.full((2, 3, 9), 2.))
assert (fake.agent_history_buffer[1, :, -1] == 2.).all()
assert (fake.agent_history_buffer[1, :, -2] == 1.).all()
fake.advance_physics_on_reset = True
update(fake, torch.full((2, 3, 9), 3.))
assert (fake.agent_history_buffer[:, :, -1] == 3.).all(), 'legacy reproduction changed'
print('PASS: tick-aligned history, idempotent observation, selective reset and legacy mode')
