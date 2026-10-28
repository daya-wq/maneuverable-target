"""A no-terminal last update must not erase earlier episode/reward fields."""
import json
from pathlib import Path
import sys
from summarize_training70 import summarize

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
first = next(r for r in rows if r.get('train/episodes_completed', 0) > 0)
last = {k: v for k, v in first.items() if not k.startswith('train/stats.')}
last['train/episodes_completed'] = 0
last['train/update'] += 1
result = summarize([first, last])
assert result['completed_episodes'] == first['train/episodes_completed']
assert result['termination_fraction'], 'last update erased earlier terminal fields'
for key, value in result['termination_fraction'].items():
    assert abs(value - first['train/stats.termination_' + key]) < 1e-6
assert result['reward_means'], 'last update erased earlier reward fields'
assert result['missing_termination_updates'] == []
print('PASS: zero-completion last update preserves weighted episode/reward summary')
