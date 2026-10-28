"""Read completed fixed-seed first-episode comparisons; never mutate models."""
import json
import math
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
folder = root / 'results/overnight70_20261021/attempt_06/flight_context_ab'
baseline = np.load(str(folder / 'baseline/trajectories.npz'))
candidate = np.load(str(folder / 'context_retry1/trajectories.npz'))
base_reason, ctx_reason = baseline['reasons'], candidate['reasons']
assert base_reason.shape == ctx_reason.shape == (128,)
improved = int(((base_reason == 7) & (ctx_reason != 7)).sum())
worsened = int(((base_reason != 7) & (ctx_reason == 7)).sum())
discordant = improved + worsened
if discordant:
    tail = sum(math.factorial(discordant) // (math.factorial(k) * math.factorial(discordant-k))
               for k in range(min(improved, worsened)+1))
    exact_p = min(1., 2. * tail / 2**discordant)
else:
    exact_p = 1.
result = dict(
    episodes=128, pairing='same evaluation seed and environment index; not a generalization test',
    baseline_oob=int((base_reason == 7).sum()), context_oob=int((ctx_reason == 7).sum()),
    baseline_capture=int((base_reason == 1).sum()), context_capture=int((ctx_reason == 1).sum()),
    oob_to_non_oob=improved, non_oob_to_oob=worsened,
    exact_two_sided_discordant_binomial_p=exact_p,
    conclusion='Do not promote on this small, zero-capture comparison alone',
)
(folder / 'paired_comparison.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
