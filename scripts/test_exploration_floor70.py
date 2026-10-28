"""Exercise the production distribution's loaded-parameter scale floor on CPU."""
import ast
import functools
import math
from numbers import Number
from pathlib import Path
from typing import Union, Tuple

import torch
from torch import nn

source = Path('omni_drones/learning/modules/distributions.py').read_text()
node = next(node for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef) and node.name == 'TanhIndependentNormalModule')
# Capture actual loc/scale supplied by production forward, without needing
# Isaac or a version-specific TorchRL distribution import for this unit test.
namespace = dict(torch=torch, nn=nn, functools=functools, Number=Number,
                 Union=Union, Tuple=Tuple, _mappings={'exp': torch.exp},
                 TanhNormalWithEntropy=lambda loc, scale, **kwargs: (loc, scale))
exec(compile(ast.Module(body=[node]), '<production-distribution>', 'exec'), namespace)
distribution = namespace['TanhIndependentNormalModule']
baseline = distribution(8, 3, event_dims=2, log_std_init=-3, log_std_min=-3)
candidate = distribution(8, 3, event_dims=2, log_std_init=-3, log_std_min=-2)
candidate.load_state_dict(baseline.state_dict())
features = torch.randn(4, 3, 10, 8)
base_loc, base_scale = baseline(features)
test_loc, test_scale = candidate(features)
assert torch.equal(base_loc, test_loc), 'Noise floor must not change action means'
assert torch.equal(candidate.log_std, baseline.log_std), 'Loaded raw weights changed'
assert torch.allclose(base_scale, torch.full((3,), math.exp(-3)))
assert torch.allclose(test_scale, torch.full((3,), math.exp(-2)))
assert torch.isfinite(test_scale).all()
assert baseline.dist_cls.keywords['event_dims'] == candidate.dist_cls.keywords['event_dims'] == 2
# Regression: a different constructor init alone is overwritten by checkpoint load.
init_only = distribution(8, 3, event_dims=2, log_std_init=-2, log_std_min=-3)
init_only.load_state_dict(baseline.state_dict())
assert torch.equal(init_only(features)[1], base_scale)
print('PASS: loaded weights/means unchanged; active scale exp(-3) -> exp(-2); init alone ineffective')
