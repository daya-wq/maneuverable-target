from torchrl.data import TensorSpec

try:
    from torchrl.data import BoundedTensorSpec
except ImportError:
    from torchrl.data import Bounded as BoundedTensorSpec

try:
    from torchrl.data import UnboundedContinuousTensorSpec
except ImportError:
    from torchrl.data import Unbounded as UnboundedContinuousTensorSpec

try:
    from torchrl.data import CompositeSpec
except ImportError:
    from torchrl.data import Composite as CompositeSpec

try:
    from torchrl.data import DiscreteTensorSpec
except ImportError:
    from torchrl.data import Categorical as DiscreteTensorSpec

try:
    from torchrl.data import MultiDiscreteTensorSpec
except ImportError:
    from torchrl.data import MultiCategorical as MultiDiscreteTensorSpec
