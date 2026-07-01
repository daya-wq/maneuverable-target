# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys

import torch
from tensordict import TensorDict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "cfg")


def _parse_cuda_ordinal(device):
    if device is None:
        return None
    device = str(device)
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        ordinal = device.split(":", 1)[1]
        if ordinal.isdigit():
            return int(ordinal)
    return None


def _get_or_default(cfg, key, default=None):
    value = cfg.get(key, default)
    if value is None:
        return default
    return value


def _patch_entry_points_api(module):
    entry_points_cls = getattr(module, "EntryPoints", None)
    if entry_points_cls is None or hasattr(entry_points_cls, "get"):
        return

    def _entry_points_get(self, group, default=None):
        selected = list(self.select(group=group))
        if selected:
            return selected
        return [] if default is None else default

    entry_points_cls.get = _entry_points_get


def _ensure_isaac_python_compat():
    modules = []
    try:
        import importlib.metadata as importlib_metadata
        modules.append(importlib_metadata)
    except ImportError:
        importlib_metadata = None
    try:
        import importlib_metadata as importlib_metadata_backport
        modules.append(importlib_metadata_backport)
    except ImportError:
        importlib_metadata_backport = None

    for module in modules:
        _patch_entry_points_api(module)


def init_simulation_app(cfg):
    isaac_path = os.environ.get("ISAAC_PATH")
    if isaac_path:
        isaac_site_packages = os.path.join(
            isaac_path, "kit", "python", "lib", "python3.7", "site-packages"
        )
        if os.path.isdir(isaac_site_packages) and isaac_site_packages not in sys.path:
            # Isaac Sim's bundled Python packages include gym and other runtime deps
            # that may be missing from the conda env used to launch training.
            sys.path.insert(0, isaac_site_packages)
    _ensure_isaac_python_compat()
    from omni.isaac.kit import SimulationApp

    sim_cfg = cfg.get("sim", {})
    inferred_gpu = _parse_cuda_ordinal(sim_cfg.get("device"))
    # Isaac Sim routes torch/physics through /physics/cudaDevice. If we only pass
    # cfg.sim.device to SimulationContext, Isaac Sim 2022.2.0 will still fall back
    # to GPU 0 unless the launcher config is updated here as well.
    config = {
        "headless": cfg["headless"],
        "anti_aliasing": 0,
        "multi_gpu": _get_or_default(sim_cfg, "multi_gpu", False),
        "active_gpu": _get_or_default(sim_cfg, "active_gpu", inferred_gpu),
        "physics_gpu": _get_or_default(sim_cfg, "physics_gpu", inferred_gpu),
    }
    # Use the lighter gym headless experience for offscreen evaluation/recording.
    app_experience_override = cfg.get("app_experience", None)
    if app_experience_override is not None:
        app_experience = str(app_experience_override)
    elif cfg.headless:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    else:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    simulation_app = SimulationApp(config, experience=app_experience)
    # simulation_app = SimulationApp(config)
    return simulation_app


def _get_shapes(self: TensorDict):
    return {
        k: v.shape if isinstance(v, torch.Tensor) else v.shapes for k, v in self.items()
    }


def _get_devices(self: TensorDict):
    return {
        k: v.device if isinstance(v, torch.Tensor) else v.devices
        for k, v in self.items()
    }


TensorDict.shapes = property(_get_shapes)
TensorDict.devices = property(_get_devices)
