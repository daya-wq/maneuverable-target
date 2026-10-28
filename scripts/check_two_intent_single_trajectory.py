"""Pure CPU/CUDA checks of the two-intent, single-trajectory contract."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Tuple
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("INTENT_TEST_DEVICE", "cpu")


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


predictor = load("dual_intent_gmm_predictor", "omni_drones/learning/predictor/dual_intent_gmm_predictor.py")
pipeline = load("data_pipeline", "omni_drones/learning/predictor/data_pipeline.py")
coop = load("single_cooperative", "omni_drones/learning/modules/cooperative_future.py")
geometry = load("single_geometry", "omni_drones/learning/predictor/gmm_encoder.py")


def environment_methods():
    tree = ast.parse((ROOT / "omni_drones/envs/hide_and_seek/hideandseek.py").read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "point_to_cylinder_distance"]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HideAndSeek")
    functions += [n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name in ("_compute_rule_strategic_probs", "_kinematic_gmm")]
    namespace = dict(torch=torch, Tuple=Tuple)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 "production_environment_methods", "exec"), namespace)
    return namespace


class IntentTests(unittest.TestCase):
    def test_model_outputs_and_removed_weight_head(self):
        cfg = predictor.StrategicGMMConfig(d_model=32, dropout=0.0)
        model = predictor.StrategicGMMTrajectoryPredictor(cfg).to(DEVICE)
        result = model(torch.randn(6, 10, 30, device=DEVICE))
        self.assertEqual(predictor.STRATEGIC_LABELS, ["Goal_Rush", "Evade"])
        self.assertEqual(tuple(result["strategic_probs"].shape), (6, 2))
        self.assertEqual(tuple(result["trajectory_mean"].shape), (6, 10, 3))
        self.assertEqual(tuple(result["gmm_means"].shape), (6, 1, 10, 3))
        self.assertTrue((result["trajectory_var"] > 0).all())
        self.assertTrue(torch.equal(result["gmm_weights"], torch.ones(6, 1, device=DEVICE)))
        self.assertFalse(any("weight_head" in name for name, _ in model.named_parameters()))
        torch.testing.assert_close(result["strategic_probs"].sum(-1), torch.ones(6, device=DEVICE))

    def test_prediction_loss_backprop(self):
        model = predictor.StrategicGMMTrajectoryPredictor(
            predictor.StrategicGMMConfig(d_model=32, dropout=0)).to(DEVICE)
        result = model(torch.randn(4, 10, 30, device=DEVICE))
        loss, metrics = predictor.StrategicGMMLoss()(
            result, torch.randn(4, 10, 3, device=DEVICE), torch.tensor([0, 1, 0, 1], device=DEVICE))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("ADE", metrics)
        self.assertIn("loss_trajectory_nll", metrics)
        self.assertNotIn("gmm_weight_entropy", metrics)
        loss.backward()
        self.assertGreater(float(model.strategic_head[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_old_architecture_rejected(self):
        for kwargs in (dict(n_components=3), dict(n_strategic=4)):
            with self.assertRaises(ValueError):
                predictor.StrategicGMMConfig(**kwargs)

    def test_online_offline_labels_match(self):
        methods = environment_methods()
        target = torch.tensor([[-30., 0, 12], [50., 0, 12], [58., 0, 12], [-30., 0, 12]], device=DEVICE)
        distances = torch.tensor([25., 15., 1., 26.], device=DEVICE)
        drones = target[:, None].repeat(1, 3, 1)
        drones[:, :, 1] += distances[:, None]
        env = SimpleNamespace(goal_region_center=torch.tensor([60., 0, 12], device=DEVICE),
                              goal_region_radius=3., goal_region_height=6.,
                              target_goal_sprint_distance=12., target_tactical_far_distance=25.,
                              target_sine_policy=SimpleNamespace(trigger_distance=25.))
        for mode, expected in (("adaptive", [1, 0, 0, 0]), ("goal_sine", [1, 1, 1, 0])):
            env.target_motion_mode = mode
            labels = methods["_compute_rule_strategic_probs"](env, drones, target).argmax(-1)
            self.assertEqual(labels.tolist(), expected)
            offline = pipeline.label_strategic_batch(
                target, torch.zeros_like(target), drones, env.goal_region_center,
                close_evade_dist=25., goal_sprint_dist=12., motion_mode=mode,
                goal_radius=3., goal_height=6.)
            self.assertEqual(offline.tolist(), expected)

    def test_fallback_is_single_constant_velocity(self):
        env = SimpleNamespace(future_prediction_obs_steps=10, prediction_dt=.05, device=DEVICE)
        target = torch.zeros(4, 3, device=DEVICE)
        velocity = torch.randn(4, 3, device=DEVICE)
        weights, means, variances = environment_methods()["_kinematic_gmm"](
            env, torch.zeros(4, 3, 3, device=DEVICE), target, velocity)
        self.assertEqual(tuple(means.shape), (4, 1, 10, 3))
        torch.testing.assert_close(means[:, 0, -1], velocity * .5)
        self.assertTrue((weights == 1).all() and (variances > 0).all())

    def test_geometry_has_no_weight_parameters(self):
        encoder = geometry.GMMTrajectoryEncoder(geometry.GMMEncoderConfig(d_model=32)).to(DEVICE)
        self.assertFalse(any("weight_encoder" in name or "weight_gate" in name
                             for name, _ in encoder.named_parameters()))
        out = encoder(torch.randn(2, 1, 10, 3, device=DEVICE),
                      torch.ones(2, 1, 10, 3, device=DEVICE), torch.ones(2, 1, device=DEVICE))
        self.assertEqual(tuple(out["point_tokens"].shape), (2, 1, 10, 32))

    def test_fusion_same_time_and_gradients(self):
        fusion = coop.TimeAlignedTrajectoryFusion(coop.CooperativeFutureConfig(d_model=32)).to(DEVICE)
        query = torch.randn(2, 3, 10, 32, device=DEVICE, requires_grad=True)
        trajectory = torch.randn(2, 1, 32, device=DEVICE)
        points = torch.randn(2, 1, 10, 32, device=DEVICE, requires_grad=True)
        first = fusion(query, trajectory, points)["queries"]
        changed = points.detach().clone()
        changed[:, :, 4, 0] += 5
        second = fusion(query, trajectory, changed)["queries"]
        unaffected = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8, 9], device=DEVICE)
        torch.testing.assert_close(first[:, :, unaffected], second[:, :, unaffected])
        self.assertGreater(float((first[:, :, 4] - second[:, :, 4]).abs().max()), 1e-5)
        first[..., 0].mean().backward()
        self.assertGreater(float(points.grad.abs().sum()), 0)
        self.assertGreater(float(query.grad.abs().sum()), 0)


if __name__ == "__main__":
    print("Two intents, single trajectory:", DEVICE, flush=True)
    unittest.main(verbosity=2)
