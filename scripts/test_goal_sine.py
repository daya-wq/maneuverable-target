"""CPU/CUDA policy and real-environment method tests without loading Isaac Sim.

Run: python scripts/test_goal_sine.py
     SINE_TEST_DEVICE=cuda:1 python scripts/test_goal_sine.py
"""
import ast
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "goal_sine", ROOT / "omni_drones/envs/hide_and_seek/goal_sine.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GoalSinePolicy = module.GoalSinePolicy
DEVICE = os.environ.get("SINE_TEST_DEVICE", "cpu")


def policy(n=4, **overrides):
    cfg = dict(trigger_distance=10.0, amplitude_range=[1.0, 2.0],
               period_range=[3.0, 5.0], lateral_position_gain=1.5)
    cfg.update(overrides)
    result = GoalSinePolicy(n, DEVICE, cfg)
    result.reset(torch.arange(n, device=DEVICE))
    return result


def inputs(n=4):
    pos = torch.zeros(n, 3, device=DEVICE)
    goal = torch.tensor([100.0, 0.0, 10.0], device=DEVICE).expand(n, -1).clone()
    distance = torch.full((n,), 20.0, device=DEVICE)
    speed = torch.full((n,), 5.0, device=DEVICE)
    return pos, goal, distance, speed


def load_environment_methods():
    """Compile actual method bodies, not a reimplementation of dispatch/labels."""
    source = ast.parse((ROOT / "omni_drones/envs/hide_and_seek/hideandseek.py").read_text())
    names = {"clip_vector_norm", "closest_point_on_cylinder", "point_to_cylinder_distance"}
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "HideAndSeek")
    methods = {"_compute_target_desired_acc", "_compute_goal_sine_desired_acc", "_compute_rule_strategic_probs"}
    nodes.append(ast.ClassDef(name="Harness", bases=[], keywords=[], decorator_list=[],
                             body=[node for node in cls.body
                                   if isinstance(node, ast.FunctionDef) and node.name in methods]))
    namespace = dict(torch=torch, TARGET_INTENT_EVADE=1, TARGET_INTENT_GOAL_RUSH=0,
                     TARGET_MANEUVER_SINE=9, TARGET_MANEUVER_STRAIGHT=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "environment_methods", "exec"), namespace)
    return namespace["Harness"]


class SineTests(unittest.TestCase):
    def test_far_points_to_goal(self):
        p = policy()
        pos, goal, dist, speed = inputs()
        got = p.desired_velocity(pos, goal, dist, speed, .01)
        expected = (goal-pos) / (goal-pos).norm(dim=-1, keepdim=True) * speed[:, None]
        torch.testing.assert_close(got, expected)
        self.assertFalse(p.active.any())

    def test_nearest_threshold_is_inclusive(self):
        p = policy()
        pos, goal, dist, speed = inputs()
        dist[:] = torch.tensor([9.9, 10.0, 10.1, 100], device=DEVICE)
        p.desired_velocity(pos, goal, dist, speed, .01)
        self.assertEqual(p.active.tolist(), [True, True, False, False])

    def test_period_and_amplitude(self):
        p = policy(1, amplitude_range=[2, 2], period_range=[4, 4])
        pos, goal, dist, speed = inputs(1)
        dist.zero_()
        p.desired_velocity(pos, goal, dist, speed, .01)
        for _ in range(100):
            p.desired_velocity(pos, goal, dist, speed, .01)
        self.assertAlmostEqual(float(p.reference_offset[0]), 2.0, places=4)
        for _ in range(100):
            p.desired_velocity(pos, goal, dist, speed, .01)
        self.assertAlmostEqual(float(p.reference_offset[0]), 0.0, places=4)
        for _ in range(100):
            p.desired_velocity(pos, goal, dist, speed, .01)
        self.assertAlmostEqual(float(p.reference_offset[0]), -2.0, places=4)

    def test_exit_and_reentry(self):
        p = policy(1)
        pos, goal, dist, speed = inputs(1)
        dist.zero_()
        p.desired_velocity(pos, goal, dist, speed, .01)
        p.desired_velocity(pos, goal, dist, speed, .01)
        dist.fill_(11)
        p.desired_velocity(pos, goal, dist, speed, .01)
        self.assertEqual(float(p.phase[0]), 0)
        self.assertEqual(float(p.reference_offset[0]), 0)
        pos[:, 1] = 7
        dist.zero_()
        p.desired_velocity(pos, goal, dist, speed, .01)
        torch.testing.assert_close(p.origin, pos)
        self.assertEqual(float(p.phase[0]), 0)

    def test_partial_reset_isolation(self):
        p = policy()
        pos, goal, dist, speed = inputs()
        dist.zero_()
        for _ in range(3):
            p.desired_velocity(pos, goal, dist, speed, .01)
        before = {name: getattr(p, name).clone() for name in (
            "amplitude", "period", "phase", "active", "origin", "lateral", "reference_offset")}
        p.reset(torch.tensor([1, 3], device=DEVICE))
        for name, value in before.items():
            torch.testing.assert_close(getattr(p, name)[[0, 2]], value[[0, 2]])
        self.assertEqual(p.active.tolist(), [True, False, True, False])

    def test_randomization_and_episode_constancy(self):
        p = policy(128)
        amplitude, period = p.amplitude.clone(), p.period.clone()
        self.assertTrue(((amplitude >= 1) & (amplitude <= 2)).all())
        self.assertTrue(((period >= 3) & (period <= 5)).all())
        self.assertGreater(float(amplitude.std()), .05)
        pos, goal, dist, speed = inputs(128)
        dist.zero_()
        for _ in range(50):
            p.desired_velocity(pos, goal, dist, speed, .01)
        torch.testing.assert_close(p.amplitude, amplitude)
        torch.testing.assert_close(p.period, period)
        p.reset(torch.arange(128, device=DEVICE))
        self.assertFalse(torch.equal(p.amplitude, amplitude))

    def test_zero_amplitude(self):
        p = policy(1, amplitude_range=[0, 0])
        pos, goal, dist, speed = inputs(1)
        dist.zero_()
        velocity = p.desired_velocity(pos, goal, dist, speed, .01)
        self.assertEqual(float(velocity[0, 1]), 0)

    def test_finite_and_bounded_at_extreme_settings(self):
        p = policy(4, amplitude_range=[1000, 1000], period_range=[.001, .001])
        pos, goal, dist, speed = inputs()
        dist.zero_()
        for _ in range(30):
            velocity = p.desired_velocity(pos, goal, dist, speed, .01)
            self.assertTrue(torch.isfinite(velocity).all())
            self.assertTrue((velocity.norm(dim=-1) <= speed + 1e-5).all())

    def test_goal_and_vertical_approach_are_finite(self):
        p = policy(2)
        pos, goal, dist, speed = inputs(2)
        goal[0].zero_()
        goal[1] = torch.tensor([0, 0, 10], device=DEVICE)
        dist.zero_()
        velocity = p.desired_velocity(pos, goal, dist, speed, .01)
        self.assertTrue(torch.isfinite(velocity).all())
        self.assertEqual(float(velocity[0].norm()), 0)
        self.assertGreater(float(velocity[1, 2]), 0)

    def test_reference_tracking_ideal_velocity_plant(self):
        p = policy(1, amplitude_range=[1, 1], period_range=[4, 4])
        pos, goal, dist, speed = inputs(1)
        goal[:, 0] = 10000
        goal[:, 2] = 0
        speed.fill_(10)
        dist.zero_()
        max_error = 0.0
        for _ in range(800):
            velocity = p.desired_velocity(pos, goal, dist, speed, .01)
            max_error = max(max_error, float((pos[:, 1]-p.reference_offset).abs().max()))
            pos += velocity * .01
        self.assertLess(max_error, .02)
        self.assertGreater(float(pos[0, 0]), 70)

    def test_invalid_config(self):
        for cfg in (dict(amplitude_range=[-1, 1]), dict(period_range=[0, 1]),
                    dict(period_range=[5, 1]), dict(trigger_distance=0),
                    dict(amplitude_range=[float("nan"), 1]),
                    dict(lateral_position_gain=float("inf"))):
            with self.assertRaises(ValueError):
                policy(**cfg)
        p = policy()
        with self.assertRaises(ValueError):
            p.desired_velocity(*inputs(), dt=0)


class EnvironmentMethodTests(unittest.TestCase):
    def setUp(self):
        self.env = load_environment_methods()()
        e = self.env
        e.target_motion_mode = "goal_sine"
        e.target_sine_policy = policy(2)
        e.target_dynamics_mode = "uav"
        e.target_vel_cmd = torch.zeros(2, 1, 3, device=DEVICE)
        e.target_max_speed = torch.full((2, 1, 1), 5.0, device=DEVICE)
        e.target_max_accel = torch.full((2, 1, 1), 3.0, device=DEVICE)
        e.target_uav_velocity_gain = 4.5
        e.target_velocity_damping = .05
        e.dt = .01
        e.goal_region_center = torch.tensor([100, 0, 10.0], device=DEVICE)
        e.goal_region_radius, e.goal_region_height = 3.0, 6.0
        e.target_tactical_intent = torch.zeros(2, dtype=torch.long, device=DEVICE)
        e.target_maneuver_id = torch.zeros_like(e.target_tactical_intent)
        e.target_maneuver_timer = torch.zeros_like(e.target_tactical_intent)
        e.target_goal_priority_active = torch.zeros(2, dtype=torch.bool, device=DEVICE)
        e.stats = {key: torch.zeros(2, 1, device=DEVICE)
                   for key in ("target_maneuver_switches", "target_tactical_switches")}
        self.dp = torch.tensor([[[8, 0, 10], [20, 0, 10], [30, 0, 10]],
                                [[20, 0, 10], [30, 0, 10], [40, 0, 10]]],
                               dtype=torch.float, device=DEVICE)
        self.tp = torch.tensor([[[0, 0, 10]], [[0, 0, 10]]], dtype=torch.float, device=DEVICE)
        e.drone = SimpleNamespace(get_world_poses=lambda *args: (self.dp, None))
        e.target = SimpleNamespace(get_world_poses=lambda *args: (self.tp, None),
                                   get_velocities=lambda: torch.zeros(2, 1, 6, device=DEVICE))
        e.get_env_poses = lambda poses: poses
        e._compute_target_boundary_force = lambda positions: torch.zeros_like(positions)

    def test_dispatch_acceleration_limits_and_labels(self):
        acceleration = self.env._compute_target_desired_acc()
        self.assertTrue(torch.isfinite(acceleration).all())
        self.assertTrue((acceleration.norm(dim=-1) <= 3.00001).all())
        self.assertEqual(self.env.target_maneuver_id.tolist(), [9, 0])
        self.assertEqual(self.env.target_tactical_intent.tolist(), [1, 0])
        probs = self.env._compute_rule_strategic_probs(self.dp, self.tp[:, 0])
        torch.testing.assert_close(probs, torch.tensor([[0., 1], [1., 0]], device=DEVICE))

    def test_no_full_acceleration_normalization(self):
        self.env.target_sine_policy.amplitude.zero_()
        self.dp[0, :, 0] += 100
        self.env.target_vel_cmd[:, :, 0] = 5
        acceleration = self.env._compute_target_desired_acc()
        self.assertLess(float(acceleration.norm(dim=-1).max()), .3)

    def test_point_mass_dispatch(self):
        self.env.target_dynamics_mode = "point_mass"
        self.assertTrue(torch.isfinite(self.env._compute_target_desired_acc()).all())

    def test_legacy_dispatch_is_unchanged(self):
        self.env.target_motion_mode = "adaptive"
        force = torch.ones(2, 1, 3, device=DEVICE)
        self.env._get_dummy_policy_prey = lambda: force
        result = self.env._compute_target_desired_acc()
        torch.testing.assert_close(result, self.env.target_max_accel * force /
                                   (force.norm(dim=-1, keepdim=True) + 1e-5))


if __name__ == "__main__":
    print("Testing goal sine on", DEVICE, flush=True)
    unittest.main(verbosity=2)
