import torch
import numpy as np
import functorch
import logging
from omni_drones.utils.torchrl.specs import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)
from tensordict.tensordict import TensorDict, TensorDictBase
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import wandb
import time
from functorch import vmap
from omni_drones.utils.torch import cpos, off_diag, others, quat_axis, quat_rotate_inverse
import torch.distributions as D
from torch.masked import masked_tensor, as_masked_tensor

import omni.isaac.core.objects as objects
# from omni.isaac.core.objects import VisualSphere, DynamicSphere, FixedCuboid, VisualCylinder, FixedCylinder, DynamicCylinder
# from omni.isaac.core.prims import RigidPrimView, GeometryPrimView
import omni.isaac.core.prims as prims
from omni_drones.views import RigidPrimView
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.controllers import PIDRateController as LowLevelPIDRateController
import omni_drones.utils.kit as kit_utils
# import omni_drones.utils.restart_sampling as rsp
from pxr import UsdGeom, Usd, UsdPhysics
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
from omni_drones.utils.scene import design_scene
import pdb
import copy
from omni_drones.utils.torch import euler_to_quaternion

from omni.isaac.debug_draw import _debug_draw
import os

from .draw import draw_traj, draw_detection, draw_catch, draw_court
from .draw_circle import Float3, _COLOR_ACCENT, _carb_float3_add, draw_court_circle
import time
import itertools
import math
from typing import Optional, Tuple
from .flight_capacity import configure_aircraft, acceleration_budget, bounded_motor_commands, segment_min_distance, specific_force, body_axis_thrust_ratio, floor_braking_prediction
from .goal_sine import GoalSinePolicy


def polygon_area_xy(points_xy: torch.Tensor) -> torch.Tensor:
    x = points_xy[..., 0]
    y = points_xy[..., 1]
    cross = x * torch.roll(y, shifts=-1, dims=-1) - y * torch.roll(x, shifts=-1, dims=-1)
    return 0.5 * torch.abs(cross.sum(dim=-1))


def point_to_cylinder_distance(
    points: torch.Tensor, center: torch.Tensor, radius: float, height: float
) -> torch.Tensor:
    radial_dist = torch.norm(points[..., :2] - center[..., :2], dim=-1)
    radial_delta = torch.relu(radial_dist - radius)
    vertical_delta = torch.relu(torch.abs(points[..., 2] - center[..., 2]) - height / 2.0)
    return torch.sqrt(radial_delta.square() + vertical_delta.square())


def point_in_cylinder(points: torch.Tensor, center: torch.Tensor, radius: float, height: float) -> torch.Tensor:
    radial_dist = torch.norm(points[..., :2] - center[..., :2], dim=-1)
    vertical_ok = torch.abs(points[..., 2] - center[..., 2]) <= height / 2.0
    return (radial_dist <= radius) & vertical_ok


def closest_point_on_cylinder(
    points: torch.Tensor, center: torch.Tensor, radius: float, height: float
) -> torch.Tensor:
    offset_xy = points[..., :2] - center[..., :2]
    radial_dist = torch.norm(offset_xy, dim=-1, keepdim=True)
    safe_radial = torch.where(radial_dist > 1e-5, radial_dist, torch.ones_like(radial_dist))
    clamped_radial = torch.minimum(radial_dist, torch.full_like(radial_dist, radius))
    closest_xy = center[..., :2] + offset_xy / safe_radial * clamped_radial
    closest_z = points[..., 2:3].clamp(center[..., 2:3] - height / 2.0, center[..., 2:3] + height / 2.0)
    return torch.cat([closest_xy, closest_z], dim=-1)


def clip_vector_norm(v: torch.Tensor, max_norm) -> torch.Tensor:
    """Clip the last-dimension norm with a scalar or broadcastable tensor."""
    norm = torch.norm(v, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    return v * scale


def safe_normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def project_to_plane(v: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    return v - (v * normal).sum(dim=-1, keepdim=True) * normal


TARGET_INTENT_GOAL_RUSH = 0
TARGET_INTENT_EVADE = 1
NUM_STRATEGIC_INTENTS = 2
NUM_TARGET_TRAJECTORIES = 1

TARGET_MANEUVER_STRAIGHT = 0
TARGET_MANEUVER_LEFT_TURN = 1
TARGET_MANEUVER_RIGHT_TURN = 2
TARGET_MANEUVER_CLIMB = 3
TARGET_MANEUVER_DIVE = 4
TARGET_MANEUVER_LEFT_CLIMB = 5
TARGET_MANEUVER_LEFT_DIVE = 6
TARGET_MANEUVER_RIGHT_CLIMB = 7
TARGET_MANEUVER_RIGHT_DIVE = 8
TARGET_NUM_MANEUVERS = 9
# Telemetry only: not a tenth candidate in the legacy nine-way optimiser.
TARGET_MANEUVER_SINE = 9

# Mutually exclusive episode termination codes.  Keep these stable because
# trajectory exports and downstream sampling analysis may store the raw code.
TERMINATION_ONGOING = 0
TERMINATION_CAPTURE = 1
TERMINATION_TARGET_GOAL = 2
TERMINATION_GROUND_COLLISION = 3
TERMINATION_DRONE_COLLISION = 4
TERMINATION_TIMEOUT_NEAR_MISS = 5
TERMINATION_TIMEOUT_NO_CLOSE = 6
TERMINATION_OUT_OF_BOUNDS = 7
TERMINATION_LANDED = 8


def compute_phi_team(
    block_i: torch.Tensor,
    dist_i: torch.Tensor,
    norm_dir: torch.Tensor,
    g_urg: torch.Tensor,
    pressure_scale: float,
):
    n = block_i.shape[1]
    device = block_i.device
    batch = block_i.shape[0]
    if n == 0:
        zeros = torch.zeros(batch, device=device)
        return zeros, zeros, zeros, zeros

    topk = min(n, 2)
    top_block = block_i.topk(topk, dim=1).values
    phi_block = 0.7 * block_i.max(dim=1).values + 0.3 * top_block.mean(dim=1)

    top_dist = dist_i.topk(topk, dim=1, largest=False).values
    phi_pressure = torch.exp(-top_dist.mean(dim=1) / max(pressure_scale, 1e-6))

    if n == 1:
        phi_spread = torch.zeros(batch, device=device)
    elif n == 2:
        cos12 = (norm_dir[:, 0, :] * norm_dir[:, 1, :]).sum(dim=-1).clamp(-1.0, 1.0)
        phi_spread = 0.5 * (1.0 - cos12)
    else:
        phi_spread = 1.0 - norm_dir.mean(dim=1).norm(dim=-1)

    phi_team = (
        0.50 * (0.4 + 0.6 * g_urg) * phi_block
        + 0.30 * phi_pressure
        + 0.20 * (1.0 - g_urg) * phi_spread
    )
    return phi_block, phi_pressure, phi_spread, phi_team


def compute_leave_one_out_di(
    block_i: torch.Tensor,
    dist_i: torch.Tensor,
    norm_dir: torch.Tensor,
    g_urg: torch.Tensor,
    pressure_scale: float,
    di_scale: float,
) -> torch.Tensor:
    n = block_i.shape[1]
    if n <= 1:
        return torch.zeros_like(block_i)

    _, _, _, phi_full = compute_phi_team(
        block_i, dist_i, norm_dir, g_urg, pressure_scale
    )
    d_i = torch.zeros_like(block_i)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        _, _, _, phi_without_i = compute_phi_team(
            block_i[:, idx],
            dist_i[:, idx],
            norm_dir[:, idx],
            g_urg,
            pressure_scale,
        )
        d_i[:, i] = phi_full - phi_without_i
    return torch.tanh(d_i / max(di_scale, 1e-6))

class HideAndSeek(IsaacEnv): 
    """
    Obstacle-free goal-defense environment with curriculum learning.

    Internal functions:

        _set_specs(self): 
            Set environment specifications for observations, states, actions, 
            rewards, statistics, infos, and initialize agent specifications

        _design_scenes(self): 
            Generate simulation scene and initialize all required objects
            
        _reset_idx(self, env_ids: torch.Tensor): 
            Reset poses of all objects, statistics and infos

        _pre_sim_step(self, tensordict: TensorDictBase):
            Process need to be completed before each step of simulation,
            including updating the APF-driven second-order target velocity

        _compute_state_and_obs(self):
            Obtain the observations and states tensor from drone state data
            Observations are organized into:
                state_self: target relative state, target prediction, self state, role encoding, time encoding
                state_others: teammate relative state plus defense-geometry features
                cooperation: shared goal-defense coordination features
            States contain centralized drone features for the critic

        _compute_reward_and_done(self):
            Compute obstacle-free goal-defense rewards from pursuer-target geometry,
            goal progress, collisions, and terminal events

        _get_dummy_policy_prey(self):
            Get APF forces for the target to move
            Force = goal attraction + pursuer repulsion + region-boundary repulsion

    """
    def __init__(self, cfg, headless):
        self.max_agents = int(cfg.task.max_agents)
        self.use_role_encoding = bool(getattr(cfg.task, "use_role_encoding", False))
        self.role_encoding_dim = 3 if self.use_role_encoding else 0
        self.future_prediction_obs_steps = int(getattr(cfg.task, "future_predcition_step", 5))
        prediction_cfg = getattr(cfg.task, "prediction", None)
        self.future_prediction_obs_steps = int(
            getattr(
                prediction_cfg,
                "prediction_horizon",
                self.future_prediction_obs_steps,
            )
        )
        self.state_self_dim = 20 + self.role_encoding_dim
        self.state_others_dim = 6 + self.role_encoding_dim
        self.cooperation_dim = 9 + 3 * self.future_prediction_obs_steps
        # _set_specs is called by IsaacEnv.__init__, so every dimension used
        # there must exist before super().__init__.
        self.agent_history_dim = 9
        self.predictor_obs_dim = int(getattr(prediction_cfg, "obs_dim", 30))
        self.target_dynamics_mode = str(
            getattr(cfg.task, "target_dynamics_mode", "point_mass")
        ).lower()
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.target_controller = None
        if self.target_dynamics_mode == "uav":
            self.target.initialize()
            self.target_controller = LowLevelPIDRateController(
                self.dt, 9.81, self.target.params
            ).to(self.device)
            self.target_hover_thrust_ratio = float(
                (self.target.gravity[0, 0] / self.target_controller.max_thrusts.sum()).item()
            )
            self.target_max_body_rate_rad_s = math.radians(
                180.0 * float(self.target_controller.target_clip)
            )
        elif self.target_dynamics_mode == "point_mass":
            self.target = RigidPrimView(
                "/World/envs/env_*/target",
                reset_xform_properties=False,
                shape=[self.num_envs, -1],
            )
            self.target.initialize()
        else:
            raise ValueError(
                "task.target_dynamics_mode must be either 'point_mass' or 'uav', "
                f"got {self.target_dynamics_mode!r}."
            )
        self.pursuer_controller = LowLevelPIDRateController(
            self.dt, 9.81, self.drone.params
        ).to(self.device)
        self.pursuer_hover_thrust_ratio = float(
            (self.drone.gravity[0, 0] / self.pursuer_controller.max_thrusts.sum()).item()
        )
        self.pursuer_max_body_rate_rad_s = math.radians(
            180.0 * float(self.pursuer_controller.target_clip)
        )
        capability = getattr(cfg.task, "flight_capability", None)
        self.thrust_fraction = float(getattr(capability, "thrust_fraction", 0.9))
        self.project_thrust_to_body_axis = bool(getattr(capability, "project_thrust_to_body_axis", False))
        self.pursuer_controller.max_thrust_ratio = self.thrust_fraction
        if self.target_controller is not None:
            self.target_controller.max_thrust_ratio = self.thrust_fraction
        self.pursuer_thrust_budget = acceleration_budget(self.pursuer_hover_thrust_ratio, self.thrust_fraction)
        self.target_motor_fraction = torch.zeros(self.num_envs, 1, device=self.device)
        self.pursuer_motor_fraction = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        
        self.time_encoding = self.cfg.task.time_encoding

        self.target_init_vel = self.target.get_velocities(clone=True)
        self.env_ids = torch.from_numpy(np.arange(0, cfg.env.num_envs))
        self.arena_size = self.cfg.task.arena_size
        self.returns = self.progress_buf * 0
        self.collision_radius = self.cfg.task.collision_radius
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.v_prey = self.cfg.task.v_drone * self.cfg.task.v_prey
        self.base_pursuer_speed = float(self.cfg.task.v_drone)
        self.base_target_speed = float(self.v_prey)
        self.current_pursuer_speed = float(self.base_pursuer_speed)
        self.current_target_speed = float(self.base_target_speed)
        self.catch_reward_coef = self.cfg.task.catch_reward_coef
        self.timeout_penalty_coef = float(getattr(self.cfg.task, "timeout_penalty_coef", 2.0))
        self.timeout_near_miss_distance = float(
            getattr(self.cfg.task, "timeout_near_miss_distance", 1.0)
        )
        if self.timeout_near_miss_distance <= float(self.cfg.task.catch_radius):
            raise ValueError(
                "timeout_near_miss_distance must be larger than catch_radius so "
                "near-miss timeouts are distinguishable from captures."
            )
        self.collision_coef = float(self.cfg.task.collision_coef)
        self.reward_profile = str(getattr(self.cfg.task, "reward_profile", "minimal_v3")).lower()
        # ── minimal_v3 reward hyper-params ──
        self.m3_progress_best_weight = float(getattr(cfg.task, "m3_progress_best_weight", 0.7))
        self.m3_progress_mean_weight = float(getattr(cfg.task, "m3_progress_mean_weight", 0.3))
        self.m3_pair_safety_weight = float(getattr(cfg.task, "m3_pair_safety_weight", 0.25))
        self.m3_env_safety_weight = float(getattr(cfg.task, "m3_env_safety_weight", 0.35))
        self.m3_control_effort_weight = float(getattr(cfg.task, "m3_control_effort_weight", 0.01))
        self.m3_control_delta_weight = float(getattr(cfg.task, "m3_control_delta_weight", 0.02))
        self.m3_time_penalty = float(getattr(cfg.task, "m3_time_penalty", 0.01))
        self.m3_terminal_collision_penalty = float(
            getattr(cfg.task, "m3_terminal_collision_penalty", 15.0)
        )
        self.m3_safety_pair_dist = float(getattr(cfg.task, "m3_safety_pair_dist", 0.5))
        self.m3_safety_env_dist = float(getattr(cfg.task, "m3_safety_env_dist", 0.5))
        self.spread_reward_coef = float(getattr(self.cfg.task, "spread_reward_coef", 0.0))
        self.speed_coef = self.cfg.task.speed_coef
        self.capture_progress_coef = float(getattr(self.cfg.task, "capture_progress_coef", 0.2))
        self.coop_phi_coef = float(getattr(self.cfg.task, "coop_phi_coef", 0.15))
        self.coop_diff_coef = float(getattr(self.cfg.task, "coop_diff_coef", 0.35))
        self.coop_reward_init = float(getattr(self.cfg.task, "coop_reward_init", 0.60))
        self.coop_reward_final = float(getattr(self.cfg.task, "coop_reward_final", 0.15))
        self.coop_anneal_portion = float(getattr(self.cfg.task, "coop_anneal_portion", 0.50))
        self.goal_progress_scale = float(getattr(self.cfg.task, "goal_progress_scale", 0.05))
        self.capture_progress_scale = float(getattr(self.cfg.task, "capture_progress_scale", 0.05))
        # Cooperative role geometry is still part of the critic observation
        # (assigned/close waypoints and motion axes), even when minimal_v3 is
        # the active reward profile. Keep these defaults local and optional;
        # this module is geometric shaping and has no OT dependency.
        self.expert2_role_reward_coef = float(
            getattr(self.cfg.task, "expert2_role_reward_coef", 0.30)
        )
        self.expert2_role_reward_scale = float(
            getattr(self.cfg.task, "expert2_role_reward_scale", 0.05)
        )
        self.expert2_role_front_side = float(
            getattr(self.cfg.task, "expert2_role_front_side", 0.40)
        )
        self.expert2_role_rear_back = float(
            getattr(self.cfg.task, "expert2_role_rear_back", 0.10)
        )
        self.expert2_role_close_enabled = bool(
            getattr(self.cfg.task, "expert2_role_close_enabled", True)
        )
        self.expert2_role_goal_dist_switch = float(
            getattr(self.cfg.task, "expert2_role_goal_dist_switch", 1.40)
        )
        self.expert2_role_far_lookahead = float(
            getattr(self.cfg.task, "expert2_role_far_lookahead", 0.18)
        )
        self.expert2_role_near_lookahead = float(
            getattr(self.cfg.task, "expert2_role_near_lookahead", 0.10)
        )
        self.expert2_intercept_pred_step = int(
            getattr(
                self.cfg.task,
                "expert_intercept_pred_step",
                getattr(self.cfg.task, "expert2_intercept_pred_step", 5),
            )
        )
        self.expert2_intercept_use_direct_pred = bool(
            getattr(
                self.cfg.task,
                "expert_intercept_use_direct_pred",
                getattr(self.cfg.task, "expert2_intercept_use_direct_pred", False),
            )
        )
        self.expert2_role_close_wp_threshold = float(
            getattr(self.cfg.task, "expert2_role_close_wp_threshold", 0.45)
        )
        self.expert2_role_chaser_close_target_threshold = float(
            getattr(self.cfg.task, "expert2_role_chaser_close_target_threshold", 0.95)
        )
        self.expert2_role_front_close_target_threshold = float(
            getattr(self.cfg.task, "expert2_role_front_close_target_threshold", 0.75)
        )
        self.expert2_role_close_chaser_back = float(
            getattr(self.cfg.task, "expert2_role_close_chaser_back", 0.10)
        )
        self.expert2_role_close_front_forward = float(
            getattr(self.cfg.task, "expert2_role_close_front_forward", 0.03)
        )
        self.expert2_role_close_front_side = float(
            getattr(self.cfg.task, "expert2_role_close_front_side", 0.24)
        )
        self.expert2_close_reward_coef = float(
            getattr(self.cfg.task, "expert2_close_reward_coef", self.expert2_role_reward_coef)
        )
        self.expert2_close_reward_scale = float(
            getattr(self.cfg.task, "expert2_close_reward_scale", self.expert2_role_reward_scale)
        )
        self.soft_separation_start = float(getattr(self.cfg.task, "soft_separation_start", 0.40))
        self.soft_separation_end = float(getattr(self.cfg.task, "soft_separation_end", 0.20))
        self.soft_separation_penalty_coef = float(
            getattr(self.cfg.task, "soft_separation_penalty_coef", 2.0)
        )
        self.di_scale = float(getattr(self.cfg.task, "di_scale", 0.15))
        self.urgent_goal_dist = float(getattr(self.cfg.task, "urgent_goal_dist", 0.8))
        self.urgent_goal_tau = float(getattr(self.cfg.task, "urgent_goal_tau", 0.15))
        self.block_margin = float(getattr(self.cfg.task, "block_margin", 0.15))
        self.block_softness = float(getattr(self.cfg.task, "block_softness", 0.10))
        self.block_between_softness = float(getattr(self.cfg.task, "block_between_softness", 0.15))
        self.block_lateral_scale = float(getattr(self.cfg.task, "block_lateral_scale", 0.35))
        self.pressure_dist_scale = float(getattr(self.cfg.task, "pressure_dist_scale", 0.8))
        self.time_penalty_coef = float(getattr(self.cfg.task, "time_penalty_coef", 0.002))
        self.smoothness_coef = self.cfg.task.smoothness_coef
        self.ground_collision_height = self.cfg.task.ground_collision_height
        self.landed_z_threshold = self.cfg.task.landed_z_threshold
        self.landed_speed_threshold = self.cfg.task.landed_speed_threshold
        self.landed_penalty_coef = self.cfg.task.landed_penalty_coef
        self.goal_progress_coef = float(getattr(self.cfg.task, "goal_progress_coef", 1.0))
        self.goal_penalty_coef = float(getattr(self.cfg.task, "goal_penalty_coef", 200.0))
        self.target_goal_attraction_coef = float(getattr(self.cfg.task, "target_goal_attraction_coef", 1.0))
        self.target_repulsion_coef = float(getattr(self.cfg.task, "target_repulsion_coef", 1.0))
        self.target_accel_limit = float(getattr(self.cfg.task, "target_accel_limit", 2.0))
        target_speed_range = tuple(
            float(v) for v in getattr(
                self.cfg.task,
                "target_max_speed_range",
                (self.base_target_speed, self.base_target_speed),
            )
        )
        target_accel_range = tuple(
            float(v) for v in getattr(
                self.cfg.task,
                "target_max_accel_range",
                (self.target_accel_limit, self.target_accel_limit),
            )
        )
        target_initial_speed_range = tuple(
            float(v) for v in getattr(
                self.cfg.task,
                "target_initial_speed_range",
                (0.5, min(2.5, target_speed_range[-1])),
            )
        )
        target_initial_pitch_range = tuple(
            float(v) for v in getattr(
                self.cfg.task,
                "target_initial_pitch_deg_range",
                (-15.0, 15.0),
            )
        )
        for name, value_range in (
            ("target_max_speed_range", target_speed_range),
            ("target_max_accel_range", target_accel_range),
            ("target_initial_speed_range", target_initial_speed_range),
            ("target_initial_pitch_deg_range", target_initial_pitch_range),
        ):
            if len(value_range) != 2 or value_range[0] > value_range[1]:
                raise ValueError(f"{name} must be [min, max], got {value_range!r}")
        if target_speed_range[0] <= 0.0 or target_accel_range[0] <= 0.0:
            raise ValueError("Target maximum speed and acceleration must stay positive.")
        if target_initial_speed_range[0] < 0.0:
            raise ValueError("target_initial_speed_range must be non-negative.")
        self.target_max_speed_range = target_speed_range
        self.target_max_accel_range = target_accel_range
        self.target_initial_speed_range = target_initial_speed_range
        self.target_initial_pitch_deg_range = target_initial_pitch_range
        self.target_velocity_damping = float(getattr(self.cfg.task, "target_velocity_damping", 0.25))
        self.target_command_tau = float(getattr(self.cfg.task, "target_command_tau", 0.15))
        self.target_uav_velocity_gain = float(getattr(self.cfg.task, "target_uav_velocity_gain", 3.2))
        self.target_uav_max_tilt_deg = float(getattr(self.cfg.task, "target_uav_max_tilt_deg", 25.0))
        self.target_uav_rate_fraction = float(getattr(self.cfg.task, "target_uav_rate_fraction", 0.65))
        self.target_uav_acc_feedforward_scale = float(getattr(self.cfg.task, "target_uav_acc_feedforward_scale", 0.50))
        self.target_min_z = float(getattr(self.cfg.task, "target_min_z", 0.2))
        self.target_motion_mode = str(getattr(self.cfg.task, "target_motion_mode", "adaptive"))
        if self.target_motion_mode not in ("adaptive", "goal_sine"):
            raise ValueError("target_motion_mode must be 'adaptive' or 'goal_sine'")
        self.target_sine_policy = None
        if self.target_motion_mode == "goal_sine":
            self.target_sine_policy = GoalSinePolicy(
                self.num_envs, self.device, getattr(self.cfg.task, "target_sine", {}))
        self.target_tactical_far_distance = float(
            getattr(self.cfg.task, "target_tactical_far_distance", 6.0)
        )
        self.target_goal_sprint_distance = float(
            getattr(self.cfg.task, "target_goal_sprint_distance", 2.0)
        )
        self.target_far_goal_gain = float(
            getattr(self.cfg.task, "target_far_goal_gain", self.target_goal_attraction_coef)
        )
        self.target_near_goal_gain = float(
            getattr(self.cfg.task, "target_near_goal_gain", self.target_goal_attraction_coef)
        )
        self.target_goal_sprint_gain = float(
            getattr(
                self.cfg.task,
                "target_goal_sprint_gain",
                self.target_goal_attraction_coef * 1.5,
            )
        )
        self.target_goal_sprint_repulsion_scale = float(
            getattr(self.cfg.task, "target_goal_sprint_repulsion_scale", 0.35)
        )
        self.target_goal_priority_gain = float(
            getattr(
                self.cfg.task,
                "target_goal_priority_gain",
                self.target_goal_attraction_coef * 2.4,
            )
        )
        self.target_goal_priority_repulsion_scale = float(
            getattr(
                self.cfg.task,
                "target_goal_priority_repulsion_scale",
                0.45,
            )
        )
        self.target_nearest_repulsion_coef = float(
            getattr(
                self.cfg.task,
                "target_nearest_repulsion_coef",
                self.target_repulsion_coef * 1.8,
            )
        )
        self.target_other_repulsion_coef = float(
            getattr(
                self.cfg.task,
                "target_other_repulsion_coef",
                self.target_repulsion_coef,
            )
        )
        self.target_maneuver_force_coef = float(
            getattr(self.cfg.task, "target_maneuver_force_coef", 0.9)
        )
        self.target_maneuver_forward_scale = float(
            getattr(self.cfg.task, "target_maneuver_forward_scale", 0.4)
        )
        self.target_maneuver_vertical_scale = float(
            getattr(self.cfg.task, "target_maneuver_vertical_scale", 0.9)
        )
        self.target_maneuver_speed_threshold = float(
            getattr(self.cfg.task, "target_maneuver_speed_threshold", 0.3)
        )
        self.target_maneuver_min_steps = int(
            getattr(self.cfg.task, "target_maneuver_min_steps", 8)
        )
        self.target_maneuver_max_steps = int(
            getattr(self.cfg.task, "target_maneuver_max_steps", 24)
        )
        self.target_opt_maneuver_evade_weight = float(
            getattr(self.cfg.task, "target_opt_maneuver_evade_weight", 1.0)
        )
        self.target_opt_maneuver_goal_weight = float(
            getattr(self.cfg.task, "target_opt_maneuver_goal_weight", 4.0)
        )
        self.target_opt_maneuver_lateral_weight = float(
            getattr(self.cfg.task, "target_opt_maneuver_lateral_weight", 0.0)
        )
        self.target_opt_maneuver_turn_weight = float(
            getattr(self.cfg.task, "target_opt_maneuver_turn_weight", 0.0)
        )
        self.target_opt_maneuver_height_weight = float(
            getattr(self.cfg.task, "target_opt_maneuver_height_weight", 3.0)
        )
        self.target_opt_maneuver_boundary_weight = float(
            getattr(self.cfg.task, "target_opt_maneuver_boundary_weight", 3.0)
        )
        self.target_maneuver_lookahead_steps = max(
            1,
            int(getattr(self.cfg.task, "target_maneuver_lookahead_steps", 1)),
        )
        self.target_maneuver_lookahead_dt = float(
            getattr(self.cfg.task, "target_maneuver_lookahead_dt", self.dt)
        )
        self.use_eval = self.cfg.task.use_eval
        self.use_partial_obs = self.cfg.task.use_partial_obs
        self.goal_region_radius = float(getattr(self.cfg.task, "goal_region_radius", 0.5))
        self.goal_region_height = float(getattr(self.cfg.task, "goal_region_height", 2.0))
        goal_region_center_cfg = getattr(self.cfg.task, "goal_region_center", None)
        if goal_region_center_cfg is None:
            self.goal_region_center = torch.tensor(
                [
                    self.arena_size - self.goal_region_radius,
                    0.0,
                    self.max_height / 2.0,
                ],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self.goal_region_center = torch.tensor(
                goal_region_center_cfg, device=self.device, dtype=torch.float32
            )
        prediction_cfg = getattr(self.cfg.task, "prediction", None)
        self.prediction_dt = float(getattr(prediction_cfg, "dt", 0.05))
        self.predictor_raw_stride = max(
            1,
            int(round(self.prediction_dt / max(float(self.dt), 1e-6))),
        )
        # Rollout transitions are now produced at prediction_dt, so the online
        # predictor target is one policy transition apart. Raw 100 Hz history
        # uses predictor_raw_stride separately.
        self.prediction_stride = 1
        self.predictor_checkpoint = str(
            getattr(prediction_cfg, "checkpoint", "") or ""
        )
        self.train_predictor = bool(getattr(prediction_cfg, "train_online", False))

        control_cfg = getattr(self.cfg.task, "trajectory_control", None)
        self.trajectory_plan_dt = float(
            getattr(control_cfg, "plan_dt", self.prediction_dt)
        )
        self.control_decimation = max(
            1,
            int(round(self.trajectory_plan_dt / max(float(self.dt), 1e-6))),
        )
        if not math.isclose(
            self.control_decimation * float(self.dt),
            self.trajectory_plan_dt,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("trajectory_control.plan_dt must be an integer multiple of sim.dt")
        if not math.isclose(
            self.trajectory_plan_dt, self.prediction_dt, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("trajectory_control.plan_dt must equal prediction.dt for time alignment")
        self.trajectory_position_gain = float(
            getattr(control_cfg, "position_gain", 8.0)
        )
        self.floor_use_braking_distance = bool(getattr(control_cfg, "floor_use_braking_distance", False))
        self.trajectory_velocity_gain = float(
            getattr(control_cfg, "velocity_gain", 4.0)
        )
        self.pursuer_accel_limit = float(
            getattr(control_cfg, "accel_limit", self.current_pursuer_speed * 2.0)
        )
        self.pursuer_max_tilt_deg = float(getattr(control_cfg, "max_tilt_deg", 35.0))
        self.pursuer_rate_fraction = float(getattr(control_cfg, "rate_fraction", 0.75))
        self.pursuer_yaw_heading_blend = float(
            getattr(control_cfg, "yaw_heading_blend", 0.0)
        )
        self.pursuer_floor_barrier_height = float(
            getattr(control_cfg, "floor_barrier_height", 0.60)
        )
        self.pursuer_floor_hard_z = float(
            getattr(control_cfg, "floor_hard_z", 0.30)
        )
        self.pursuer_floor_target_margin = float(
            getattr(control_cfg, "floor_target_margin", 0.10)
        )
        self.pursuer_floor_prediction_time = float(
            getattr(control_cfg, "floor_prediction_time", 0.25)
        )
        self.pursuer_floor_recovery_margin = float(
            getattr(control_cfg, "floor_recovery_margin", 0.55)
        )
        self.pursuer_floor_barrier_max_accel = float(
            getattr(control_cfg, "floor_barrier_max_accel", 5.5)
        )
        self.pursuer_floor_xy_scale_min = float(
            getattr(control_cfg, "floor_xy_scale_min", 0.25)
        )
        self.pursuer_floor_barrier_gain = float(
            getattr(control_cfg, "floor_barrier_gain", 8.0)
        )
        self.pursuer_floor_barrier_damping = float(
            getattr(control_cfg, "floor_barrier_damping", 3.0)
        )
        self.pursuer_vertical_accel_limit = float(
            getattr(control_cfg, "vertical_accel_limit", self.pursuer_floor_barrier_max_accel)
        )
        self.pursuer_horizontal_accel_limit = float(
            getattr(control_cfg, "horizontal_accel_limit", self.pursuer_accel_limit)
        )

        self.predictor_loaded_checkpoint = False
        self.predictor = self._build_predictor()
        if self.predictor_checkpoint:
            self._load_predictor(self.predictor_checkpoint)
        self.training_progress = 1.0
        self.position_scale = torch.tensor(
            [self.arena_size, self.arena_size, self.max_height],
            device=self.device,
            dtype=torch.float32,
        )
        self.goal_distance_norm = max(self.arena_size, 1e-6)
        self.velocity_scale = max(float(self.current_pursuer_speed), 1e-6)
        self.target_velocity_scale = max(
            float(self.current_target_speed),
            self.target_max_speed_range[1],
            1e-6,
        )
        reset_cfg = getattr(self.cfg.task, "reset_profile", None)
        # Legacy reset advanced the entire PhysX world, including live envs.
        # Keep legacy reproducibility; isolated configs use only IsaacEnv flush.
        self.advance_physics_on_reset = bool(getattr(reset_cfg, "advance_physics_on_reset", True))
        legacy_curr_cfg = getattr(self.cfg.task, "curriculum", None)
        curr_cfg = legacy_curr_cfg
        spawn_cfg = reset_cfg if reset_cfg is not None else legacy_curr_cfg
        default_curriculum_stages = [
            {"distance_range": (0.3, 0.8), "pursuer_speed": 0.8, "target_speed": 0.5},
            {"distance_range": (0.5, 1.5), "pursuer_speed": 1.0, "target_speed": 1.0},
            {"distance_range": (1.0, 2.5), "pursuer_speed": 1.5, "target_speed": 1.5},
            {"distance_range": (2.0, 3.5), "pursuer_speed": 1.5, "target_speed": 1.7},
        ]
        self.curriculum_enabled = bool(
            getattr(curr_cfg, "enabled", False)
        ) and self.scenario_flag == "goal_defense"
        self.curriculum_eval_uses_fixed_layout = bool(getattr(curr_cfg, "eval_uses_fixed_layout", True))
        eval_promote_cfg = getattr(curr_cfg, "eval_promote", None) if curr_cfg is not None else None
        self.curriculum_eval_promote_enabled = bool(getattr(eval_promote_cfg, "enabled", False))
        self.curriculum_start_stage = int(getattr(curr_cfg, "start_stage", 1)) - 1
        self.curriculum_ema_alpha = float(getattr(curr_cfg, "ema_alpha", 0.10))
        self.curriculum_min_stage_episodes = int(getattr(curr_cfg, "min_stage_episodes", max(self.num_envs, 100)))
        self.curriculum_pursuer_spawn_mode = str(getattr(spawn_cfg, "pursuer_spawn_mode", "uniform"))
        self.curriculum_goal_side_angle = math.radians(float(getattr(spawn_cfg, "goal_side_angle_deg", 65.0)))
        self.curriculum_goal_side_angle_jitter = math.radians(
            float(getattr(spawn_cfg, "goal_side_angle_jitter_deg", 10.0))
        )
        self.curriculum_target_spawn_x = tuple(getattr(spawn_cfg, "target_spawn_x", (-0.9, -0.2)))
        self.curriculum_target_spawn_y = tuple(getattr(spawn_cfg, "target_spawn_y", (-1.0, 1.0)))
        self.curriculum_front_box_target_x = tuple(getattr(spawn_cfg, "front_box_target_x", (-3.0, -2.5)))
        self.curriculum_front_box_target_y = tuple(getattr(spawn_cfg, "front_box_target_y", (-1.0, 1.0)))
        self.curriculum_front_box_drone_x = tuple(getattr(spawn_cfg, "front_box_drone_x", (1.0, 2.0)))
        self.curriculum_front_box_drone_y = tuple(getattr(spawn_cfg, "front_box_drone_y", (-1.0, 1.0)))
        self.curriculum_front_box_max_z_diff = float(getattr(spawn_cfg, "front_box_max_z_diff", 1.0))
        self.curriculum_target_z_jitter = float(getattr(spawn_cfg, "target_z_jitter", 0.15))
        self.curriculum_drone_z_jitter = float(getattr(spawn_cfg, "drone_z_jitter", 0.15))
        self.curriculum_drone_min_separation = float(
            getattr(spawn_cfg, "drone_min_separation", max(4.0 * self.collision_radius, 0.30))
        )
        self.curriculum_max_sample_attempts = int(getattr(spawn_cfg, "max_sample_attempts", 64))
        stage_source_cfg = spawn_cfg if spawn_cfg is not None else curr_cfg
        if stage_source_cfg is not None and getattr(stage_source_cfg, "stages", None):
            self.curriculum_stages = []
            for stage_cfg in stage_source_cfg.stages:
                dist_range = tuple(float(v) for v in stage_cfg.pursuer_target_dist)
                parsed_stage = {
                    "distance_range": dist_range,
                    "pursuer_speed": float(getattr(stage_cfg, "pursuer_speed", self.base_pursuer_speed)),
                    "target_speed": float(getattr(stage_cfg, "target_speed", self.base_target_speed)),
                }
                if hasattr(stage_cfg, "promote_success"):
                    parsed_stage["promote_success"] = float(stage_cfg.promote_success)
                if hasattr(stage_cfg, "promote_capture_step"):
                    parsed_stage["promote_capture_step"] = float(stage_cfg.promote_capture_step)
                self.curriculum_stages.append(parsed_stage)
        else:
            self.curriculum_stages = default_curriculum_stages
        self.curriculum_stage = 0
        self.curriculum_stage_episodes = 0
        self.curriculum_success_ema = 0.0
        self.curriculum_capture_step_ema = float(self.max_episode_length)
        self.curriculum_capture_ema_initialized = False
        initial_stage = self.curriculum_start_stage if self.curriculum_enabled else len(self.curriculum_stages) - 1
        self._set_curriculum_stage(initial_stage, reset_metrics=True, announce=False)
        self.capture = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.min_dist = torch.ones(self.num_envs, 1, device=self.device) * float(torch.inf) # for teacher evaluation
        self.episode_min_target_distance = torch.full(
            (self.num_envs, 1), float("inf"), device=self.device
        )
        # prev_target_dist for distance progress reward: [num_envs, num_agents]
        self.prev_target_dist = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.prev_goal_dist = torch.zeros(self.num_envs, 1, device=self.device)
        # ── minimal_v3 reward trackers ──
        self.m3_prev_target_dist = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.m3_action_magnitude = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.m3_action_delta = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.prev_expert2_role_dist = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.prev_expert2_role_dist_ready = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.prev_expert2_close_triggered = torch.zeros(
            self.num_envs, self.num_agents, dtype=torch.bool, device=self.device
        )
        self.expert2_role_permutations = torch.tensor(
            list(itertools.permutations(range(self.num_agents))),
            device=self.device,
            dtype=torch.long,
        )
        self.expert2_prev_assignment = torch.full(
            (self.num_envs, self.num_agents),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.expert2_assignment_ready = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device
        )
        self._expert2_cached_active_waypoint = None
        self._expert2_cached_close_triggered = None
        self.target_acc_cmd = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.target_vel_cmd = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.target_max_speed = torch.full(
            (self.num_envs, 1, 1),
            self.target_max_speed_range[1],
            device=self.device,
        )
        self.target_max_accel = torch.full(
            (self.num_envs, 1, 1),
            self.target_max_accel_range[1],
            device=self.device,
        )
        self.target_initial_speed = torch.zeros(
            self.num_envs, 1, device=self.device
        )
        self.target_pid_reset = torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.pursuer_pid_reset = torch.ones(
            self.num_envs, self.num_agents, dtype=torch.bool, device=self.device
        )
        self.target_tactical_intent = torch.full(
            (self.num_envs,),
            TARGET_INTENT_GOAL_RUSH,
            dtype=torch.long,
            device=self.device,
        )
        self.target_maneuver_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.target_maneuver_timer = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.target_goal_priority_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Internal execution phases preserve the old manoeuvre timer resets;
        # they are not extra strategic classes or network inputs.
        self.target_controller_stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        
        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )

        self.init_drone_pos_dist = D.Uniform(
            torch.tensor([0.1, -self.arena_size / math.sqrt(2.0) + 0.1], device=self.device),
            torch.tensor([self.arena_size / math.sqrt(2.0) - 0.1, self.arena_size / math.sqrt(2.0) - 0.1], device=self.device)
        )
        self.init_target_pos_dist = D.Uniform(
            torch.tensor([-self.arena_size / math.sqrt(2.0) + 0.1, -self.arena_size / math.sqrt(2.0) + 0.1], device=self.device),
            torch.tensor([-0.1, self.arena_size / math.sqrt(2.0) - 0.1], device=self.device)
        )

        self.init_drone_pos_dist_z = D.Uniform(
            torch.tensor([self.max_height / 2 - 0.1], device=self.device),
            torch.tensor([self.max_height / 2 + 0.1], device=self.device)
        )
        self.init_target_pos_dist_z = D.Uniform(
            torch.tensor([self.max_height / 2 - 0.1], device=self.device),
            torch.tensor([self.max_height / 2 + 0.1], device=self.device)
        )

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.2, -0.2, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 0.2], device=self.device) * torch.pi
        )

        if self.use_eval:
            self.init_rpy_dist = D.Uniform(
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi,
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi
            )

        self.mask_value = -5
        self.draw = None
        if self._should_render(0):
            try:
                self.draw = _debug_draw.acquire_debug_draw_interface()
            except RuntimeError as exc:
                logging.warning("Debug draw is unavailable; continuing without render overlays: %s", exc)
        # The former single-trajectory TP network is retired. Future target
        # motion is represented exclusively by the three-mode GMM predictor.
        self.use_TP_net = False
        
        # use self.masked_drone_pos to expand drone_pos
        self.masked_drone_pos = self.mask_value * torch.ones(self.num_envs, self.max_agents - self.num_agents, 3, device=self.device)

        # TP net
        # t, target pos, target vel, padded drone_pos
        if self.use_TP_net:
            from omni_drones.learning import TP_net

            self.TP = TP_net(
                input_dim=1 + 3 + 3 + 3 * self.max_agents,
                output_dim=3 * self.future_predcition_step,
                future_predcition_step=self.future_predcition_step,
                window_step=self.window_step,
            ).to(self.device)
        else:
            self.TP = None
        self.history_step = int(self.cfg.task.history_step)
        self.tp_frame_dim = 1 + 3 + 3 + 3 * self.max_agents
        # Keep per-env TP history so partial resets do not leak stale traces
        # from finished environments into freshly reset ones.
        self.tp_history_buffer = torch.zeros(
            self.num_envs,
            self.history_step,
            self.tp_frame_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.tp_history_initialized = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.target_pos_predicted = None
        self.predictor_strategic_probs = torch.zeros(self.num_envs, NUM_STRATEGIC_INTENTS, device=self.device)
        self.predictor_strategic_probs[:, 0] = 1.0
        self.predictor_gmm_weights = torch.zeros(self.num_envs, NUM_TARGET_TRAJECTORIES, device=self.device)
        self.predictor_gmm_weights[:, 0] = 1.0
        self.predictor_gmm_means = torch.zeros(
            self.num_envs, NUM_TARGET_TRAJECTORIES, self.future_prediction_obs_steps, 3, device=self.device
        )
        self.predictor_gmm_vars = torch.ones_like(self.predictor_gmm_means)
        self.predictor_history = torch.zeros(
            self.num_envs,
            self.future_prediction_obs_steps,
            self.predictor_obs_dim,
            device=self.device,
        )
        self.predictor_initialized = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.predictor_raw_history_len = (
            (self.future_prediction_obs_steps - 1) * self.predictor_raw_stride + 1
        )
        self.predictor_raw_history = torch.zeros(
            self.num_envs,
            self.predictor_raw_history_len,
            self.predictor_obs_dim,
            device=self.device,
        )
        self.predictor_raw_initialized = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.predictor_history_indices = torch.arange(
            0,
            self.predictor_raw_history_len,
            self.predictor_raw_stride,
            device=self.device,
            dtype=torch.long,
        )
        self.predictor_sample_valid = torch.zeros(
            self.num_envs,
            1,
            device=self.device,
        )
        self.predictor_target_pos_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.predictor_strategic_label = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.long,
            device=self.device,
        )
        self.agent_history_buffer = torch.zeros(
            self.num_envs,
            self.num_agents,
            self.history_step,
            self.agent_history_dim,
            device=self.device,
        )
        self.agent_history_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.agent_history_last_step = torch.full((self.num_envs,), -1.0, device=self.device)
        self.previous_plan = torch.zeros(
            self.num_envs,
            self.num_agents,
            self.future_prediction_obs_steps,
            6,  # physical position/velocity plan, independent of latent action dimension
            device=self.device,
        )
        self.previous_plan_mask = torch.zeros(
            self.num_envs,
            self.num_agents,
            self.future_prediction_obs_steps,
            device=self.device,
        )
        from omni_drones.learning.modules.cooperative_future import (
            BoundedTrajectoryReferenceDecoder,
        )

        self.trajectory_decoder = BoundedTrajectoryReferenceDecoder(
            dt=self.prediction_dt,
            max_speed=self.current_pursuer_speed,
            reference_mode=str(getattr(control_cfg, "reference_mode", "position_velocity")),
            accel_limits=(
                self.pursuer_horizontal_accel_limit,
                self.pursuer_horizontal_accel_limit,
                self.pursuer_vertical_accel_limit,
            ),
        ).to(self.device)
        # self.debug_list = []

        # for deployment
        self.prev_actions = torch.zeros(
            self.num_envs,
            self.num_agents,
            self.future_prediction_obs_steps,
            self.action_reference_dim,
            device=self.device,
        )

    def set_training_progress(self, fraction: float):
        self.training_progress = float(max(0.0, min(1.0, fraction)))

    def _set_curriculum_stage(self, stage: int, reset_metrics: bool = False, announce: bool = True):
        stage = int(max(0, min(stage, len(self.curriculum_stages) - 1)))
        self.curriculum_stage = stage
        stage_cfg = self.curriculum_stages[stage]
        if self.curriculum_enabled:
            self.current_pursuer_speed = float(stage_cfg.get("pursuer_speed", self.base_pursuer_speed))
            self.current_target_speed = float(stage_cfg["target_speed"])
        else:
            # When curriculum is disabled, respect the caller-provided base target speed
            # instead of silently snapping to the YAML stage preset.
            self.current_pursuer_speed = float(self.base_pursuer_speed)
            self.current_target_speed = float(self.base_target_speed)
        self.velocity_scale = max(self.current_pursuer_speed, 1e-6)
        # Use one fixed normalization scale while the physical limit is sampled
        # independently per episode.  Otherwise the same world velocity would
        # acquire a different observation value solely due to hidden domain
        # randomization.
        self.target_velocity_scale = max(
            self.current_target_speed,
            self.target_max_speed_range[1],
            1e-6,
        )
        if reset_metrics:
            self.curriculum_stage_episodes = 0
            self.curriculum_success_ema = 0.0
            self.curriculum_capture_step_ema = float(self.max_episode_length)
            self.curriculum_capture_ema_initialized = False
        if announce and self.curriculum_enabled:
            logging.info(
                "[Curriculum] switched to stage %d | pursuer-target dist=(%.2f, %.2f) | pursuer_speed=%.2f | target_speed=%.2f",
                stage + 1,
                stage_cfg["distance_range"][0],
                stage_cfg["distance_range"][1],
                self.current_pursuer_speed,
                self.current_target_speed,
            )

    def _maybe_advance_curriculum(self):
        if not self.curriculum_enabled:
            return
        if self.curriculum_eval_promote_enabled:
            return
        if self.curriculum_stage >= len(self.curriculum_stages) - 1:
            return
        if self.curriculum_stage_episodes < self.curriculum_min_stage_episodes:
            return

        stage_cfg = self.curriculum_stages[self.curriculum_stage]
        success_ready = self.curriculum_success_ema >= float(stage_cfg.get("promote_success", 1.1))
        capture_ready = True
        if "promote_capture_step" in stage_cfg:
            capture_ready = (
                self.curriculum_capture_ema_initialized
                and self.curriculum_capture_step_ema <= float(stage_cfg["promote_capture_step"])
            )
        if success_ready and capture_ready:
            self._set_curriculum_stage(self.curriculum_stage + 1, reset_metrics=True, announce=True)

    def _update_curriculum_metrics(self, done: torch.Tensor, capture_before_goal: torch.Tensor):
        if not self.curriculum_enabled or not self.training:
            return
        done_mask = done.squeeze(-1)
        if not torch.any(done_mask):
            return

        alpha = self.curriculum_ema_alpha
        success_batch = capture_before_goal.squeeze(-1)[done_mask].float().mean().item()
        if self.curriculum_stage_episodes == 0:
            self.curriculum_success_ema = success_batch
        else:
            self.curriculum_success_ema = (1.0 - alpha) * self.curriculum_success_ema + alpha * success_batch
        self.curriculum_stage_episodes += int(done_mask.sum().item())

        successful_done = done_mask & capture_before_goal.squeeze(-1)
        if torch.any(successful_done):
            capture_step_batch = self.progress_buf[successful_done].float().mean().item()
            if not self.curriculum_capture_ema_initialized:
                self.curriculum_capture_step_ema = capture_step_batch
                self.curriculum_capture_ema_initialized = True
            else:
                self.curriculum_capture_step_ema = (
                    (1.0 - alpha) * self.curriculum_capture_step_ema + alpha * capture_step_batch
                )
        self._maybe_advance_curriculum()

    def _record_termination_stats(
        self,
        *,
        capture: torch.Tensor,
        target_goal: torch.Tensor,
        ground_collision: torch.Tensor,
        drone_collision: torch.Tensor,
        timeout: torch.Tensor,
        out_of_bounds: torch.Tensor = None,
        landed: torch.Tensor = None,
    ):
        """Record exactly one terminal reason for every completed episode.

        Inputs may overlap on the same physics step.  This method applies one
        documented priority so one-hot termination rates always sum to one:
        capture > target goal > ground collision > landed > drone collision >
        out of bounds > timeout.  A timeout is split using the minimum
        pursuer-target distance observed over the complete episode.
        """
        zeros = torch.zeros_like(capture, dtype=torch.bool)
        out_of_bounds = zeros if out_of_bounds is None else out_of_bounds.bool()
        landed = zeros if landed is None else landed.bool()
        capture = capture.bool()
        target_goal = target_goal.bool()
        ground_collision = ground_collision.bool()
        drone_collision = drone_collision.bool()
        timeout = timeout.bool()

        done = (
            capture
            | target_goal
            | ground_collision
            | landed
            | drone_collision
            | out_of_bounds
            | timeout
        )
        fresh = done & self.stats["termination_reason"].eq(float(TERMINATION_ONGOING))
        remaining = fresh

        reason_capture = remaining & capture
        remaining = remaining & ~reason_capture
        reason_target_goal = remaining & target_goal
        remaining = remaining & ~reason_target_goal
        reason_ground = remaining & ground_collision
        remaining = remaining & ~reason_ground
        reason_landed = remaining & landed
        remaining = remaining & ~reason_landed
        reason_drone = remaining & drone_collision
        remaining = remaining & ~reason_drone
        reason_out_of_bounds = remaining & out_of_bounds
        remaining = remaining & ~reason_out_of_bounds
        reason_timeout = remaining & timeout
        reason_timeout_near_miss = reason_timeout & (
            self.episode_min_target_distance <= self.timeout_near_miss_distance
        )
        reason_timeout_no_close = reason_timeout & ~reason_timeout_near_miss

        reason_code = torch.zeros_like(self.stats["termination_reason"])
        reason_code = torch.where(
            reason_capture,
            torch.full_like(reason_code, float(TERMINATION_CAPTURE)),
            reason_code,
        )
        reason_code = torch.where(
            reason_target_goal,
            torch.full_like(reason_code, float(TERMINATION_TARGET_GOAL)),
            reason_code,
        )
        reason_code = torch.where(
            reason_ground,
            torch.full_like(reason_code, float(TERMINATION_GROUND_COLLISION)),
            reason_code,
        )
        reason_code = torch.where(
            reason_drone,
            torch.full_like(reason_code, float(TERMINATION_DRONE_COLLISION)),
            reason_code,
        )
        reason_code = torch.where(
            reason_timeout_near_miss,
            torch.full_like(reason_code, float(TERMINATION_TIMEOUT_NEAR_MISS)),
            reason_code,
        )
        reason_code = torch.where(
            reason_timeout_no_close,
            torch.full_like(reason_code, float(TERMINATION_TIMEOUT_NO_CLOSE)),
            reason_code,
        )
        reason_code = torch.where(
            reason_out_of_bounds,
            torch.full_like(reason_code, float(TERMINATION_OUT_OF_BOUNDS)),
            reason_code,
        )
        reason_code = torch.where(
            reason_landed,
            torch.full_like(reason_code, float(TERMINATION_LANDED)),
            reason_code,
        )

        self.stats["termination_reason"].copy_(
            torch.where(fresh, reason_code, self.stats["termination_reason"])
        )
        one_hot = {
            "termination_capture": reason_capture,
            "termination_target_goal": reason_target_goal,
            "termination_ground_collision": reason_ground,
            "termination_drone_collision": reason_drone,
            "termination_timeout_near_miss": reason_timeout_near_miss,
            "termination_timeout_no_close": reason_timeout_no_close,
            "termination_out_of_bounds": reason_out_of_bounds,
            "termination_landed": reason_landed,
        }
        for key, mask in one_hot.items():
            self.stats[key].copy_(
                torch.logical_or(self.stats[key].bool(), mask).float()
            )

        self.stats["episode_min_target_distance"].copy_(
            torch.where(
                fresh,
                self.episode_min_target_distance,
                self.stats["episode_min_target_distance"],
            )
        )

        # Preserve the established terminal_* names for old analysis scripts.
        legacy_masks = {
            "terminal_capture": reason_capture,
            "terminal_goal": reason_target_goal,
            "terminal_floor_collision": reason_ground,
            "terminal_landed": reason_landed,
            "terminal_drone_collision": reason_drone,
            "terminal_out_of_bounds": reason_out_of_bounds,
            "terminal_timeout": reason_timeout,
            "success": reason_capture,
        }
        for key, mask in legacy_masks.items():
            self.stats[key].copy_(
                torch.logical_or(self.stats[key].bool(), mask).float()
            )

    def _sample_goal_defense_curriculum_positions(self, env_ids: torch.Tensor):
        num_envs = len(env_ids)
        goal_z = float(self.goal_region_center[2].item())
        stage_cfg = self.curriculum_stages[self.curriculum_stage]
        dist_min, dist_max = stage_cfg["distance_range"]
        target_x_min, target_x_max = self.curriculum_target_spawn_x
        target_y_min, target_y_max = self.curriculum_target_spawn_y

        target_pos = torch.zeros(num_envs, 1, 3, device=self.device)
        target_pos[..., 0].uniform_(min(target_x_min, target_x_max), max(target_x_min, target_x_max))
        target_pos[..., 1].uniform_(min(target_y_min, target_y_max), max(target_y_min, target_y_max))
        target_pos[..., 2].uniform_(goal_z - self.curriculum_target_z_jitter, goal_z + self.curriculum_target_z_jitter)
        target_pos[..., 2].clamp_(self.target_min_z, self.max_height - 0.1)

        drone_pos = torch.zeros(num_envs, self.num_agents, 3, device=self.device)
        if self.curriculum_pursuer_spawn_mode == "front_box":
            tx_min, tx_max = self.curriculum_front_box_target_x
            ty_min, ty_max = self.curriculum_front_box_target_y
            dx_min, dx_max = self.curriculum_front_box_drone_x
            dy_min, dy_max = self.curriculum_front_box_drone_y
            target_pos[..., 0].uniform_(min(tx_min, tx_max), max(tx_min, tx_max))
            target_pos[..., 1].uniform_(min(ty_min, ty_max), max(ty_min, ty_max))
            for env_idx in range(num_envs):
                target_z = target_pos[env_idx, 0, 2]
                placed = False
                for _ in range(self.curriculum_max_sample_attempts):
                    candidate = torch.zeros(self.num_agents, 3, device=self.device)
                    candidate[:, 0].uniform_(min(dx_min, dx_max), max(dx_min, dx_max))
                    candidate[:, 1].uniform_(min(dy_min, dy_max), max(dy_min, dy_max))
                    candidate[:, 2].uniform_(
                        target_z - self.curriculum_front_box_max_z_diff,
                        target_z + self.curriculum_front_box_max_z_diff,
                    )
                    candidate[:, 2].clamp_(0.3, self.max_height - 0.1)
                    if self.num_agents > 1:
                        pairwise = torch.cdist(candidate[:, :2], candidate[:, :2])
                        pairwise.fill_diagonal_(float("inf"))
                        separated = bool((pairwise >= self.curriculum_drone_min_separation).all())
                    else:
                        separated = True
                    if separated:
                        drone_pos[env_idx] = candidate
                        placed = True
                        break
                if not placed:
                    candidate = torch.zeros(self.num_agents, 3, device=self.device)
                    candidate[:, 0] = torch.linspace(min(dx_min, dx_max), max(dx_min, dx_max), self.num_agents, device=self.device)
                    candidate[:, 1] = torch.linspace(min(dy_min, dy_max), max(dy_min, dy_max), self.num_agents, device=self.device)
                    candidate[:, 2] = target_z
                    drone_pos[env_idx] = candidate
            return drone_pos, target_pos

        fallback_angle = (
            self.curriculum_goal_side_angle
            if self.curriculum_pursuer_spawn_mode == "goal_side_arc"
            else 0.5 * math.pi
        )
        fallback_offsets = torch.linspace(
            -fallback_angle, fallback_angle, self.num_agents, device=self.device
        ) if self.num_agents > 1 else torch.zeros(1, device=self.device)

        for env_idx in range(num_envs):
            target_xy = target_pos[env_idx, 0, :2]
            target_z = target_pos[env_idx, 0, 2]
            placed = False
            for _ in range(self.curriculum_max_sample_attempts):
                radii = torch.empty(self.num_agents, device=self.device).uniform_(dist_min, dist_max)
                if self.curriculum_pursuer_spawn_mode == "goal_side_arc":
                    goal_dir = safe_normalize(
                        (self.goal_region_center[:2] - target_xy).unsqueeze(0)
                    ).squeeze(0)
                    base_angle = torch.atan2(goal_dir[1], goal_dir[0])
                    angle_offsets = torch.linspace(
                        -self.curriculum_goal_side_angle,
                        self.curriculum_goal_side_angle,
                        self.num_agents,
                        device=self.device,
                    ) if self.num_agents > 1 else torch.zeros(1, device=self.device)
                    angle_jitter = torch.empty(self.num_agents, device=self.device).uniform_(
                        -self.curriculum_goal_side_angle_jitter,
                        self.curriculum_goal_side_angle_jitter,
                    )
                    angles = base_angle + angle_offsets + angle_jitter
                else:
                    angles = torch.empty(self.num_agents, device=self.device).uniform_(0.0, 2.0 * math.pi)
                z_offsets = torch.empty(self.num_agents, device=self.device).uniform_(
                    -self.curriculum_drone_z_jitter, self.curriculum_drone_z_jitter
                )
                candidate = torch.zeros(self.num_agents, 3, device=self.device)
                candidate[:, 0] = target_xy[0] + radii * torch.cos(angles)
                candidate[:, 1] = target_xy[1] + radii * torch.sin(angles)
                candidate[:, 2] = target_z + z_offsets

                in_arena = (candidate[:, 0] ** 2 + candidate[:, 1] ** 2) <= (self.arena_size - 0.10) ** 2
                valid_height = (candidate[:, 2] >= 0.3) & (candidate[:, 2] <= self.max_height - 0.1)
                if self.num_agents > 1:
                    pairwise = torch.cdist(candidate[:, :2], candidate[:, :2])
                    pairwise.fill_diagonal_(float("inf"))
                    separated = bool((pairwise >= self.curriculum_drone_min_separation).all())
                else:
                    separated = True
                if bool(in_arena.all() and valid_height.all()) and separated:
                    drone_pos[env_idx] = candidate
                    placed = True
                    break

            if not placed:
                goal_dir = safe_normalize(
                    (self.goal_region_center[:2] - target_xy).unsqueeze(0)
                ).squeeze(0)
                base_angle = torch.atan2(goal_dir[1], goal_dir[0])
                radii = torch.empty(self.num_agents, device=self.device).uniform_(dist_min, dist_max)
                jitter = torch.empty(self.num_agents, device=self.device).uniform_(-0.20, 0.20)
                angles = base_angle + fallback_offsets + jitter
                candidate = torch.zeros(self.num_agents, 3, device=self.device)
                candidate[:, 0] = target_xy[0] + radii * torch.cos(angles)
                candidate[:, 1] = target_xy[1] + radii * torch.sin(angles)
                candidate[:, 2] = target_z + torch.empty(self.num_agents, device=self.device).uniform_(
                    -self.curriculum_drone_z_jitter, self.curriculum_drone_z_jitter
                )
                candidate[:, 2].clamp_(0.3, self.max_height - 0.1)
                radial_norm = torch.norm(candidate[:, :2], dim=-1, keepdim=True)
                radial_scale = torch.clamp((self.arena_size - 0.10) / (radial_norm + 1e-6), max=1.0)
                candidate[:, :2] = candidate[:, :2] * radial_scale
                drone_pos[env_idx] = candidate

        return drone_pos, target_pos

    def _set_specs(self):        
        self.use_flight_context = bool(getattr(self.cfg.task, "actor_flight_context", False))
        reference_cfg = getattr(self.cfg.task, "trajectory_control", None)
        self.action_reference_dim = int(getattr(reference_cfg, "action_reference_dim", 6))
        reference_mode = str(getattr(reference_cfg, "reference_mode", "position_velocity"))
        if self.action_reference_dim not in (3, 6) or (self.action_reference_dim == 3 and reference_mode != "acceleration_sequence"):
            raise ValueError("Three controls require acceleration_sequence; legacy references require six.")
        drone_state_dim = self.drone.state_spec.shape.numel()
        self.time_encoding_dim = 1 if self.cfg.task.time_encoding else 0
        self.future_predcition_step = self.future_prediction_obs_steps
        self.history_step = self.cfg.task.history_step
        self.window_step = self.cfg.task.window_step
        state_self_dim = self.state_self_dim
        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, state_self_dim)),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, self.state_others_dim)),
            "cooperation": UnboundedContinuousTensorSpec((1, self.cooperation_dim)),
            "agent_history": UnboundedContinuousTensorSpec(
                (self.history_step, self.agent_history_dim)
            ),
            "current_pos": UnboundedContinuousTensorSpec(3),
            "current_vel": UnboundedContinuousTensorSpec(3),
            "prev_plan": UnboundedContinuousTensorSpec(
                (self.future_prediction_obs_steps, 6)
            ),
            "prev_plan_mask": UnboundedContinuousTensorSpec(
                self.future_prediction_obs_steps
            ),
            "gmm_weights": UnboundedContinuousTensorSpec(NUM_TARGET_TRAJECTORIES),
            "strategic_probs": UnboundedContinuousTensorSpec(NUM_STRATEGIC_INTENTS),
            "gmm_means": UnboundedContinuousTensorSpec(
                (NUM_TARGET_TRAJECTORIES, self.future_prediction_obs_steps, 3)
            ),
            "gmm_vars": UnboundedContinuousTensorSpec(
                (NUM_TARGET_TRAJECTORIES, self.future_prediction_obs_steps, 3)
            ),
        }).to(self.device)
        state_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, state_self_dim + self.cooperation_dim)),
        }).to(self.device)
        
        TP_spec = CompositeSpec({
            "TP_input": UnboundedContinuousTensorSpec((self.history_step, 1 + 3 + 3 + self.max_agents * 3)),
            "TP_groundtruth": UnboundedContinuousTensorSpec(3),
            "TP_done": UnboundedContinuousTensorSpec(1),
        }).to(self.device)
        prediction_spec = CompositeSpec({
            "pred_input": UnboundedContinuousTensorSpec(
                (self.future_prediction_obs_steps, self.predictor_obs_dim)
            ),
            "target_pos_world": UnboundedContinuousTensorSpec(3),
            "strategic_label": UnboundedContinuousTensorSpec(1),
            "valid": UnboundedContinuousTensorSpec(1),
            "strategic_probs": UnboundedContinuousTensorSpec(NUM_STRATEGIC_INTENTS),
            "gmm_weights": UnboundedContinuousTensorSpec(NUM_TARGET_TRAJECTORIES),
            "gmm_means": UnboundedContinuousTensorSpec(
                (NUM_TARGET_TRAJECTORIES, self.future_prediction_obs_steps, 3)
            ),
            "gmm_vars": UnboundedContinuousTensorSpec(
                (NUM_TARGET_TRAJECTORIES, self.future_prediction_obs_steps, 3)
            ),
        }).to(self.device)
        if self.use_flight_context:
            observation_spec["flight_context"] = UnboundedContinuousTensorSpec(6, device=self.device)
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": observation_spec.expand(self.drone.n),
                "state": state_spec,
                "TP": TP_spec,
                "prediction": prediction_spec,
            })
        }).expand(self.num_envs).to(self.device)
        action_spec = UnboundedContinuousTensorSpec(
            (self.drone.n, self.future_prediction_obs_steps, self.action_reference_dim),
            device=self.device,
        )
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": action_spec,
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((self.drone.n, 1)),                
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state"),
        )

        # stats and infos
        stats_spec = CompositeSpec({
            "success": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "terminal_reward": UnboundedContinuousTensorSpec(1),
            "goal_progress_reward": UnboundedContinuousTensorSpec(1),
            "capture_progress_reward": UnboundedContinuousTensorSpec(1),
            "catch_reward": UnboundedContinuousTensorSpec(1),
            "goal_penalty": UnboundedContinuousTensorSpec(1),
            "goal_reached": UnboundedContinuousTensorSpec(1),
            "terminal_capture": UnboundedContinuousTensorSpec(1),
            "terminal_goal": UnboundedContinuousTensorSpec(1),
            "terminal_floor_collision": UnboundedContinuousTensorSpec(1),
            "terminal_landed": UnboundedContinuousTensorSpec(1),
            "terminal_drone_collision": UnboundedContinuousTensorSpec(1),
            "terminal_out_of_bounds": UnboundedContinuousTensorSpec(1),
            "terminal_timeout": UnboundedContinuousTensorSpec(1),
            # termination_reason is one stable integer code per completed
            # episode; the following one-hot fields are scalar rates in logs.
            "termination_reason": UnboundedContinuousTensorSpec(1),
            "termination_capture": UnboundedContinuousTensorSpec(1),
            "termination_target_goal": UnboundedContinuousTensorSpec(1),
            "termination_ground_collision": UnboundedContinuousTensorSpec(1),
            "termination_drone_collision": UnboundedContinuousTensorSpec(1),
            "termination_timeout_near_miss": UnboundedContinuousTensorSpec(1),
            "termination_timeout_no_close": UnboundedContinuousTensorSpec(1),
            "termination_out_of_bounds": UnboundedContinuousTensorSpec(1),
            "termination_landed": UnboundedContinuousTensorSpec(1),
            "episode_min_target_distance": UnboundedContinuousTensorSpec(1),
            "speed_reward": UnboundedContinuousTensorSpec(1),
            "time_penalty": UnboundedContinuousTensorSpec(1),
            "coop_reward": UnboundedContinuousTensorSpec(1),
            "spread_reward": UnboundedContinuousTensorSpec(1),
            "close_reward": UnboundedContinuousTensorSpec(1),
            "role_reward": UnboundedContinuousTensorSpec(1),
            "phi_team": UnboundedContinuousTensorSpec(1),
            "phi_block": UnboundedContinuousTensorSpec(1),
            "phi_pressure": UnboundedContinuousTensorSpec(1),
            "phi_spread": UnboundedContinuousTensorSpec(1),
            "d_i_mean": UnboundedContinuousTensorSpec(1),
            "d_i_min": UnboundedContinuousTensorSpec(1),
            "urgent_block_rate": UnboundedContinuousTensorSpec(1),
            "n_agents_ahead_mean": UnboundedContinuousTensorSpec(1),
            "collision_penalty": UnboundedContinuousTensorSpec(1),
            "separation_penalty": UnboundedContinuousTensorSpec(1),
            "collision_wall": UnboundedContinuousTensorSpec(1),
            "collision_floor": UnboundedContinuousTensorSpec(1),
            "collision_drone": UnboundedContinuousTensorSpec(1),
            "smoothness_reward": UnboundedContinuousTensorSpec(1),
            "landed_penalty": UnboundedContinuousTensorSpec(1),
            "any_landed": UnboundedContinuousTensorSpec(1),
            "smoothness_mean": UnboundedContinuousTensorSpec(1),
            "smoothness_max": UnboundedContinuousTensorSpec(1),
            "first_capture_step": UnboundedContinuousTensorSpec(1),
            "curriculum_stage": UnboundedContinuousTensorSpec(1),
            "curriculum_success_ema": UnboundedContinuousTensorSpec(1),
            "curriculum_capture_step_ema": UnboundedContinuousTensorSpec(1),
            "curriculum_pursuer_speed": UnboundedContinuousTensorSpec(1),
            "curriculum_target_speed": UnboundedContinuousTensorSpec(1),
            "target_max_accel": UnboundedContinuousTensorSpec(1),
            "target_initial_speed": UnboundedContinuousTensorSpec(1),
            "sum_detect_step": UnboundedContinuousTensorSpec(1),
            "return": UnboundedContinuousTensorSpec(1),
            "action_error_order1_mean": UnboundedContinuousTensorSpec(1),
            "action_error_order1_max": UnboundedContinuousTensorSpec(1),
            "target_predicted_error": UnboundedContinuousTensorSpec(1),
            "distance_threshold_L": UnboundedContinuousTensorSpec(1),
            "out_of_arena": UnboundedContinuousTensorSpec(1),
            "smoothness_coef": UnboundedContinuousTensorSpec(1),
            "pursuer_collisions_count": UnboundedContinuousTensorSpec(1),
            "strategic_goal_rush": UnboundedContinuousTensorSpec(1),
            "strategic_evade": UnboundedContinuousTensorSpec(1),
            # ── minimal_v3 reward stats ──
            "m3_progress": UnboundedContinuousTensorSpec(1),
            "m3_progress_best_delta": UnboundedContinuousTensorSpec(1),
            "m3_safety_pair": UnboundedContinuousTensorSpec(1),
            "m3_safety_env": UnboundedContinuousTensorSpec(1),
            "m3_control": UnboundedContinuousTensorSpec(1),
            "m3_time": UnboundedContinuousTensorSpec(1),
            "m3_terminal": UnboundedContinuousTensorSpec(1),
            "m3_dense_raw": UnboundedContinuousTensorSpec(1),
            "m3_reward": UnboundedContinuousTensorSpec(1),
            "episode_duration_s": UnboundedContinuousTensorSpec(1),
            "target_maneuver_switches": UnboundedContinuousTensorSpec(1),
            "target_tactical_switches": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        prev_action_spec = UnboundedContinuousTensorSpec(
            (self.drone.n, self.future_prediction_obs_steps, self.action_reference_dim),
            device=self.device,
        )
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "target_state": UnboundedContinuousTensorSpec((1, 6), device=self.device),
            # Target telemetry is kept outside the policy observation.  It is
            # used to verify that the commanded manoeuvres are also realised
            # by the multirotor dynamics, rather than inferred from animation.
            "target_acc_cmd": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            "target_max_speed": UnboundedContinuousTensorSpec((1,), device=self.device),
            "target_max_accel": UnboundedContinuousTensorSpec((1,), device=self.device),
            "target_initial_speed": UnboundedContinuousTensorSpec((1,), device=self.device),
            "pursuer_motor_fraction": UnboundedContinuousTensorSpec((self.drone.n,), device=self.device),
            "target_motor_fraction": UnboundedContinuousTensorSpec((1,), device=self.device),
            "target_tactical_intent": UnboundedContinuousTensorSpec((1,), device=self.device),
            "target_maneuver_id": UnboundedContinuousTensorSpec((1,), device=self.device),
            "target_sine_amplitude_m": UnboundedContinuousTensorSpec((1,), device=self.device),
            "target_sine_period_s": UnboundedContinuousTensorSpec((1,), device=self.device),
            "target_sine_active": UnboundedContinuousTensorSpec((1,), device=self.device),
            "target_sine_reference_offset_m": UnboundedContinuousTensorSpec((1,), device=self.device),
            "prev_action": prev_action_spec,
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()
        
    def _design_scene(self): # for render
        self.use_local_usd = self.cfg.use_local_usd
        self.num_agents = self.cfg.task.num_agents
        self.drone_detect_radius = self.cfg.task.drone_detect_radius
        self.target_detect_radius = self.cfg.task.target_detect_radius
        self.catch_radius = self.cfg.task.catch_radius
        self.arena_size = self.cfg.task.arena_size
        self.max_height = self.cfg.task.max_height
        self.scenario_flag = self.cfg.task.scenario_flag
        self.boundary = self.arena_size - 0.1
        self.use_TP_net = False

        # init
        drone_pos = torch.tensor([
                            [0.6000,  0.0000, 0.5],
                            [0.8000,  0.0000, 0.5],
                            [0.8000, -0.2000, 0.5],
                            [0.8000,  0.2000, 0.5],
                        ], device=self.device)[:self.num_agents]
        target_pos = torch.tensor([
                            [-0.8000,  0.0000, 0.5],
                        ], device=self.device)
        if self.scenario_flag != "goal_defense":
            raise NotImplementedError(
                f"Obstacle-free build only supports scenario_flag='goal_defense', got {self.scenario_flag!r}."
            )

        # init drone
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        cfg.rigid_props.max_linear_velocity = self.cfg.task.v_drone
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        capability = getattr(self.cfg.task, "flight_capability", None)
        configure_aircraft(self.drone, getattr(capability, "pursuer_thrust_to_weight", None))
        self.drone.spawn(drone_pos)

        target_physics_max_velocity = float(
            getattr(
                self.cfg.task,
                "target_physics_max_velocity",
                max(5.0, self.cfg.task.v_drone * self.cfg.task.v_prey * 3.0),
            )
        )
        if self.target_dynamics_mode == "uav":
            target_model_name = str(getattr(self.cfg.task, "target_drone_model", self.cfg.task.drone_model))
            target_model = MultirotorBase.REGISTRY[target_model_name]
            target_cfg = target_model.cfg_cls(force_sensor=False)
            target_cfg.rigid_props.max_linear_velocity = target_physics_max_velocity
            self.target: MultirotorBase = target_model(name="target_drone", cfg=target_cfg)
            configure_aircraft(self.target, getattr(capability, "target_thrust_to_weight", None))
            self.target.spawn(
                target_pos,
                prim_paths=["/World/envs/env_0/target_drone_0"],
            )
        else:
            # init prey as a bounded double-integrator point mass.
            objects.DynamicSphere(
                prim_path="/World/envs/env_0/target",
                name="target",
                translation=target_pos,
                radius=0.05,
                color=torch.tensor([1.0, 0.0, 0.0]),
                mass=1.0,
            )
            kit_utils.set_rigid_body_properties(
                prim_path="/World/envs/env_0/target",
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=target_physics_max_velocity,
            )
    
        # kit_utils.create_ground_plane(
        #     "/World/defaultGroundPlane",
        #     static_friction=1.0,
        #     dynamic_friction=1.0,
        #     restitution=0.0,
        # )

        if self.use_local_usd:
            # use local usd resources
            usd_path = os.path.join(os.path.dirname(__file__), os.pardir, "assets", "default_environment.usd")
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                usd_path=usd_path
            )
        else:
            # use online usd resources
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )

        return ["/World/defaultGroundPlane"]

    def _sample_target_episode_dynamics(
        self, env_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one hidden dynamics profile and initial velocity per episode.

        The limits are deliberately telemetry-only rather than privileged
        policy observations.  The pursuers must infer the target capability
        from its observed history, which matches real deployment.
        """
        n = int(env_ids.numel())
        speed_limit = torch.empty(n, 1, 1, device=self.device).uniform_(
            self.target_max_speed_range[0], self.target_max_speed_range[1]
        )
        accel_limit = torch.empty(n, 1, 1, device=self.device).uniform_(
            self.target_max_accel_range[0], self.target_max_accel_range[1]
        )
        initial_speed = torch.empty(n, 1, device=self.device)
        if self.target_initial_speed_range[0] == self.target_initial_speed_range[1]:
            initial_speed.fill_(self.target_initial_speed_range[0])
        else:
            initial_speed.uniform_(
                self.target_initial_speed_range[0],
                self.target_initial_speed_range[1],
            )
        initial_speed = torch.minimum(initial_speed, speed_limit.squeeze(-1))

        yaw = torch.empty(n, 1, device=self.device).uniform_(-math.pi, math.pi)
        pitch_deg = torch.empty(n, 1, device=self.device)
        if self.target_initial_pitch_deg_range[0] == self.target_initial_pitch_deg_range[1]:
            pitch_deg.fill_(self.target_initial_pitch_deg_range[0])
        else:
            pitch_deg.uniform_(
                self.target_initial_pitch_deg_range[0],
                self.target_initial_pitch_deg_range[1],
            )
        pitch = torch.deg2rad(pitch_deg)
        cos_pitch = torch.cos(pitch)
        direction = torch.cat(
            [
                cos_pitch * torch.cos(yaw),
                cos_pitch * torch.sin(yaw),
                torch.sin(pitch),
            ],
            dim=-1,
        )
        initial_velocity = direction * initial_speed

        self.target_max_speed[env_ids] = speed_limit
        self.target_max_accel[env_ids] = accel_limit
        self.target_initial_speed[env_ids] = initial_speed
        return initial_velocity, yaw

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        if self.target_dynamics_mode == "uav":
            self.target._reset_idx(env_ids)

        use_goal_defense_curriculum = (
            self.curriculum_enabled
            and self.scenario_flag == "goal_defense"
            and (self.training or not self.curriculum_eval_uses_fixed_layout)
        )
        use_goal_defense_random_profile = (
            (not self.curriculum_enabled)
            and self.scenario_flag == "goal_defense"
            and self.curriculum_pursuer_spawn_mode in ("front_box", "goal_side_arc")
        )
        use_goal_defense_random_reset = use_goal_defense_curriculum or use_goal_defense_random_profile

        # init, fixed xy and randomize z
        if use_goal_defense_random_reset:
            drone_pos, target_pos = self._sample_goal_defense_curriculum_positions(env_ids)
        elif not self.use_eval:
            # random pos
            drone_pos = self.init_drone_pos_dist.sample((*env_ids.shape, self.num_agents))
            target_pos =  self.init_target_pos_dist.sample((*env_ids.shape, 1))
            drone_pos_z = self.init_drone_pos_dist_z.sample((*env_ids.shape, self.num_agents))
            target_pos_z = self.init_target_pos_dist_z.sample((*env_ids.shape, 1))
            drone_pos = torch.concat([drone_pos, drone_pos_z], dim=-1)
            target_pos = torch.concat([target_pos, target_pos_z], dim=-1)
        else:
            goal_z = float(self.goal_region_center[2].item())
            drone_pos = torch.tensor(
                [
                    [1.2000, 0.0000, goal_z],
                    [1.5000, 0.6000, goal_z],
                    [1.5000, -0.6000, goal_z],
                    [1.8000, 0.0000, goal_z],
                ],
                device=self.device,
            )[: self.num_agents]
            target_pos = torch.tensor(
                [[-1.6000, 0.0000, goal_z]],
                device=self.device,
            )

        # Add position noise to fixed initial positions for exploration diversity
        if self.use_eval and not use_goal_defense_random_reset:
            pos_noise_xy = 0.15  # ±0.15m noise on x,y
            pos_noise_z = 0.1   # ±0.10m noise on z
            drone_noise = torch.zeros(len(env_ids), self.num_agents, 3, device=self.device)
            drone_noise[..., :2].uniform_(-pos_noise_xy, pos_noise_xy)
            drone_noise[..., 2].uniform_(-pos_noise_z, pos_noise_z)
            drone_pos = drone_pos.unsqueeze(0).expand(len(env_ids), -1, -1) + drone_noise
            # clamp to arena bounds
            drone_pos[..., :2].clamp_(-self.arena_size + 0.05, self.arena_size - 0.05)
            drone_pos[..., 2].clamp_(0.2, self.max_height - 0.1)

            target_noise = torch.zeros(len(env_ids), 1, 3, device=self.device)
            target_noise[..., :2].uniform_(-pos_noise_xy, pos_noise_xy)
            target_noise[..., 2].uniform_(-pos_noise_z, pos_noise_z)
            target_pos = target_pos.unsqueeze(0).expand(len(env_ids), -1, -1) + target_noise
            target_pos[..., :2].clamp_(-self.arena_size + 0.05, self.arena_size - 0.05)
            target_pos[..., 2].clamp_(self.target_min_z, self.max_height - 0.1)

        # drone_pos = self.init_drone_pos_dist.sample((*env_ids.shape, self.num_agents))
        rpy = self.init_rpy_dist.sample((*env_ids.shape, self.num_agents))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        drone_init_velocities = torch.zeros(len(env_ids) * self.num_agents, 6, device=self.device)
        self.drone.set_velocities(drone_init_velocities, env_ids)

        target_initial_velocity, target_initial_yaw = (
            self._sample_target_episode_dynamics(env_ids)
        )
        target_rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        target_rpy[..., 2] = target_initial_yaw
        target_rot = euler_to_quaternion(target_rpy)
        self.target.set_world_poses(
            positions=target_pos + self.envs_positions[env_ids].unsqueeze(1),
            orientations=target_rot,
            env_indices=env_ids,
        )
        target_init_velocities = torch.zeros(len(env_ids), 6, device=self.device)
        target_init_velocities[..., :3] = target_initial_velocity
        self.target.set_velocities(target_init_velocities, env_ids)
        self.target_acc_cmd[env_ids] = 0.0
        self.target_vel_cmd[env_ids] = target_initial_velocity.unsqueeze(1)
        self.target_pid_reset[env_ids] = True
        self.pursuer_pid_reset[env_ids] = True
        self.target_tactical_intent[env_ids] = TARGET_INTENT_GOAL_RUSH
        self.target_maneuver_id[env_ids] = TARGET_MANEUVER_STRAIGHT
        self.target_maneuver_timer[env_ids] = 0
        self.target_goal_priority_active[env_ids] = False
        self.target_controller_stage[env_ids] = 0
        if self.target_sine_policy is not None:
            self.target_sine_policy.reset(env_ids)

        # reset stats
        self.stats[env_ids] = 0.
        self.episode_min_target_distance[env_ids] = float("inf")
        self.stats['first_capture_step'][env_ids] = self.max_episode_length
        self.stats["curriculum_stage"][env_ids] = float(self.curriculum_stage + 1)
        self.stats["curriculum_success_ema"][env_ids] = float(self.curriculum_success_ema)
        self.stats["curriculum_capture_step_ema"][env_ids] = float(self.curriculum_capture_step_ema)
        self.stats["curriculum_pursuer_speed"][env_ids] = float(self.current_pursuer_speed)
        self.stats["curriculum_target_speed"][env_ids] = self.target_max_speed[
            env_ids
        ].squeeze(-1)
        self.stats["target_max_accel"][env_ids] = self.target_max_accel[
            env_ids
        ].squeeze(-1)
        self.stats["target_initial_speed"][env_ids] = self.target_initial_speed[
            env_ids
        ]

        # init prev_actions
        self.info['prev_action'][env_ids] = 0.0
        self.prev_actions[env_ids] = self.info['prev_action'][env_ids].clone()

        # reset prev_target_dist for distance progress reward
        # drone_pos: [len(env_ids), num_agents, 3], target_pos: [len(env_ids), 1, 3] — broadcasts naturally
        self.prev_target_dist[env_ids] = torch.norm(target_pos - drone_pos, dim=-1)
        self.m3_prev_target_dist[env_ids] = self.prev_target_dist[env_ids].clone()
        self.m3_action_magnitude[env_ids] = 0.0
        self.m3_action_delta[env_ids] = 0.0
        self.prev_expert2_role_dist[env_ids] = 0.0
        self.prev_expert2_role_dist_ready[env_ids] = False
        self.prev_expert2_close_triggered[env_ids] = False
        self.expert2_prev_assignment[env_ids] = -1
        self.expert2_assignment_ready[env_ids] = False
        self.tp_history_buffer[env_ids] = 0.0
        self.tp_history_initialized[env_ids] = False
        self.predictor_strategic_probs[env_ids] = 0.0
        self.predictor_strategic_probs[env_ids, 0] = 1.0
        self.predictor_gmm_weights[env_ids] = 0.0
        self.predictor_gmm_weights[env_ids, 0] = 1.0
        self.predictor_gmm_means[env_ids] = 0.0
        self.predictor_gmm_vars[env_ids] = 1.0
        self.predictor_history[env_ids] = 0.0
        self.predictor_initialized[env_ids] = False
        self.predictor_raw_history[env_ids] = 0.0
        self.predictor_raw_initialized[env_ids] = False
        self.predictor_sample_valid[env_ids] = 0.0
        self.predictor_target_pos_world[env_ids] = 0.0
        self.predictor_strategic_label[env_ids] = 0
        self.agent_history_buffer[env_ids] = 0.0
        self.agent_history_initialized[env_ids] = False
        self.previous_plan[env_ids] = 0.0
        self.previous_plan_mask[env_ids] = 0.0
        self._expert2_cached_active_waypoint = None
        self._expert2_cached_close_triggered = None
        goal_dist = point_to_cylinder_distance(
            target_pos.squeeze(1), self.goal_region_center, self.goal_region_radius, self.goal_region_height
        )
        self.prev_goal_dist[env_ids] = goal_dist.unsqueeze(-1)
        
        if self.use_eval and self._should_render(0):
            self._draw_court_circle()
        
        self._advance_physics_after_reset()

    def _advance_physics_after_reset(self):
        if self.advance_physics_on_reset:
            self.sim.step(self._should_render(0))

    def _compute_target_desired_acc(self) -> torch.Tensor:
        if self.target_motion_mode == "goal_sine":
            return self._compute_goal_sine_desired_acc()
        forces_target = self._get_dummy_policy_prey()
        return self.target_max_accel * forces_target / (
            torch.norm(forces_target, dim=-1, keepdim=True) + 1e-5
        )

    def _compute_goal_sine_desired_acc(self) -> torch.Tensor:
        """Use the same acceleration filter and physical target controller as before."""
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses(False))
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        goal_position = closest_point_on_cylinder(
            target_pos[:, 0], self.goal_region_center,
            self.goal_region_radius, self.goal_region_height)
        nearest_distance = (drone_pos - target_pos).norm(dim=-1).min(dim=1).values
        sine = self.target_sine_policy
        desired_velocity = sine.desired_velocity(
            target_pos[:, 0], goal_position, nearest_distance,
            self.target_max_speed.reshape(-1), float(self.dt)).unsqueeze(1)
        # UAV mode integrates a velocity reference; point-mass mode integrates
        # actual velocity. Feedback must use the state integrated by that mode.
        velocity = (self.target_vel_cmd if self.target_dynamics_mode == "uav"
                    else self.target.get_velocities()[..., :3])
        desired_acc = (self.target_uav_velocity_gain * (desired_velocity - velocity)
                       + self.target_velocity_damping * velocity)
        # Retain arena/floor protection, but NOT the old pursuer repulsion,
        # look-ahead manoeuvre optimisation, goal-priority or sprint override.
        desired_acc += self.target_max_accel * self._compute_target_boundary_force(target_pos)
        previous_maneuver = self.target_maneuver_id.clone()
        previous_tactical = self.target_tactical_intent.clone()
        self.target_tactical_intent.copy_(torch.where(
            sine.active, TARGET_INTENT_EVADE, TARGET_INTENT_GOAL_RUSH))
        self.target_maneuver_id.copy_(torch.where(
            sine.active, TARGET_MANEUVER_SINE, TARGET_MANEUVER_STRAIGHT))
        self.target_goal_priority_active.zero_()
        self.target_maneuver_timer.zero_()
        self.stats["target_maneuver_switches"].add_(
            (self.target_maneuver_id != previous_maneuver).float().unsqueeze(-1))
        self.stats["target_tactical_switches"].add_(
            (self.target_tactical_intent != previous_tactical).float().unsqueeze(-1))
        # Unlike the legacy force-direction law, preserve small magnitudes.
        # Always normalising to max acceleration would destroy a sine reference.
        return clip_vector_norm(desired_acc, self.target_max_accel).float()

    def _sample_target_maneuver(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        n = int(env_ids.numel())
        self.target_maneuver_id[env_ids] = torch.randint(
            0, TARGET_NUM_MANEUVERS, (n,), device=self.device
        )
        self.target_maneuver_timer[env_ids] = torch.randint(
            self.target_maneuver_min_steps,
            self.target_maneuver_max_steps + 1,
            (n,),
            device=self.device,
        )

    def _compute_target_goal_force(
        self, target_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        goal_center = self.goal_region_center.view(1, 1, 3)
        goal_closest = closest_point_on_cylinder(
            target_pos, goal_center, self.goal_region_radius, self.goal_region_height
        )
        goal_offset = goal_closest - target_pos
        goal_dist = torch.norm(goal_offset, dim=-1, keepdim=True)
        goal_force = goal_offset / (goal_dist + 1e-5)
        return goal_force, goal_dist.squeeze(-1), goal_offset

    def _compute_target_boundary_force(self, target_pos: torch.Tensor) -> torch.Tensor:
        force_region = torch.zeros_like(target_pos)
        target_origin_dist = torch.norm(target_pos[..., :2], dim=-1)
        force_r_xy_direction = -target_pos[..., :2] / (
            target_origin_dist.unsqueeze(-1) + 1e-5
        )
        out_of_arena = (
            target_pos[..., 0] ** 2 + target_pos[..., 1] ** 2 > self.arena_size**2
        )
        self.stats["out_of_arena"].copy_(
            torch.logical_or(
                self.stats["out_of_arena"].bool(), out_of_arena
            ).float()
        )

        force_region[..., 0] = (
            out_of_arena.float() * force_r_xy_direction[..., 0] * (1 / 1e-5)
            + (~out_of_arena).float()
            * force_r_xy_direction[..., 0]
            * (1 / ((self.arena_size - target_origin_dist) + 1e-5))
        )
        force_region[..., 1] = (
            out_of_arena.float() * force_r_xy_direction[..., 1] * (1 / 1e-5)
            + (~out_of_arena).float()
            * force_r_xy_direction[..., 1]
            * (1 / ((self.arena_size - target_origin_dist) + 1e-5))
        )

        higher_than_z = target_pos[..., 2] > self.max_height
        force_region[..., 2] = higher_than_z.float() * (-1 / 1e-5) + (
            ~higher_than_z
        ).float() * -(
            self.max_height - target_pos[..., 2]
        ) / ((self.max_height - target_pos[..., 2]) ** 2 + 1e-5)
        lower_than_ground = target_pos[..., 2] < self.target_min_z
        force_region[..., 2] += lower_than_ground.float() * (1 / 1e-5) + (
            ~lower_than_ground
        ).float() * -(
            self.target_min_z - target_pos[..., 2]
        ) / ((self.target_min_z - target_pos[..., 2]) ** 2 + 1e-5)
        return force_region

    def _target_maneuver_components(
        self,
        target_vel: torch.Tensor,
        goal_force: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forward = safe_normalize(target_vel)
        goal_forward = safe_normalize(goal_force)
        speed = torch.norm(target_vel, dim=-1, keepdim=True)
        use_goal_frame = speed < self.target_maneuver_speed_threshold
        forward = torch.where(use_goal_frame, goal_forward, forward)
        forward = safe_normalize(forward)

        world_up = torch.zeros_like(forward)
        world_up[..., 2] = 1.0
        lateral = torch.cross(world_up, forward, dim=-1)
        lateral_norm = torch.norm(lateral, dim=-1, keepdim=True)
        goal_horizontal = project_to_plane(goal_forward, world_up)
        goal_horizontal = safe_normalize(goal_horizontal)
        fallback_lateral = torch.cross(world_up, goal_horizontal, dim=-1)
        fallback_lateral = safe_normalize(fallback_lateral)
        lateral = torch.where(lateral_norm > 1e-4, lateral, fallback_lateral)
        lateral = safe_normalize(lateral)

        forward_component = (
            self.target_maneuver_force_coef
            * self.target_maneuver_forward_scale
            * forward
        )
        lateral_component = self.target_maneuver_force_coef * lateral
        vertical_component = (
            self.target_maneuver_force_coef
            * self.target_maneuver_vertical_scale
            * world_up
        )
        return forward_component, lateral_component, vertical_component

    def _compute_target_maneuver_candidates(
        self,
        target_vel: torch.Tensor,
        goal_force: torch.Tensor,
    ) -> torch.Tensor:
        forward_component, lateral_component, vertical_component = self._target_maneuver_components(
            target_vel,
            goal_force,
        )
        return torch.stack(
            [
                forward_component,
                lateral_component,
                -lateral_component,
                vertical_component,
                -vertical_component,
                lateral_component + vertical_component,
                lateral_component - vertical_component,
                -lateral_component + vertical_component,
                -lateral_component - vertical_component,
            ],
            dim=1,
        )

    def _compute_target_maneuver_force(
        self,
        target_vel: torch.Tensor,
        goal_force: torch.Tensor,
    ) -> torch.Tensor:
        candidates = self._compute_target_maneuver_candidates(target_vel, goal_force)
        maneuver_force = torch.zeros_like(target_vel)
        ids = self.target_maneuver_id
        for maneuver_id in range(TARGET_NUM_MANEUVERS):
            maneuver_force[ids == maneuver_id] = candidates[ids == maneuver_id, maneuver_id]
        return maneuver_force.unsqueeze(1)

    def _select_target_optimal_maneuver_force(
        self,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        drone_pos: torch.Tensor,
        drone_vel: torch.Tensor,
        base_force: torch.Tensor,
        goal_force: torch.Tensor,
        goal_dist_now: torch.Tensor,
        max_accel: torch.Tensor,
        max_speed: torch.Tensor,
        env_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Only evaluate nine candidates when the physical hold timer expires.
        # Reconstruct the held bias in the current velocity frame every step.
        if env_ids is not None:
            refresh = self.target_maneuver_timer[env_ids] <= 0
            candidates = self._compute_target_maneuver_candidates(target_vel.squeeze(1), goal_force.squeeze(1))
            chosen = self.target_maneuver_id[env_ids].clone()
            if bool(refresh.any()):
                _, fresh_id = self._select_target_optimal_maneuver_force(
                    target_pos[refresh], target_vel[refresh], drone_pos[refresh], drone_vel[refresh],
                    base_force[refresh], goal_force[refresh], goal_dist_now[refresh],
                    max_accel[refresh], max_speed[refresh], env_ids=None)
                chosen[refresh] = fresh_id
                self.target_maneuver_timer[env_ids[refresh]] = torch.randint(
                    self.target_maneuver_min_steps, self.target_maneuver_max_steps + 1,
                    (int(refresh.sum()),), device=self.device)
            return candidates[torch.arange(len(env_ids), device=self.device), chosen].unsqueeze(1), chosen
        target_pos_flat = target_pos.squeeze(1)
        target_vel_flat = target_vel.squeeze(1)
        candidates = self._compute_target_maneuver_candidates(
            target_vel_flat,
            goal_force.squeeze(1),
        )
        base_force = base_force.reshape(-1, 1, 3)
        total_force = base_force + candidates
        desired_acc = max_accel * safe_normalize(total_force)

        lookahead_dt = max(float(self.target_maneuver_lookahead_dt), float(self.dt))
        pos = target_pos_flat.unsqueeze(1).expand(-1, TARGET_NUM_MANEUVERS, -1).clone()
        vel = target_vel_flat.unsqueeze(1).expand(-1, TARGET_NUM_MANEUVERS, -1).clone()
        min_pursuer_dist = torch.full(
            (target_pos_flat.shape[0], TARGET_NUM_MANEUVERS),
            float("inf"),
            dtype=target_pos_flat.dtype,
            device=target_pos_flat.device,
        )
        height_penalty = torch.zeros_like(min_pursuer_dist)
        boundary_penalty = torch.zeros_like(min_pursuer_dist)

        pursuer_pos0 = drone_pos.unsqueeze(1)
        pursuer_vel = drone_vel.unsqueeze(1)
        for step_idx in range(self.target_maneuver_lookahead_steps):
            vel = vel + (
                desired_acc - self.target_velocity_damping * vel
            ) * lookahead_dt
            vel = clip_vector_norm(vel, max_speed)
            pos = pos + vel * lookahead_dt
            pursuer_pos = pursuer_pos0 + pursuer_vel * ((step_idx + 1) * lookahead_dt)
            dist = torch.norm(
                pos.unsqueeze(2) - pursuer_pos,
                dim=-1,
            )
            min_pursuer_dist = torch.minimum(min_pursuer_dist, dist.min(dim=-1).values)
            low_height = torch.relu((self.target_min_z + 0.10) - pos[..., 2])
            high_height = torch.relu(pos[..., 2] - (self.max_height - 0.10))
            arena_excess = torch.relu(
                torch.norm(pos[..., :2], dim=-1) - (self.arena_size - 0.05)
            )
            height_penalty = torch.maximum(height_penalty, low_height + high_height)
            boundary_penalty = torch.maximum(boundary_penalty, arena_excess)

        goal_dist_next = point_to_cylinder_distance(
            pos.reshape(-1, 3),
            self.goal_region_center,
            self.goal_region_radius,
            self.goal_region_height,
        ).view(pos.shape[0], TARGET_NUM_MANEUVERS)
        goal_progress = goal_dist_now.reshape(-1, 1) - goal_dist_next
        goal_axis = safe_normalize(goal_force.squeeze(1)).unsqueeze(1)
        lateral_disp = torch.norm(
            project_to_plane(pos - target_pos_flat.unsqueeze(1), goal_axis),
            dim=-1,
        )
        speed_axis = safe_normalize(target_vel_flat).unsqueeze(1)
        speed_axis = torch.where(
            torch.norm(target_vel_flat, dim=-1, keepdim=True).unsqueeze(1)
            > self.target_maneuver_speed_threshold,
            speed_axis,
            goal_axis,
        )
        turn_score = torch.norm(
            project_to_plane(desired_acc, speed_axis), dim=-1
        ) / max_accel.squeeze(-1).clamp_min(1e-6)
        utility = (
            self.target_opt_maneuver_evade_weight * min_pursuer_dist
            + self.target_opt_maneuver_goal_weight * goal_progress
            + self.target_opt_maneuver_lateral_weight * lateral_disp
            + self.target_opt_maneuver_turn_weight * turn_score
            - self.target_opt_maneuver_height_weight * height_penalty
            - self.target_opt_maneuver_boundary_weight * boundary_penalty
        )
        best_id = utility.argmax(dim=-1)
        if env_ids is not None:
            refresh = self.target_maneuver_timer[env_ids] <= 0
            best_id = torch.where(refresh, best_id, self.target_maneuver_id[env_ids])
            durations = torch.randint(self.target_maneuver_min_steps, self.target_maneuver_max_steps + 1,
                                      (len(env_ids),), device=self.device)
            self.target_maneuver_timer[env_ids] = torch.where(refresh, durations, self.target_maneuver_timer[env_ids])
        best_force = candidates[
            torch.arange(candidates.shape[0], device=candidates.device),
            best_id,
        ].unsqueeze(1)
        return best_force, best_id

    def _step_point_mass_target(self):
        target_vel = self.target.get_velocities()
        target_lin_vel = target_vel[..., :3]
        desired_acc = self._compute_target_desired_acc()
        accel_alpha = min(1.0, self.dt / max(self.target_command_tau, self.dt))
        self.target_acc_cmd.lerp_(desired_acc, accel_alpha)
        next_target_vel = target_lin_vel + (
            self.target_acc_cmd - self.target_velocity_damping * target_lin_vel
        ) * self.dt
        next_target_vel = clip_vector_norm(next_target_vel, self.target_max_speed)
        target_vel[..., :3] = next_target_vel
        target_vel[..., 3:] = 0.0
        self.target.set_velocities(target_vel.type(torch.float32), self.env_ids)

    def _step_uav_target(self):
        if self.target_controller is None:
            raise RuntimeError("target_dynamics_mode='uav' requires a target_controller.")

        target_state = self.target.get_state()[..., :13]
        target_vel_actual = target_state[..., 7:10]
        desired_acc = self._compute_target_desired_acc()
        accel_alpha = min(1.0, self.dt / max(self.target_command_tau, self.dt))
        self.target_acc_cmd.lerp_(desired_acc, accel_alpha)

        prev_vel_cmd = self.target_vel_cmd.clone()
        target_vel_cmd = self.target_vel_cmd + (
            self.target_acc_cmd - self.target_velocity_damping * self.target_vel_cmd
        ) * self.dt
        self.target_vel_cmd.copy_(clip_vector_norm(target_vel_cmd, self.target_max_speed))

        target_acc_ff = (self.target_vel_cmd - prev_vel_cmd) / max(float(self.dt), 1e-6)
        target_acc_ff = clip_vector_norm(target_acc_ff, self.target_max_accel)
        target_acc_ff = target_acc_ff * self.target_uav_acc_feedforward_scale

        goal_vec = self.goal_region_center.view(1, 1, 3) - target_state[..., :3]
        pidrate_action = self._target_velocity_to_pidrate_action(
            desired_vel=self.target_vel_cmd,
            current_vel=target_vel_actual,
            current_quat=target_state[..., 3:7],
            feedforward_acc=target_acc_ff,
            heading_hint=goal_vec,
        )
        target_rate_norm, target_thrust_norm = pidrate_action.split([3, 1], dim=-1)
        target_rate = target_rate_norm * 180.0 * float(self.target_controller.target_clip)
        target_thrust_ratio = torch.clamp(
            (target_thrust_norm + 1.0) / 2.0,
            min=float(self.target_controller.min_thrust_ratio),
            max=float(self.target_controller.max_thrust_ratio),
        )
        target_thrust = target_thrust_ratio * 2**16

        cmds, _ = self.target_controller(
            target_state,
            target_rate=target_rate,
            target_thrust=target_thrust,
            reset_pid=self.target_pid_reset,
        )
        self.target_pid_reset.fill_(False)
        torch.nan_to_num_(cmds, 0.0)
        cmds = bounded_motor_commands(cmds, self.thrust_fraction)
        self.target_motor_fraction.copy_(((cmds + 1.0) / 2.0).max(-1).values)
        self.target.apply_action(cmds)

    def _target_velocity_to_pidrate_action(
        self,
        desired_vel: torch.Tensor,
        current_vel: torch.Tensor,
        current_quat: torch.Tensor,
        feedforward_acc: torch.Tensor,
        heading_hint: torch.Tensor,
    ) -> torch.Tensor:
        desired_vel = clip_vector_norm(desired_vel, self.target_max_speed)
        vel_error = desired_vel - current_vel
        acc_cmd = self.target_uav_velocity_gain * vel_error + feedforward_acc
        acc_cmd = clip_vector_norm(acc_cmd, self.target_max_accel)

        max_tilt_tan = math.tan(math.radians(self.target_uav_max_tilt_deg))
        a_xy_max = torch.clamp((9.81 + acc_cmd[..., 2:3]) * max_tilt_tan, min=0.1)
        acc_xy = acc_cmd[..., :2]
        acc_xy_norm = torch.norm(acc_xy, dim=-1, keepdim=True)
        acc_cmd[..., :2] = acc_xy * (a_xy_max / acc_xy_norm.clamp(min=1e-6)).clamp(max=1.0)

        e3 = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=desired_vel.dtype).view(1, 1, 3)
        thrust_world = specific_force(acc_cmd, current_quat, current_vel, self.target,
                                      self.thrust_fraction, self.target_uav_max_tilt_deg)
        thrust_norm = torch.norm(thrust_world, dim=-1, keepdim=True).clamp(min=1e-6)
        z_des = thrust_world / thrust_norm
        thrust_ratio = self.target_hover_thrust_ratio * (thrust_norm / 9.81)
        if self.project_thrust_to_body_axis:
            thrust_ratio = body_axis_thrust_ratio(thrust_world, current_quat, self.target_hover_thrust_ratio)
        thrust_ratio = thrust_ratio.clamp(
            float(self.target_controller.min_thrust_ratio),
            float(self.target_controller.max_thrust_ratio),
        )

        x_cur = quat_axis(current_quat, axis=0)
        y_cur = quat_axis(current_quat, axis=1)
        z_cur = quat_axis(current_quat, axis=2)

        x_ref = project_to_plane(x_cur, z_des)
        fallback_x = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=desired_vel.dtype).view(1, 1, 3)
        x_ref = torch.where(x_ref.norm(dim=-1, keepdim=True) > 1e-5, x_ref, fallback_x.expand_as(x_ref))

        heading = torch.where(
            desired_vel.norm(dim=-1, keepdim=True) > 0.05,
            desired_vel,
            heading_hint,
        )
        heading_proj = project_to_plane(heading, z_des)
        x_ref = torch.where(
            heading_proj.norm(dim=-1, keepdim=True) > 1e-5,
            0.5 * x_ref + 0.5 * heading_proj,
            x_ref,
        )

        x_des = safe_normalize(x_ref)
        y_des = torch.linalg.cross(z_des, x_des, dim=-1)
        fallback_y = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=desired_vel.dtype).view(1, 1, 3)
        y_des = torch.where(y_des.norm(dim=-1, keepdim=True) > 1e-5, y_des, fallback_y.expand_as(y_des))
        y_des = safe_normalize(y_des)
        x_des = safe_normalize(torch.linalg.cross(y_des, z_des, dim=-1))

        e_rot = 0.5 * (
            torch.linalg.cross(x_cur, x_des, dim=-1)
            + torch.linalg.cross(y_cur, y_des, dim=-1)
            + torch.linalg.cross(z_cur, z_des, dim=-1)
        )
        omega_world = 2.0 * e_rot
        omega_body = quat_rotate_inverse(current_quat, omega_world)
        omega_body = clip_vector_norm(
            omega_body,
            self.target_uav_rate_fraction * self.target_max_body_rate_rad_s,
        )

        stopped = desired_vel.norm(dim=-1, keepdim=True) < 1e-5
        omega_body = torch.where(stopped, torch.zeros_like(omega_body), omega_body)
        thrust_ratio = torch.where(
            stopped,
            torch.full_like(thrust_ratio, self.target_hover_thrust_ratio),
            thrust_ratio,
        )

        rate_norm = (omega_body / max(self.target_max_body_rate_rad_s, 1e-6)).clamp(-0.999, 0.999)
        thrust_norm_action = (2.0 * thrust_ratio - 1.0).clamp(-0.999, 0.999)
        return torch.cat([rate_norm, thrust_norm_action], dim=-1)

    def _acceleration_to_pidrate_action(
        self,
        acc_cmd: torch.Tensor,
        current_quat: torch.Tensor,
        heading_hint: torch.Tensor,
        hover_thrust_ratio: float,
        max_body_rate_rad_s: float,
        max_tilt_deg: float,
        rate_fraction: float,
        controller,
        heading_blend: float = 0.0,
    ) -> torch.Tensor:
        acc_cmd = clip_vector_norm(acc_cmd, self.pursuer_accel_limit)
        acc_cmd = acc_cmd.clone()
        acc_cmd[..., 2].clamp_(
            -float(self.pursuer_vertical_accel_limit),
            float(self.pursuer_vertical_accel_limit),
        )

        max_tilt_tan = math.tan(math.radians(max_tilt_deg))
        a_xy_max = torch.clamp((9.81 + acc_cmd[..., 2:3]) * max_tilt_tan, min=0.1)
        acc_xy = acc_cmd[..., :2]
        acc_xy_norm = torch.norm(acc_xy, dim=-1, keepdim=True)
        acc_cmd[..., :2] = acc_xy * (a_xy_max / acc_xy_norm.clamp(min=1e-6)).clamp(max=1.0)

        e3 = torch.zeros_like(acc_cmd)
        e3[..., 2] = 1.0
        thrust_world = specific_force(acc_cmd, current_quat, self.drone.get_velocities()[..., :3],
                                      self.drone, self.thrust_fraction, max_tilt_deg)
        thrust_norm = torch.norm(thrust_world, dim=-1, keepdim=True).clamp(min=1e-6)
        z_des = thrust_world / thrust_norm
        thrust_ratio = hover_thrust_ratio * (thrust_norm / 9.81)
        if self.project_thrust_to_body_axis:
            thrust_ratio = body_axis_thrust_ratio(thrust_world, current_quat, hover_thrust_ratio)
        thrust_ratio = thrust_ratio.clamp(
            float(controller.min_thrust_ratio),
            float(controller.max_thrust_ratio),
        )

        x_cur = quat_axis(current_quat, axis=0)
        y_cur = quat_axis(current_quat, axis=1)
        z_cur = quat_axis(current_quat, axis=2)
        x_ref = project_to_plane(x_cur, z_des)
        fallback_x = torch.tensor(
            [1.0, 0.0, 0.0],
            device=acc_cmd.device,
            dtype=acc_cmd.dtype,
        ).view(1, 1, 3)
        x_ref = torch.where(
            x_ref.norm(dim=-1, keepdim=True) > 1e-5,
            x_ref,
            fallback_x.expand_as(x_ref),
        )
        if heading_blend > 0.0:
            heading_proj = project_to_plane(heading_hint, z_des)
            x_ref = torch.where(
                heading_proj.norm(dim=-1, keepdim=True) > 1e-5,
                (1.0 - float(heading_blend)) * x_ref + float(heading_blend) * heading_proj,
                x_ref,
            )
        x_des = x_ref
        x_des = safe_normalize(x_des)
        y_des = torch.linalg.cross(z_des, x_des, dim=-1)
        fallback_y = quat_axis(current_quat, axis=1)
        y_des = torch.where(y_des.norm(dim=-1, keepdim=True) > 1e-5, y_des, fallback_y)
        y_des = safe_normalize(y_des)
        x_des = safe_normalize(torch.linalg.cross(y_des, z_des, dim=-1))

        e_rot = 0.5 * (
            torch.linalg.cross(x_cur, x_des, dim=-1)
            + torch.linalg.cross(y_cur, y_des, dim=-1)
            + torch.linalg.cross(z_cur, z_des, dim=-1)
        )
        omega_world = 2.0 * e_rot
        omega_body = quat_rotate_inverse(current_quat, omega_world)
        omega_body = clip_vector_norm(
            omega_body,
            rate_fraction * max_body_rate_rad_s,
        )
        rate_norm = (omega_body / max(max_body_rate_rad_s, 1e-6)).clamp(-0.999, 0.999)
        thrust_norm_action = (2.0 * thrust_ratio - 1.0).clamp(-0.999, 0.999)
        return torch.cat([rate_norm, thrust_norm_action], dim=-1)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """One 20 Hz policy step containing five 100 Hz physics/control steps."""
        self._pre_sim_step(tensordict)
        dp, _ = self.get_env_poses(self.drone.get_world_poses())
        tp, _ = self.get_env_poses(self.target.get_world_poses())
        prev_relative = dp - tp
        self._control_min_target_distance = prev_relative.norm(dim=-1).min(-1, keepdim=True).values
        for substep in range(self.control_decimation):
            self._apply_trajectory_tracking_control(substep)
            if self.target_dynamics_mode == "uav":
                self._step_uav_target()
            else:
                self._step_point_mass_target()
            self.sim.step(self._should_render(substep))
            dp, _ = self.get_env_poses(self.drone.get_world_poses())
            tp, _ = self.get_env_poses(self.target.get_world_poses())
            relative = dp - tp
            swept_distance = segment_min_distance(prev_relative, relative).min(-1, keepdim=True).values
            self._control_min_target_distance.copy_(torch.minimum(self._control_min_target_distance, swept_distance))
            prev_relative = relative
            self._append_predictor_raw_frame()
        self._post_sim_step(tensordict)
        self.progress_buf += 1
        output = TensorDict({"next": {}}, self.batch_size)
        output["next"].update(self._compute_state_and_obs())
        output["next"].update(self._compute_reward_and_done())
        # Freeze terminal telemetry before the collector can reset this env.
        output["next", "stats"] = self.stats.clone()
        output["next", "info"] = self.info.clone()
        return output

    def _pre_sim_step(self, tensordict: TensorDictBase):
        """Decode one bounded 10-knot trajectory and cache its first segment."""
        actions = tensordict[("agents", "action")]
        prev_action = tensordict[("info", "prev_action")]
        try:
            action_error_order1 = tensordict[("stats", "action_error_order1")]
        except KeyError:
            action_error_order1 = torch.norm(actions - prev_action, dim=-1).mean(dim=-1)
        if action_error_order1.ndim > 2:
            action_error_order1 = action_error_order1.mean(dim=-1)
        self.info["prev_action"].copy_(actions.detach())
        self.prev_actions = self.info["prev_action"].clone()
        self.action_error_order1 = action_error_order1.clone()
        self.m3_action_delta.copy_(action_error_order1)
        self.m3_action_magnitude.copy_(torch.norm(actions, dim=-1).mean(dim=-1))
        self.stats["action_error_order1_mean"].add_(
            self.action_error_order1.mean(dim=-1).unsqueeze(-1)
        )
        self.stats["action_error_order1_max"].copy_(
            torch.max(
                self.stats["action_error_order1_max"],
                self.action_error_order1.mean(dim=-1).unsqueeze(-1),
            )
        )

        if self.pursuer_controller is None:
            raise RuntimeError("Trajectory control requires a pursuer PIDRateController.")
        drone_state = self.drone.get_state()[..., :13]
        drone_pos = drone_state[..., :3]
        drone_vel = drone_state[..., 7:10]
        self.trajectory_decoder.max_speed = float(self.current_pursuer_speed)
        decoded_plan = self.trajectory_decoder(
            normalized_reference=actions,
            initial_pos=drone_pos,
            initial_vel=drone_vel,
        )
        normalized_plan = torch.cat(
            [
                decoded_plan["positions_relative"]
                / self.position_scale.view(1, 1, 1, 3),
                decoded_plan["velocities"] / max(self.velocity_scale, 1e-6),
            ],
            dim=-1,
        )
        self.previous_plan.copy_(normalized_plan.detach())
        self.previous_plan_mask.fill_(1.0)

        self._trajectory_start_pos = drone_pos.detach().clone()
        self._trajectory_start_vel = drone_vel.detach().clone()
        self._trajectory_endpoint_pos = decoded_plan["positions_world"][..., 0, :].detach()
        self._trajectory_endpoint_vel = decoded_plan["velocities"][..., 0, :].detach()

    def _apply_trajectory_tracking_control(self, substep: int) -> None:
        """Track the first 0.05 s plan segment with a 100 Hz reference."""
        drone_state = self.drone.get_state()[..., :13]
        drone_pos = drone_state[..., :3]
        drone_vel = drone_state[..., 7:10]
        if self.trajectory_decoder.reference_mode == "bounded_position_velocity":
            # Independent position/velocity setpoints: interpolate commands,
            # never infer an arbitrarily large Hermite acceleration from them.
            alpha = (substep + 1) / float(self.control_decimation)
            reference = {
                "position": torch.lerp(self._trajectory_start_pos, self._trajectory_endpoint_pos, alpha),
                "velocity": torch.lerp(self._trajectory_start_vel, self._trajectory_endpoint_vel, alpha),
                "acceleration": torch.zeros_like(drone_vel),
            }
        elif self.trajectory_decoder.reference_mode == "acceleration_sequence":
            reference = self.trajectory_decoder.sample_constant_acceleration(
                self._trajectory_start_pos, self._trajectory_start_vel,
                self._trajectory_endpoint_vel,
                elapsed=(substep + 1) * float(self.dt), duration=self.trajectory_plan_dt)
        else:
            reference = self.trajectory_decoder.sample_cubic_hermite(
                self._trajectory_start_pos,
                self._trajectory_start_vel,
                self._trajectory_endpoint_pos,
                self._trajectory_endpoint_vel,
                elapsed=(substep + 1) * float(self.dt),
                duration=self.trajectory_plan_dt,
            )
        desired_acc_raw = (
            reference["acceleration"]
            + self.trajectory_position_gain * (reference["position"] - drone_pos)
            + self.trajectory_velocity_gain * (reference["velocity"] - drone_vel)
        )
        self._apply_pursuer_acceleration(desired_acc_raw, drone_state)

    def _apply_pursuer_acceleration(
        self, desired_acc_raw: torch.Tensor, drone_state: torch.Tensor
    ) -> None:
        """Apply physical safety bounds and convert acceleration to PIDRate commands."""
        drone_pos = drone_state[..., :3]
        drone_quat = drone_state[..., 3:7]
        drone_vel = drone_state[..., 7:10]
        target_state = self.target.get_state()[..., :13]
        target_pos = target_state[..., :3]
        target_z = target_pos[..., 2:3]
        if target_z.dim() == 3:
            target_z = target_z.squeeze(1).unsqueeze(1)
        nominal_safe_z = torch.full_like(
            drone_pos[..., 2:3], float(self.pursuer_floor_barrier_height)
        )
        target_compatible_z = target_z - float(self.pursuer_floor_target_margin)
        safe_z = torch.minimum(nominal_safe_z, target_compatible_z).clamp_min(
            float(self.pursuer_floor_hard_z)
        )
        predicted_z = drone_pos[..., 2:3] + drone_vel[..., 2:3] * float(
            self.pursuer_floor_prediction_time
        )
        if self.floor_use_braking_distance:
            predicted_z = floor_braking_prediction(
                drone_pos[..., 2:3], drone_vel[..., 2:3],
                self.pursuer_floor_barrier_max_accel, self.pursuer_floor_prediction_time)
        height_error = torch.maximum(
            safe_z - drone_pos[..., 2:3], safe_z - predicted_z
        ).clamp_min(0.0)
        floor_active = height_error > 0.0
        upward_acc = (
            self.pursuer_floor_barrier_gain * height_error
            + self.pursuer_floor_barrier_damping * torch.relu(-drone_vel[..., 2:3])
        ).clamp(max=float(self.pursuer_floor_barrier_max_accel))
        vertical_acc = torch.where(
            floor_active,
            torch.maximum(desired_acc_raw[..., 2:3], upward_acc),
            desired_acc_raw[..., 2:3],
        ).clamp(
            min=-float(self.pursuer_vertical_accel_limit),
            max=float(self.pursuer_vertical_accel_limit),
        )
        recovery_progress = (
            (drone_pos[..., 2:3] - safe_z)
            / max(float(self.pursuer_floor_recovery_margin), 1e-6)
        ).clamp(0.0, 1.0)
        xy_scale = float(self.pursuer_floor_xy_scale_min) + (
            1.0 - float(self.pursuer_floor_xy_scale_min)
        ) * recovery_progress
        xy_cmd = desired_acc_raw[..., :2] * xy_scale
        xy_norm = torch.norm(xy_cmd, dim=-1, keepdim=True)
        xy_cmd = xy_cmd * (
            float(self.pursuer_horizontal_accel_limit) / xy_norm.clamp_min(1e-6)
        ).clamp(max=1.0)
        desired_acc = clip_vector_norm(
            torch.cat([xy_cmd, vertical_acc], dim=-1), self.pursuer_accel_limit
        )
        self.pursuer_safe_z = safe_z.detach()
        self.pursuer_floor_active = floor_active.to(desired_acc.dtype).detach()
        self.pursuer_desired_acc_cmd = desired_acc.detach()
        heading_hint = target_pos - drone_pos
        pidrate_action = self._acceleration_to_pidrate_action(
            desired_acc,
            drone_quat,
            heading_hint,
            hover_thrust_ratio=self.pursuer_hover_thrust_ratio,
            max_body_rate_rad_s=self.pursuer_max_body_rate_rad_s,
            max_tilt_deg=self.pursuer_max_tilt_deg,
            rate_fraction=self.pursuer_rate_fraction,
            controller=self.pursuer_controller,
            heading_blend=self.pursuer_yaw_heading_blend,
        )
        target_rate_norm, target_thrust_norm = pidrate_action.split([3, 1], dim=-1)
        target_rate = target_rate_norm * 180.0 * float(self.pursuer_controller.target_clip)
        target_thrust_ratio = torch.clamp(
            (target_thrust_norm + 1.0) / 2.0,
            min=float(self.pursuer_controller.min_thrust_ratio),
            max=float(self.pursuer_controller.max_thrust_ratio),
        )
        target_thrust = target_thrust_ratio * 2**16
        reset_pid = getattr(self, "pursuer_pid_reset", None)
        if reset_pid is None or reset_pid.shape[:2] != target_rate.shape[:2]:
            reset_pid = torch.ones(
                self.num_envs, self.num_agents, dtype=torch.bool, device=self.device
            )
            self.pursuer_pid_reset = reset_pid
        cmds, _ = self.pursuer_controller(
            drone_state,
            target_rate=target_rate,
            target_thrust=target_thrust,
            reset_pid=reset_pid,
        )
        self.pursuer_pid_reset.fill_(False)
        torch.nan_to_num_(cmds, 0.0)
        cmds = bounded_motor_commands(cmds, self.thrust_fraction)
        self.pursuer_motor_fraction.copy_(((cmds + 1.0) / 2.0).max(-1).values)
        self.effort = self.drone.apply_action(cmds)

    def _append_predictor_raw_frame(self) -> None:
        """Append one 100 Hz state frame; the GMM later samples every fifth frame."""
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses())
        drone_vel = self.drone.get_velocities()[..., :3]
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        target_vel = self.target.get_velocities()[..., :3]
        frame = self._build_predictor_frame(
            drone_pos,
            drone_vel,
            target_pos.squeeze(1),
            target_vel.squeeze(1),
        )
        initialized = self.predictor_raw_initialized
        if bool(initialized.any()):
            self.predictor_raw_history[initialized] = torch.roll(
                self.predictor_raw_history[initialized], shifts=-1, dims=1
            )
            self.predictor_raw_history[initialized, -1] = frame[initialized]
        fresh = ~initialized
        if bool(fresh.any()):
            self.predictor_raw_history[fresh] = frame[fresh].unsqueeze(1).expand(
                -1, self.predictor_raw_history_len, -1
            )
            self.predictor_raw_initialized[fresh] = True

    def _select_expert2_intercept_prediction(self, target_flat: torch.Tensor) -> torch.Tensor:
        """Select the same TP prediction step used by the current Expert2 policy."""
        pred = getattr(self, "target_pos_predicted", None)
        if pred is None:
            return target_flat
        if pred.dim() == 2:
            return pred.to(device=target_flat.device, dtype=target_flat.dtype)
        if pred.dim() != 3 or pred.shape[1] <= 0:
            return target_flat
        pred_idx = max(
            0,
            min(
                int(self.expert2_intercept_pred_step) - 1,
                int(pred.shape[1]) - 1,
            ),
        )
        return pred[:, pred_idx].to(device=target_flat.device, dtype=target_flat.dtype)

    def _compute_expert2_role_assignment_and_targets(
        self,
        drone_pos: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        target_pos_pred: Optional[torch.Tensor] = None,
        update_assignment_state: bool = False,
    ):
        """Return Expert2 role ids and normal/close targets used by observations and reward."""
        target_flat = target_pos.squeeze(1) if target_pos.dim() == 3 else target_pos
        target_vel_flat = target_vel[..., :3]
        if target_vel_flat.dim() == 3:
            target_vel_flat = target_vel_flat.squeeze(1)

        goal_pos = self.goal_region_center.view(1, 3).to(target_flat.device, target_flat.dtype)
        goal_vec = goal_pos - target_flat
        goal_dist = goal_vec.norm(dim=-1)
        goal_dir = safe_normalize(goal_vec)

        lookahead_time = torch.where(
            goal_dist > self.expert2_role_goal_dist_switch,
            torch.full_like(goal_dist, self.expert2_role_far_lookahead),
            torch.full_like(goal_dist, self.expert2_role_near_lookahead),
        )
        if target_pos_pred is None:
            target_pos_pred = self._select_expert2_intercept_prediction(target_flat)
        elif target_pos_pred.dim() == 3:
            pred_idx = max(
                0,
                min(
                    int(self.expert2_intercept_pred_step) - 1,
                    int(target_pos_pred.shape[1]) - 1,
                ),
            )
            target_pos_pred = target_pos_pred[:, pred_idx]
        target_pos_pred = target_pos_pred.to(device=target_flat.device, dtype=target_flat.dtype)
        if self.expert2_intercept_use_direct_pred:
            intercept_seed = target_pos_pred
        else:
            intercept_seed = target_pos_pred + target_vel_flat * lookahead_time.unsqueeze(-1)

        motion_hint = target_vel_flat.clone()
        motion_norm = motion_hint.norm(dim=-1)
        motion_hint = torch.where(
            (motion_norm < 1e-4).unsqueeze(-1),
            intercept_seed - target_flat,
            motion_hint,
        )
        motion_norm = motion_hint.norm(dim=-1)
        motion_hint = torch.where(
            (motion_norm < 1e-4).unsqueeze(-1),
            goal_dir,
            motion_hint,
        )
        forward_dir = safe_normalize(motion_hint)

        up = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=drone_pos.dtype).view(1, 3)
        lateral = torch.linalg.cross(up.expand_as(forward_dir), forward_dir, dim=-1)
        fallback_lateral = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=drone_pos.dtype).view(1, 3)
        lateral = torch.where(
            lateral.norm(dim=-1, keepdim=True) > 1e-5,
            lateral,
            fallback_lateral.expand_as(lateral),
        )
        lateral = safe_normalize(lateral)

        rear = intercept_seed - self.expert2_role_rear_back * forward_dir
        front_left = intercept_seed + self.expert2_role_front_side * lateral
        front_right = intercept_seed - self.expert2_role_front_side * lateral
        anchors = torch.stack([rear, front_left, front_right], dim=1)

        dist = torch.cdist(drone_pos, anchors)
        batch_size = drone_pos.shape[0]
        perm = self.expert2_role_permutations.unsqueeze(0).expand(batch_size, -1, -1)
        dist_expanded = dist.unsqueeze(1).expand(-1, perm.shape[1], -1, -1)
        assigned = torch.gather(dist_expanded, 3, perm.unsqueeze(-1)).squeeze(-1)
        cost = assigned.sum(dim=-1)
        if self.expert2_prev_assignment.shape[0] == batch_size:
            ready = self.expert2_assignment_ready.squeeze(-1)
            changed = (perm != self.expert2_prev_assignment.unsqueeze(1)).float().sum(dim=-1)
            cost = cost + 0.20 * changed * ready.float().unsqueeze(-1)
        assignment = perm[
            torch.arange(batch_size, device=self.device),
            cost.argmin(dim=-1),
        ]
        if update_assignment_state and self.expert2_prev_assignment.shape[0] == batch_size:
            self.expert2_prev_assignment.copy_(assignment)
            self.expert2_assignment_ready.fill_(True)
        normal_targets = torch.gather(anchors, 1, assignment.unsqueeze(-1).expand(-1, -1, 3))
        role_dist_normal = torch.norm(normal_targets - drone_pos, dim=-1)

        is_chaser = assignment == 0
        dist_to_target = torch.norm(drone_pos - target_flat.unsqueeze(1), dim=-1)
        target_close_threshold = torch.where(
            is_chaser,
            torch.full_like(role_dist_normal, self.expert2_role_chaser_close_target_threshold),
            torch.full_like(role_dist_normal, self.expert2_role_front_close_target_threshold),
        )
        close_triggered = (
            (role_dist_normal < self.expert2_role_close_wp_threshold)
            | (dist_to_target < target_close_threshold)
        )
        if not self.expert2_role_close_enabled:
            close_triggered = torch.zeros_like(close_triggered, dtype=torch.bool)

        side_axis = project_to_plane(
            normal_targets - target_flat.unsqueeze(1),
            forward_dir.unsqueeze(1),
        )
        side_axis = torch.where(
            side_axis.norm(dim=-1, keepdim=True) > 1e-5,
            safe_normalize(side_axis),
            torch.zeros_like(side_axis),
        )
        close_chaser = target_pos_pred.unsqueeze(1) - self.expert2_role_close_chaser_back * forward_dir.unsqueeze(1)
        close_interceptor = (
            target_pos_pred.unsqueeze(1)
            + self.expert2_role_close_front_forward * forward_dir.unsqueeze(1)
            + self.expert2_role_close_front_side * side_axis
        )
        close_targets = torch.where(is_chaser.unsqueeze(-1), close_chaser, close_interceptor)
        active_targets = torch.where(close_triggered.unsqueeze(-1), close_targets, normal_targets)

        return (
            assignment,
            normal_targets,
            close_targets,
            close_triggered,
            active_targets,
            forward_dir,
            lateral,
        )

    def _build_predictor(self):
        from omni_drones.learning.predictor.dual_intent_gmm_predictor import (
            StrategicGMMConfig,
            StrategicGMMTrajectoryPredictor,
        )

        predictor_cfg = StrategicGMMConfig(
            obs_dim=self.predictor_obs_dim,
            n_components=NUM_TARGET_TRAJECTORIES,
            n_strategic=NUM_STRATEGIC_INTENTS,
            pred_horizon=self.future_prediction_obs_steps,
        )
        predictor = StrategicGMMTrajectoryPredictor(predictor_cfg).to(self.device)
        predictor._train_updates = 0
        predictor.eval()
        return predictor

    def _load_predictor(self, checkpoint_path: str) -> None:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logging.warning("Predictor checkpoint not found (%s); using kinematic GMM.", checkpoint_path)
            return
        try:
            predictor = self._build_predictor()
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                for key in ("model", "model_state_dict", "state_dict", "predictor"):
                    if key in checkpoint:
                        checkpoint = checkpoint[key]
                        break
            predictor.load_state_dict(checkpoint, strict=False)
            predictor.eval()
            self.predictor = predictor
            self.predictor_loaded_checkpoint = True
            logging.info("Loaded strategic GMM predictor from %s", checkpoint_path)
        except Exception as exc:
            raise RuntimeError(
                "Predictor checkpoint is incompatible with the two-intent/single-trajectory "
                "branch. Use a matching checkpoint or train from scratch; old 4-intent/3-mode "
                "checkpoints remain usable on archive/pv-sine-four-intent-20261028."
            ) from exc

    def _compute_rule_strategic_probs(
        self,
        drone_pos: torch.Tensor,
        target_pos_flat: torch.Tensor,
    ) -> torch.Tensor:
        goal_dist = point_to_cylinder_distance(
            target_pos_flat,
            self.goal_region_center,
            self.goal_region_radius,
            self.goal_region_height,
        )
        pursuer_dist = torch.norm(drone_pos - target_pos_flat.unsqueeze(1), dim=-1)
        min_pursuer_dist = pursuer_dist.min(dim=1).values
        if self.target_motion_mode == "goal_sine":
            near = min_pursuer_dist <= self.target_sine_policy.trigger_distance
            return torch.stack((~near, near), dim=-1).to(drone_pos.dtype)
        sprint = goal_dist <= self.target_goal_sprint_distance
        near = min_pursuer_dist <= self.target_tactical_far_distance
        close_evade = near & (~sprint) & (goal_dist > min_pursuer_dist)
        return torch.stack((~close_evade, close_evade), dim=-1).to(drone_pos.dtype)

    def _build_predictor_frame(
        self,
        drone_pos: torch.Tensor,
        drone_vel: torch.Tensor,
        target_pos_flat: torch.Tensor,
        target_vel_flat: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = target_pos_flat.shape[0]
        target_pos_norm = target_pos_flat / self.position_scale.view(1, 3)
        target_vel_norm = target_vel_flat / max(self.target_velocity_scale, 1e-6)
        goal_vec_norm = (
            self.goal_region_center.view(1, 3) - target_pos_flat
        ) / self.position_scale.view(1, 3)
        rel_pos = (drone_pos - target_pos_flat.unsqueeze(1)) / self.position_scale.view(1, 1, 3)
        rel_vel = (drone_vel - target_vel_flat.unsqueeze(1)) / max(self.velocity_scale, 1e-6)
        goal_dist = point_to_cylinder_distance(
            target_pos_flat,
            self.goal_region_center,
            self.goal_region_radius,
            self.goal_region_height,
        ).unsqueeze(-1) / max(self.arena_size, 1e-6)
        min_pursuer_dist = torch.norm(
            drone_pos - target_pos_flat.unsqueeze(1), dim=-1
        ).min(dim=1, keepdim=True).values / max(self.arena_size, 1e-6)
        progress_buf = self.progress_buf if env_ids is None else self.progress_buf[env_ids]
        progress = progress_buf.float().unsqueeze(-1) / max(float(self.max_episode_length), 1.0)
        frame = torch.cat(
            [
                target_pos_norm,
                target_vel_norm,
                goal_vec_norm,
                rel_pos.reshape(batch, -1),
                rel_vel.reshape(batch, -1),
                progress,
                goal_dist,
                min_pursuer_dist,
            ],
            dim=-1,
        )
        if frame.shape[-1] < self.predictor_obs_dim:
            pad = torch.zeros(
                batch,
                self.predictor_obs_dim - frame.shape[-1],
                device=self.device,
                dtype=frame.dtype,
            )
            frame = torch.cat([frame, pad], dim=-1)
        elif frame.shape[-1] > self.predictor_obs_dim:
            frame = frame[..., : self.predictor_obs_dim]
        return frame

    def _kinematic_gmm(
        self,
        drone_pos: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single constant-velocity fallback while the predictor warms up."""
        batch = target_pos.shape[0]
        time = torch.arange(
            1,
            self.future_prediction_obs_steps + 1,
            device=self.device,
            dtype=target_pos.dtype,
        ).view(1, 1, -1, 1) * self.prediction_dt
        means = target_vel[:, None, None, :] * time
        weights = torch.ones(batch, 1, device=self.device, dtype=target_pos.dtype)
        variance = (0.04 + 0.02 * time).expand(batch, 1, -1, 3)
        return weights, means, variance

    def _predict_gmm(
        self,
        drone_pos: torch.Tensor,
        drone_vel: torch.Tensor,
        target_pos_flat: torch.Tensor,
        target_vel_flat: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(
                target_pos_flat.shape[0],
                device=self.device,
                dtype=torch.long,
            )
        frame = self._build_predictor_frame(
            drone_pos,
            drone_vel,
            target_pos_flat,
            target_vel_flat,
            env_ids=env_ids,
        )
        raw_history = self.predictor_raw_history[env_ids].clone()
        initialized = self.predictor_raw_initialized[env_ids]
        fresh = ~initialized
        if bool(fresh.any()):
            raw_history[fresh] = frame[fresh].unsqueeze(1).expand(
                -1,
                self.predictor_raw_history_len,
                -1,
            )
            self.predictor_raw_history[env_ids[fresh]] = raw_history[fresh]
            self.predictor_raw_initialized[env_ids[fresh]] = True
        history = raw_history.index_select(1, self.predictor_history_indices)
        self.predictor_history[env_ids] = history
        self.predictor_initialized[env_ids] = True

        rule_probs = self._compute_rule_strategic_probs(drone_pos, target_pos_flat)
        train_updates = int(getattr(self.predictor, "_train_updates", 0)) if self.predictor else 0
        use_learned_prediction = self.predictor is not None and (
            self.predictor_loaded_checkpoint or train_updates > 0
        )
        if use_learned_prediction:
            with torch.no_grad():
                self.predictor.eval()
                pred = self.predictor(history)
            strategic_probs = pred["strategic_probs"].to(dtype=drone_pos.dtype)
            gmm_weights = pred["gmm_weights"].to(dtype=drone_pos.dtype)
            gmm_means = pred["gmm_means"].to(dtype=drone_pos.dtype)
            gmm_vars = pred["gmm_vars"].to(dtype=drone_pos.dtype)
        else:
            strategic_probs = rule_probs
            gmm_weights, gmm_means, gmm_vars = self._kinematic_gmm(
                drone_pos, target_pos_flat, target_vel_flat
            )

        self.predictor_sample_valid[env_ids] = 1.0
        self.predictor_target_pos_world[env_ids] = target_pos_flat.detach()
        self.predictor_strategic_label[env_ids, 0] = rule_probs.argmax(dim=-1)
        self.predictor_strategic_probs[env_ids] = strategic_probs.detach()
        self.predictor_gmm_weights[env_ids] = gmm_weights.detach()
        self.predictor_gmm_means[env_ids] = gmm_means.detach()
        self.predictor_gmm_vars[env_ids] = gmm_vars.detach()
        return strategic_probs, gmm_weights, gmm_means, gmm_vars
     
    def _record_agent_history(self, history_frame):
        history_fresh = ~self.agent_history_initialized
        history_ready = self.agent_history_initialized.clone()
        if not self.advance_physics_on_reset:
            # IsaacEnv recomputes observations for ALL envs after a partial
            # reset. Live envs must not acquire an extra, zero-time history row.
            history_ready &= self.agent_history_last_step.ne(self.progress_buf)
        if bool(history_ready.any()):
            self.agent_history_buffer[history_ready] = torch.roll(
                self.agent_history_buffer[history_ready], shifts=-1, dims=2
            )
            self.agent_history_buffer[history_ready, :, -1] = history_frame[history_ready]
        if bool(history_fresh.any()):
            self.agent_history_buffer[history_fresh] = history_frame[
                history_fresh
            ].unsqueeze(2).expand(-1, -1, self.history_step, -1)
            self.agent_history_initialized[history_fresh] = True
        self.agent_history_last_step.copy_(self.progress_buf)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        self.info["drone_state"][:] = self.drone_states[..., :13]
        drone_pos, drone_quat = self.get_env_poses(self.drone.get_world_poses())
        drone_vel_full = self.drone.get_velocities()
        drone_vel = drone_vel_full[..., :3]
        drone_body_rates = quat_rotate_inverse(drone_quat, drone_vel_full[..., 3:6])
        self.drone_rpos = vmap(off_diag)(vmap(cpos)(drone_pos, drone_pos))
        self.drone_rvel = vmap(off_diag)(vmap(cpos)(drone_vel, drone_vel))

        obs = TensorDict({}, [self.num_envs, self.drone.n])

        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        target_vel = self.target.get_velocities()
        self.info["target_state"][..., :3] = target_pos
        self.info["target_state"][..., 3:6] = target_vel[..., :3]
        self.info["target_acc_cmd"].copy_(self.target_acc_cmd)
        self.info["target_max_speed"].copy_(self.target_max_speed.squeeze(-1))
        self.info["target_max_accel"].copy_(self.target_max_accel.squeeze(-1))
        self.info["target_initial_speed"].copy_(self.target_initial_speed)
        self.info["pursuer_motor_fraction"].copy_(self.drone.throttle.square().max(-1).values)
        self.info["target_motor_fraction"].copy_(self.target.throttle.square().max(-1).values)
        self.info["target_tactical_intent"][..., 0].copy_(
            self.target_tactical_intent.to(self.info["target_tactical_intent"].dtype)
        )
        self.info["target_maneuver_id"][..., 0].copy_(
            self.target_maneuver_id.to(self.info["target_maneuver_id"].dtype)
        )
        if self.target_sine_policy is not None:
            sine = self.target_sine_policy
            self.info["target_sine_amplitude_m"][..., 0].copy_(sine.amplitude)
            self.info["target_sine_period_s"][..., 0].copy_(sine.period)
            self.info["target_sine_active"][..., 0].copy_(sine.active)
            self.info["target_sine_reference_offset_m"][..., 0].copy_(sine.reference_offset)
        target_rpos = vmap(cpos)(drone_pos, target_pos)
        in_detection_range = torch.norm(target_rpos, dim=-1) < self.drone_detect_radius
        self.broadcast_detect = torch.any(in_detection_range, dim=1)

        target_mask = (~self.broadcast_detect).unsqueeze(-1).expand_as(target_pos)
        target_pos_norm = target_pos / self.position_scale.view(1, 1, 3)
        target_pos_masked = target_pos_norm.clone()
        target_pos_masked.masked_fill_(target_mask, self.mask_value)
        target_vel_norm = target_vel[..., :3] / self.target_velocity_scale
        target_vel_masked = target_vel_norm.clone()
        target_vel_masked.masked_fill_(target_mask, self.mask_value)

        relative_pos = (drone_pos - target_pos) / self.position_scale.view(1, 1, 3)
        relative_vel = (
            drone_vel - target_vel[..., :3]
        ) / max(self.velocity_scale, 1e-6)
        desired_acc = getattr(self, "pursuer_desired_acc_cmd", None)
        if desired_acc is None or desired_acc.shape != drone_pos.shape:
            desired_acc = torch.zeros_like(drone_pos)
        normalized_acc = desired_acc / max(self.pursuer_accel_limit, 1e-6)
        history_frame = torch.cat(
            [relative_pos, relative_vel, normalized_acc], dim=-1
        )
        self._record_agent_history(history_frame)

        TP = TensorDict({}, [self.num_envs])

        if self.use_TP_net:
            expanded_drone_pos = torch.concat(
                [drone_pos / self.position_scale.view(1, 1, 3), self.masked_drone_pos], dim=1
            )
            frame_state = torch.concat(
                [
                    self.progress_buf.unsqueeze(-1) / max(self.max_episode_length, 1),
                    target_pos_norm.reshape(self.num_envs, -1),
                    target_vel_norm.squeeze(1),
                    expanded_drone_pos.reshape(self.num_envs, -1),
                ],
                dim=-1,
            )
            if (
                self.tp_history_buffer.shape[0] != self.num_envs
                or self.tp_history_buffer.shape[1] != self.history_step
                or self.tp_history_buffer.shape[2] != frame_state.shape[-1]
            ):
                self.tp_history_buffer = torch.zeros(
                    self.num_envs,
                    self.history_step,
                    frame_state.shape[-1],
                    device=self.device,
                    dtype=frame_state.dtype,
                )
                self.tp_history_initialized = torch.zeros(
                    self.num_envs,
                    dtype=torch.bool,
                    device=self.device,
                )

            # Initialized envs: shift history and append latest frame.
            initialized = self.tp_history_initialized
            if bool(initialized.any()):
                self.tp_history_buffer[initialized] = torch.roll(
                    self.tp_history_buffer[initialized], shifts=-1, dims=1
                )
                self.tp_history_buffer[initialized, -1] = frame_state[initialized]

            # Freshly reset envs: bootstrap full history with current frame.
            fresh = ~initialized
            if bool(fresh.any()):
                self.tp_history_buffer[fresh] = frame_state[fresh].unsqueeze(1).expand(
                    -1, self.history_step, -1
                )
                self.tp_history_initialized[fresh] = True

            TP["TP_input"] = self.tp_history_buffer.clone()
            # Environment observations use TP as a predictor, not as part of
            # the rollout computation graph. Keeping this under no_grad avoids
            # evaluation-time graph accumulation and CUDA OOM.
            with torch.no_grad():
                self.target_pos_predicted = self.TP(TP["TP_input"]).reshape(
                    self.num_envs, self.future_predcition_step, -1
                )
            self.target_pos_predicted[..., :2] = self.target_pos_predicted[..., :2] * self.arena_size
            self.target_pos_predicted[..., 2] = (
                (self.target_pos_predicted[..., 2] + 1.0) / 2.0 * self.max_height
            )
            self.stats["target_predicted_error"].add_(
                torch.norm(target_pos.squeeze(1) - self.target_pos_predicted[:, 0], dim=-1).unsqueeze(-1)
            )
            TP["TP_done"] = (
                self.progress_buf <= (self.max_episode_length - self.future_predcition_step)
            ).unsqueeze(-1)
            TP["TP_groundtruth"] = target_pos.squeeze(1).clone()
            TP["TP_groundtruth"][..., :2] = TP["TP_groundtruth"][..., :2] / self.arena_size
            TP["TP_groundtruth"][..., 2] = TP["TP_groundtruth"][..., 2] / self.max_height * 2.0 - 1.0
            tp_escape_dir = safe_normalize(self.target_pos_predicted[:, 0] - target_pos.squeeze(1))
        else:
            self.target_pos_predicted = None
            tp_escape_dir = None

        target_pos_flat = target_pos.squeeze(1)
        goal_vec = self.goal_region_center.view(1, 3) - target_pos_flat
        goal_threat_dir = safe_normalize(goal_vec)
        if tp_escape_dir is None:
            tp_escape_dir = goal_threat_dir
        else:
            tp_escape_dir = torch.where(
                self.broadcast_detect.expand_as(goal_threat_dir),
                tp_escape_dir,
                goal_threat_dir,
            )

        agent_from_target = drone_pos - target_pos
        self.target_dist = torch.norm(agent_from_target, dim=-1)
        norm_dir = safe_normalize(agent_from_target)
        coverage_quality = 1.0 - norm_dir.mean(dim=1).norm(dim=-1, keepdim=True)
        coverage_quality = coverage_quality.expand(-1, self.num_agents)

        goal_expanded = goal_threat_dir.unsqueeze(1).expand(-1, self.num_agents, -1)
        tp_expanded = tp_escape_dir.unsqueeze(1).expand(-1, self.num_agents, -1)
        self.my_ahead_proj_goal = (agent_from_target * goal_expanded).sum(dim=-1)
        self.my_ahead_proj_tp = (agent_from_target * tp_expanded).sum(dim=-1)
        lateral_vec_goal = agent_from_target - self.my_ahead_proj_goal.unsqueeze(-1) * goal_expanded
        self.my_lateral_dist_goal = lateral_vec_goal.norm(dim=-1)
        goal_to_target_dist = goal_vec.norm(dim=-1)
        goal_dist_boundary = point_to_cylinder_distance(
            target_pos_flat, self.goal_region_center, self.goal_region_radius, self.goal_region_height
        )
        self.g_urg = torch.sigmoid(
            (self.urgent_goal_dist - goal_dist_boundary) / max(self.urgent_goal_tau, 1e-6)
        )
        self.block_i = (
            torch.sigmoid((self.my_ahead_proj_goal - self.block_margin) / max(self.block_softness, 1e-6))
            * torch.sigmoid(
                (goal_to_target_dist.unsqueeze(1) - self.my_ahead_proj_goal)
                / max(self.block_between_softness, 1e-6)
            )
            * torch.exp(-self.my_lateral_dist_goal / max(self.block_lateral_scale, 1e-6))
        )
        self.phi_block, self.phi_pressure, self.phi_spread, self.phi_team = compute_phi_team(
            self.block_i,
            self.target_dist,
            norm_dir,
            self.g_urg,
            self.pressure_dist_scale,
        )
        self.d_i_clip = compute_leave_one_out_di(
            self.block_i,
            self.target_dist,
            norm_dir,
            self.g_urg,
            self.pressure_dist_scale,
            self.di_scale,
        )

        target_vel_flat = target_vel[..., :3].squeeze(1)
        _, gmm_weights, gmm_means, _ = self._predict_gmm(
            drone_pos=drone_pos,
            drone_vel=drone_vel,
            target_pos_flat=target_pos_flat,
            target_vel_flat=target_vel_flat,
        )
        mixed_displacement = (
            gmm_weights.unsqueeze(-1).unsqueeze(-1) * gmm_means
        ).sum(dim=1)
        target_pred_world = target_pos_flat.unsqueeze(1) + mixed_displacement
        self.target_pos_predicted = target_pred_world

        obs["agent_history"] = self.agent_history_buffer.detach().clone()
        obs["current_pos"] = relative_pos
        obs["current_vel"] = relative_vel
        if self.use_flight_context:
            # Local arena frame, not the global coordinates of tiled envs.
            # Relative pursuit geometry alone does not specify ceiling/wall margins.
            obs["flight_context"] = torch.cat([
                drone_pos / self.position_scale.view(1, 1, 3),
                drone_vel / max(self.velocity_scale, 1e-6),
            ], dim=-1).detach().clone()
        obs["prev_plan"] = self.previous_plan.detach().clone()
        obs["prev_plan_mask"] = self.previous_plan_mask.detach().clone()
        # ``expand`` alone returns a zero-stride view into the predictor's
        # mutable rollout buffers.  Those buffers are overwritten next step;
        # retaining the view corrupts PPO's stored observations and makes the
        # recomputed log-probability disagree with the behaviour log-prob.
        # Materialise an immutable per-step snapshot before handing it to the
        # collector.
        obs["gmm_weights"] = self.predictor_gmm_weights.detach().unsqueeze(1).expand(
            -1, self.num_agents, -1
        ).clone()
        obs["strategic_probs"] = self.predictor_strategic_probs.detach().unsqueeze(1).expand(
            -1, self.num_agents, -1
        ).clone()
        obs["gmm_means"] = self.predictor_gmm_means.detach().unsqueeze(1).expand(
            -1, self.num_agents, -1, -1, -1
        ).clone()
        obs["gmm_vars"] = self.predictor_gmm_vars.detach().unsqueeze(1).expand(
            -1, self.num_agents, -1, -1, -1
        ).clone()

        (
            role_assignment,
            assigned_waypoint_world,
            close_waypoint_world,
            close_triggered,
            _active_waypoint_world,
            forward_dir_world,
            lateral_world,
        ) = self._compute_expert2_role_assignment_and_targets(
            drone_pos,
            target_pos,
            target_vel[..., :3],
            target_pos_pred=self._select_expert2_intercept_prediction(target_pos_flat),
            update_assignment_state=True,
        )
        self._expert2_cached_active_waypoint = _active_waypoint_world.detach()
        self._expert2_cached_close_triggered = close_triggered.detach()

        if self.role_encoding_dim > 0:
            role_encoding = torch.zeros(
                self.num_envs,
                self.num_agents,
                self.role_encoding_dim,
                device=self.device,
                dtype=drone_pos.dtype,
            )
            role_encoding.scatter_(-1, role_assignment.unsqueeze(-1), 1.0)
        else:
            role_encoding = torch.zeros(
                self.num_envs,
                self.num_agents,
                0,
                device=self.device,
                dtype=drone_pos.dtype,
            )

        cooperation_env = torch.cat(
            [
                target_vel[..., :3].squeeze(1),
                target_pred_world.reshape(self.num_envs, -1),
                forward_dir_world,
                lateral_world,
            ],
            dim=-1,
        )
        cooperation = (
            cooperation_env.unsqueeze(1)
            .expand(-1, self.num_agents, -1)
            .unsqueeze(2)
            .clone()
        )
        obs["cooperation"] = cooperation

        obs["state_self"] = torch.cat(
            [
                drone_pos,
                drone_vel,
                drone_body_rates,
                drone_quat,
                assigned_waypoint_world,
                close_waypoint_world,
                close_triggered.unsqueeze(-1).to(drone_pos.dtype),
                role_encoding,
            ],
            dim=-1,
        ).unsqueeze(2)

        if self.drone.n > 1:
            other_role_encoding = vmap(others)(role_encoding)
            obs["state_others"] = torch.cat(
                [
                    -self.drone_rpos,
                    -self.drone_rvel,
                    other_role_encoding,
                ],
                dim=-1,
            )
        else:
            obs["state_others"] = torch.zeros(
                self.num_envs,
                self.num_agents,
                0,
                self.state_others_dim,
                device=self.device,
            )

        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = torch.cat(
            [
                obs["state_self"].squeeze(2),
                cooperation.squeeze(2),
            ],
            dim=-1,
        )

        prediction = TensorDict(
            {
                "pred_input": self.predictor_history.detach().clone(),
                "target_pos_world": self.predictor_target_pos_world.detach().clone(),
                "strategic_label": self.predictor_strategic_label.to(torch.float32),
                "valid": self.predictor_sample_valid.detach().clone(),
                "strategic_probs": self.predictor_strategic_probs.detach().clone(),
                "gmm_weights": self.predictor_gmm_weights.detach().clone(),
                "gmm_means": self.predictor_gmm_means.detach().clone(),
                "gmm_vars": self.predictor_gmm_vars.detach().clone(),
            },
            [self.num_envs],
        )
        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "state": state,
                    "prediction": prediction,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.batch_size,
        )

    def _compute_expert2_waypoint_rewards(
        self,
        drone_pos: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-agent step rewards for normal-waypoint mode and close-waypoint mode."""
        cached_targets = getattr(self, "_expert2_cached_active_waypoint", None)
        cached_close = getattr(self, "_expert2_cached_close_triggered", None)
        if (
            cached_targets is not None
            and cached_close is not None
            and cached_targets.shape == drone_pos.shape
            and cached_close.shape == drone_pos.shape[:2]
        ):
            role_targets = cached_targets.to(device=drone_pos.device, dtype=drone_pos.dtype)
            close_triggered = cached_close.to(device=drone_pos.device)
        else:
            target_flat = target_pos.squeeze(1) if target_pos.dim() == 3 else target_pos
            (
                _assignment,
                _normal_targets,
                _close_targets,
                close_triggered,
                role_targets,
                _forward_dir,
                _lateral,
            ) = self._compute_expert2_role_assignment_and_targets(
                drone_pos,
                target_pos,
                target_vel,
                target_pos_pred=self._select_expert2_intercept_prediction(target_flat),
                update_assignment_state=False,
            )
        role_dist = torch.norm(role_targets - drone_pos, dim=-1)
        ready_mask = self.prev_expert2_role_dist_ready.expand_as(role_dist)
        mode_switch = (~ready_mask) | (close_triggered != self.prev_expert2_close_triggered)
        prev_role_dist = torch.where(ready_mask, self.prev_expert2_role_dist, role_dist)
        effective_prev_role_dist = torch.where(mode_switch, role_dist, prev_role_dist)
        role_progress = effective_prev_role_dist - role_dist
        self.prev_expert2_role_dist.copy_(role_dist)
        self.prev_expert2_role_dist_ready.fill_(True)
        self.prev_expert2_close_triggered.copy_(close_triggered)

        normal_reward = (
            self.expert2_role_reward_coef
            * torch.tanh(role_progress / max(self.expert2_role_reward_scale, 1e-6))
            * (~close_triggered).float()
        )
        close_reward = (
            self.expert2_close_reward_coef
            * torch.tanh(role_progress / max(self.expert2_close_reward_scale, 1e-6))
            * close_triggered.float()
        )
        return normal_reward, close_reward

    # ══════════════════════════════════════════════════════════════════
    #  minimal_v3 reward  (r_terminal + r_progress + r_safety +
    #                       r_control + r_time)
    # ══════════════════════════════════════════════════════════════════
    def _compute_reward_minimal_v3(self):
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses())
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        drone_vel = self.drone.get_velocities()[..., :3]
        target_vel = self.target.get_velocities()[..., :3].squeeze(1)

        v_max_sum = self.current_pursuer_speed + self.target_max_speed.squeeze(-1)
        dt = max(float(self.trajectory_plan_dt), 1e-6)
        d_max_step = v_max_sum * dt
        inv_d_max_step = d_max_step.clamp_min(1e-6).reciprocal()

        # ── 1.  Progress:  distance-based  ───────────────────────────
        d_curr = torch.norm(target_pos - drone_pos, dim=-1)          # [E,N]
        self.episode_min_target_distance.copy_(
            torch.minimum(
                self.episode_min_target_distance,
                d_curr.min(dim=-1, keepdim=True).values,
            )
        )
        d_prev = self.m3_prev_target_dist                            # [E,N]
        delta_i = ((d_prev - d_curr) * inv_d_max_step).clamp(-1.0, 1.0)  # [E,N]

        d_curr_best, _ = d_curr.min(dim=-1)                          # [E]
        d_prev_best, _ = d_prev.min(dim=-1)                          # [E]
        delta_best = (
            (d_prev_best - d_curr_best) * inv_d_max_step.squeeze(-1)
        ).clamp(-1.0, 1.0).unsqueeze(-1)                              # [E,1]

        r_progress = (
            self.m3_progress_best_weight * delta_best
            + self.m3_progress_mean_weight * delta_i.mean(dim=-1, keepdim=True)
        )  # [E,1]
        self.m3_prev_target_dist.copy_(d_curr)

        # ── 2.  Safety  ─────────────────────────────────────────────
        # 2a.  Pair separation  (quadratic penalty)
        # off_diag already removes self-pairs, so this is [E, N, N-1].
        drone_pairwise_dist = torch.norm(self.drone_rpos, dim=-1)
        d_pair = self.m3_safety_pair_dist
        f_pair = ((d_pair - drone_pairwise_dist) / max(d_pair, 1e-6)).clamp(0.0, 1.0) ** 2
        p_pair = f_pair.sum(dim=(1, 2)) / max(self.num_agents * (self.num_agents - 1), 1)

        # 2b.  Environment safety  (floor + wall boundary)
        arena_r = self.arena_size
        d_env_xy = arena_r - torch.norm(drone_pos[..., :2], dim=-1)  # radial margin
        d_env_z_top = self.max_height - drone_pos[..., 2]             # ceiling margin
        d_env_z_bot = drone_pos[..., 2] - self.ground_collision_height  # floor margin
        d_env = torch.minimum(torch.minimum(d_env_xy, d_env_z_top), d_env_z_bot)  # [E,N]
        d_env_ref = self.m3_safety_env_dist
        f_env = ((d_env_ref - d_env) / max(d_env_ref, 1e-6)).clamp(0.0, 1.0) ** 2
        p_env = f_env.mean(dim=-1)                                    # [E]

        r_safety = (
            -self.m3_pair_safety_weight * p_pair.unsqueeze(-1)
            - self.m3_env_safety_weight * p_env.unsqueeze(-1)
        )  # [E,1]

        # ── 3.  Control  (on executed portion only) ─────────────────
        p_u = self.m3_action_magnitude                              # [E,N]  (set in _pre_sim_step)
        p_du = self.m3_action_delta                                 # [E,N]  (set in _pre_sim_step)
        r_control = (
            -self.m3_control_effort_weight * p_u.mean(dim=-1, keepdim=True)
            - self.m3_control_delta_weight * p_du.mean(dim=-1, keepdim=True)
        )  # [E,1]

        # ── 4.  Time  ───────────────────────────────────────────────
        r_time = torch.full(
            (self.num_envs, 1),
            -self.m3_time_penalty,
            device=self.device,
            dtype=drone_pos.dtype,
        )

        # ── 5.  Terminal events  (per-step check) ───────────────────
        catch_radius = float(self.catch_radius)
        control_min = getattr(self, "_control_min_target_distance", d_curr.min(-1, keepdim=True).values)
        self.episode_min_target_distance.copy_(torch.minimum(self.episode_min_target_distance, control_min))
        capture = control_min <= catch_radius

        goal_reached = point_in_cylinder(
            target_pos.squeeze(1),
            self.goal_region_center,
            self.goal_region_radius,
            self.goal_region_height,
        ).unsqueeze(-1)

        collision_drone = (
            (self.drone_rpos.norm(dim=-1) < 2.0 * self.collision_radius).float().sum(-1) > 0
        ).any(dim=-1, keepdim=True)
        out_of_bounds = (
            (drone_pos[..., -1] > self.max_height)
            | ((drone_pos[..., 0] ** 2 + drone_pos[..., 1] ** 2) > arena_r ** 2)
        ).any(dim=-1, keepdim=True)
        collision_floor = (
            drone_pos[..., -1] < self.ground_collision_height
        ).any(dim=-1, keepdim=True)
        landed = (
            (drone_pos[..., -1] < self.landed_z_threshold)
            & (drone_vel[..., :3].norm(dim=-1) < self.landed_speed_threshold)
        ).any(dim=-1, keepdim=True)

        timeout = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        )

        # Priority-ordered terminal detection (first true wins)
        done_capture = capture
        done_goal = ~done_capture & goal_reached
        done_ground = ~done_capture & ~done_goal & collision_floor
        done_landed = ~done_capture & ~done_goal & ~done_ground & landed
        done_drone = (
            ~done_capture & ~done_goal & ~done_ground & ~done_landed
            & collision_drone
        )
        done_out_of_bounds = (
            ~done_capture & ~done_goal & ~done_ground & ~done_landed
            & ~done_drone & out_of_bounds
        )
        done_timeout = (
            ~done_capture & ~done_goal & ~done_ground & ~done_landed
            & ~done_drone & ~done_out_of_bounds & timeout
        )

        r_terminal = torch.zeros(self.num_envs, 1, device=self.device, dtype=drone_pos.dtype)
        r_terminal = torch.where(done_capture, torch.tensor(30.0, device=self.device, dtype=r_terminal.dtype), r_terminal)
        r_terminal = torch.where(done_goal, torch.tensor(-30.0, device=self.device, dtype=r_terminal.dtype), r_terminal)
        collision_terminal_reward = torch.tensor(
            -self.m3_terminal_collision_penalty,
            device=self.device,
            dtype=r_terminal.dtype,
        )
        r_terminal = torch.where(done_ground, collision_terminal_reward, r_terminal)
        r_terminal = torch.where(done_landed, collision_terminal_reward, r_terminal)
        r_terminal = torch.where(done_drone, collision_terminal_reward, r_terminal)
        # Leaving the legal flight volume is an explicit terminal failure with
        # exactly the same penalty as a collision.
        r_terminal = torch.where(done_out_of_bounds, collision_terminal_reward, r_terminal)
        r_terminal = torch.where(done_timeout, torch.tensor(-3.0, device=self.device, dtype=r_terminal.dtype), r_terminal)

        terminal_mask = (
            done_capture | done_goal | done_ground | done_landed
            | done_drone | done_out_of_bounds | done_timeout
        )

        # ── 6.  Assemble  ───────────────────────────────────────────
        # Dense reward is passed to PPO without any clipping, so the
        # logged m3_dense_raw is the true sum of the dense terms.
        dense = r_progress + r_safety + r_control + r_time           # [E,1]

        reward = torch.where(terminal_mask, r_terminal, dense)       # [E,1]
        reward = reward.expand(-1, self.num_agents).clone()          # [E,N]

        # ── 7.  Done flags  ─────────────────────────────────────────
        done = terminal_mask

        # ── 8.  Stats  ──────────────────────────────────────────────
        # A terminal step executes r_terminal *instead of* the dense reward.
        # Log only reward terms that were actually active so that
        # m3_dense_raw + m3_terminal equals m3_reward for every episode.
        zero_dense = torch.zeros_like(dense)
        logged_progress = torch.where(terminal_mask, zero_dense, r_progress)
        logged_safety_pair = torch.where(
            terminal_mask,
            zero_dense,
            -self.m3_pair_safety_weight * p_pair.unsqueeze(-1),
        )
        logged_safety_env = torch.where(
            terminal_mask,
            zero_dense,
            -self.m3_env_safety_weight * p_env.unsqueeze(-1),
        )
        logged_control = torch.where(terminal_mask, zero_dense, r_control)
        logged_time = torch.where(terminal_mask, zero_dense, r_time)
        logged_dense = torch.where(terminal_mask, zero_dense, dense)

        self.stats["m3_progress"].add_(logged_progress)
        self.stats["m3_progress_best_delta"].add_(delta_best)
        self.stats["m3_safety_pair"].add_(logged_safety_pair)
        self.stats["m3_safety_env"].add_(logged_safety_env)
        self.stats["m3_control"].add_(logged_control)
        self.stats["m3_time"].add_(logged_time)
        self.stats["m3_terminal"].add_(r_terminal)
        self.stats["m3_dense_raw"].add_(logged_dense)
        self.stats["m3_reward"].add_(reward.mean(dim=-1, keepdim=True))
        self._record_termination_stats(
            capture=done_capture,
            target_goal=done_goal,
            ground_collision=done_ground,
            landed=done_landed,
            drone_collision=done_drone,
            out_of_bounds=done_out_of_bounds,
            timeout=done_timeout,
        )

        capture_before_goal = done_capture
        self._update_curriculum_metrics(done, capture_before_goal)

        ep_len = torch.clamp(self.progress_buf.unsqueeze(-1), min=1)

        # averaged stats
        self.stats["episode_duration_s"].copy_(torch.where(
            done, ep_len.float() * self.trajectory_plan_dt, self.stats["episode_duration_s"]))
        for key in (
            "m3_progress", "m3_safety_pair", "m3_safety_env",
            "m3_control", "m3_time", "m3_terminal", "m3_dense_raw", "m3_reward",
            "m3_progress_best_delta",
        ):
            self.stats[key].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))

        self.stats["return"].add_(reward.mean(-1).unsqueeze(-1))

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "reward": {
                    "agents": {
                        "reward": reward.unsqueeze(-1),
                    },
                },
                "done": done,
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        if self.reward_profile == "minimal_v3":
            return self._compute_reward_minimal_v3()
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses())
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        target_vel = self.target.get_velocities()[..., :3].squeeze(1)

        target_dist = torch.norm(target_pos - drone_pos, dim=-1)
        self.episode_min_target_distance.copy_(
            torch.minimum(
                self.episode_min_target_distance,
                target_dist.min(dim=-1, keepdim=True).values,
            )
        )
        goal_dist = point_to_cylinder_distance(
            target_pos.squeeze(1), self.goal_region_center, self.goal_region_radius, self.goal_region_height
        )
        goal_progress_reward_val = self.goal_progress_coef * torch.tanh(
            (goal_dist.unsqueeze(-1) - self.prev_goal_dist) / max(self.goal_progress_scale, 1e-6)
        )
        self.prev_goal_dist.copy_(goal_dist.unsqueeze(-1))
        goal_reached = point_in_cylinder(
            target_pos.squeeze(1), self.goal_region_center, self.goal_region_radius, self.goal_region_height
        ).unsqueeze(-1)
        self.stats["goal_reached"].copy_(
            torch.logical_or(self.stats["goal_reached"].bool(), goal_reached).float()
        )

        topk = min(self.num_agents, 2)
        prev_team_dist = self.prev_target_dist.topk(topk, dim=1, largest=False).values.mean(dim=1, keepdim=True)
        curr_team_dist = target_dist.topk(topk, dim=1, largest=False).values.mean(dim=1, keepdim=True)
        capture_progress_reward_val = self.capture_progress_coef * torch.tanh(
            (prev_team_dist - curr_team_dist) / max(self.capture_progress_scale, 1e-6)
        )
        self.prev_target_dist.copy_(target_dist)

        self.capture = target_dist < self.catch_radius
        capture_done = torch.any(self.capture, dim=-1).unsqueeze(-1)

        drone_vel = self.drone.get_velocities()
        drone_speed_norm = torch.norm(drone_vel[..., :3], dim=-1)
        speed_excess = torch.relu(drone_speed_norm / max(self.current_pursuer_speed, 1e-6) - 1.0)
        speed_reward = -self.speed_coef * speed_excess

        drone_pos_dist = torch.norm(self.drone_rpos, dim=-1)
        collision_drone = (drone_pos_dist < 2.0 * self.collision_radius).float().sum(-1)
        self.stats["collision_drone"].add_(collision_drone.mean(-1).unsqueeze(-1))
        self.stats["pursuer_collisions_count"].add_((collision_drone.sum(-1) / 2).unsqueeze(-1))
        closest_drone_dist = drone_pos_dist.min(dim=-1).values
        soft_separation_span = max(self.soft_separation_start - self.soft_separation_end, 1e-6)
        soft_separation_ratio = (
            (self.soft_separation_start - closest_drone_dist) / soft_separation_span
        ).clamp(0.0, 1.0)
        separation_penalty = (
            -self.soft_separation_penalty_coef
            * soft_separation_ratio
            * (collision_drone == 0).float()
        )

        collision_wall = (
            (drone_pos[..., -1] > self.max_height).float()
            + ((drone_pos[..., 0] ** 2 + drone_pos[..., 1] ** 2) > self.arena_size ** 2).float()
        )
        collision_floor = (drone_pos[..., -1] < self.ground_collision_height).float()
        collision_binary = (
            (collision_drone > 0)
            | (collision_wall > 0)
            | (collision_floor > 0)
        ).float()
        collision_penalty = -self.collision_coef * collision_binary
        collision_flag = collision_binary.any(dim=1, keepdim=True)
        self.stats["collision"].add_(collision_flag.float())
        self.stats["collision_wall"].add_(collision_wall.mean(-1).unsqueeze(-1))
        self.stats["collision_floor"].add_(collision_floor.mean(-1).unsqueeze(-1))
        floor_collision_done = (collision_floor > 0).any(dim=1, keepdim=True)

        landed = (
            (drone_pos[..., -1] < self.landed_z_threshold)
            & (drone_speed_norm < self.landed_speed_threshold)
        )
        any_landed = landed.any(dim=1, keepdim=True)
        landed_penalty = -self.landed_penalty_coef * any_landed.expand_as(target_dist).float()
        self.stats["any_landed"].add_(any_landed.float())

        drone_collision_done = (collision_drone > 0).any(dim=1, keepdim=True)
        timeout = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            & ~capture_done
            & ~goal_reached
            & ~floor_collision_done
            & ~any_landed
            & ~drone_collision_done
        )
        capture_before_goal = (
            capture_done
            & ~goal_reached
            & ~floor_collision_done
            & ~any_landed
            & ~drone_collision_done
        )
        catch_reward = self.catch_reward_coef * capture_before_goal.expand_as(target_dist).float()
        goal_penalty_val = -self.goal_penalty_coef * goal_reached.expand_as(target_dist).float()
        timeout_penalty_val = -self.timeout_penalty_coef * timeout.expand_as(target_dist).float()
        terminal_reward = catch_reward + goal_penalty_val + timeout_penalty_val

        coop_anneal = 1.0
        if self.coop_anneal_portion > 0:
            coop_anneal = min(self.training_progress / self.coop_anneal_portion, 1.0)
        coop_weight = self.coop_reward_init + (
            self.coop_reward_final - self.coop_reward_init
        ) * coop_anneal
        coop_reward = coop_weight * (
            self.coop_diff_coef * self.d_i_clip
            + self.coop_phi_coef * self.phi_team.unsqueeze(-1).expand(-1, self.num_agents)
        )

        airborne_mask = (drone_pos[..., -1] > self.landed_z_threshold).float()
        smoothness_reward = self.smoothness_coef * (torch.exp(-self.action_error_order1) - 1.0) * airborne_mask
        # Penalize longer episodes more strongly while keeping the maximum
        # per-step scale controlled by time_penalty_coef.
        step_fraction = (
            self.progress_buf.unsqueeze(-1).float()
            / max(float(self.max_episode_length), 1.0)
        ).clamp(0.0, 1.0)
        time_penalty = -self.time_penalty_coef * step_fraction.expand_as(target_dist)

        spread_reward = self.spread_reward_coef * self.phi_spread.unsqueeze(-1).expand(-1, self.num_agents)
        role_reward, close_reward = self._compute_expert2_waypoint_rewards(
            drone_pos,
            target_pos,
            target_vel,
        )

        if self.reward_profile == "expert2_minimal":
            reward_collision_penalty = -self.collision_coef * collision_binary
            reward = (
                catch_reward
                + goal_penalty_val
                + timeout_penalty_val
                + reward_collision_penalty
                + landed_penalty
                + spread_reward
                + close_reward
                + time_penalty
            )
            terminal_reward = catch_reward + goal_penalty_val + timeout_penalty_val
            collision_penalty = reward_collision_penalty
        elif self.reward_profile == "expert2_role":
            reward_collision_penalty = -self.collision_coef * collision_binary
            reward = (
                catch_reward
                + goal_penalty_val
                + timeout_penalty_val
                + separation_penalty
                + role_reward
                + close_reward
                + reward_collision_penalty
                + landed_penalty
            )
            terminal_reward = catch_reward + goal_penalty_val + timeout_penalty_val
            collision_penalty = reward_collision_penalty + separation_penalty
        else:
            reward = (
                terminal_reward
                + goal_progress_reward_val.expand(-1, self.num_agents)
                + capture_progress_reward_val.expand(-1, self.num_agents)
                + coop_reward
                + collision_penalty
                + landed_penalty
                + speed_reward
                + smoothness_reward
                + time_penalty
            )

        # Success is a terminal outcome, not "was briefly close before still
        # losing to the goal". It is updated after terminal cause arbitration.
        progress_step = self.progress_buf.unsqueeze(-1).float()
        current_capture_step = (
            capture_before_goal.float() * progress_step
            + (~capture_before_goal).float() * self.max_episode_length
        )
        self.stats["first_capture_step"].copy_(
            torch.min(self.stats["first_capture_step"], current_capture_step)
        )
        self.stats["terminal_reward"].add_(terminal_reward.mean(-1).unsqueeze(-1))
        self.stats["goal_progress_reward"].add_(goal_progress_reward_val)
        self.stats["capture_progress_reward"].add_(capture_progress_reward_val)
        self.stats["catch_reward"].add_(catch_reward.mean(-1).unsqueeze(-1))
        self.stats["goal_penalty"].add_(goal_penalty_val.mean(-1).unsqueeze(-1))
        self.stats["speed_reward"].add_(speed_reward.mean(-1).unsqueeze(-1))
        self.stats["time_penalty"].add_(time_penalty.mean(-1).unsqueeze(-1))
        self.stats["coop_reward"].add_(coop_reward.mean(-1).unsqueeze(-1))
        self.stats["spread_reward"].add_(spread_reward.mean(-1).unsqueeze(-1))
        self.stats["close_reward"].add_(close_reward.mean(-1).unsqueeze(-1))
        self.stats["role_reward"].add_(role_reward.mean(-1).unsqueeze(-1))
        self.stats["strategic_goal_rush"].add_(self.predictor_strategic_probs[:, 0:1])
        self.stats["strategic_evade"].add_(self.predictor_strategic_probs[:, 1:2])
        self.stats["phi_team"].add_(self.phi_team.unsqueeze(-1))
        self.stats["phi_block"].add_(self.phi_block.unsqueeze(-1))
        self.stats["phi_pressure"].add_(self.phi_pressure.unsqueeze(-1))
        self.stats["phi_spread"].add_(self.phi_spread.unsqueeze(-1))
        self.stats["d_i_mean"].add_(self.d_i_clip.mean(dim=-1, keepdim=True))
        self.stats["d_i_min"].add_(self.d_i_clip.min(dim=-1, keepdim=True).values)
        urgent_block = ((self.g_urg > 0.5) & (self.phi_block > 0.6)).float().unsqueeze(-1)
        self.stats["urgent_block_rate"].add_(urgent_block)
        n_agents_ahead = (self.my_ahead_proj_goal > 0).float().sum(dim=1, keepdim=True)
        self.stats["n_agents_ahead_mean"].add_(n_agents_ahead)
        self.stats["collision_penalty"].add_(collision_penalty.mean(-1).unsqueeze(-1))
        self.stats["separation_penalty"].add_(separation_penalty.mean(-1).unsqueeze(-1))
        self.stats["landed_penalty"].add_(landed_penalty.mean(-1).unsqueeze(-1))
        self.stats["smoothness_reward"].add_(smoothness_reward.mean(-1).unsqueeze(-1))
        self.stats["smoothness_mean"].add_(self.drone.throttle_difference.mean(-1).unsqueeze(-1))
        self.stats["smoothness_max"].copy_(
            torch.max(
                self.drone.throttle_difference.max(-1).values.unsqueeze(-1),
                self.stats["smoothness_max"],
            )
        )

        done = (
            timeout
            | capture_before_goal
            | goal_reached
            | floor_collision_done
            | any_landed
            | drone_collision_done
        )
        self._record_termination_stats(
            capture=capture_before_goal,
            target_goal=goal_reached,
            ground_collision=floor_collision_done,
            landed=any_landed,
            drone_collision=drone_collision_done,
            timeout=timeout,
        )
        self._update_curriculum_metrics(done, capture_before_goal)

        ep_len = torch.clamp(self.progress_buf.unsqueeze(-1), min=1)
        averaged_stats = [
            "collision",
            "action_error_order1_mean",
            "target_predicted_error",
            "smoothness_mean",
            "smoothness_reward",
            "terminal_reward",
            "goal_progress_reward",
            "capture_progress_reward",
            "catch_reward",
            "goal_penalty",
            "speed_reward",
            "time_penalty",
            "coop_reward",
            "spread_reward",
            "close_reward",
            "role_reward",
            "strategic_goal_rush",
            "strategic_evade",
            "phi_team",
            "phi_block",
            "phi_pressure",
            "phi_spread",
            "d_i_mean",
            "d_i_min",
            "urgent_block_rate",
            "n_agents_ahead_mean",
            "collision_penalty",
            "separation_penalty",
            "collision_wall",
            "collision_floor",
            "collision_drone",
            "landed_penalty",
        ]
        for key in averaged_stats:
            self.stats[key].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))

        self.stats["return"].add_(reward.mean(-1).unsqueeze(-1))

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "reward": {
                    "agents": {
                        "reward": reward.unsqueeze(-1),
                    },
                },
                "done": done,
            },
            self.batch_size,
        )
        
    def _get_dummy_policy_prey(self):
        previous_maneuver = self.target_maneuver_id.clone()
        previous_tactical = self.target_tactical_intent.clone()
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses(False))
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        drone_vel = self.drone.get_velocities()[..., :3]
        target_vel = self.target.get_velocities()[..., :3]
        goal_force_dir, goal_dist, _goal_offset = self._compute_target_goal_force(target_pos)
        boundary_force = self._compute_target_boundary_force(target_pos)

        pursuer_offset = drone_pos - target_pos
        pursuer_dist = torch.norm(pursuer_offset, dim=-1)
        min_pursuer_dist, nearest_idx = pursuer_dist.min(dim=1)
        repulsion_dir = safe_normalize(target_pos - drone_pos)
        repulsion_force = repulsion_dir / (pursuer_dist.unsqueeze(-1) + 1e-5)

        nearest_mask = torch.nn.functional.one_hot(
            nearest_idx, num_classes=self.num_agents
        ).unsqueeze(-1).to(repulsion_force.dtype)
        nearest_repulsion = (repulsion_force * nearest_mask).sum(dim=1, keepdim=True)
        other_repulsion = repulsion_force.sum(dim=1, keepdim=True) - nearest_repulsion

        sprint_mask = goal_dist.squeeze(-1) <= self.target_goal_sprint_distance
        evade_mask = (~sprint_mask) & (
            min_pursuer_dist <= self.target_tactical_far_distance
        )
        far_mask = ~(sprint_mask | evade_mask)
        goal_priority_mask = goal_dist.squeeze(-1) <= min_pursuer_dist
        evade_goal_priority_mask = evade_mask & goal_priority_mask
        normal_evade_mask = evade_mask & (~goal_priority_mask)

        new_tactical = torch.where(normal_evade_mask, TARGET_INTENT_EVADE, TARGET_INTENT_GOAL_RUSH)
        execution_stage = torch.zeros_like(self.target_controller_stage)
        execution_stage[normal_evade_mask] = 1
        execution_stage[evade_goal_priority_mask] = 2
        execution_stage[sprint_mask] = 3
        tactical_changed = self.target_controller_stage != execution_stage
        self.target_controller_stage.copy_(execution_stage)
        self.target_maneuver_timer.sub_(1)
        self.target_maneuver_timer[tactical_changed] = 0
        self.target_tactical_intent.copy_(new_tactical)

        entered_goal_priority = evade_goal_priority_mask & (~self.target_goal_priority_active)
        if entered_goal_priority.any():
            priority_env_ids = entered_goal_priority.nonzero(as_tuple=False).squeeze(-1)
            self.target_maneuver_id[priority_env_ids] = TARGET_MANEUVER_STRAIGHT
            self.target_maneuver_timer[priority_env_ids] = 0
        non_evade_env_ids = (~evade_mask).nonzero(as_tuple=False).squeeze(-1)
        if non_evade_env_ids.numel() > 0:
            self.target_maneuver_id[non_evade_env_ids] = TARGET_MANEUVER_STRAIGHT
            self.target_maneuver_timer[non_evade_env_ids] = 0
        self.target_goal_priority_active.copy_(evade_goal_priority_mask)

        force = boundary_force.clone()
        if far_mask.any():
            force[far_mask] += (
                self.target_far_goal_gain * goal_force_dir[far_mask]
            )
        if normal_evade_mask.any():
            normal_base_force = (
                boundary_force[normal_evade_mask]
                + self.target_near_goal_gain * goal_force_dir[normal_evade_mask]
                + self.target_nearest_repulsion_coef * nearest_repulsion[normal_evade_mask]
                + self.target_other_repulsion_coef * other_repulsion[normal_evade_mask]
            )
            maneuver_force, maneuver_ids = self._select_target_optimal_maneuver_force(
                target_pos=target_pos[normal_evade_mask],
                target_vel=target_vel[normal_evade_mask],
                drone_pos=drone_pos[normal_evade_mask],
                drone_vel=drone_vel[normal_evade_mask],
                base_force=normal_base_force.squeeze(1),
                goal_force=goal_force_dir[normal_evade_mask],
                goal_dist_now=goal_dist[normal_evade_mask],
                max_accel=self.target_max_accel[normal_evade_mask],
                max_speed=self.target_max_speed[normal_evade_mask],
                env_ids=normal_evade_mask.nonzero(as_tuple=False).squeeze(-1),
            )
            force[normal_evade_mask] += (
                self.target_near_goal_gain * goal_force_dir[normal_evade_mask]
                + self.target_nearest_repulsion_coef * nearest_repulsion[normal_evade_mask]
                + self.target_other_repulsion_coef * other_repulsion[normal_evade_mask]
                + maneuver_force
            )
            normal_evade_env_ids = normal_evade_mask.nonzero(as_tuple=False).squeeze(-1)
            self.target_maneuver_id[normal_evade_env_ids] = maneuver_ids
        if evade_goal_priority_mask.any():
            priority_base_force = (
                boundary_force[evade_goal_priority_mask]
                + self.target_goal_priority_gain * goal_force_dir[evade_goal_priority_mask]
                + self.target_goal_priority_repulsion_scale
                * (
                    self.target_nearest_repulsion_coef * nearest_repulsion[evade_goal_priority_mask]
                    + self.target_other_repulsion_coef * other_repulsion[evade_goal_priority_mask]
                )
            )
            maneuver_force, maneuver_ids = self._select_target_optimal_maneuver_force(
                target_pos=target_pos[evade_goal_priority_mask],
                target_vel=target_vel[evade_goal_priority_mask],
                drone_pos=drone_pos[evade_goal_priority_mask],
                drone_vel=drone_vel[evade_goal_priority_mask],
                base_force=priority_base_force.squeeze(1),
                goal_force=goal_force_dir[evade_goal_priority_mask],
                goal_dist_now=goal_dist[evade_goal_priority_mask],
                max_accel=self.target_max_accel[evade_goal_priority_mask],
                max_speed=self.target_max_speed[evade_goal_priority_mask],
                env_ids=evade_goal_priority_mask.nonzero(as_tuple=False).squeeze(-1),
            )
            force[evade_goal_priority_mask] += (
                self.target_goal_priority_gain * goal_force_dir[evade_goal_priority_mask]
                + self.target_goal_priority_repulsion_scale
                * (
                    self.target_nearest_repulsion_coef * nearest_repulsion[evade_goal_priority_mask]
                    + self.target_other_repulsion_coef * other_repulsion[evade_goal_priority_mask]
                )
                + maneuver_force
            )
            priority_env_ids = evade_goal_priority_mask.nonzero(as_tuple=False).squeeze(-1)
            self.target_maneuver_id[priority_env_ids] = maneuver_ids
        if sprint_mask.any():
            sprint_repulsion = self.target_goal_sprint_repulsion_scale * (
                self.target_nearest_repulsion_coef * nearest_repulsion[sprint_mask]
                + self.target_other_repulsion_coef * other_repulsion[sprint_mask]
            )
            sprint_base_force = (
                boundary_force[sprint_mask]
                + self.target_goal_sprint_gain * goal_force_dir[sprint_mask]
                + sprint_repulsion
            )
            maneuver_force, maneuver_ids = self._select_target_optimal_maneuver_force(
                target_pos=target_pos[sprint_mask],
                target_vel=target_vel[sprint_mask],
                drone_pos=drone_pos[sprint_mask],
                drone_vel=drone_vel[sprint_mask],
                base_force=sprint_base_force.squeeze(1),
                goal_force=goal_force_dir[sprint_mask],
                goal_dist_now=goal_dist[sprint_mask],
                max_accel=self.target_max_accel[sprint_mask],
                max_speed=self.target_max_speed[sprint_mask],
                env_ids=sprint_mask.nonzero(as_tuple=False).squeeze(-1),
            )
            force[sprint_mask] += (
                self.target_goal_sprint_gain * goal_force_dir[sprint_mask]
                + sprint_repulsion
                + maneuver_force
            )
            sprint_env_ids = sprint_mask.nonzero(as_tuple=False).squeeze(-1)
            self.target_maneuver_id[sprint_env_ids] = maneuver_ids

        self.stats["target_maneuver_switches"].add_(
            (self.target_maneuver_id != previous_maneuver).float().unsqueeze(-1))
        self.stats["target_tactical_switches"].add_(
            (self.target_tactical_intent != previous_tactical).float().unsqueeze(-1))
        return force.type(torch.float32)

    # visualize functions
    def _draw_court_circle(self):
        if self.draw is None:
            return
        self.draw.clear_lines()

        arena_p1, arena_p2, arena_colors, arena_sizes = draw_court_circle(
            self.arena_size,
            self.max_height,
            color_edge=(0.82, 0.82, 0.82, 1.0),
            color_wall=(0.82, 0.82, 0.82, 1.0),
            line_size=3.0,
            wall_alpha=0.015,
        )

        goal_halo_edge = (1.0, 0.78, 0.28, 1.0)
        goal_halo_wall = (1.0, 0.74, 0.24, 1.0)
        halo_p1, halo_p2, halo_colors, halo_sizes = draw_court_circle(
            self.goal_region_radius + 0.06,
            self.goal_region_height,
            color_edge=goal_halo_edge,
            color_wall=goal_halo_wall,
            line_size=10.0,
            wall_alpha=0.18,
        )

        goal_edge = (1.0, 0.35, 0.12, 1.0)
        goal_wall = (1.0, 0.42, 0.14, 1.0)
        goal_p1, goal_p2, goal_colors, goal_sizes = draw_court_circle(
            self.goal_region_radius,
            self.goal_region_height,
            color_edge=goal_edge,
            color_wall=goal_wall,
            line_size=9.0,
            wall_alpha=0.42,
        )
        # `draw_court_circle` draws a cylinder spanning z in [0, H], while the
        # task logic treats `goal_region_center` as the midpoint with height H.
        # Shift the draw origin down by H/2 so the rendered guard zone matches
        # the actual in-cylinder test used for rewards/done conditions.
        goal_draw_origin_world = _carb_float3_add(
            self.central_env_pos,
            Float3(
                float(self.goal_region_center[0].item()),
                float(self.goal_region_center[1].item()),
                float((self.goal_region_center[2] - self.goal_region_height / 2.0).item()),
            ),
        )
        goal_center_world = _carb_float3_add(
            self.central_env_pos,
            Float3(
                float(self.goal_region_center[0].item()),
                float(self.goal_region_center[1].item()),
                float(self.goal_region_center[2].item()),
            ),
        )
        goal_top_world = Float3(
            goal_center_world.x,
            goal_center_world.y,
            float(goal_center_world.z + self.goal_region_height / 2.0),
        )
        beacon_top_world = Float3(
            goal_center_world.x,
            goal_center_world.y,
            float(self.central_env_pos.z + self.max_height + 0.8),
        )
        beacon_arm = float(self.goal_region_radius * 0.55)
        floor_marker_z = float(self.central_env_pos.z + 0.05)
        floor_arm = float(self.goal_region_radius * 1.2)

        arena_p1 = [
            _carb_float3_add(p, self.central_env_pos) for p in arena_p1
        ]
        arena_p2 = [
            _carb_float3_add(p, self.central_env_pos) for p in arena_p2
        ]
        halo_p1 = [
            _carb_float3_add(p, goal_draw_origin_world) for p in halo_p1
        ]
        halo_p2 = [
            _carb_float3_add(p, goal_draw_origin_world) for p in halo_p2
        ]
        goal_p1 = [
            _carb_float3_add(p, goal_draw_origin_world) for p in goal_p1
        ]
        goal_p2 = [
            _carb_float3_add(p, goal_draw_origin_world) for p in goal_p2
        ]
        beacon_p1 = [
            goal_top_world,
            Float3(goal_center_world.x - beacon_arm, goal_center_world.y, beacon_top_world.z),
            Float3(goal_center_world.x, goal_center_world.y - beacon_arm, beacon_top_world.z),
            Float3(goal_center_world.x - beacon_arm * 0.75, goal_center_world.y - beacon_arm * 0.75, beacon_top_world.z),
            Float3(goal_center_world.x - floor_arm, goal_center_world.y, floor_marker_z),
            Float3(goal_center_world.x, goal_center_world.y - floor_arm, floor_marker_z),
        ]
        beacon_p2 = [
            beacon_top_world,
            Float3(goal_center_world.x + beacon_arm, goal_center_world.y, beacon_top_world.z),
            Float3(goal_center_world.x, goal_center_world.y + beacon_arm, beacon_top_world.z),
            Float3(goal_center_world.x + beacon_arm * 0.75, goal_center_world.y + beacon_arm * 0.75, beacon_top_world.z),
            Float3(goal_center_world.x + floor_arm, goal_center_world.y, floor_marker_z),
            Float3(goal_center_world.x, goal_center_world.y + floor_arm, floor_marker_z),
        ]
        beacon_colors = [
            (1.0, 0.08, 0.02, 1.0),
            (1.0, 0.22, 0.08, 1.0),
            (1.0, 0.22, 0.08, 1.0),
            (1.0, 0.36, 0.10, 1.0),
            (1.0, 0.10, 0.02, 1.0),
            (1.0, 0.10, 0.02, 1.0),
        ]
        beacon_sizes = [14.0, 12.0, 12.0, 10.0, 10.0, 10.0]

        point_list_1 = arena_p1 + halo_p1 + goal_p1 + beacon_p1
        point_list_2 = arena_p2 + halo_p2 + goal_p2 + beacon_p2
        colors = arena_colors + halo_colors + goal_colors + beacon_colors
        sizes = arena_sizes + halo_sizes + goal_sizes + beacon_sizes
        self.draw.draw_lines(point_list_1, point_list_2, colors, sizes)

    def _draw_traj(self):
        if self.draw is None:
            return
        drone_pos = self.drone_states[..., :3]
        drone_vel = self.drone.get_velocities()[..., :3]
        point_list1, point_list2, colors, sizes = draw_traj(
            drone_pos[self.central_env_idx, :], drone_vel[self.central_env_idx, :], dt=0.02, size=4.0
        )
        point_list1 = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list1
        ]
        point_list2 = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list2
        ]
        self.draw.draw_lines(point_list1, point_list2, colors, sizes)   
    
    def _draw_detection(self):
        if self.draw is None:
            return
        self.draw.clear_points()

        # drone detection
        drone_pos = self.drone_states[..., :3]
        drone_ori = self.drone_states[..., 3:7]
        drone_xaxis = quat_axis(drone_ori, 0)
        drone_yaxis = quat_axis(drone_ori, 1)
        drone_zaxis = quat_axis(drone_ori, 2)
        drone_point_list, drone_colors, drone_sizes = draw_detection(
            pos=drone_pos[self.central_env_idx, :],
            xaxis=drone_xaxis[self.central_env_idx, 0, :],
            yaxis=drone_yaxis[self.central_env_idx, 0, :],
            zaxis=drone_zaxis[self.central_env_idx, 0, :],
            drange=self.drone_detect_radius,
        )

        # target detection
        target_pos, target_ori = self.get_env_poses(self.target.get_world_poses())
        target_xaxis = quat_axis(target_ori, 0)
        target_yaxis = quat_axis(target_ori, 1)
        target_zaxis = quat_axis(target_ori, 2)
        target_point_list, target_colors, target_sizes = draw_detection(
            pos=target_pos[self.central_env_idx, :],
            xaxis=target_xaxis[self.central_env_idx, 0, :],
            yaxis=target_yaxis[self.central_env_idx, 0, :],
            zaxis=target_zaxis[self.central_env_idx, 0, :],
            drange=self.target_detect_radius,
        )
        
        point_list = drone_point_list + target_point_list
        colors = drone_colors + target_colors
        sizes = drone_sizes + target_sizes
        point_list = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list
        ]
        self.draw.draw_points(point_list, colors, sizes)

    def _draw_catch(self):
        if self.draw is None:
            return
        self.draw.clear_points()
        # drone detection
        drone_pos = self.drone_states[..., :3]
        drone_ori = self.drone_states[..., 3:7]
        drone_xaxis = quat_axis(drone_ori, 0)
        drone_yaxis = quat_axis(drone_ori, 1)
        drone_zaxis = quat_axis(drone_ori, 2)
        # catch
        point_list, colors, sizes = draw_catch(
            pos=drone_pos[self.central_env_idx, :],
            xaxis=drone_xaxis[self.central_env_idx, 0, :],
            yaxis=drone_yaxis[self.central_env_idx, 0, :],
            zaxis=drone_zaxis[self.central_env_idx, 0, :],
            drange=self.catch_radius,
        )
        # predicted target
        if self.use_TP_net:
            for step in range(self.target_pos_predicted.shape[1]):
                point_list.append(Float3(self.target_pos_predicted[self.central_env_idx, step].cpu().numpy().tolist()))
                colors.append((1.0, 1.0, 0.0, 0.3))
                sizes.append(20.0)
        point_list = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list
        ]
        # catch, green
        catch_mask = self.capture[self.central_env_idx].unsqueeze(1).expand(-1, 400).reshape(-1)
        for idx in range(len(catch_mask)):
            if catch_mask[idx]:
                colors[idx] = (0.0, 1.0, 0.0, 0.3)
        self.draw.draw_points(point_list, colors, sizes)
