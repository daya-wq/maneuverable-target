"""Measure the physical effect of resetting only environment zero."""
import torch


@torch.no_grad()
def measure_partial_reset(env, base_env):
    results = {}
    original_mode = base_env.advance_physics_on_reset
    original_step = base_env.sim.step
    try:
        for name, mode in [('legacy', True), ('isolated', False)]:
            base_env.advance_physics_on_reset = mode
            count = [0]
            def counted_step(*args, **kwargs):
                count[0] += 1
                return original_step(*args, **kwargs)
            base_env.sim.step = counted_step
            positions = base_env.drone.get_world_poses(clone=True)[0]
            velocities = base_env.drone.get_velocities(clone=True)
            target_positions = base_env.target.get_world_poses(clone=True)[0]
            history = base_env.agent_history_buffer.clone()
            mask = torch.zeros(base_env.num_envs, 1, dtype=torch.bool, device=base_env.device)
            mask[0] = True
            env.reset(env.fake_tensordict().set('_reset', mask))
            delta_p = (base_env.drone.get_world_poses(clone=True)[0] - positions)[1:]
            delta_v = (base_env.drone.get_velocities(clone=True) - velocities)[1:]
            delta_t = (base_env.target.get_world_poses(clone=True)[0] - target_positions)[1:]
            results[name] = dict(
                global_physics_steps=count[0],
                other_drone_position_change_max_m=float(delta_p.norm(dim=-1).max()),
                other_drone_velocity_change_max=float(delta_v.norm(dim=-1).max()),
                other_target_position_change_max_m=float(delta_t.norm(dim=-1).max()),
                other_agent_history_change_max=float((base_env.agent_history_buffer-history)[1:].abs().max()),
            )
    finally:
        base_env.sim.step = original_step
        base_env.advance_physics_on_reset = original_mode
    return results
