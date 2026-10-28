"""First-episode evaluation with joint pursuer/target trajectory evidence."""
import json
from pathlib import Path
import numpy as np
import torch

REASON_NAMES = ('ongoing', 'capture', 'target_goal', 'ground_collision',
                'drone_collision', 'timeout_near_miss', 'timeout_no_close',
                'out_of_bounds', 'landed')


def summarize_reward_outcomes(reasons, finished, lengths, dt, returns, discounted, components):
    result = {}
    for code, name in enumerate(REASON_NAMES):
        mask = finished & (reasons == code)
        if not mask.any():
            continue
        item = dict(episodes=int(mask.sum()), mean_return=float(returns[mask].mean()),
                    mean_discounted_return=float(discounted[mask].mean()),
                    mean_duration_s=float(lengths[mask].float().mean()) * dt)
        item.update({'mean_' + key + '_sum': float(value[mask].mean())
                     for key, value in components.items()})
        result[name] = item
    error = None
    if finished.any() and 'dense_raw' in components and 'terminal' in components:
        error = float((returns-components['dense_raw']-components['terminal'])[finished].abs().max())
    return result, error


@torch.no_grad()
def evaluate_with_trajectories(env, base_env, policy, output_dir, seed=20261022, rule_baseline=False, hover_test=False, stochastic=False, reset_finished=False, reset_integrity_audit=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env.eval()
    base_env.eval()
    env.set_seed(seed)
    if hasattr(policy, 'reset_prev_action_cache'):
        policy.reset_prev_action_cache()
    td = env.reset()
    n = base_env.num_envs
    device = base_env.device
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    reasons = torch.zeros(n, dtype=torch.long, device=device)
    lengths = torch.zeros(n, dtype=torch.long, device=device)
    episode_returns = torch.zeros(n, device=device)
    discounted_returns = torch.zeros_like(episode_returns)
    logged_returns = torch.zeros_like(episode_returns)
    gamma = float(getattr(policy, 'gae_gamma', .995))
    component_keys = ('progress', 'safety_pair', 'safety_env', 'control', 'time',
                      'terminal', 'dense_raw') if base_env.reward_profile == 'minimal_v3' else ()
    component_sums = {key: torch.zeros_like(episode_returns) for key in component_keys}
    frames, targets, tactics, maneuvers = [], [], [], []
    speeds, accelerations, turns, motors, actual_motors = [], [], [], [], []
    best_distance = torch.full((n,), float('inf'), device=device)
    terminal_pursuer_z = torch.zeros(n, base_env.num_agents, device=device)
    terminal_pursuer_radius = torch.zeros_like(terminal_pursuer_z)
    pursuer_speeds, applied_vertical_accels = [], []
    saved = min(n, 12)
    switches = torch.zeros(n, dtype=torch.long, device=device)
    tactical_switches = torch.zeros_like(switches)
    previous_vel = previous_id = previous_tactical = None
    dt = float(base_env.trajectory_plan_dt)
    start_dp, _ = base_env.get_env_poses(base_env.drone.get_world_poses())
    start_tp, _ = base_env.get_env_poses(base_env.target.get_world_poses())
    initial_distances = (start_dp-start_tp).norm(dim=-1)
    for step in range(base_env.max_episode_length):
        if hover_test:
            action_td = td
            action_td['agents','action'] = torch.zeros(n,base_env.num_agents,base_env.future_prediction_obs_steps,base_env.action_reference_dim,device=device)
        elif rule_baseline:
            dp, _ = base_env.get_env_poses(base_env.drone.get_world_poses())
            tp, _ = base_env.get_env_poses(base_env.target.get_world_poses())
            dv = base_env.drone.get_velocities()[..., :3]
            tv = base_env.target.get_velocities()[..., :3]
            relative = tp - dp
            lead = (relative.norm(dim=-1, keepdim=True) / (base_env.current_pursuer_speed + 1e-6)).clamp(.1, 1.0)
            aim = relative + tv * lead
            desired_velocity = base_env.current_pursuer_speed * aim / aim.norm(dim=-1, keepdim=True).clamp_min(1.)
            acceleration = 2.0 * (desired_velocity-dv)
            limits = base_env.trajectory_decoder.accel_limits
            normalized = torch.tanh(acceleration / limits)
            control = normalized if base_env.action_reference_dim == 3 else torch.cat([normalized, normalized],-1)
            command = control.unsqueeze(-2).expand(-1,-1,base_env.future_prediction_obs_steps,-1).clone()
            action_td = td
            action_td['agents','action'] = command
        else:
            action_td = policy(td, deterministic=not stochastic)
        transition = env.step(action_td)
        td = transition['next']
        alive = ~finished
        dp, _ = base_env.get_env_poses(base_env.drone.get_world_poses())
        tp, _ = base_env.get_env_poses(base_env.target.get_world_poses())
        velocity = base_env.target.get_velocities()[..., :3].squeeze(1)
        mid = base_env.target_maneuver_id.clone()
        tactical = base_env.target_tactical_intent.clone()
        frames.append(dp[:saved].cpu().numpy())
        targets.append(tp[:saved, 0].cpu().numpy())
        maneuvers.append(mid[:saved].cpu().numpy())
        tactics.append(tactical[:saved].cpu().numpy())
        speeds.extend(velocity[alive].norm(dim=-1).cpu().tolist())
        motors.extend(base_env.pursuer_motor_fraction[alive].reshape(-1).cpu().tolist())
        motors.extend(base_env.target_motor_fraction[alive].reshape(-1).cpu().tolist())
        actual_motors.extend(base_env.drone.throttle[alive].square().reshape(-1).cpu().tolist())
        actual_motors.extend(base_env.target.throttle[alive].square().reshape(-1).cpu().tolist())
        lengths[alive] += 1
        step_reward = td['agents', 'reward'].reshape(n, -1).mean(-1)
        episode_returns[alive] += step_reward[alive]
        discounted_returns[alive] += step_reward[alive] * gamma ** (lengths[alive].float()-1)
        best_distance = torch.where(alive, torch.minimum(best_distance,
            base_env.episode_min_target_distance.reshape(n)), best_distance)
        if previous_vel is not None:
            accelerations.extend(((velocity-previous_vel)[alive].norm(dim=-1)/dt).cpu().tolist())
            s0, s1 = previous_vel.norm(dim=-1), velocity.norm(dim=-1)
            valid = alive & (s0 > .25) & (s1 > .25)
            cosine = (previous_vel * velocity).sum(-1)/(s0*s1).clamp_min(1e-6)
            turns.extend((torch.rad2deg(torch.acos(cosine.clamp(-1,1)))[valid]/dt).cpu().tolist())
            switches += ((mid != previous_id) & alive).long()
            tactical_switches += ((tactical != previous_tactical) & alive).long()
        previous_vel, previous_id, previous_tactical = velocity.clone(), mid, tactical
        newly_done = td['done'].reshape(n).bool() & alive
        reasons[newly_done] = td['stats','termination_reason'].reshape(n)[newly_done].long()
        logged_returns[newly_done] = td['stats', 'return'].reshape(n)[newly_done]
        for key, values in component_sums.items():
            values[newly_done] = td['stats', 'm3_' + key].reshape(n)[newly_done] * lengths[newly_done]
        terminal_pursuer_z[newly_done] = dp[newly_done, :, 2]
        terminal_pursuer_radius[newly_done] = dp[newly_done, :, :2].norm(dim=-1)
        pursuer_speeds.extend(base_env.drone.get_velocities()[alive, :, :3].norm(dim=-1).reshape(-1).cpu().tolist())
        applied_vertical_accels.extend(base_env.pursuer_desired_acc_cmd[alive, :, 2].reshape(-1).cpu().tolist())
        finished |= newly_done
        if finished.all():
            break
        if reset_finished and td['done'].any():
            # Diagnostic only: retain first-episode metrics but match collector
            # partial resets for terminated environments. Default is unchanged.
            td.set('_reset', td['done'].clone())
            td = env.reset(td)
    def percentile(x):
        return float(np.percentile(x,95)) if x else 0.0
    result = {'episodes': n, 'seed': seed, 'rule_baseline': rule_baseline, 'hover_test': hover_test,
              'stochastic': stochastic,
              'reset_finished': reset_finished,
              'advance_physics_on_reset': base_env.advance_physics_on_reset,
              'project_thrust_to_body_axis': bool(base_env.project_thrust_to_body_axis),
              'trajectory_reference_mode': base_env.trajectory_decoder.reference_mode,
              'action_controls_per_knot': base_env.action_reference_dim,
              'floor_use_braking_distance': bool(base_env.floor_use_braking_distance),
              'pursuer_hover_ratio': base_env.pursuer_hover_thrust_ratio,
              'target_hover_ratio': base_env.target_hover_thrust_ratio,
              'capture_rate': float((reasons==1).float().mean()),
              'mean_duration_s': float(lengths.float().mean()*dt),
              'mean_min_distance_m': float(best_distance.mean()),
              'maneuver_switches_per_episode': float(switches.float().mean()),
              'tactical_switches_per_episode': float(tactical_switches.float().mean()),
              'target_speed_p95_mps': percentile(speeds),
              'target_accel_p95_mps2': percentile(accelerations),
              'target_turn_rate_p95_deg_s': percentile(turns),
              'motor_thrust_fraction_max': max(motors) if motors else 0.0,
              'actual_motor_thrust_fraction_max': max(actual_motors) if actual_motors else 0.0,
              'initial_distance_min_m': float(initial_distances.min()),
              'initial_distance_max_m': float(initial_distances.max())}
    result.update(
        pursuer_speed_p95_mps=percentile(pursuer_speeds),
        pursuer_applied_vertical_accel_mean_mps2=float(np.mean(applied_vertical_accels)) if applied_vertical_accels else 0.0,
        terminal_pursuer_z_min_m=float(terminal_pursuer_z.min()),
        terminal_pursuer_z_max_m=float(terminal_pursuer_z.max()),
        terminal_pursuer_radius_max_m=float(terminal_pursuer_radius.max()),
        episodes_with_pursuer_above_ceiling=int((terminal_pursuer_z > base_env.max_height).any(-1).sum()),
        episodes_with_pursuer_beyond_radius=int((terminal_pursuer_radius > float(base_env.arena_size)).any(-1).sum()),
    )
    captured = reasons == 1
    result['capture_time_mean_s'] = float(lengths[captured].float().mean()*dt) if captured.any() else None
    result['termination_counts'] = {str(i): int((reasons==i).sum()) for i in range(9)}
    reason_names = REASON_NAMES
    result['termination_counts_named'] = {name: int((reasons==i).sum()) for i, name in enumerate(reason_names)}
    result['discount_gamma'] = gamma
    result['return_by_reason'], result['reward_accounting_abs_error_max'] = summarize_reward_outcomes(
        reasons, finished, lengths, dt, episode_returns, discounted_returns, component_sums)
    result['logged_return_abs_error_max'] = float((episode_returns-logged_returns)[finished].abs().max()) if finished.any() else None
    if reset_integrity_audit:
        from reset_integrity70 import measure_partial_reset
        result['partial_reset_integrity'] = measure_partial_reset(env, base_env)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    np.savez_compressed(str(output/'trajectories.npz'), pursuers=np.stack(frames),
        target=np.stack(targets), maneuver=np.stack(maneuvers), tactical=np.stack(tactics),
        lengths=lengths.cpu().numpy(), reasons=reasons.cpu().numpy(), dt=dt,
        terminal_pursuer_z=terminal_pursuer_z.cpu().numpy(),
        terminal_pursuer_radius=terminal_pursuer_radius.cpu().numpy(),
        goal=base_env.goal_region_center.cpu().numpy(),
        episode_return=episode_returns.cpu().numpy(), discounted_return=discounted_returns.cpu().numpy(),
        **{'episode_' + key + '_sum': value.cpu().numpy() for key, value in component_sums.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    for e in range(min(saved, 6)):
        end = int(lengths[e])
        d = np.stack(frames)[:end,e]
        t = np.stack(targets)[:end,e]
        fig = plt.figure(figsize=(13,5))
        ax = fig.add_subplot(121)
        ax.plot(t[:,0],t[:,1],color='black',label='Target',linewidth=2)
        for j in range(d.shape[1]):
            ax.plot(d[:,j,0],d[:,j,1],label='Pursuer %d'%(j+1))
            ax.scatter(*d[0,j,:2],marker='o',s=16)
        goal = base_env.goal_region_center.cpu().numpy()
        ax.add_patch(Circle(goal[:2],base_env.goal_region_radius,color='gold',alpha=.35))
        ax.set_aspect('equal',adjustable='datalim'); ax.grid(alpha=.25)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.legend(fontsize=8)
        ax3 = fig.add_subplot(122,projection='3d')
        ax3.plot(t[:,0],t[:,1],t[:,2],color='black',linewidth=2)
        for j in range(d.shape[1]): ax3.plot(d[:,j,0],d[:,j,1],d[:,j,2])
        ax3.set_xlabel('x (m)'); ax3.set_ylabel('y (m)'); ax3.set_zlabel('z (m)')
        mode = 'rule' if rule_baseline else ('hover' if hover_test else ('stochastic' if stochastic else 'deterministic'))
        fig.suptitle('Episode %d | %s | %s | %.2f s'%(e,mode,reason_names[int(reasons[e])],end*dt))
        fig.tight_layout(); fig.savefig(str(output/('trajectory_%02d.png'%e)),dpi=150); plt.close(fig)
    if hasattr(policy, 'reset_prev_action_cache'): policy.reset_prev_action_cache()
    return {'eval/'+k:v for k,v in result.items() if isinstance(v,(int,float))}
