"""Unopposed goal sprint: original motor model, zero initial velocity, 100 Hz."""
import importlib
import itertools
import json
from pathlib import Path
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name='train')
@torch.no_grad()
def main(cfg):
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.resolve(cfg)
    app = init_simulation_app(cfg)
    module = importlib.import_module('omni_drones.envs.hide_and_seek.hideandseek')
    env = module.HideAndSeek(cfg, headless=True)
    output = Path(str(cfg.arrival_output))
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.yaml').write_text(OmegaConf.to_yaml(cfg))
    cases = list(itertools.product([-48.,-38.],[-10.,0.,10.],[15.,17.],[10.,12.]))
    assert env.num_envs == len(cases)
    grid = torch.tensor(cases,device=env.device)
    # Use normal production reset and immediately replace only the diagnostic
    # initial conditions. Keep pursuers far from the target route, at floor level.
    original_sample = env._sample_target_episode_dynamics
    def sample_capability(ids):
        result = original_sample(ids)
        env.target_max_speed[ids] = grid[ids,2].view(-1,1,1)
        env.target_max_accel[ids] = grid[ids,3].view(-1,1,1)
        return result
    def sample_positions(ids):
        dp = torch.zeros(len(ids),3,3,device=env.device)
        dp[...,1] = torch.tensor([60.,70.,80.],device=env.device)
        dp[...,2] = 1.
        tp = torch.zeros(len(ids),1,3,device=env.device)
        tp[:,0,:2] = grid[ids,:2]
        tp[...,2] = 12.
        return dp,tp
    env._sample_target_episode_dynamics = sample_capability
    env._sample_goal_defense_curriculum_positions = sample_positions
    # Remove only evasive/obstacle/vertical maneuver intent for this named
    # unopposed straight-line diagnostic. The UAV controller is unchanged.
    def straight_acceleration():
        state = env.target.get_state()
        force,_,_ = env._compute_target_goal_force(state[...,:3])
        return env.target_max_accel*force
    env._compute_target_desired_acc = straight_acceleration
    env.eval(); env.reset()
    initial = env.target.get_state()[..., :3].clone()
    assert torch.allclose(initial[:,0,:2],grid[:,:2],atol=.02), 'Reset did not apply requested locations'
    assert env.target.get_velocities()[...,:3].norm(dim=-1).max() < 1e-5
    arrival = torch.full((len(cases),),-1.,device=env.device)
    records = []
    motor_max = 0.
    for step in range(int(env.max_episode_length*env.control_decimation)):
        env.drone.get_state()
        env.drone.apply_action(torch.full((len(cases),3,4),-1.,device=env.device))
        env._step_uav_target()
        env.sim.step(render=False)
        state = env.target.get_state()[...,:13]
        motor_max = max(motor_max,float(env.target.throttle.square().max()))
        inside = module.point_in_cylinder(state[:,0,:3],env.goal_region_center,
                    env.goal_region_radius,env.goal_region_height)
        newly = inside & (arrival < 0.)
        arrival[newly] = (step+1)*env.dt
        records.append(state.cpu().numpy().copy())
        if step % 500 == 0:
            print('ARRIVAL_STEP',step,'arrived',int((arrival>=0).sum()),flush=True)
        if bool((arrival>=0).all()):
            break
    result = dict(planner_dt=env.trajectory_plan_dt,control_dt=env.dt,
                  max_episode_steps=env.max_episode_length,
                  episode_seconds=env.max_episode_length*env.trajectory_plan_dt,
                  height_limit=env.max_height,thrust_fraction=env.thrust_fraction,
                  hover_fraction=env.target_hover_thrust_ratio,
                  actual_motor_max=motor_max,cases=[])
    for i,(x,y,speed,accel) in enumerate(cases):
        distance=float(((x-60.)**2+y*y)**.5-3.)
        t_ideal=distance/speed+speed/(2.*accel)
        result['cases'].append(dict(x=x,y=y,vmax=speed,amax=accel,
              distance_to_guard_m=distance,ideal_seconds=t_ideal,
              measured_seconds=float(arrival[i])))
    valid=arrival[arrival>=0.]
    result.update(arrived=int(len(valid)),total=len(cases),
                  measured_min_seconds=float(valid.min()) if len(valid) else None,
                  measured_max_seconds=float(valid.max()) if len(valid) else None)
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(str(output/'trajectories.npz'),target=np.stack(records),dt=env.dt)
    print(json.dumps(result,indent=2),flush=True)
    env.close(); app.close()


if __name__=='__main__':
    main()
