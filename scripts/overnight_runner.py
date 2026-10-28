"""Bounded overnight experiment supervisor; checkpoint-based independent evals.

Run inside the sim+Isaac environment. No shell commands are constructed from
log contents. Every attempt has an independent run directory and diagnostics.
"""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import fcntl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/overnight70_20261021'
OUT.mkdir(parents=True, exist_ok=True)
DEADLINE = dt.datetime(2026,10,22,0,0,tzinfo=dt.timezone.utc).timestamp()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def stop_process(proc):
    if proc is None or proc.poll() is not None: return
    os.killpg(proc.pid,signal.SIGINT)
    try: proc.wait(timeout=35)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=10)


def read_progress(path):
    if path is None or not path.exists(): return []
    rows=[]
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except ValueError: continue
    return rows


def plot_curves(rows, folder):
    if not rows: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=[['drone/approx_kl','drone/ratio_clip_fraction'],
            ['drone/policy_loss','drone/value_loss'],
            ['drone/advantages_mean','drone/advantages_std'],
            ['drone/explained_var','drone/value_abs_error'],
            ['drone/predictor_minADE','drone/predictor_FDE'],
            ['train/stats.termination_capture','train/stats.termination_out_of_bounds',
             'train/stats.termination_ground_collision']]
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for ax,keys in zip(axes.flat,groups):
        for key in keys:
            valid=[r for r in rows if key in r and math.isfinite(r[key])]
            ax.plot([r['train/update'] for r in valid],[r[key] for r in valid],label=key.split('/')[-1],linewidth=.8)
        ax.grid(alpha=.2); ax.legend(fontsize=7); ax.set_xlabel('PPO update')
    fig.tight_layout(); fig.savefig(str(folder/'training_curves.png'),dpi=150); plt.close(fig)


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Supervisor stopped by signal %s' % signum)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    experiment_lock=(OUT/'supervisor.lock').open('w')
    fcntl.flock(experiment_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT/'supervisor.pid').write_text(str(os.getpid()))
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--eval-gpu',type=int,default=1)
    parser.add_argument('--envs',type=int,default=128)
    parser.add_argument('--rollout',type=int,default=32)
    parser.add_argument('--updates',type=int,default=10000)
    parser.add_argument('--start-attempt',type=int,default=1)
    parser.add_argument('--task',default='HideAndSeek70')
    parser.add_argument('--initial-checkpoint',default=None)
    parser.add_argument('--deadline-utc',default=None,
                        help='Timezone-aware ISO runtime limit; default is the original 08:00 Beijing limit')
    parser.add_argument('--actor-lr',type=float,default=1e-6)
    parser.add_argument('--critic-lr',type=float,default=5e-5)
    parser.add_argument('--log-std',type=float,default=-1.5)
    parser.add_argument('--log-std-min',type=float,default=-3.)
    parser.add_argument('--collision-penalty',type=float)
    parser.add_argument('--safety-env-dist',type=float)
    args=parser.parse_args()
    deadline=DEADLINE
    if args.deadline_utc:
        deadline_time=dt.datetime.fromisoformat(args.deadline_utc)
        if deadline_time.tzinfo is None:
            parser.error('--deadline-utc must include a timezone')
        deadline=deadline_time.timestamp()
    common=['task=%s'%args.task,'algo.num_minibatches=4','algo.predictor.num_minibatches=2',
            'algo.predictor.max_speed=17','algo.predictor.max_accel=12',
            'algo.actor.log_std_init=%s'%args.log_std,'algo.actor.log_std_min=%s'%args.log_std_min,
            'algo.actor.lr_scheduler=null','algo.critic.lr_scheduler=null',
            'eval_interval=-1','wandb.mode=disabled']
    if args.collision_penalty is not None:
        common.append('task.m3_terminal_collision_penalty=%s'%args.collision_penalty)
    if args.safety_env_dist is not None:
        common.append('+task.m3_safety_env_dist=%s'%args.safety_env_dist)
    events=OUT/'supervisor_events.jsonl'
    def event(kind, **data):
        data.update(event=kind,utc=dt.datetime.now(dt.timezone.utc).isoformat())
        with events.open('a') as f: f.write(json.dumps(data)+'\n')
    actor_lr,critic_lr=args.actor_lr,args.critic_lr
    warm_checkpoint=args.initial_checkpoint
    if warm_checkpoint is not None:
        warm_checkpoint=str(Path(warm_checkpoint).resolve(strict=True))
    best_checkpoint=None
    best_success=-1.
    eval_proc=eval_file=None
    eval_context=None
    last_rows=[]
    active_train=None
    run_dir=None
    try:
        for attempt in range(args.start_attempt,args.start_attempt+3):
            if time.time()>=deadline: break
            folder=OUT/('attempt_%02d'%attempt); folder.mkdir(exist_ok=True)
            log_path=folder/'train.log'
            command=[sys.executable,'scripts/train.py']+common+[
                'task.sim.device=cuda:%d'%args.gpu,'task.sim.physics_gpu=%d'%args.gpu,
                'task.sim.active_gpu=%d'%args.gpu,'task.env.num_envs=%d'%args.envs,
                'algo.train_every=%d'%args.rollout,'max_iters=%d'%args.updates,
                'save_interval=250','algo.actor.lr=%s'%actor_lr,'algo.critic.lr=%s'%critic_lr,
                '+experiment_attempt=%d'%attempt,
                '+trajectory_evaluation=true','+evaluation_output=%s'%(folder/'final_evaluation')]
            if warm_checkpoint: command.append('model_dir=%s'%warm_checkpoint)
            (folder/'command.json').write_text(json.dumps(command,indent=2))
            log_file=log_path.open('w')
            active_train=subprocess.Popen(command,cwd=str(ROOT),stdout=log_file,stderr=subprocess.STDOUT,start_new_session=True)
            event('train_started',attempt=attempt,pid=active_train.pid,command=command)
            run_dir=None; last_update=0; scheduled=set(); reason=None
            while active_train.poll() is None:
                if run_dir is None:
                    text=log_path.read_text(errors='replace')
                    match=re.search(r'Run output dir: (\S+)',text)
                    if match: run_dir=Path(match.group(1))
                rows=read_progress(run_dir/'progress.jsonl' if run_dir else None)
                if rows:
                    last_rows=rows
                    current=rows[-1]; update=int(current['train/update'])
                    tensorboard_dir=OUT/'tensorboard'
                    tensorboard_dir.mkdir(exist_ok=True)
                    target=Path(current['tb_log_dir']).resolve()
                    link=tensorboard_dir/('attempt_%02d'%attempt)
                    if target.parent==(ROOT/'runs').resolve() and not link.exists():
                        link.symlink_to(target,target_is_directory=True)
                    if update!=last_update:
                        last_update=update
                        finite=all(math.isfinite(current.get(k,0)) for k in (
                            'drone/approx_kl','drone/value_loss','drone/policy_loss','drone/predictor_loss_total'))
                        if not finite: reason='non_finite_training_metric'; break
                        recent=rows[-30:]
                        if update>100 and len(recent)==30 and all(r.get('drone/actor_update_skipped',0)>.95 for r in recent):
                            reason='actor_KL_guard_skipped_30_updates'; break
                        if update%50==0: plot_curves(rows,folder)
                        status={'state':'training','attempt':attempt,'pid':active_train.pid,
                                'run_dir':str(run_dir),'latest':current,'eval_pending':eval_context,
                                'best_capture_rate':best_success,'deadline_utc':deadline}
                        write_json(OUT/'status.json',status)
                    # Use an exact saved checkpoint. Evaluation never resets the training collector.
                    for milestone in (250,1000,2500,5000,7500,10000):
                        if update>=milestone and milestone not in scheduled and eval_proc is None:
                            checkpoint=run_dir/('checkpoint_%d.pt'%(milestone*args.envs*args.rollout))
                            if not checkpoint.exists(): continue
                            eval_dir=folder/('eval_%05d'%milestone)
                            eval_command=[sys.executable,'scripts/train.py']+common+[
                                'task.sim.device=cuda:%d'%args.eval_gpu,'task.sim.physics_gpu=%d'%args.eval_gpu,
                                'task.sim.active_gpu=%d'%args.eval_gpu,'task.env.num_envs=64',
                                'model_dir=%s'%checkpoint,'+evaluate_only=true','+trajectory_evaluation=true',
                                '+evaluation_output=%s'%eval_dir]
                            eval_file=(folder/('eval_%05d.log'%milestone)).open('w')
                            eval_proc=subprocess.Popen(eval_command,cwd=str(ROOT),stdout=eval_file,stderr=subprocess.STDOUT,start_new_session=True)
                            eval_context={'attempt':attempt,'update':milestone,'output':str(eval_dir),'checkpoint':str(checkpoint)}
                            scheduled.add(milestone); event('evaluation_started',**eval_context)
                            break
                if eval_proc is not None and eval_proc.poll() is not None:
                    eval_file.close()
                    summary_path=Path(eval_context['output'])/'summary.json'
                    if summary_path.exists():
                        summary=json.loads(summary_path.read_text())
                        event('evaluation_completed',context=eval_context,summary=summary)
                        success=summary['capture_rate']
                        if success>best_success:
                            best_success=success; best_checkpoint=eval_context['checkpoint']
                        elif best_success>=.2 and success<best_success-.2:
                            reason='capture_rate_drop_over_20pp_vs_best'; break
                        if eval_context['update']>=5000 and success<.05:
                            event('low_capture_at_5000',note='poor learning, not numerical collapse',context=eval_context)
                    else: event('evaluation_failed',context=eval_context,returncode=eval_proc.returncode)
                    eval_proc=None; eval_context=None
                if time.time()>=deadline:
                    reason='morning_deadline'; break
                time.sleep(5)
            if reason is None:
                reason='completed' if active_train.returncode==0 else 'process_exit_%s'%active_train.returncode
            stop_process(active_train); log_file.close()
            plot_curves(last_rows,folder)
            event('attempt_stopped',attempt=attempt,reason=reason,last_update=last_update)
            if reason in ('morning_deadline','completed'): break
            stop_process(eval_proc)
            if eval_file is not None and not eval_file.closed: eval_file.close()
            eval_proc=None; eval_context=None
            actor_lr*=.5; critic_lr*=.5
            warm_checkpoint=best_checkpoint
            if warm_checkpoint is None and run_dir:
                files=list(run_dir.glob('checkpoint_*.pt'))
                files=[p for p in files if p.stem.split('_')[-1].isdigit()]
                if files: warm_checkpoint=str(max(files,key=lambda p:int(p.stem.split('_')[-1])))
            event('parameters_changed',actor_lr=actor_lr,critic_lr=critic_lr,
                  weights_checkpoint=warm_checkpoint,note='new run; optimizer state restored when present, environment and update counter reset; not exact resume')
        stop_process(eval_proc)
        # Produce a final independent evaluation from the latest durable checkpoint.
        if run_dir is not None:
            checkpoints=[p for p in run_dir.glob('checkpoint_*.pt')
                         if p.stem.split('_')[-1].isdigit()]
            if checkpoints:
                checkpoint=max(checkpoints,key=lambda p:int(p.stem.split('_')[-1]))
                final_eval_dir=OUT/'morning_evaluation'
                final_eval_command=[sys.executable,'scripts/train.py']+common+[
                    'task.sim.device=cuda:%d'%args.eval_gpu,'task.sim.physics_gpu=%d'%args.eval_gpu,
                    'task.sim.active_gpu=%d'%args.eval_gpu,'task.env.num_envs=128',
                    'model_dir=%s'%checkpoint,'+evaluate_only=true','+trajectory_evaluation=true',
                    '+evaluation_output=%s'%final_eval_dir]
                with (OUT/'morning_evaluation.log').open('w') as final_log:
                    completed=subprocess.run(final_eval_command,cwd=str(ROOT),stdout=final_log,
                                             stderr=subprocess.STDOUT,timeout=900)
                event('morning_evaluation_finished',checkpoint=str(checkpoint),returncode=completed.returncode)
        write_json(OUT/'finished.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                   'best_capture_rate':best_success,'best_checkpoint':best_checkpoint,
                   'last_update':last_rows[-1]['train/update'] if last_rows else 0})
        summaries=[]
        for p in sorted(OUT.glob('attempt_*/eval_*/summary.json')):
            summaries.append((str(p.parent),json.loads(p.read_text())))
        morning=OUT/'morning_evaluation/summary.json'
        if morning.exists(): summaries.append((str(morning.parent),json.loads(morning.read_text())))
        lines=['# Overnight experiment: actual results','',
               'Generated UTC: '+dt.datetime.now(dt.timezone.utc).isoformat(),
               '','Last local update: '+str(last_rows[-1]['train/update'] if last_rows else 0),
               'Best evaluated capture rate: '+str(best_success),'',
               'See supervisor_events.jsonl for failed attempts and parameter changes.',
               'New runs reset environment and update count; saved optimizer state is restored when present. Do not add counts as if identical physics or exact resume.','']
        for name,summary in summaries:
            lines += ['## '+name,'','```json',json.dumps(summary,indent=2),'```','']
        (OUT/'MORNING_REPORT.md').write_text('\n'.join(lines))
    finally:
        stop_process(active_train); stop_process(eval_proc)


if __name__=='__main__': main()
