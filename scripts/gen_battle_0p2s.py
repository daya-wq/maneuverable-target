#!/usr/bin/env python3
"""
生成0.2s固定切换机动的10张3D对抗轨迹图
保存到 results/battle_0p2s_switch/
"""
import sys, os
import numpy as np
import torch
import types

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

for name, path in [
    ("omni_drones", "omni_drones"),
    ("omni_drones.learning", "omni_drones/learning"),
    ("omni_drones.envs", "omni_drones/envs"),
    ("omni_drones.envs.hide_and_seek", "omni_drones/envs/hide_and_seek"),
]:
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(project_root, path)]
    mod.__package__ = name
    sys.modules[name] = mod

from omni_drones.envs.hide_and_seek.target_fsm import TargetFSM, STRATEGY_NAMES
from omni_drones.envs.hide_and_seek.proportional_navigation import proportional_navigation_3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ===========================================================================
# 参数
# ===========================================================================
ARENA_RADIUS = 10.0
MAX_HEIGHT   = 5.0
MAX_SPEED    = 3.0
MAX_ACCEL    = 3.0
DT           = 0.01        # 100Hz
MAX_STEPS    = 2000        # 20s
CAPTURE_R    = 0.4
GOAL         = torch.tensor([8.0, 0.0, 2.5], dtype=torch.float32)

SWITCH_STEPS = 20          # 0.2s @ 100Hz，固定切换间隔

STRATEGY_COLORS = {
    "APF_Rush":     "#2ecc71",
    "Serpentine":   "#3498db",
    "Spiral_Climb": "#9b59b6",
    "Sharp_Turn":   "#e74c3c",
    "Dive_Rush":    "#e67e22",
}
PURSUER_COLORS = ["#e74c3c", "#3498db", "#f39c12"]

SAVE_DIR = os.path.join(project_root, "results", "battle_0p2s_switch")
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================================
# 投影
# ===========================================================================
def proj(pts, elev=25, azim=-55):
    e, a = np.radians(elev), np.radians(azim)
    Rz = np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    Rx = np.array([[1,0,0],[0,np.cos(e),-np.sin(e)],[0,np.sin(e),np.cos(e)]])
    R  = Rx @ Rz
    r  = np.array(pts) @ R.T
    return r[:, 0], r[:, 2]


# ===========================================================================
# 仿真
# ===========================================================================
def run_battle(seed, fsm):
    torch.manual_seed(seed); np.random.seed(seed)

    # 目标初始
    a = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(1.0, 3.0)
    tp = torch.tensor([r*np.cos(a), r*np.sin(a),
                        np.random.uniform(1.5, 3.5)], dtype=torch.float32)
    tv = torch.zeros(3, dtype=torch.float32)
    va = np.random.uniform(0, 2*np.pi)
    tv[0] = float(np.random.uniform(1, 2.5)*np.cos(va))
    tv[1] = float(np.random.uniform(1, 2.5)*np.sin(va))
    tv[2] = float(np.random.uniform(-0.5, 0.5))

    # 追捕方
    d0 = np.random.uniform(7.0, 8.0)
    pp = torch.zeros(3, 3, dtype=torch.float32)
    pv = torch.zeros(3, 3, dtype=torch.float32)
    for i in range(3):
        pa = a + np.pi + np.random.uniform(-0.5, 0.5) + i*(2*np.pi/3)
        pr = d0 * np.random.uniform(0.9, 1.1)
        pp[i] = torch.tensor([tp[0].item()+pr*np.cos(pa),
                               tp[1].item()+pr*np.sin(pa),
                               np.random.uniform(1.0, 4.0)], dtype=torch.float32)
        xy = torch.norm(pp[i,:2])
        if xy > ARENA_RADIUS: pp[i,:2] = pp[i,:2]/xy*(ARENA_RADIUS-0.1)
        pp[i,2] = pp[i,2].clamp(0.1, MAX_HEIGHT)

    min_d0 = min(torch.norm(tp - pp[i]).item() for i in range(3))
    fsm.reset(torch.tensor([0]))
    fsm.set_initial_distance(torch.tensor([0]), torch.tensor([min_d0]))
    # 固定切换间隔
    fsm.close_min_steps = SWITCH_STEPS
    fsm.close_max_steps = SWITCH_STEPS

    traj_t, traj_p, strats, min_ds = [], [], [], []
    captured = False; cap_step = -1
    damp = 0.3

    for step in range(MAX_STEPS):
        traj_t.append(tp.clone().numpy())
        traj_p.append(pp.clone().numpy())

        # APF
        go = GOAL - tp
        fg = go / torch.norm(go).clamp(min=1e-5)
        fr = torch.zeros(3, dtype=torch.float32)
        for i in range(3):
            d = (tp - pp[i]); dn = torch.norm(d).clamp(min=1e-5)
            fr += 1.5 * d / (dn**2 + 1e-5)
        fb = torch.zeros(3, dtype=torch.float32)
        xyd = torch.norm(tp[:2])
        if xyd > ARENA_RADIUS*0.85: fb[:2] = -tp[:2]/(xyd+1e-5)*3.0
        if tp[2] > MAX_HEIGHT*0.9: fb[2] -= 3.0
        if tp[2] < 0.3: fb[2] += 3.0

        fsm_f, strat = fsm.compute_strategy_force(
            tp.unsqueeze(0), tv.unsqueeze(0), pp.unsqueeze(0), goal_center=GOAL)
        fsm_f = fsm_f.squeeze(0)
        strats.append(strat[0].item())

        md = min(torch.norm(tp - pp[i]).item() for i in range(3))
        min_ds.append(md)

        apf = fg + fr + fb
        if md < min_d0 * 0.4:
            total = 0.3 * apf + fsm_f
        else:
            total = apf + fsm_f

        fn = torch.norm(total).clamp(min=1e-6)
        accel = MAX_ACCEL * total / fn
        tv = tv + (accel - damp*tv)*DT
        sp = torch.norm(tv)
        if sp > MAX_SPEED: tv = tv/sp*MAX_SPEED
        tp = tp + tv*DT
        xyd2 = torch.norm(tp[:2])
        if xyd2 > ARENA_RADIUS: tp[:2] = tp[:2]/xyd2*ARENA_RADIUS
        tp[2] = tp[2].clamp(0.1, MAX_HEIGHT)

        for i in range(3):
            ap = proportional_navigation_3d(
                pp[i].unsqueeze(0), pv[i].unsqueeze(0),
                tp.unsqueeze(0), tv.unsqueeze(0), N=4.0, max_accel=MAX_ACCEL).squeeze(0)
            pv[i] = pv[i] + (ap - damp*pv[i])*DT
            ps = torch.norm(pv[i])
            if ps > MAX_SPEED: pv[i] = pv[i]/ps*MAX_SPEED
            pp[i] = pp[i] + pv[i]*DT
            pxy = torch.norm(pp[i,:2])
            if pxy > ARENA_RADIUS: pp[i,:2] = pp[i,:2]/pxy*ARENA_RADIUS
            pp[i,2] = pp[i,2].clamp(0.1, MAX_HEIGHT)

        if md < CAPTURE_R:
            captured = True; cap_step = step; break

    return {
        "traj_t": np.array(traj_t),
        "traj_p": np.array(traj_p),
        "strats": strats,
        "min_ds": min_ds,
        "captured": captured,
        "cap_step": cap_step,
        "d0": min_d0,
        "T": len(strats),
    }


# ===========================================================================
# 绘图
# ===========================================================================
def plot_battle(data, sc_id, azim):
    T      = data["T"]
    tpos   = data["traj_t"]
    ppos   = data["traj_p"]
    strats = data["strats"]
    d0     = data["d0"]

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_facecolor("#f5f5f7")

    # 场地边界
    th = np.linspace(0, 2*np.pi, 90)
    fl = np.column_stack([ARENA_RADIUS*np.cos(th),
                           ARENA_RADIUS*np.sin(th), np.zeros(90)])
    fx, fy = proj(fl, azim=azim)
    ax.plot(fx, fy, "--", color="#aaa", lw=0.8, alpha=0.4)

    # XYZ坐标轴
    orig = np.array([-ARENA_RADIUS-0.5, -ARENA_RADIUS-0.5, 0.0])
    ox, oy = proj([orig], azim=azim)
    for i, (lbl, clr) in enumerate(zip("XYZ", ["#e74c3c","#2ecc71","#3498db"])):
        end = orig.copy(); end[i] += 2.0
        ex, ey = proj([end], azim=azim)
        ax.annotate("", xy=(ex[0],ey[0]), xytext=(ox[0],oy[0]),
                    arrowprops=dict(arrowstyle="->", color=clr, lw=1.8))
        ax.text(ex[0]+0.1, ey[0]+0.1, lbl, color=clr, fontsize=11, fontweight="bold")

    # 目标轨迹 (按策略着色)
    tx, ty = proj(tpos, azim=azim)
    for t in range(T-1):
        sid  = strats[t]
        name = STRATEGY_NAMES[sid] if 0 <= sid < 5 else "APF_Rush"
        col  = STRATEGY_COLORS[name]
        ax.plot(tx[t:t+2], ty[t:t+2], color=col, lw=2.8, alpha=0.9, solid_capstyle="round")

    # 时间戳
    interval = max(1, T//10)
    for t in range(0, T, interval):
        ax.plot(tx[t], ty[t], "o", color="#555", ms=3.5, alpha=0.6, zorder=6)
        ax.annotate(f"{t*DT:.1f}s", (tx[t], ty[t]),
                    xytext=(4,4), textcoords="offset points", fontsize=7, color="#444")

    # 目标起止
    ax.plot(tx[0], ty[0], "^", color="#111", ms=14, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5, label="Target START")
    ax.plot(tx[-1], ty[-1], "v", color="#111", ms=14, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5, label="Target END")

    # 围捕标记
    if data["captured"]:
        cs = data["cap_step"]
        cx, cy = proj(tpos[cs:cs+1], azim=azim)
        ax.plot(cx[0], cy[0], "*", color="#e74c3c", ms=28, zorder=15,
                markeredgecolor="white", markeredgewidth=1)
        ax.annotate(f"CAPTURED\n{cs*DT:.2f}s",
                    (cx[0], cy[0]), xytext=(15, -25),
                    textcoords="offset points", fontsize=10,
                    color="#e74c3c", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))

    # 追捕方
    for i in range(3):
        px, py = proj(ppos[:, i, :], azim=azim)
        ax.plot(px, py, "--", color=PURSUER_COLORS[i], lw=1.4, alpha=0.55)
        ax.plot(px[0], py[0], "o", color=PURSUER_COLORS[i], ms=11, zorder=9,
                markeredgecolor="white", markeredgewidth=1.5,
                label=f"Pursuer {i+1} START")
        ax.plot(px[-1], py[-1], "s", color=PURSUER_COLORS[i], ms=8, zorder=9,
                markeredgecolor="white", markeredgewidth=1)

    # 图例
    patches = [mpatches.Patch(color=v, label=k, alpha=0.9)
               for k, v in STRATEGY_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=9,
              framealpha=0.92, edgecolor="#ccc")

    status = "CAPTURED" if data["captured"] else "ESCAPED"
    n_sw   = sum(1 for t in range(1,T) if strats[t]!=strats[t-1])
    ax.set_title(
        f"Scenario {sc_id}  —  {status}\n"
        f"Duration: {T*DT:.2f}s  |  Init dist: {d0:.1f}m  |  "
        f"Min dist: {min(data['min_ds']):.2f}m  |  "
        f"Maneuver switches: {n_sw}  (每 0.2s 切换)",
        fontsize=13, fontweight="bold"
    )
    ax.set_aspect("equal")
    ax.tick_params(length=0, labelbottom=False, labelleft=False)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, f"battle_0p2s_{sc_id:02d}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return n_sw, status, T*DT, min(data["min_ds"])


# ===========================================================================
# 主函数
# ===========================================================================
def main():
    print("="*55)
    print("  生成0.2s固定切换机动对抗轨迹  (10 scenarios)")
    print("="*55)

    fsm = TargetFSM(num_envs=1, device="cpu",
                    max_speed=MAX_SPEED, max_accel=MAX_ACCEL,
                    arena_radius=ARENA_RADIUS, max_height=MAX_HEIGHT)

    seeds  = [42, 77, 123, 200, 314, 500, 618, 720, 888, 999]
    azims  = [-55 + i*8 for i in range(10)]
    summary = []

    for idx, (seed, azim) in enumerate(zip(seeds, azims), start=1):
        print(f"\n[{idx}/10] seed={seed} ...")
        data = run_battle(seed, fsm)
        n_sw, status, dur, md = plot_battle(data, idx, azim)
        summary.append((idx, status, dur, data["d0"], md, n_sw))

    print("\n" + "="*55)
    print(f"  {'#':<4}{'Result':<12}{'Time':>8}{'d0':>8}{'MinD':>8}{'Switches':>10}")
    print("  " + "-"*50)
    for sc, st, dur, d0, md, sw in summary:
        print(f"  {sc:<4}{st:<12}{dur:>7.2f}s{d0:>7.1f}m{md:>7.2f}m{sw:>10}")
    print("="*55)
    print(f"\nAll plots → {SAVE_DIR}/")


if __name__ == "__main__":
    main()
