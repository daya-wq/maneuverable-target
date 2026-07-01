#!/usr/bin/env python3
"""
6-DOF SO(3) 对抗轨迹生成 (0.2s机动切换)

双方都使用非线性六自由度模型:
    外环: APF/FSM/PN → 期望加速度
    内环: SO(3) 几何控制器 → 推力 + 角速度 → 力矩
    动力学: 刚体积分 (平动 + 转动)

保存到 results/battle_6dof_0p2s/
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
from omni_drones.envs.hide_and_seek.quadrotor_6dof import Quadrotor6DOF

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
DT           = 0.01
MAX_STEPS    = 2000       # 20s
CAPTURE_R    = 0.4
GOAL         = torch.tensor([8.0, 0.0, 2.5], dtype=torch.float32)
SWITCH_STEPS = 20         # 0.2s

# 6-DOF 参数
MASS         = 1.0        # kg
J_DIAG       = (0.01, 0.01, 0.02)  # kg·m²
GRAVITY      = 9.81
MAX_THRUST   = 25.0       # N (~2.5g)

STRATEGY_COLORS = {
    "APF_Rush":     "#2ecc71",
    "Serpentine":   "#3498db",
    "Spiral_Climb": "#9b59b6",
    "Sharp_Turn":   "#e74c3c",
    "Dive_Rush":    "#e67e22",
}
PURSUER_STYLES = [
    {"color": "#e74c3c", "ls": "--"},
    {"color": "#3498db", "ls": "-."},
    {"color": "#f39c12", "ls": ":"},
]

SAVE_DIR = os.path.join(project_root, "results", "battle_6dof_0p2s")
os.makedirs(SAVE_DIR, exist_ok=True)


def proj(pts, elev=25, azim=-55):
    e, a = np.radians(elev), np.radians(azim)
    Rz = np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    Rx = np.array([[1,0,0],[0,np.cos(e),-np.sin(e)],[0,np.sin(e),np.cos(e)]])
    R  = Rx @ Rz
    r  = np.array(pts) @ R.T
    return r[:, 0], r[:, 2]


# ===========================================================================
# APF (目标的外环)
# ===========================================================================
def compute_target_apf(tp, pp, goal):
    """计算目标的APF加速度 [3]"""
    go = goal - tp
    fg = go / torch.norm(go).clamp(min=1e-5)

    fr = torch.zeros(3)
    for i in range(pp.shape[0]):
        d = tp - pp[i]
        dn = torch.norm(d).clamp(min=1e-5)
        fr += 1.5 * d / (dn**2 + 1e-5)

    fb = torch.zeros(3)
    xyd = torch.norm(tp[:2])
    if xyd > ARENA_RADIUS * 0.85:
        fb[:2] = -tp[:2] / (xyd + 1e-5) * 3.0
    if tp[2] > MAX_HEIGHT * 0.9:
        fb[2] -= 3.0
    if tp[2] < 0.3:
        fb[2] += 3.0

    return fg + fr + fb


# ===========================================================================
# 仿真
# ===========================================================================
def run_battle_6dof(seed, fsm):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 目标初始状态 ---
    a = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(1.0, 3.0)
    t_pos = torch.tensor([r*np.cos(a), r*np.sin(a),
                           np.random.uniform(1.5, 3.5)], dtype=torch.float32)
    va = np.random.uniform(0, 2*np.pi)
    t_vel = torch.tensor([
        np.random.uniform(1, 2.5)*np.cos(va),
        np.random.uniform(1, 2.5)*np.sin(va),
        np.random.uniform(-0.5, 0.5)
    ], dtype=torch.float32)

    # --- 追捕方初始状态 ---
    d0_range = np.random.uniform(7.0, 8.0)
    p_pos = torch.zeros(3, 3, dtype=torch.float32)
    p_vel = torch.zeros(3, 3, dtype=torch.float32)
    for i in range(3):
        pa = a + np.pi + np.random.uniform(-0.5, 0.5) + i*(2*np.pi/3)
        pr = d0_range * np.random.uniform(0.9, 1.1)
        p_pos[i] = torch.tensor([
            t_pos[0].item() + pr*np.cos(pa),
            t_pos[1].item() + pr*np.sin(pa),
            np.random.uniform(1.0, 4.0)
        ])
        xy = torch.norm(p_pos[i, :2])
        if xy > ARENA_RADIUS:
            p_pos[i, :2] = p_pos[i, :2] / xy * (ARENA_RADIUS - 0.1)
        p_pos[i, 2] = p_pos[i, 2].clamp(0.1, MAX_HEIGHT)

    # 初始距离
    min_d0 = min(torch.norm(t_pos - p_pos[i]).item() for i in range(3))

    # --- 创建6-DOF模型 ---
    target_quad = Quadrotor6DOF(
        num_envs=1, mass=MASS, J_diag=J_DIAG, gravity=GRAVITY,
        max_thrust=MAX_THRUST, max_speed=MAX_SPEED, arena_radius=ARENA_RADIUS,
        max_height=MAX_HEIGHT, k_R=8.0, k_omega=2.5, k_drag=0.2)

    # 3个追捕方共用一个batch
    pursuer_quads = []
    for i in range(3):
        pq = Quadrotor6DOF(
            num_envs=1, mass=MASS, J_diag=J_DIAG, gravity=GRAVITY,
            max_thrust=MAX_THRUST, max_speed=MAX_SPEED, arena_radius=ARENA_RADIUS,
            max_height=MAX_HEIGHT, k_R=10.0, k_omega=3.0, k_drag=0.2)
        pq.reset(torch.tensor([0]), pos=p_pos[i:i+1], vel=p_vel[i:i+1])
        pursuer_quads.append(pq)

    target_quad.reset(torch.tensor([0]), pos=t_pos.unsqueeze(0), vel=t_vel.unsqueeze(0))

    # FSM
    fsm.reset(torch.tensor([0]))
    fsm.set_initial_distance(torch.tensor([0]), torch.tensor([min_d0]))
    fsm.close_min_steps = SWITCH_STEPS
    fsm.close_max_steps = SWITCH_STEPS

    # --- 记录 ---
    traj_t, traj_p = [], []
    strats, min_ds = [], []
    euler_t, euler_p = [], []
    omega_t, omega_p = [], []
    captured = False
    cap_step = -1

    for step in range(MAX_STEPS):
        # 当前状态
        tp = target_quad.pos[0]             # [3]
        tv = target_quad.vel[0]             # [3]
        pp = torch.stack([pq.pos[0] for pq in pursuer_quads])   # [3, 3]
        pv = torch.stack([pq.vel[0] for pq in pursuer_quads])   # [3, 3]

        traj_t.append(tp.detach().numpy().copy())
        traj_p.append(pp.detach().numpy().copy())
        euler_t.append(target_quad.get_euler_angles()[0].detach().numpy().copy())
        omega_t.append(target_quad.get_body_rates()[0].detach().numpy().copy())

        # --- 目标外环 ---
        apf = compute_target_apf(tp, pp, GOAL)

        fsm_f, strat = fsm.compute_strategy_force(
            tp.unsqueeze(0), tv.unsqueeze(0), pp.unsqueeze(0), goal_center=GOAL)
        fsm_f = fsm_f.squeeze(0)
        strats.append(strat[0].item())

        md = min(torch.norm(tp - pp[i]).item() for i in range(3))
        min_ds.append(md)

        # 近距时策略力主导
        if md < min_d0 * 0.4:
            a_des_target = (0.3 * apf + fsm_f) / MASS
        else:
            a_des_target = (apf + fsm_f) / MASS

        # 限幅
        an = torch.norm(a_des_target).clamp(min=1e-6)
        a_des_target = a_des_target * (MAX_ACCEL / an).clamp(max=1.0)

        # --- 追捕方外环 (PN) ---
        for i, pq in enumerate(pursuer_quads):
            a_pn = proportional_navigation_3d(
                pq.pos, pq.vel,
                tp.unsqueeze(0), tv.unsqueeze(0),
                N=4.0, max_accel=MAX_ACCEL
            ).squeeze(0)
            a_des_pursuer = a_pn / MASS
            pq.step(a_des_pursuer, dt=DT)

        # --- 目标内环 (6-DOF + SO3) ---
        target_quad.step(a_des_target.unsqueeze(0), dt=DT)

        # 捕获检测
        if md < CAPTURE_R:
            captured = True
            cap_step = step
            break

    return {
        "traj_t": np.array(traj_t),
        "traj_p": np.array(traj_p),
        "strats": strats,
        "min_ds": min_ds,
        "euler_t": np.array(euler_t),
        "omega_t": np.array(omega_t),
        "captured": captured,
        "cap_step": cap_step,
        "d0": min_d0,
        "T": len(strats),
    }


# ===========================================================================
# 绘图
# ===========================================================================
def plot_battle(data, sc_id, azim):
    T = data["T"]
    tpos = data["traj_t"]
    ppos = data["traj_p"]
    strats = data["strats"]
    d0 = data["d0"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9),
                              gridspec_kw={"width_ratios": [3, 1]})

    # ========== 左: 3D对抗轨迹 ==========
    ax = axes[0]
    ax.set_facecolor("#f5f5f7")

    # 场地边界
    th = np.linspace(0, 2*np.pi, 90)
    fl = np.column_stack([ARENA_RADIUS*np.cos(th),
                           ARENA_RADIUS*np.sin(th), np.zeros(90)])
    fx, fy = proj(fl, azim=azim)
    ax.plot(fx, fy, "--", color="#aaa", lw=0.8, alpha=0.4)

    # 坐标轴
    orig = np.array([-ARENA_RADIUS-0.5, -ARENA_RADIUS-0.5, 0.0])
    ox, oy = proj([orig], azim=azim)
    for i, (lbl, clr) in enumerate(zip("XYZ", ["#e74c3c","#2ecc71","#3498db"])):
        end = orig.copy(); end[i] += 2.0
        ex, ey = proj([end], azim=azim)
        ax.annotate("", xy=(ex[0],ey[0]), xytext=(ox[0],oy[0]),
                    arrowprops=dict(arrowstyle="->", color=clr, lw=1.8))
        ax.text(ex[0]+0.1, ey[0]+0.1, lbl, color=clr, fontsize=11, fontweight="bold")

    # 目标轨迹
    tx, ty = proj(tpos, azim=azim)
    for t in range(T-1):
        sid = strats[t]
        name = STRATEGY_NAMES[sid] if 0 <= sid < 5 else "APF_Rush"
        col = STRATEGY_COLORS[name]
        ax.plot(tx[t:t+2], ty[t:t+2], color=col, lw=2.8, alpha=0.9, solid_capstyle="round")

    # 时间标记
    interval = max(1, T//10)
    for t in range(0, T, interval):
        ax.plot(tx[t], ty[t], "o", color="#555", ms=3.5, alpha=0.6, zorder=6)
        ax.annotate(f"{t*DT:.1f}s", (tx[t], ty[t]),
                    xytext=(4,4), textcoords="offset points", fontsize=7, color="#444")

    # 起止
    ax.plot(tx[0], ty[0], "^", color="#111", ms=14, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(tx[-1], ty[-1], "v", color="#111", ms=14, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5)

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
        ax.plot(px, py, PURSUER_STYLES[i]["ls"],
                color=PURSUER_STYLES[i]["color"], lw=1.4, alpha=0.55)
        ax.plot(px[0], py[0], "o", color=PURSUER_STYLES[i]["color"], ms=10,
                zorder=9, markeredgecolor="white", markeredgewidth=1.5)
        ax.plot(px[-1], py[-1], "s", color=PURSUER_STYLES[i]["color"], ms=7,
                zorder=9, markeredgecolor="white", markeredgewidth=1)

    patches = [mpatches.Patch(color=v, label=k, alpha=0.9)
               for k, v in STRATEGY_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=9,
              framealpha=0.92, edgecolor="#ccc")

    status = "CAPTURED" if data["captured"] else "ESCAPED"
    n_sw = sum(1 for t in range(1,T) if strats[t]!=strats[t-1])
    ax.set_title(
        f"Scenario {sc_id} [6-DOF + SO(3)]  -  {status}\n"
        f"Duration: {T*DT:.2f}s | Init dist: {d0:.1f}m | "
        f"Min dist: {min(data['min_ds']):.2f}m | Switches: {n_sw}",
        fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(length=0, labelbottom=False, labelleft=False)

    # ========== 右: 姿态角 ==========
    ax2 = axes[1]
    euler = data["euler_t"]  # [T, 3] = [roll, pitch, yaw]
    t_axis = np.arange(T) * DT

    ax2.plot(t_axis, np.degrees(euler[:, 0]), color="#e74c3c", lw=1.2, alpha=0.8, label="Roll")
    ax2.plot(t_axis, np.degrees(euler[:, 1]), color="#2ecc71", lw=1.2, alpha=0.8, label="Pitch")
    ax2.plot(t_axis, np.degrees(euler[:, 2]), color="#3498db", lw=1.2, alpha=0.8, label="Yaw")
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("Euler Angles (deg)", fontsize=10)
    ax2.set_title("Target Attitude (SO3)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, f"battle_6dof_{sc_id:02d}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return n_sw, status, T*DT, min(data["min_ds"])


# ===========================================================================
# main
# ===========================================================================
def main():
    print("=" * 60)
    print("  6-DOF SO(3) Battle Trajectories  (0.2s switch, 10 scenes)")
    print("=" * 60)

    fsm = TargetFSM(num_envs=1, device="cpu",
                    max_speed=MAX_SPEED, max_accel=MAX_ACCEL,
                    arena_radius=ARENA_RADIUS, max_height=MAX_HEIGHT)

    seeds = [42, 77, 123, 200, 314, 500, 618, 720, 888, 999]
    azims = [-55 + i*8 for i in range(10)]
    summary = []

    for idx, (seed, azim) in enumerate(zip(seeds, azims), start=1):
        print(f"\n[{idx}/10] seed={seed} ...")
        data = run_battle_6dof(seed, fsm)
        n_sw, status, dur, md = plot_battle(data, idx, azim)
        summary.append((idx, status, dur, data["d0"], md, n_sw))

    print("\n" + "=" * 60)
    print(f"  {'#':<4}{'Result':<12}{'Time':>8}{'d0':>8}{'MinD':>8}{'Switches':>10}")
    print("  " + "-" * 54)
    for sc, st, dur, d0, md, sw in summary:
        print(f"  {sc:<4}{st:<12}{dur:>7.2f}s{d0:>7.1f}m{md:>7.2f}m{sw:>10}")
    print("=" * 60)
    print(f"\nAll plots -> {SAVE_DIR}/")


if __name__ == "__main__":
    main()
