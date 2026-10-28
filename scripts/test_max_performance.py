#!/usr/bin/env python3
"""
无人机最大性能测试脚本
=======================
基于 YAML 参数文件中的物理参数，纯计算方式分析各无人机机型在以下条件下的极限性能：
  - 所有电机满油门 (throttle = 1.0)
  - 无倾角 (tilt = 0°)，即推力全部沿 z 轴向上
  - 考虑/不考虑空气阻力

计算内容：
  1. 单电机最大推力 & 总推力
  2. 推重比 (thrust-to-weight ratio)
  3. 最大垂直加速度 (扣除重力)
  4. 终端速度 (阻力平衡时的最大速度)
  5. 0→终端速度的时间历程 (数值积分)
  6. 如果允许倾斜（对比参考），不同倾角下的水平分量

不修改任何项目代码文件。
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# ─── 配置 ───────────────────────────────────────────────────────────────────────
ASSET_DIR = Path(__file__).resolve().parent.parent / "omni_drones" / "robots" / "assets" / "usd"

YAML_FILES = {
    "Crazyflie":   ASSET_DIR / "crazyflie.yaml",
    "Hummingbird": ASSET_DIR / "hummingbird.yaml",
    "Firefly":     ASSET_DIR / "firefly.yaml",
    "Iris":        ASSET_DIR / "iris.yaml",
    "Neo11":       ASSET_DIR / "neo11.yaml",
}

SIM_DT = 0.01       # 仿真步长 (s)，来自 sim_base.yaml
G = 9.81             # 重力加速度 (m/s²)
INTEGRATION_DT = 0.001  # 数值积分步长 (s)
SIM_TIME = 10.0      # 模拟总时长 (s)


# ─── 数据结构 ────────────────────────────────────────────────────────────────────
@dataclass
class DroneParams:
    """从 YAML 解析出的无人机参数"""
    name: str
    mass: float                     # kg
    inertia: Dict[str, float]       # xx, yy, zz
    num_rotors: int
    force_constants: List[float]    # N/(rad/s)^2
    max_rot_velocities: List[float] # rad/s
    moment_constants: List[float]
    arm_lengths: List[float]
    drag_coef: float
    time_constant: Optional[float] = None

    @classmethod
    def from_yaml(cls, name: str, path: Path) -> "DroneParams":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        rc = data["rotor_configuration"]
        return cls(
            name=name,
            mass=data["mass"],
            inertia=data["inertia"],
            num_rotors=rc["num_rotors"],
            force_constants=rc["force_constants"],
            max_rot_velocities=rc["max_rotation_velocities"],
            moment_constants=rc["moment_constants"],
            arm_lengths=rc["arm_lengths"],
            drag_coef=data.get("drag_coef", 0.0),
            time_constant=rc.get("time_constant", None),
        )


@dataclass
class PerformanceResult:
    """性能分析结果"""
    name: str
    mass: float
    weight: float
    num_rotors: int
    
    # 推力
    kf_per_rotor: List[float]       # KF = omega_max^2 * force_constant
    max_thrust_per_rotor: List[float]  # N (= KF, 因为满油门 throttle=1, f(1)=1^2=1)
    total_max_thrust: float
    thrust_to_weight: float
    
    # 加速度
    max_vertical_accel: float       # m/s² (扣除重力)
    max_vertical_accel_g: float     # 以 g 为单位
    
    # 终端速度
    drag_coef: float
    terminal_velocity_no_drag: str  # 无阻力 -> 无限
    terminal_velocity_with_drag: float  # m/s
    
    # 时间历程
    time_to_5ms: float
    time_to_10ms: float
    time_to_terminal_90pct: float


# ─── 核心计算 ────────────────────────────────────────────────────────────────────

def compute_kf(force_constant: float, max_rot_vel: float) -> float:
    """
    计算 KF = max_rotation_velocity^2 * force_constant
    与 RotorGroup.__init__ 中的逻辑一致:
        self.KF = nn.Parameter(max_rot_vels.square() * force_constants)
    """
    return max_rot_vel ** 2 * force_constant


def compute_max_thrust(kf: float) -> float:
    """
    满油门时推力 = KF * f(throttle) = KF * throttle^2
    throttle = 1.0 时, f(1) = 1.0
    所以最大推力 = KF (N)
    """
    return kf


def simulate_vertical_flight(
    total_thrust: float,
    mass: float,
    drag_coef: float,
    dt: float = INTEGRATION_DT,
    t_max: float = SIM_TIME,
) -> Dict:
    """
    数值积分模拟垂直飞行 (无倾角)
    
    动力学方程 (来自 multirotor.py apply_action):
      F_total = T - m*g + F_drag
      F_drag  = drag_coef * mass * velocity  (线性阻力模型)
      a = F_total / m = T/m - g + drag_coef * v
    
    注意: 在代码中 drag_coef 的符号表示:
      self.forces[:] += (self.drag_coef * self.masses) * self.vel[..., :3]
    drag_coef 的物理含义是 负的阻尼系数（因为阻力方向与速度相反）。
    如果 drag_coef < 0，则是真阻力；如果 drag_coef >= 0（如 crazyflie 的 0.0），
    则表示无阻力或模型中阻力通过其他方式处理。
    
    这里我们按物理直觉处理：阻力 = -|drag_coef| * m * v
    """
    n_steps = int(t_max / dt)
    t = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    pos = np.zeros(n_steps + 1)
    acc = np.zeros(n_steps + 1)
    
    # 净向上推力加速度 (不含阻力)
    a_thrust = total_thrust / mass - G  # m/s²
    
    drag = abs(drag_coef)
    
    for i in range(n_steps):
        # 加速度 = 推力/质量 - 重力 - 阻力*速度
        # 注意: 这里 drag_coef * v 是减速效应
        acc[i] = a_thrust - drag * v[i]
        v[i + 1] = v[i] + acc[i] * dt
        v[i + 1] = max(v[i + 1], 0.0)  # 不允许负速度（只考虑向上）
        pos[i + 1] = pos[i] + v[i + 1] * dt
        t[i + 1] = t[i] + dt
    acc[-1] = a_thrust - drag * v[-1]
    
    # 终端速度 (理论值: 当 acc = 0 时)
    if drag > 1e-10:
        v_terminal = a_thrust / drag
    else:
        v_terminal = float('inf')
    
    # 找到达到特定速度的时间
    def find_time_to_speed(target_v):
        for i in range(len(v)):
            if v[i] >= target_v:
                return t[i]
        return float('inf')
    
    time_to_5 = find_time_to_speed(5.0)
    time_to_10 = find_time_to_speed(10.0)
    time_to_90pct = find_time_to_speed(0.9 * v_terminal) if v_terminal != float('inf') else float('inf')
    
    return {
        "t": t, "v": v, "pos": pos, "acc": acc,
        "v_terminal": v_terminal,
        "time_to_5": time_to_5,
        "time_to_10": time_to_10,
        "time_to_90pct": time_to_90pct,
        "max_acc": a_thrust,
    }


def analyze_tilt_performance(total_thrust: float, mass: float, drag_coef: float):
    """
    分析不同倾角下的水平性能 (作为参考对比)
    
    倾角 theta 时:
      - 垂直推力: T*cos(theta) 需要 >= m*g 才能维持高度
      - 水平推力: T*sin(theta)
      - 最大可用倾角: arccos(m*g / T)
      - 水平加速度: T*sin(theta)/m - drag*v_h
    """
    weight = mass * G
    max_tilt_rad = np.arccos(min(weight / total_thrust, 1.0))
    max_tilt_deg = np.degrees(max_tilt_rad)
    
    results = []
    for deg in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, max_tilt_deg]:
        if deg > max_tilt_deg + 0.1:
            continue
        rad = np.radians(deg)
        vert_thrust = total_thrust * np.cos(rad)
        horiz_thrust = total_thrust * np.sin(rad)
        vert_accel = vert_thrust / mass - G
        horiz_accel = horiz_thrust / mass
        
        # 水平终端速度 (如果有阻力)
        drag = abs(drag_coef)
        if drag > 1e-10:
            v_h_terminal = horiz_accel / drag
        else:
            v_h_terminal = float('inf')
        
        results.append({
            "tilt_deg": round(deg, 1),
            "vert_thrust_N": round(vert_thrust, 4),
            "horiz_thrust_N": round(horiz_thrust, 4),
            "vert_accel_ms2": round(vert_accel, 3),
            "horiz_accel_ms2": round(horiz_accel, 3),
            "horiz_terminal_v_ms": round(v_h_terminal, 3) if v_h_terminal != float('inf') else "∞",
            "can_hover": vert_thrust >= weight,
        })
    
    return results, max_tilt_deg


# ─── 主程序 ──────────────────────────────────────────────────────────────────────

def main():
    separator = "=" * 80
    thin_sep = "-" * 80
    
    print(separator)
    print("  无人机最大性能理论分析")
    print(f"  仿真步长 dt = {SIM_DT}s | 重力 g = {G} m/s²")
    print(separator)
    
    all_results = {}
    
    for drone_name, yaml_path in YAML_FILES.items():
        if not yaml_path.exists():
            print(f"\n⚠️  {drone_name}: YAML 文件未找到 ({yaml_path})")
            continue
        
        params = DroneParams.from_yaml(drone_name, yaml_path)
        
        # 1. 计算 KF 和最大推力
        kf_list = [compute_kf(fc, mrv) for fc, mrv in zip(params.force_constants, params.max_rot_velocities)]
        max_thrust_list = [compute_max_thrust(kf) for kf in kf_list]
        total_max_thrust = sum(max_thrust_list)
        weight = params.mass * G
        t2w = total_max_thrust / weight
        
        # 2. 最大垂直加速度 (无倾角)
        max_vert_accel = total_max_thrust / params.mass - G  # m/s²
        
        # 3. 数值积分模拟
        sim_result = simulate_vertical_flight(
            total_thrust=total_max_thrust,
            mass=params.mass,
            drag_coef=params.drag_coef,
        )
        
        # 4. 倾角分析
        tilt_results, max_tilt = analyze_tilt_performance(
            total_max_thrust, params.mass, params.drag_coef
        )
        
        # 打印结果
        print(f"\n{separator}")
        print(f"  🚁 {drone_name}")
        print(f"{separator}")
        
        print(f"\n  📋 基本参数:")
        print(f"     质量:          {params.mass:.4f} kg")
        print(f"     重量:          {weight:.4f} N")
        print(f"     电机数:        {params.num_rotors}")
        print(f"     惯性矩 (xx):   {params.inertia['xx']:.6e}")
        print(f"     惯性矩 (yy):   {params.inertia['yy']:.6e}")
        print(f"     惯性矩 (zz):   {params.inertia['zz']:.6e}")
        print(f"     阻力系数:      {params.drag_coef}")
        if params.time_constant is not None:
            print(f"     电机时间常数:  {params.time_constant}s")
        
        print(f"\n  ⚡ 推力参数:")
        for i in range(params.num_rotors):
            print(f"     电机 {i}: force_const = {params.force_constants[i]:.6e}, "
                  f"ω_max = {params.max_rot_velocities[i]:.1f} rad/s, "
                  f"KF = {kf_list[i]:.6f} N")
        print(f"     {'─' * 60}")
        print(f"     总最大推力:    {total_max_thrust:.4f} N")
        print(f"     推重比:        {t2w:.3f}")
        
        print(f"\n  🚀 最大性能 (垂直, 无倾角, 满油门):")
        print(f"     最大加速度:          {max_vert_accel:.3f} m/s²  ({max_vert_accel/G:.2f} g)")
        if sim_result["v_terminal"] == float('inf'):
            print(f"     终端速度(理论):      ∞ (无阻力)")
        else:
            print(f"     终端速度(理论):      {sim_result['v_terminal']:.3f} m/s")
        print(f"     0→5 m/s 用时:        {sim_result['time_to_5']:.3f} s")
        print(f"     0→10 m/s 用时:       {sim_result['time_to_10']:.3f} s")
        if sim_result["v_terminal"] != float('inf'):
            print(f"     0→90%终端速度 用时:  {sim_result['time_to_90pct']:.3f} s")
        
        # 某些时间点的速度
        print(f"\n  📈 速度-时间采样 (前5秒):")
        print(f"     {'时间(s)':>8}  {'速度(m/s)':>10}  {'加速度(m/s²)':>12}  {'高度(m)':>8}")
        for check_t in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
            idx = min(int(check_t / INTEGRATION_DT), len(sim_result["v"]) - 1)
            print(f"     {sim_result['t'][idx]:8.2f}  {sim_result['v'][idx]:10.3f}  "
                  f"{sim_result['acc'][idx]:12.3f}  {sim_result['pos'][idx]:8.2f}")
        
        # 倾角分析
        print(f"\n  🔄 不同倾角性能对比 (最大可倾斜角: {max_tilt:.1f}°):")
        print(f"     {'倾角(°)':>8}  {'垂直加速度':>10}  {'水平加速度':>10}  {'水平终端速度':>12}  {'可悬停':>6}")
        for tr in tilt_results:
            hover_str = "✅" if tr["can_hover"] else "❌"
            vt_str = f"{tr['horiz_terminal_v_ms']}" if isinstance(tr['horiz_terminal_v_ms'], str) else f"{tr['horiz_terminal_v_ms']:.1f} m/s"
            print(f"     {tr['tilt_deg']:8.1f}  {tr['vert_accel_ms2']:8.3f} m/s²  "
                  f"{tr['horiz_accel_ms2']:8.3f} m/s²  {vt_str:>12}  {hover_str:>6}")
        
        all_results[drone_name] = {
            "params": params,
            "total_thrust": total_max_thrust,
            "t2w": t2w,
            "max_accel": max_vert_accel,
            "terminal_v": sim_result["v_terminal"],
            "sim": sim_result,
            "tilt": tilt_results,
            "max_tilt": max_tilt,
        }
    
    # ─── 横向对比总结 ────────────────────────────────────────────────────────────
    print(f"\n\n{separator}")
    print("  📊 所有机型横向对比总结")
    print(f"{separator}")
    
    header = f"  {'机型':>12}  {'质量(kg)':>8}  {'推重比':>6}  {'最大加速度(m/s²)':>16}  {'最大加速度(g)':>12}  {'终端速度(m/s)':>12}  {'0→5m/s(s)':>10}"
    print(header)
    print(f"  {thin_sep}")
    
    for name, res in all_results.items():
        v_term = f"{res['terminal_v']:.1f}" if res["terminal_v"] != float('inf') else "∞"
        print(f"  {name:>12}  {res['params'].mass:8.4f}  {res['t2w']:6.2f}  "
              f"{res['max_accel']:16.3f}  {res['max_accel']/G:12.2f}  "
              f"{v_term:>12}  {res['sim']['time_to_5']:10.3f}")
    
    # ─── 特别关注 Crazyflie（你的主要目标机型）─────────────────────────────────
    if "Crazyflie" in all_results:
        cf = all_results["Crazyflie"]
        print(f"\n\n{separator}")
        print("  🎯 Crazyflie 详细性能卡片")
        print(f"{separator}")
        print(f"""
    质量:             {cf['params'].mass} kg ({cf['params'].mass*1000:.1f} g)
    电机数:           {cf['params'].num_rotors}
    单电机 KF:        {cf['total_thrust']/cf['params'].num_rotors:.6f} N
    总最大推力:       {cf['total_thrust']:.4f} N
    重量:             {cf['params'].mass * G:.4f} N
    推重比:           {cf['t2w']:.3f}
    净向上推力:       {cf['total_thrust'] - cf['params'].mass * G:.4f} N
    
    === 垂直(无倾角)极限 ===
    最大加速度:       {cf['max_accel']:.3f} m/s² ({cf['max_accel']/G:.2f} g)
    终端速度:         {'∞ (无阻力模型)' if cf['terminal_v'] == float('inf') else f"{cf['terminal_v']:.2f} m/s"}
    
    === 考虑电机响应(time_constant={cf['params'].time_constant}s) ===
    满油门响应时间:   ~{3*cf['params'].time_constant if cf['params'].time_constant else 'N/A'}s (达到95%)
    
    === 最大倾角下的水平性能 ===
    最大倾角:         {cf['max_tilt']:.1f}°
    该倾角下水平加速度: {cf['tilt'][-1]['horiz_accel_ms2']:.3f} m/s²
    """)
    
    print(f"\n{separator}")
    print("  ✅ 分析完成")
    print(f"  注: 以上分析基于 YAML 参数的理论计算")
    print(f"  实际仿真中还需考虑: 电机响应延迟、仿真步长离散化、碰撞等")
    print(f"{separator}")


if __name__ == "__main__":
    main()
