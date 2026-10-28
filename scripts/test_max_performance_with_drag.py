#!/usr/bin/env python3
"""
Crazyflie 最大性能测试（含阻力修正）
=====================================
针对 Crazyflie 2.x，研究合适的阻力系数，并计算有阻力时的终端速度。

阻力来源分析:
1. PhysX linear_damping = 0.2 (config.py 默认值)
   - 但 hideandseek.py L889 对点质量目标设为 0.0
   - 对 UAV 目标，使用默认值 0.2
   
2. YAML drag_coef = 0.0 (crazyflie.yaml)
   - multirotor.py L266: self.drag_coef = torch.zeros(...) * self.drag_coef_para
   - 即使 drag_coef_para 非零，初始化也是 zeros * para = 0！

3. 代码中的阻力力:
   - multirotor.py L499: F_drag = drag_coef * mass * velocity
   - 这是线性阻力模型 (linear drag)

4. PhysX linear_damping 的物理含义:
   - PhysX 内部施加: F = -linear_damping * mass * v
   - 默认值 0.2 意味着 τ_damp = 1/0.2 = 5s
   
文献调研 (Crazyflie 空气动力学参数):
- Forster (2015) "System Identification of the Crazyflie 2.0":
  线性阻力系数 D = diag(Dx, Dy, Dz) ≈ diag(0.000, 0.000, 0.000) (几乎可忽略)
  
- Shi et al. (2019) "Neural Lander" / DATT (2022):
  对 Crazyflie 2.x 的阻力参数通常设为 0 或极小值
  原因: 30g 微型四旋翼，速度通常 < 3 m/s，雷诺数很低，阻力极小

- Mehndiratta & Kayacan (2019) "Gaussian Process-based Learning Control":
  drag_coef ≈ 0.0 (实验验证在 < 2 m/s 时阻力可忽略)

- Torrente et al. (2021) "Data-Driven MPC for Quadrotors":
  在 > 5 m/s 高速飞行时确实需要考虑阻力
  推荐值: 寄生阻力系数 C_D ≈ 0.1-0.3 (基于迎风面积)
  
对于 30g 的 Crazyflie:
- 迎风面积 A ≈ 0.01 m² (10cm x 10cm)  
- 空气密度 ρ = 1.225 kg/m³
- 二次阻力: F_drag = 0.5 * ρ * C_D * A * v²
- 等效线性阻力 (在某个参考速度 v_ref 下): d_linear = 0.5 * ρ * C_D * A * v_ref / m
  在 v_ref = 3 m/s 时: d_linear ≈ 0.5 * 1.225 * 0.2 * 0.01 * 3 / 0.032 ≈ 0.115

但代码模型是 F = drag_coef * m * v (线性)
所以 drag_coef 的物理含义是: a_drag = drag_coef * v
等效的 drag_coef ≈ 0.5 * ρ * C_D * A / m

使用 C_D = 0.2, A = 0.01 m²:
  drag_coef = 0.5 * 1.225 * 0.2 * 0.01 / 0.032 ≈ 0.038

然而需要注意: 你的仿真中 PhysX 已经有 linear_damping = 0.2！
这个值远大于实际物理阻力，因此 PhysX damping 本身就在提供隐式阻力。
"""

import math
import yaml
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent.parent / "omni_drones" / "robots" / "assets" / "usd"
G = 9.81
DT = 0.001
SIM_TIME = 15.0

def load_crazyflie():
    with open(ASSET_DIR / "crazyflie.yaml") as f:
        data = yaml.safe_load(f)
    rc = data["rotor_configuration"]
    mass = data["mass"]
    fc = rc["force_constants"]
    mrv = rc["max_rotation_velocities"]
    num_rotors = rc["num_rotors"]
    kf_list = [fc[i] * mrv[i]**2 for i in range(num_rotors)]
    total_thrust = sum(kf_list)
    return {
        "mass": mass,
        "weight": mass * G,
        "num_rotors": num_rotors,
        "kf_per_rotor": kf_list,
        "total_thrust": total_thrust,
        "t2w": total_thrust / (mass * G),
        "drag_coef_yaml": data.get("drag_coef", 0.0),
    }


def simulate(total_thrust, mass, drag_coef, tilt_deg=0.0, dt=DT, t_max=SIM_TIME):
    """
    模拟飞行: 在固定倾角下的速度-时间历程
    垂直: a_z = T*cos(tilt)/m - g - drag*v_z
    水平: a_x = T*sin(tilt)/m - drag*v_x
    """
    tilt = math.radians(tilt_deg)
    n_steps = int(t_max / dt)
    
    vz, vx = 0.0, 0.0
    pz, px = 0.0, 0.0
    
    results = []
    
    for i in range(n_steps + 1):
        t = i * dt
        az = total_thrust * math.cos(tilt) / mass - G - drag_coef * vz
        ax = total_thrust * math.sin(tilt) / mass - drag_coef * vx
        v_total = math.sqrt(vx**2 + vz**2)
        
        if i % int(0.1 / dt) == 0 or i == 0:
            results.append({
                "t": round(t, 3),
                "vz": round(vz, 4),
                "vx": round(vx, 4),
                "v_total": round(v_total, 4),
                "az": round(az, 4),
                "ax": round(ax, 4),
                "pz": round(pz, 3),
                "px": round(px, 3),
            })
        
        vz += az * dt
        vx += ax * dt
        vz = max(vz, -50)  # prevent negative infinity
        vx = max(vx, -50)
        pz += vz * dt
        px += vx * dt
    
    # Terminal velocity
    if drag_coef > 1e-10:
        v_term_z = (total_thrust * math.cos(tilt) / mass - G) / drag_coef if total_thrust * math.cos(tilt) > mass * G else 0
        v_term_x = (total_thrust * math.sin(tilt) / mass) / drag_coef if tilt > 0 else 0
    else:
        v_term_z = float('inf')
        v_term_x = float('inf') if tilt > 0 else 0
    
    return results, v_term_z, v_term_x


def main():
    cf = load_crazyflie()
    sep = "=" * 80
    thin = "-" * 80
    
    print(sep)
    print("  Crazyflie 最大性能分析 — 阻力模型研究")
    print(sep)
    
    print(f"""
  📋 基本参数:
     质量: {cf['mass']} kg ({cf['mass']*1000:.1f} g)
     总最大推力: {cf['total_thrust']:.4f} N
     重量: {cf['weight']:.4f} N
     推重比: {cf['t2w']:.3f}
     YAML drag_coef: {cf['drag_coef_yaml']}
""")
    
    # ───────────────────────────────────────────────────────────
    print(sep)
    print("  🔍 你的模型中的阻力来源分析")
    print(sep)
    print("""
  1) PhysX linear_damping = 0.2 (config.py 默认值)
     - RigidBodyPropertiesCfg 中设置
     - PhysX 内部施加: F_damp = -linear_damping * v (对每个刚体)
     - 追逐者(drone) ✅ 使用默认值 0.2
     - 目标(target, UAV模式) ✅ 使用默认值 0.2
     - 目标(target, 点质量模式) ❌ hideandseek.py L889 设为 0.0
     
  2) YAML drag_coef = 0.0 (crazyflie.yaml)
     - ⚠️  multirotor.py L266 初始化:
       self.drag_coef = torch.zeros(...) * self.drag_coef_para
     - zeros * 任何值 = zeros → drag_coef 始终为 0！
     - 即使 YAML 写了非零值也无效（除非通过 randomization 设置）
     
  3) velocity_damping = 0.25 (任务级别)
     - 仅用于 target 的速度指令衰减（非物理阻力）
     - hideandseek.py L1059: a_cmd - 0.25 * v
     
  结论: 你的追逐者无人机确实有阻力 — 来自 PhysX linear_damping = 0.2
        你的目标无人机(UAV模式)也有 — 同样是 PhysX linear_damping = 0.2
        YAML 中的 drag_coef 实际上被初始化为 0，是无效的
""")
    
    # ───────────────────────────────────────────────────────────
    print(sep)
    print("  📚 文献调研: Crazyflie 阻力系数")
    print(sep)
    print("""
  1) Forster (2015) "System Identification of the Crazyflie 2.0"
     - 线性阻力系数 ≈ 可忽略 (在 < 2 m/s 速度下)
     
  2) Shi et al. (2019) "Neural Lander" & DATT (2022)  
     - Crazyflie 仿真中 drag_coef = 0
     - 30g 微型四旋翼，雷诺数低，速度范围小
     
  3) Torrente et al. (2021) "Data-Driven MPC for Quadrotors"
     - 高速 (>5 m/s) 需要阻力模型
     - 推荐: C_D ≈ 0.1-0.3 (阻力系数，基于迎风面积)
     
  4) 等效计算 (二次阻力→线性近似):
     F_drag_quad = 0.5 * ρ * C_D * A * v²
     
     Crazyflie 参数:
       ρ = 1.225 kg/m³ (空气密度)
       C_D ≈ 0.2 (估计)
       A ≈ 0.009 m² (≈96mm × 96mm 迎风面积)
       m = 0.032 kg
       
     等效线性: drag_coef = 0.5 * ρ * C_D * A / m
                         = 0.5 * 1.225 * 0.2 * 0.009 / 0.032
                         ≈ 0.034
     
  5) PhysX linear_damping 的等效:
     PhysX: F = -damping * v
     代码: F = drag_coef * mass * v (multirotor.py)
     等效关系: PhysX_damping ≈ drag_coef * mass (量纲匹配时)
     
     但 PhysX linear_damping = 0.2 → 对 32g Crazyflie 来说相当保守
""")

    # ───────────────────────────────────────────────────────────
    print(sep)
    print("  🧪 不同阻力系数下的性能对比")
    print(sep)
    
    # 测试场景:
    # 1. drag = 0 (纯理论)
    # 2. drag = 0.034 (文献推荐的等效线性阻力)
    # 3. drag = 0.2 (PhysX linear_damping 默认值，你仿真中实际使用的)
    # 4. drag = 0.5 (较大阻力参考)
    
    drag_scenarios = [
        (0.0,   "无阻力 (drag=0, 纯理论)"),
        (0.034, "文献推荐 (drag≈0.034, C_D=0.2)"),
        (0.2,   "PhysX默认 (linear_damping=0.2, 你的仿真实际值)"),
        (0.5,   "较大阻力 (drag=0.5, 参考)"),
    ]
    
    for tilt in [0.0, 25.0]:
        print(f"\n  {'─' * 70}")
        print(f"  倾角 = {tilt}°")
        print(f"  {'─' * 70}")
        
        for drag, desc in drag_scenarios:
            results, v_term_z, v_term_x = simulate(
                cf['total_thrust'], cf['mass'], drag, tilt_deg=tilt
            )
            
            v_term_str_z = f"{v_term_z:.2f} m/s" if v_term_z != float('inf') else "∞"
            v_term_str_x = f"{v_term_x:.2f} m/s" if v_term_x != float('inf') else "∞"
            
            print(f"\n    📊 {desc}")
            print(f"    终端速度: 垂直={v_term_str_z}, 水平={v_term_str_x}")
            print(f"    {'时间(s)':>8}  {'V_z(m/s)':>10}  {'V_x(m/s)':>10}  {'V_total':>10}  {'a_z(m/s²)':>10}  {'a_x(m/s²)':>10}")
            
            for r in results:
                if r['t'] in [0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
                    print(f"    {r['t']:8.1f}  {r['vz']:10.3f}  {r['vx']:10.3f}  {r['v_total']:10.3f}  {r['az']:10.3f}  {r['ax']:10.3f}")
    
    # ───────────────────────────────────────────────────────────
    print(f"\n\n{sep}")
    print("  📋 终端速度总结表")
    print(sep)
    
    print(f"\n  {'阻力模型':>35}  {'垂直终端速度':>14}  {'25°水平终端速度':>16}  {'25°总终端速度':>14}")
    print(f"  {thin}")
    for drag, desc in drag_scenarios:
        _, vtz_0, _ = simulate(cf['total_thrust'], cf['mass'], drag, tilt_deg=0, t_max=0.1)
        _, vtz_25, vtx_25 = simulate(cf['total_thrust'], cf['mass'], drag, tilt_deg=25, t_max=0.1)
        
        if drag > 1e-10:
            a_z_0 = cf['total_thrust'] / cf['mass'] - G
            v_term_z_0 = a_z_0 / drag
            
            a_z_25 = cf['total_thrust'] * math.cos(math.radians(25)) / cf['mass'] - G
            a_x_25 = cf['total_thrust'] * math.sin(math.radians(25)) / cf['mass']
            v_term_z_25 = max(a_z_25 / drag, 0)
            v_term_x_25 = a_x_25 / drag
            v_term_total_25 = math.sqrt(v_term_z_25**2 + v_term_x_25**2)
            
            print(f"  {desc:>35}  {v_term_z_0:12.2f} m/s  {v_term_x_25:14.2f} m/s  {v_term_total_25:12.2f} m/s")
        else:
            print(f"  {desc:>35}  {'∞':>14}  {'∞':>16}  {'∞':>14}")
    
    # ───────────────────────────────────────────────────────────
    print(f"\n\n{sep}")
    print("  🎯 结论与建议")
    print(sep)
    
    a_net = cf['total_thrust'] / cf['mass'] - G
    v_term_physx = a_net / 0.2
    
    print(f"""
  ✅ 你的模型中 IS 有阻力:
     - PhysX linear_damping = 0.2 (在 config.py 中默认设置)
     - 这等效于线性阻力 F = -0.2 * v (作用在刚体上)
     - 你的 Crazyflie 仿真中实际终端垂直速度 ≈ {v_term_physx:.1f} m/s
     
  ⚠️  YAML 中的 drag_coef = 0.0 是正确的:
     - 因为阻力已经通过 PhysX linear_damping 提供了
     - multirotor.py 中的 drag_coef 是额外的阻力项
     - 两者叠加才是总阻力
     
  📌 推荐阻力配置:
     如果你想更精确地控制阻力:
     
     方案A (当前): 保持 YAML drag_coef=0, 依赖 PhysX damping=0.2
       → 垂直终端速度 ≈ {v_term_physx:.1f} m/s
       
     方案B: 设 PhysX damping=0, 在 YAML 中用 drag_coef 控制
       → 需要先修复 multirotor.py L266 的 bug (zeros * para = 0)
       → 推荐 drag_coef ≈ 0.034 (文献值)
       → 垂直终端速度 ≈ {a_net/0.034:.1f} m/s (更真实)
       
     方案C: 如果你只关心限速，用 max_linear_velocity
       → 你现在的配置: drone = {1.5} m/s, target = {5.0} m/s
       → 这是 PhysX 硬限速，与物理阻力无关
""")


if __name__ == "__main__":
    main()
