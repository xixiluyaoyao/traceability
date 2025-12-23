import numpy as np
import matplotlib.pyplot as plt
from generate2_2 import RobustUSVSimulator  # 确保这个文件名对上了


def verify_physics_strictly():
    """
    严格物理验证：控制变量法 (Control Variable Experiment)
    强制使用相同的背景、相同的源强，只改变距离。
    """
    print("🔬 正在进行严格物理规律验证...")

    # 初始化模拟器 (不需要真实背景，因为我们要手动造纯净背景)
    sim = RobustUSVSimulator(pkl_path='none')

    # === 设定控制变量 ===
    # 1. 模拟一条理想河道，背景非常干净且恒定
    duration_hours = 24
    dt_min = 1
    t_steps = int(duration_hours * 60 / dt_min)
    t_axis = np.linspace(0, duration_hours, t_steps)  # 小时

    # 背景 COD 设为恒定 5.0 mg/L，只有微小噪声
    bg_cod = np.full(t_steps, 5.0) + np.random.normal(0, 0.1, t_steps)

    # 2. 设定相同的污染源参数
    mass_mg = 100000 * 100 * 1000  # 100,000 mg/L * 100 m3 (大事故)
    U = 0.8  # 流速 0.8 m/s
    W = 50.0  # 河宽 50 m
    Q = U * W * 3.0  # 深度 3m
    D = 1.0 * U * W  # 扩散系数
    k = 0.2  # 衰减系数

    # === 对比实验：近场 vs 中场 vs 远场 ===
    distances = [3.0, 15.0, 45.0]  # km
    colors = ['red', 'orange', 'blue']
    labels = ['Near Field (3km)', 'Mid Field (15km)', 'Far Field (45km)']

    plt.figure(figsize=(14, 6))

    # 绘制全时间段的波形 (Ground Truth)
    for dist, col, lbl in zip(distances, colors, labels):
        dist_m = dist * 1000

        # 调用 ADE 方程
        curve = sim.solve_ade(t_axis, dist_m, mass_mg, U, Q, D, k)
        total_cod = bg_cod + curve

        plt.plot(t_axis * 60, total_cod, color=col, linewidth=2, label=lbl)

        # 标记峰值位置
        peak_idx = np.argmax(total_cod)
        peak_time = t_axis[peak_idx] * 60
        peak_val = total_cod[peak_idx]
        plt.scatter(peak_time, peak_val, color=col, s=50, zorder=5)
        plt.text(peak_time, peak_val + 5, f"Peak: {peak_val:.1f}", color=col, fontweight='bold')

    plt.title("Physics Verification: Same Source, Different Distances (Controlled Experiment)", fontsize=14)
    plt.xlabel("Time since spill (minutes)", fontsize=12)
    plt.ylabel("COD Concentration (mg/L)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # 插入子图：模拟无人船采样看到的景象 (30分钟窗口)
    # -------------------------------------------------
    # 我们在每个波形的峰值附近截取 30分钟，看看"局部"长什么样
    ax_ins = plt.axes([0.65, 0.4, 0.25, 0.25])  # [left, bottom, width, height]
    ax_ins.set_title("What Agent Sees (30min Window)", fontsize=10)

    for dist, col in zip(distances, colors):
        dist_m = dist * 1000
        curve = sim.solve_ade(t_axis, dist_m, mass_mg, U, Q, D, k)

        # 找到峰值并截取前后 15min
        peak_idx = np.argmax(curve)
        start = max(0, peak_idx - 15)
        end = min(len(curve), peak_idx + 15)

        segment = curve[start:end] + bg_cod[start:end]
        # 添加一点传感器噪声模拟实战
        segment += np.random.normal(0, 0.5, len(segment))

        ax_ins.plot(range(len(segment)), segment, color=col, alpha=0.8)

    plt.show()


if __name__ == "__main__":
    verify_physics_strictly()