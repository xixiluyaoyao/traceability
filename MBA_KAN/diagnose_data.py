import numpy as np
import matplotlib.pyplot as plt
import os


def check_physics_correlation():
    # 1. 强制加载你刚才生成的 v3 数据
    path = 'ultimate_dataset_v3.npz'
    if not os.path.exists(path):
        print(f"❌ 找不到 {path}，请确认生成脚本是否执行成功！")
        return

    print(f"📂 正在尸检数据: {path} ...")
    data = np.load(path)
    X = data['X']  # [N, 4, 30]
    y_dist = data['y_dist']  # [N]

    # 检查是否有 Mass 标签
    if 'y_mass' in data:
        y_mass = data['y_mass']
        print(f"✅ Mass 数据存在，范围: {y_mass.min():.1f} - {y_mass.max():.1f} kg")
        # ⚠️ 关键检查点：Mass 是固定的吗？
        mass_std = np.std(y_mass)
        if mass_std < 1.0:
            print("🟢 状态确认: Mass 已固定 (控制变量成功)")
        else:
            print("🔴 严重警告: Mass 依然是随机的！(标准差 > 1.0)")
            print("   👉 在随机 Mass 下使用全局归一化，模型必挂！")
    else:
        print("⚠️ 警告: 数据中没有 Mass 标签")

    # 2. 提取关键物理量：COD 峰值强度
    # 既然我们用了全局归一化，那么 COD 的绝对最大值应该和距离呈现 1/x 关系
    print("⚡ 正在分析 [峰值浓度] vs [真实距离] 的关系...")

    cod_channels = X[:, 0, :]  # [N, 30]
    # 取每个样本 30 个点里的最大值
    peak_vals = np.max(cod_channels, axis=1)

    # 3. 画图诊断
    plt.figure(figsize=(12, 5))

    # 子图1: 物理相关性 (距离 vs 浓度)
    plt.subplot(1, 2, 1)
    plt.scatter(y_dist, peak_vals, alpha=0.3, s=5, c='purple')
    plt.xlabel("True Distance (km)")
    plt.ylabel("Peak COD (mg/L)")
    plt.title("Correlation Check: Distance vs. Intensity")
    plt.yscale('log')  # 浓度通常是指数衰减，用对数坐标看
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # 子图2: 波形宽度相关性 (距离 vs 偏度/峰度)
    # 理论上：距离越远，扩散越厉害，波形越宽，峰度(Kurtosis)越低
    from scipy.stats import kurtosis
    kurt_vals = kurtosis(cod_channels, axis=1)

    plt.subplot(1, 2, 2)
    plt.scatter(y_dist, kurt_vals, alpha=0.3, s=5, c='teal')
    plt.xlabel("True Distance (km)")
    plt.ylabel("Kurtosis (Shape Sharpness)")
    plt.title("Correlation Check: Distance vs. Shape")
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()

    print("\n🔍 【诊断指南】")
    print("1. 看左图：是否能看到一条清晰的下降曲线？")
    print("   - 如果是一团乱糟糟的云 -> 数据物理性缺失 (Mass未固定 或 采样没采到峰值)")
    print("   - 如果是一条清晰的线 -> 数据没问题，是模型读入有问题")
    print("2. 看 Mass 标准差：")
    print("   - 如果显示红色警告 -> 请立刻修改 generate_micro_data.py 固定 Mass 并重新生成！")


if __name__ == "__main__":
    check_physics_correlation()