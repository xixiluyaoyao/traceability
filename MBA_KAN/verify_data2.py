import numpy as np
import matplotlib.pyplot as plt
import os

# 设置绘图风格，使其看起来像论文插图
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('ggplot')


def verify_dataset(file_path='train_dataset_v3.npz'):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"📂 正在加载数据集: {file_path} ...")
    data = np.load(file_path)
    X = data['X']  # (N, 4, Window)
    y_dist = data['y_dist']  # (N,)
    y_pos = data['y_pos']  # (N,)

    print(f"✅ 加载完成. 样本数: {X.shape[0]}, 窗口长度: {X.shape[2]}")
    print("-" * 50)

    # ==========================================
    # 1. 物理规律验证: 近场 vs 远场
    # ==========================================
    print("🔍 验证物理扩散效应 (Dispersion Effect)...")

    # 筛选近场 (< 5km) 和 远场 (> 40km) 的样本
    near_indices = np.where(y_dist < 5.0)[0]
    far_indices = np.where(y_dist > 40.0)[0]

    if len(near_indices) > 0 and len(far_indices) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

        # 绘制近场案例
        idx_near = near_indices[0]
        axes[0].plot(X[idx_near, 0, :], 'r-', linewidth=2, label='COD (mg/L)')
        axes[0].set_title(f"Near Field (Dist={y_dist[idx_near]:.1f}km)\nExpect: Sharp Peak & High Conc")
        axes[0].set_xlabel("Time Steps (min)")
        axes[0].set_ylabel("COD (mg/L)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 绘制远场案例
        idx_far = far_indices[0]
        axes[1].plot(X[idx_far, 0, :], 'b-', linewidth=2, label='COD (mg/L)')
        axes[1].set_title(f"Far Field (Dist={y_dist[idx_far]:.1f}km)\nExpect: Flat/Plateau & Low Conc")
        axes[1].set_xlabel("Time Steps (min)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ 样本中未找到极近或极远的对比样本。")

    # ==========================================
    # 2. 鲁棒性验证: 寻找并展示"脏数据"
    # ==========================================
    print("🔍 验证非理想传感器工况 (Robustness)...")

    # 随机抽取 9 个样本，通常能看到丢包或噪声
    indices = np.random.choice(len(X), 9, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    fig.suptitle("Random Samples Check (Look for Noise, Dropouts, Saturation)", fontsize=14)

    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        cod_seq = X[idx, 0, :]
        dist = y_dist[idx]
        pos = y_pos[idx]  # 相对位置

        # 根据相对位置生成可读标签
        pos_str = "Head" if pos < 0.3 else ("Tail" if pos > 0.7 else "Body")

        ax.plot(cod_seq, color='darkcyan')
        ax.set_title(f"D:{dist:.1f}km | Pos:{pos_str}", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 标记可能的丢包 (0值)
        if np.min(cod_seq) < 0.1:
            ax.text(0.5, 0.5, "DROPOUT?", transform=ax.transAxes, color='red',
                    ha='center', alpha=0.5, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 3. 多通道详情: 查看单个样本的所有参数
    # ==========================================
    print("🔍 查看单样本多参数联动 (Multi-channel)...")

    # 找一个 COD 波动比较大的样本
    var_scores = np.var(X[:, 0, :], axis=1)
    target_idx = np.argmax(var_scores)  # 选方差最大的那个

    sample = X[target_idx]
    dist = y_dist[target_idx]

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    # COD
    axes[0].plot(sample[0], color='tab:red', linewidth=2)
    axes[0].set_ylabel('COD (mg/L)')
    axes[0].set_title(f"Detailed View of Sample #{target_idx} (Distance: {dist:.2f} km)")
    axes[0].grid(True)

    # pH
    axes[1].plot(sample[1], color='tab:blue', linewidth=2)
    axes[1].set_ylabel('pH')
    axes[1].grid(True)

    # DO
    axes[2].plot(sample[2], color='tab:green', linewidth=2)
    axes[2].set_ylabel('DO (mg/L)')
    axes[2].grid(True)

    # Velocity
    axes[3].plot(sample[3], color='tab:purple', linewidth=2, linestyle='--')
    axes[3].set_ylabel('Velocity (m/s)')
    axes[3].set_xlabel('Time Steps (minutes)')
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    verify_dataset()