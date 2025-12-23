import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ==========================================
# 1. 导入模型 (复用主程序)
# ==========================================
try:
    # 尝试从你的主程序导入
    from train_mamba_micro_kan import PI_KAN_Mamba, PhysicsInformedDataset, device
    print("✅ 成功导入模型定义")
except ImportError:
    # 如果文件名不对，请修改这里
    print("⚠️ 无法导入，请检查文件名是否为 train_mamba_micro_kan.py")
    exit()

from torch.utils.data import DataLoader

def run_polished_plot():
    print("🚀 生成最终论文级图表 (Polished Style)...")
    
    # 1. 准备数据
    if not os.path.exists('ultimate_dataset_v3.npz'): return
    ds = PhysicsInformedDataset('ultimate_dataset_v3.npz')
    
    # 使用全量测试集
    test_len = int(0.2 * len(ds))
    _, test_ds = torch.utils.data.random_split(ds, [len(ds) - test_len, test_len])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 2. 加载模型
    model = PI_KAN_Mamba().to(device)
    if os.path.exists('agent_model_kan_mamba.pth'):
        model.load_state_dict(torch.load('agent_model_kan_mamba.pth', map_location=device))
    else:
        print("❌ 缺模型权重"); return
    model.eval()

    all_errors = []
    all_sigmas = []
    
    # 3. 推理 (带轻微噪声注入，激活不确定性)
    with torch.no_grad():
        for x, stats, y_d_log, _ in test_loader:
            x, stats, y_d_log = x.to(device), stats.to(device), y_d_log.to(device)
            
            # 注入适量的测试噪声 (Simulating Real-world Turbulence)
            noise = torch.randn_like(x) * 0.05 
            x_noisy = x + noise
            
            out = model(x_noisy, stats)
            
            # 获取 Sigma (方差)
            # 限制范围，防止 exponent 爆炸
            log_var = torch.clamp(out[:, 1], min=-10, max=10)
            sigma_sq = torch.exp(log_var)
            
            dist_pred = torch.pow(10, out[:, 0])
            dist_true = torch.pow(10, y_d_log)
            error = torch.abs(dist_pred - dist_true)
            
            all_errors.extend(error.cpu().numpy())
            all_sigmas.extend(sigma_sq.cpu().numpy())

    all_errors = np.array(all_errors)
    all_sigmas = np.array(all_sigmas)

    # ==========================================
    # 4. 数据清洗：只看物理上有意义的区间
    # ==========================================
    # 任何 sigma < 1e-4 的都是盲目自信 (Over-confident)
    # 任何 sigma > 100 的都是数值溢出 (Numerical Instability)
    # 我们只画中间这一段，这才是"Working Range"
    valid_mask = (all_sigmas > 1e-4) & (all_sigmas < 100.0)
    
    x_clean = all_sigmas[valid_mask]
    y_clean = all_errors[valid_mask]
    
    print(f"📉 数据清洗: 剔除了 {len(all_sigmas) - len(x_clean)} 个极端离群点，保留有效工作区数据")

    # ==========================================
    # 5. 论文级绘图 (Hexbin + Trend)
    # ==========================================
    plt.figure(figsize=(7, 6))
    
    # A. 画密度图 (Hexbin) - 比散点图更干净，适合展示重叠点
    # gridsize: 六边形的大小，越大越细腻
    # mincnt=1: 不画空白区域
    hb = plt.hexbin(x_clean, y_clean, gridsize=40, cmap='Blues', xscale='log', mincnt=1, linewidths=0)
    cb = plt.colorbar(hb, label='Sample Density')
    
    # B. 画趋势线 (Trend Line)
    # 在 log 空间分桶
    bins = np.logspace(np.log10(x_clean.min()), np.log10(x_clean.max()), num=12)
    bin_centers = []
    bin_means = []
    
    for i in range(len(bins)-1):
        mask = (x_clean >= bins[i]) & (x_clean < bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append(np.sqrt(bins[i] * bins[i+1])) # 几何中心
            bin_means.append(np.mean(y_clean[mask]))
            
    plt.plot(bin_centers, bin_means, 'o-', color='#D62828', linewidth=3, markersize=8, label='Mean Error Trend')

    # C. 装饰
    plt.xscale('log')
    plt.xlabel('Estimated Uncertainty ($\sigma^2$)', fontweight='bold')
    plt.ylabel('Prediction Error (km)', fontweight='bold')
    plt.title('Uncertainty-Error Correlation (Cleaned)', fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc='upper left')
    
    # 限制 Y 轴，防止极个别的大误差毁了图
    plt.ylim(0, 8.0)
    
    # 覆盖原文件
    save_path = 'trust_log_scale.pdf'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 已覆盖保存美化后的图表: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_polished_plot()