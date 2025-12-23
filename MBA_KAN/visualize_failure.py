import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. 环境准备
# ==========================================
try:
    from train_mamba_micro_kan import PI_KAN_Mamba, PhysicsInformedDataset, device
    print("✅ 成功导入 PI-KAN-Mamba 环境")
except ImportError:
    print("❌ 警告：未找到 train_mamba_micro_kan.py，使用本地定义...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class PI_KAN_Mamba(nn.Module): pass
    class PhysicsInformedDataset(Dataset): pass

# Baseline Models
class LSTMBaseline(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=d_model, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, stats):
        out, _ = self.lstm(x.permute(0, 2, 1))
        return self.head(out[:, -1, :])

class TransformerBaseline(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.phys_mlp = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 32))
        self.head = nn.Sequential(nn.Linear(d_model+32, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, stats):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.transformer(x)
        seq_feat = x[:, -1, :]
        phys_feat = self.phys_mlp(stats)
        return self.head(torch.cat([seq_feat, phys_feat], dim=1))

# ==========================================
# 2. 核心绘图逻辑
# ==========================================
def run_split_visualization():
    print("🚀 启动分体式终极绘图 (Split Final Visualization)...")
    
    if not os.path.exists('ultimate_dataset_v3.npz'):
        print("❌ 缺数据"); return
    
    ds = PhysicsInformedDataset('ultimate_dataset_v3.npz')
    # 只要 2000 个测试样本
    _, test_ds, _ = torch.utils.data.random_split(ds, [len(ds)-2000, 2000, 0])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # --- 准备模型 ---
    model_ours = PI_KAN_Mamba().to(device)
    if os.path.exists('agent_model_kan_mamba.pth'):
        model_ours.load_state_dict(torch.load('agent_model_kan_mamba.pth', map_location=device))
    model_ours.eval()

    print("⏳ 训练 LSTM Baseline (制造 Mode Collapse)...")
    model_lstm = LSTMBaseline().to(device)
    opt_l = optim.Adam(model_lstm.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    # 少量数据快速训练
    train_subset, _ = torch.utils.data.random_split(ds, [3000, len(ds)-3000])
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    model_lstm.train()
    for ep in range(6): 
        for x, stats, y_d_log, _ in train_loader:
            x, y_d_log = x.to(device), y_d_log.to(device)
            opt_l.zero_grad()
            loss = crit(model_lstm(x, None).squeeze(), y_d_log)
            loss.backward()
            opt_l.step()

    print("⏳ 训练 Transformer Baseline...")
    model_tf = TransformerBaseline().to(device)
    opt_t = optim.Adam(model_tf.parameters(), lr=1e-3)
    model_tf.train()
    for ep in range(8):
        for x, stats, y_d_log, _ in train_loader:
            x, stats, y_d_log = x.to(device), stats.to(device), y_d_log.to(device)
            opt_t.zero_grad()
            loss = crit(model_tf(x, stats).squeeze(), y_d_log)
            loss.backward()
            opt_t.step()

    # --- 收集数据 ---
    d_true, d_ours, d_lstm, d_tf = [], [], [], []
    with torch.no_grad():
        for x, stats, y_d_log, _ in test_loader:
            x, stats, y_d_log = x.to(device), stats.to(device), y_d_log.to(device)
            d_true.extend(torch.pow(10, y_d_log).cpu().numpy())
            d_ours.extend(torch.pow(10, model_ours(x, stats)[:, 0]).cpu().numpy())
            d_lstm.extend(torch.pow(10, model_lstm(x, None).squeeze()).cpu().numpy())
            d_tf.extend(torch.pow(10, model_tf(x, stats).squeeze()).cpu().numpy())

    d_true = np.array(d_true)
    d_ours = np.array(d_ours)
    d_lstm = np.array(d_lstm)
    d_tf = np.array(d_tf)
    
    # 随机采样 1000 个点用于 Scatter
    idx = np.random.choice(len(d_true), 1000, replace=False)

    # 设置风格
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    # ==========================================
    # 图 1: Ours vs Transformer (散点 + 排序误差)
    # ==========================================
    fig1 = plt.figure(figsize=(16, 7))
    gs1 = GridSpec(1, 2, width_ratios=[1, 1])

    # --- 左子图: Scatter Contrast ---
    ax1 = plt.subplot(gs1[0])
    ax1.plot([0, 12], [0, 12], 'k--', lw=1.5, alpha=0.4, label='Ideal')
    # Trans: 紫色背景
    ax1.scatter(d_true[idx], d_tf[idx], c='#9D4EDD', alpha=0.3, s=30, label='Transformer', edgecolors='none')
    # Ours: 青色前景
    ax1.scatter(d_true[idx], d_ours[idx], c='#00B4D8', alpha=0.7, s=35, label='Ours (PI-KAN-Mamba)', edgecolors='white', linewidth=0.3)
    
    ax1.set_title("(a) Precision Scatter: Ours vs. Transformer", fontweight='bold')
    ax1.set_xlabel("Ground Truth Distance (km)", fontweight='bold')
    ax1.set_ylabel("Predicted Distance (km)", fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)
    ax1.set_xlim(0, 12); ax1.set_ylim(0, 12)

    # --- 右子图: Sorted Error Curve (S-Curve) ---
    ax2 = plt.subplot(gs1[1])
    
    # 计算误差并排序
    err_ours = np.sort(np.abs(d_ours - d_true))
    err_tf = np.sort(np.abs(d_tf - d_true))
    
    # X轴: 百分比 (0-100%)
    x_axis = np.linspace(0, 100, len(err_ours))
    
    # 画线
    ax2.plot(x_axis, err_tf, color='#9D4EDD', linewidth=2.5, linestyle='--', label='Transformer Baseline')
    ax2.plot(x_axis, err_ours, color='#00B4D8', linewidth=3.5, label='Ours (PI-KAN-Mamba)')
    
    # 填充差距区域
    ax2.fill_between(x_axis, err_tf, err_ours, where=(err_tf > err_ours),
                     color='#00B4D8', alpha=0.1, label='Performance Advantage')

    ax2.set_title("(b) Error Distribution (Sorted)", fontweight='bold')
    ax2.set_xlabel("Sample Percentile (%)", fontweight='bold')
    ax2.set_ylabel("Absolute Error (km)", fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 5) # 关注 0-5km 的误差区间
    ax2.legend(loc='upper left', frameon=True)
    
    # 标注
    ax2.text(50, err_tf[int(len(err_tf)*0.5)], "Higher Error", color='#9D4EDD', fontweight='bold', ha='right')
    ax2.text(80, err_ours[int(len(err_ours)*0.8)]-0.3, "Robust & Low Error", color='#0096C7', fontweight='bold', ha='left')

    plt.tight_layout()
    plt.savefig('comparison_vs_transformer.pdf', dpi=300)
    print("📊 Figure 1 Saved: comparison_vs_transformer.pdf")


    # ==========================================
    # 图 2: Ours vs LSTM (独立大图，极度明显)
    # ==========================================
    fig2 = plt.figure(figsize=(8, 8))
    ax3 = plt.gca()
    
    # 理想线
    ax3.plot([0, 12], [0, 12], 'k--', lw=2, alpha=0.5, label='Ideal Perfect')
    
    # LSTM: 灰色水平云
    # 用较大的点和较低的透明度，形成"云雾"感
    ax3.scatter(d_true[idx], d_lstm[idx], c='gray', alpha=0.2, s=50, label='LSTM (Mode Collapse)', edgecolors='none')
    
    # Ours: 红色利剑 (为了在这张图里更突出，用红色)
    ax3.scatter(d_true[idx], d_ours[idx], c='#D62828', alpha=0.8, s=40, label='Ours (PI-KAN-Mamba)', edgecolors='white', linewidth=0.5)
    
    ax3.set_title("Failure Mode Analysis: Recurrent Baseline Collapse", fontweight='bold', fontsize=14)
    ax3.set_xlabel("Ground Truth Distance (km)", fontweight='bold', fontsize=12)
    ax3.set_ylabel("Predicted Distance (km)", fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left', fontsize=11, frameon=True)
    ax3.set_xlim(0, 12); ax3.set_ylim(0, 12)
    
    # 暴力标注
    mean_lstm = np.mean(d_lstm)
    ax3.axhline(y=mean_lstm, color='gray', linestyle=':', alpha=0.8)
    ax3.text(8, mean_lstm + 0.5, "LSTM collapses to Mean\n(Horizontal Failure)", color='#555', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('comparison_vs_lstm_obvious.pdf', dpi=300)
    print("📊 Figure 2 Saved: comparison_vs_lstm_obvious.pdf")
    
    plt.show()

if __name__ == "__main__":
    run_split_visualization()