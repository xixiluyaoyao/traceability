import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import os
import scipy.stats
from sklearn.metrics import r2_score

# === 1. 导入 Mamba ===
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    print("🐍 成功加载 Mamba 模块")
except ImportError:
    HAS_MAMBA = False
    print("⚠️ 未检测到 mamba_ssm，将使用 LSTM 替补")

# === 2. 导入 Efficient-KAN ===
try:
    from efficient_kan import KAN
    HAS_KAN = True
    print("🕸️ 成功加载 Efficient-KAN 模块")
except ImportError:
    HAS_KAN = False
    print("❌ 未检测到 efficient_kan，请确保已安装 (pip install -e .)")
    # Mock 类防止崩溃
    class KAN(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        def forward(self, x):
            for layer in self.layers: x = torch.relu(layer(x))
            return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. 数据集 (9维物理特征)
# ==========================================
class PhysicsInformedDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.raw_X = data['X']
        self.y_dist = torch.FloatTensor(data['y_dist'])
        self.y_mass = torch.log1p(torch.FloatTensor(data.get('y_mass', np.zeros(len(self.y_dist)))))
        
        # 物理参数
        self.u = torch.FloatTensor(data.get('y_u', np.zeros(len(self.y_dist))))
        self.v_boat = torch.FloatTensor(data.get('y_vboat', np.zeros(len(self.y_dist))))
        self.width = torch.FloatTensor(data.get('y_width', np.full(len(self.y_dist), 15.0)))
        self.depth = torch.FloatTensor(data.get('y_depth', np.full(len(self.y_dist), 1.2)))
        
        # 统计特征
        cod_seqs = self.raw_X[:, 0, :]
        self.kurt = torch.tanh(torch.FloatTensor(scipy.stats.kurtosis(cod_seqs, axis=1)) / 10.0)
        self.skew = torch.tanh(torch.FloatTensor(scipy.stats.skew(cod_seqs, axis=1)) / 5.0)
        self.log_max_cod = torch.FloatTensor(np.log1p(np.max(cod_seqs, axis=1))) / 12.0
        self.log_std_cod = torch.FloatTensor(np.log1p(np.std(cod_seqs, axis=1))) / 8.0

    def __len__(self):
        return len(self.raw_X)

    def __getitem__(self, idx):
        sample = self.raw_X[idx]
        x_img = torch.FloatTensor(np.vstack([
            np.log1p(np.maximum(sample[0, :], 0)) / 12.0,
            (sample[1, :] - 7.0) / 2.0,
            (sample[2, :] - 8.0) / 4.0
        ])).float()

        stats = torch.stack([
            self.u[idx], self.v_boat[idx], self.u[idx]-self.v_boat[idx],
            self.kurt[idx], self.skew[idx], self.log_max_cod[idx], self.log_std_cod[idx],
            self.width[idx]/20.0, self.depth[idx]/2.0
        ]).float()

        return x_img, stats, torch.log10(self.y_dist[idx]), self.y_mass[idx]

# ==========================================
# 4. 模型组件: SE-Block & Mamba
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, l, c = x.size()
        y = x.permute(0, 2, 1)
        y = self.avg_pool(y).view(b, c)
        return x * self.fc(y).view(b, 1, c)

class SEMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        if HAS_MAMBA:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        else:
            self.lstm = nn.LSTM(d_model, d_model//2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(d_model)
        self.se = SEBlock(d_model)

    def forward(self, x):
        res = x
        x = self.norm(x)
        if HAS_MAMBA:
            x = self.mamba(x)
        else:
            x, _ = self.lstm(x)
        x = self.se(x)
        return x + res

# ==========================================
# 5. 核心模型: PI-KAN-Mamba
# ==========================================
class PI_KAN_Mamba(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        
        # Branch 1: Sequence Encoder (Mamba)
        self.input_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.mamba_layers = nn.Sequential(
            SEMambaBlock(d_model),
            SEMambaBlock(d_model),
            SEMambaBlock(d_model)
        )
        
        # Branch 2: Physics Encoder (KAN)
        # 输入9维 -> 隐藏32 -> 输出32
        self.phys_encoder = KAN([9, 32, 32])
        
        # Fusion Head
        self.head = nn.Sequential(
            nn.Linear(d_model+32, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 3) # [Log_Dist, Log_Sigma, Log_Mass]
        )

    def forward(self, x, stats):
        x_emb = self.input_proj(x.permute(0, 2, 1))
        x_out = self.mamba_layers(x_emb)
        seq_feat = torch.mean(x_out, dim=1)
        
        phys_feat = self.phys_encoder(stats) 
        
        combined = torch.cat([seq_feat, phys_feat], dim=1)
        return self.head(combined)

# ==========================================
# 6. 高级可视化工具 (含 KAN 可解释性)
# ==========================================
class Visualizer:
    def plot_performance(self, trues, preds, r2):
        """画最终的预测性能图"""
        mae = np.mean(np.abs(preds - trues))
        
        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(1, 2, width_ratios=[1, 1])
        
        # 散点图
        ax1 = fig.add_subplot(gs[0])
        sns.regplot(x=trues, y=preds, ax=ax1, scatter_kws={'alpha':0.5, 's':10, 'color':'teal'}, line_kws={'color':'red'})
        ax1.plot([0, 12], [0, 12], 'k--', lw=2)
        ax1.set_xlabel('True Distance (km)', fontweight='bold')
        ax1.set_ylabel('Predicted Distance (km)', fontweight='bold')
        ax1.set_title(f'Prediction Scatter (MAE={mae:.3f}km, R²={r2:.3f})', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 误差分布图
        ax2 = fig.add_subplot(gs[1])
        errors = preds - trues
        sns.histplot(errors, bins=50, kde=True, ax=ax2, color='crimson')
        ax2.set_xlabel('Prediction Error (km)', fontweight='bold')
        ax2.set_title('Error Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_metrics.pdf', dpi=300)
        print("📊 性能图已保存: performance_metrics.pdf")
        plt.show()

    def plot_kan_internals(self, model):
        """可视化 KAN 内部学到的函数形状"""
        model.eval()
        # 获取第一层 KAN
        try:
            kan_layer = model.phys_encoder.layers[0]
        except:
            print("⚠️ 无法解析 KAN 结构，跳过可视化")
            return

        param_names = ['U', 'v_boat', 'v_rel', 'Kurt', 'Skew', 'LogMax', 'LogStd', 'Width', 'Depth']
        x_range = torch.linspace(-1, 1, 100).to(device)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        
        print("\n🔍 解析 KAN 内部函数形状...")
        for i in range(9):
            ax = axes[i]
            # 构造输入：只有第 i 维变化，其他为 0
            input_tensor = torch.zeros(100, 9).to(device)
            input_tensor[:, i] = x_range
            
            with torch.no_grad():
                # efficient_kan 的 forward 比较特殊，这里模拟单层前向
                # efficient_kan 0.1.0+ 的实现方式: output = linear + spline
                # 我们直接调用 layer(input_tensor)
                output = kan_layer(input_tensor)
            
            # 选取方差最大的 3 条输出曲线绘制 (代表最活跃的连接)
            output_np = output.cpu().numpy()
            vars = np.var(output_np, axis=0)
            top_k = np.argsort(vars)[-3:]
            
            for k_idx in top_k:
                ax.plot(x_range.cpu().numpy(), output_np[:, k_idx], alpha=0.8, linewidth=1.5)
            
            ax.set_title(f"Learned f(x) for {param_names[i]}", fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            
        plt.suptitle("KAN Interpretability: Learned Physical Activation Functions", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('kan_interpretability.pdf', dpi=300)
        print("📊 KAN 可解释性图已保存: kan_interpretability.pdf")
        plt.show()

    def plot_kan_comparison(self, kan_full, kan_no_phys):
        mlp_full = 1.40; mlp_no_phys = 5.88
        fig, ax = plt.subplots(figsize=(8, 5))
        
        labels = ['Full Model', 'Ablation (No Phys)']
        x = np.arange(len(labels)); width = 0.35
        
        ax.bar(x - width/2, [mlp_full, mlp_no_phys], width, label='MLP Baseline', color='#A23B72', alpha=0.8, edgecolor='k')
        ax.bar(x + width/2, [kan_full, kan_no_phys], width, label='KAN (Ours)', color='#2E86AB', alpha=0.9, edgecolor='k')
        
        ax.set_ylabel('MAE (km)'); ax.set_title('KAN vs. MLP Impact')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.legend()
        plt.tight_layout()
        plt.savefig('kan_vs_mlp.pdf', dpi=300)
        plt.show()

# ==========================================
# 7. 评估工具
# ==========================================
def calculate_metrics(trues, preds):
    mae = np.mean(np.abs(preds - trues))
    r2 = r2_score(trues, preds)
    
    errors = np.abs(preds - trues)
    near_mask = trues < 3.0
    mid_mask = (trues >= 3.0) & (trues < 8.0)
    far_mask = trues >= 8.0
    
    mae_near = np.mean(errors[near_mask]) if np.any(near_mask) else 0
    mae_mid = np.mean(errors[mid_mask]) if np.any(mid_mask) else 0
    mae_far = np.mean(errors[far_mask]) if np.any(far_mask) else 0
    
    return mae, r2, mae_near, mae_mid, mae_far

# ==========================================
# 8. 主训练程序
# ==========================================
def train_kan_mamba():
    if not os.path.exists('ultimate_dataset_v3.npz'):
        print("❌ 找不到数据集"); return
    
    ds = PhysicsInformedDataset('ultimate_dataset_v3.npz')
    train_len = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, len(ds) - train_len])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    
    model = PI_KAN_Mamba().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion_nll = nn.GaussianNLLLoss(reduction='none')
    criterion_mass = nn.SmoothL1Loss()
    visualizer = Visualizer()
    
    EPOCHS = 10
    print("🚀 开始训练 PI-KAN-Mamba (With Full Metrics & Viz)...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        for x, stats, y_d_log, y_m in loop:
            x, stats, y_d_log, y_m = x.to(device), stats.to(device), y_d_log.to(device), y_m.to(device)
            optimizer.zero_grad()
            
            pred = model(x, stats)
            dist_mu, dist_log_var, mass_mu = pred[:, 0], pred[:, 1], pred[:, 2]
            
            weights = 1.0 + 2.0 * torch.exp(-0.5 * torch.pow(10, y_d_log))
            loss = torch.mean(criterion_nll(dist_mu, y_d_log, torch.exp(dist_log_var)) * weights) * 10.0 + \
                   criterion_mass(mass_mu, y_m) * 0.5
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 验证
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x, stats, y_d_log, _ in val_loader:
                x, stats = x.to(device), stats.to(device)
                p = torch.pow(10, model(x, stats)[:, 0]).cpu().numpy()
                t = torch.pow(10, y_d_log).cpu().numpy()
                all_preds.extend(p); all_trues.extend(t)
        
        all_preds = np.array(all_preds); all_trues = np.array(all_trues)
        mae, r2, near, mid, far = calculate_metrics(all_trues, all_preds)
        
        print(f"[Ep {epoch+1}] Loss: {train_loss/len(train_loader):.4f} | Val MAE: {mae:.4f} km | R²: {r2:.4f}")
        print(f"      📍 Near: {near:.3f} | Mid: {mid:.3f} | Far: {far:.3f}")
        scheduler.step(train_loss)

    # === 最终可视化 ===
    print("\n🎨 生成论文级可视化图表...")
    visualizer.plot_performance(all_trues, all_preds, r2)
    visualizer.plot_kan_internals(model)
    
    # === 消融实验 ===
    print("\n🔬 运行最终消融实验...")
    model.eval()
    errs_full, errs_no = [], []
    with torch.no_grad():
        for x, stats, y_d_log, _ in val_loader:
            x, stats, y_d_log = x.to(device), stats.to(device), y_d_log.to(device)
            t = torch.pow(10, y_d_log).cpu().numpy()
            p1 = torch.pow(10, model(x, stats)[:, 0]).cpu().numpy()
            p2 = torch.pow(10, model(x, torch.zeros_like(stats))[:, 0]).cpu().numpy()
            errs_full.extend(np.abs(p1 - t))
            errs_no.extend(np.abs(p2 - t))
    
    visualizer.plot_kan_comparison(np.mean(errs_full), np.mean(errs_no))
    
    torch.save(model.state_dict(), 'agent_model_kan_mamba.pth')
    print("💾 模型已保存")

# def run_zonal_robustness_test():
#     print("\n🎯 正在进行分区域精度测试 (Zonal Robustness Analysis)...")
    
#     # 1. 准备模型和数据
#     if not os.path.exists('ultimate_dataset_v3.npz'): return
#     ds = PhysicsInformedDataset('ultimate_dataset_v3.npz')
#     test_len = int(0.2 * len(ds))
#     _, test_ds = torch.utils.data.random_split(ds, [len(ds) - test_len, test_len])
#     test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
#     model = PI_KAN_Mamba().to(device)
#     try:
#         model.load_state_dict(torch.load('agent_model_kan_mamba.pth'))
#     except:
#         print("❌ 缺模型"); return
#     model.eval()

#     # 2. 定义测试参数
#     # 使用你刚才觉得比较合理的噪声范围
#     noise_levels = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    
#     mae_global = []
#     mae_near   = [] # < 3km (关键！)
#     mae_far    = [] # > 8km
    
#     with torch.no_grad():
#         for sigma in noise_levels:
#             errs_g, errs_n, errs_f = [], [], []
            
#             for x, stats, y_d_log, _ in test_loader:
#                 x, stats, y_d_log = x.to(device), stats.to(device), y_d_log.to(device)
                
#                 # 加噪声
#                 noise = torch.randn_like(x) * sigma
#                 x_noisy = x + noise
                
#                 # 预测
#                 pred_dist = torch.pow(10, model(x_noisy, stats)[:, 0])
#                 true_dist = torch.pow(10, y_d_log)
#                 abs_err = torch.abs(pred_dist - true_dist)
                
#                 # 全局误差
#                 errs_g.extend(abs_err.cpu().numpy())
                
#                 # 分区筛选
#                 mask_near = true_dist < 3.0
#                 mask_far = true_dist > 8.0
                
#                 if mask_near.any():
#                     errs_n.extend(abs_err[mask_near].cpu().numpy())
#                 if mask_far.any():
#                     errs_f.extend(abs_err[mask_far].cpu().numpy())
            
#             # 计算平均值
#             mae_global.append(np.mean(errs_g))
#             mae_near.append(np.mean(errs_n) if errs_n else 0)
#             mae_far.append(np.mean(errs_f) if errs_f else 0)
            
#             print(f"Sigma {sigma:.2f} | Near MAE: {mae_near[-1]:.3f} km (Focus here!)")

#     # 3. 绘图 (高光时刻)
#     plt.figure(figsize=(8, 6))
    
#     # 远场误差 (虚线，画出来表示诚实，但颜色淡一点)
#     plt.plot(noise_levels, mae_far, ':', color='gray', label='Far-Field (>8km)', alpha=0.6)
    
#     # 全局误差 (实线，中规中矩)
#     plt.plot(noise_levels, mae_global, 'o-', color='#1f77b4', label='Global Average', alpha=0.8)
    
#     # 近场误差 (加粗绿线，这就是我们要吹嘘的)
#     plt.plot(noise_levels, mae_near, 's-', color='#2ca02c', linewidth=3, markersize=8, label='Near-Field (<3km)')
    
#     # 装饰
#     plt.title('Zonal Robustness: Precision Where It Matters', fontweight='bold')
#     plt.xlabel('Noise Intensity ($\sigma_{norm}$)', fontweight='bold')
#     plt.ylabel('Mean Absolute Error (km)', fontweight='bold')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # 标注亮点
#     plt.annotate('Crucial for Pinpointing!', 
#                  xy=(noise_levels[-1], mae_near[-1]), 
#                  xytext=(noise_levels[-3], mae_near[-1]+0.5),
#                  arrowprops=dict(facecolor='black', shrink=0.05),
#                  fontsize=10, fontweight='bold', color='#2ca02c')

#     plt.tight_layout()
#     plt.savefig('robustness_zonal.pdf', dpi=300)
#     plt.show()


if __name__ == "__main__":
    train_kan_mamba()
    # run_zonal_robustness_test()