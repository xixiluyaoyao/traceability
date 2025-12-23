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
from mamba_ssm import Mamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🐍 Using device: {device}")

# ==========================================
# SE-Attention 模块
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
        attention_weights = self.fc(y).view(b, 1, c)
        return x * attention_weights, attention_weights.squeeze(1)

# ==========================================
# 数据集
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
# SE-Mamba Block
# ==========================================
class SEMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.norm = nn.LayerNorm(d_model)
        self.se = SEBlock(d_model)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.mamba(x)
        x, attn_weights = self.se(x)
        return x + res, attn_weights

# ==========================================
# PI-SE-Mamba 模型
# ==========================================
class PI_SE_Mamba(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.mamba_layers = nn.ModuleList([SEMambaBlock(d_model) for _ in range(3)])
        self.phys_encoder = nn.Sequential(
            nn.Linear(9, 32), nn.BatchNorm1d(32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU()
        )
        self.head = nn.Sequential(
            nn.Linear(d_model+32, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3)
        )

    def forward(self, x, stats, return_attention=False):
        x_emb = self.input_proj(x.permute(0, 2, 1))
        attention_maps = []
        for layer in self.mamba_layers:
            x_emb, attn = layer(x_emb)
            attention_maps.append(attn)
        
        combined = torch.cat([torch.mean(x_emb, dim=1), self.phys_encoder(stats)], dim=1)
        output = self.head(combined)
        return (output, torch.stack(attention_maps)) if return_attention else output

# ==========================================
# 可视化工具
# ==========================================
class Visualizer:
    def __init__(self):
        sns.set_style("whitegrid")
        self.colors = {'full': '#2E86AB', 'no_phys': '#A23B72', 'no_wave': '#F18F01'}
    
    def plot_training_curves(self, history):
        """训练曲线 - 双子图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2.5, color='#E63946')
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2.5, color='#457B9D')
        axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold')
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(history['val_mae'], linewidth=2.5, color='#1D3557', marker='o', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('MAE (km)', fontsize=13, fontweight='bold')
        axes[1].set_title('Validation MAE', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.pdf', dpi=300, bbox_inches='tight')
        print("   ✅ 已保存: training_curves.pdf")
        plt.show()
    
    def plot_predictions(self, true_vals, pred_vals, mae, range_errors):
        """预测结果 - 散点图 + 距离范围性能"""
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[2, 2, 1])
        
        # 主散点图
        ax1 = fig.add_subplot(gs[0, :2])
        errors = np.abs(pred_vals - true_vals)
        scatter = ax1.scatter(true_vals, pred_vals, alpha=0.6, s=25, c=errors, 
                             cmap='YlOrRd', edgecolors='k', linewidth=0.3, vmin=0, vmax=2)
        ax1.plot([0, 12], [0, 12], 'k--', lw=2.5, label='Perfect Prediction', alpha=0.7)
        ax1.set_xlabel('True Distance (km)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Predicted Distance (km)', fontsize=13, fontweight='bold')
        ax1.set_title(f'Prediction Performance (MAE = {mae:.3f} km)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, 12)
        ax1.set_ylim(0, 12)
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Absolute Error (km)', fontsize=11)
        
        # 距离范围性能柱状图
        ax2 = fig.add_subplot(gs[0, 2])
        ranges = list(range_errors.keys())
        means = [np.mean(range_errors[r]) for r in ranges]
        stds = [np.std(range_errors[r]) for r in ranges]
        
        colors_range = ['#06D6A0', '#118AB2', '#EF476F']
        bars = ax2.bar(range(len(ranges)), means, yerr=stds, capsize=5,
                      color=colors_range, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.set_xticks(range(len(ranges)))
        ax2.set_xticklabels(['Near\n(<3km)', 'Mid\n(3-8km)', 'Far\n(>8km)'], fontsize=10)
        ax2.set_ylabel('MAE (km)', fontsize=12, fontweight='bold')
        ax2.set_title('Range Performance', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('predictions.pdf', dpi=300, bbox_inches='tight')
        print("   ✅ 已保存: predictions.pdf")
        plt.show()
    
    def plot_ablation(self, ablation_results):
        """消融实验 - 条形图 + 箱线图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        models = list(ablation_results.keys())
        maes = [ablation_results[m]['mae'] for m in models]
        
        # 条形图
        bars = axes[0].bar(models, maes, color=[self.colors[m] for m in models], 
                          edgecolor='black', linewidth=2, alpha=0.85)
        axes[0].set_ylabel('MAE (km)', fontsize=13, fontweight='bold')
        axes[0].set_title('Ablation Study Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim(0, max(maes) * 1.2)
        
        for bar, mae in zip(bars, maes):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{mae:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 箱线图
        error_data = [ablation_results[m]['errors'] for m in models]
        bp = axes[1].boxplot(error_data, labels=models, patch_artist=True, 
                            showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=7))
        
        for patch, model in zip(bp['boxes'], models):
            patch.set_facecolor(self.colors[model])
            patch.set_alpha(0.7)
            patch.set_linewidth(1.5)
        
        axes[1].set_ylabel('Absolute Error (km)', fontsize=13, fontweight='bold')
        axes[1].set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ablation.pdf', dpi=300, bbox_inches='tight')
        print("   ✅ 已保存: ablation.pdf")
        plt.show()
    
    def plot_attention(self, attention_weights):
        """SE-Attention 权重热图"""
        avg_attention = attention_weights.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(avg_attention, aspect='auto', cmap='RdYlBu_r', interpolation='bilinear')
        
        ax.set_xlabel('Channel Index', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mamba Layer', fontsize=13, fontweight='bold')
        ax.set_title('SE-Attention Weights across Layers', fontsize=14, fontweight='bold')
        ax.set_yticks(range(3))
        ax.set_yticklabels(['Layer 1', 'Layer 2', 'Layer 3'], fontsize=12)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('attention.pdf', dpi=300, bbox_inches='tight')
        print("   ✅ 已保存: attention.pdf")
        plt.show()

# ==========================================
# 评估函数
# ==========================================
def evaluate_model(model, val_loader, return_details=False):
    """评估模型性能"""
    model.eval()
    all_preds, all_trues = [], []
    errors = {'Near (<3km)': [], 'Mid (3-8km)': [], 'Far (>8km)': []}
    attention_sum = None
    n_samples = 0
    
    with torch.no_grad():
        for x, stats, y_d_log, _ in val_loader:
            x, stats = x.to(device), stats.to(device)
            pred, attn = model(x, stats, return_attention=True)
            
            pred_km = torch.pow(10, pred[:, 0]).cpu().numpy()
            true_km = torch.pow(10, y_d_log).cpu().numpy()
            
            all_preds.extend(pred_km)
            all_trues.extend(true_km)
            
            # 累积平均注意力权重
            if attention_sum is None:
                attention_sum = attn.sum(dim=1)
            else:
                attention_sum += attn.sum(dim=1)
            n_samples += attn.size(1)
            
            abs_err = np.abs(pred_km - true_km)
            for i, d in enumerate(true_km):
                if d < 3.0: 
                    errors['Near (<3km)'].append(abs_err[i])
                elif d < 8.0: 
                    errors['Mid (3-8km)'].append(abs_err[i])
                else: 
                    errors['Far (>8km)'].append(abs_err[i])
    
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    mae = np.mean(np.abs(all_preds - all_trues))
    
    if return_details:
        avg_attention = attention_sum / n_samples
        return mae, all_preds, all_trues, errors, avg_attention
    return mae

# ==========================================
# 消融实验
# ==========================================
def run_ablation_study(model, val_loader):
    """消融实验 - 对比完整模型、无物理特征、无波形特征"""
    print("\n🔬 Running Ablation Study...")
    model.eval()
    ablation_results = {}
    
    with torch.no_grad():
        all_errors = {'full': [], 'no_phys': [], 'no_wave': []}
        
        for x, stats, y_d_log, _ in val_loader:
            x, stats, y_d_log = x.to(device), stats.to(device), y_d_log.to(device)
            true_dist = torch.pow(10, y_d_log).cpu().numpy()
            
            # 完整模型
            pred_full = torch.pow(10, model(x, stats)[:, 0]).cpu().numpy()
            all_errors['full'].extend(np.abs(pred_full - true_dist))
            
            # 无物理特征
            pred_no_phys = torch.pow(10, model(x, torch.zeros_like(stats))[:, 0]).cpu().numpy()
            all_errors['no_phys'].extend(np.abs(pred_no_phys - true_dist))
            
            # 无波形特征
            pred_no_wave = torch.pow(10, model(torch.zeros_like(x), stats)[:, 0]).cpu().numpy()
            all_errors['no_wave'].extend(np.abs(pred_no_wave - true_dist))
    
    for key in all_errors:
        ablation_results[key] = {'mae': np.mean(all_errors[key]), 'errors': all_errors[key]}
        print(f"   {'✅' if key=='full' else '🔻'} {key:10s}: MAE = {ablation_results[key]['mae']:.4f} km")
    
    return ablation_results

# ==========================================
# 主训练函数
# ==========================================
def train_model():
    if not os.path.exists('ultimate_dataset_v3.npz'):
        print("❌ 数据集不存在"); return
    
    # 数据加载
    ds = PhysicsInformedDataset('ultimate_dataset_v3.npz')
    train_len = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, len(ds) - train_len])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    
    # 模型初始化
    model = PI_SE_Mamba().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion_nll = nn.GaussianNLLLoss(reduction='none')
    criterion_mass = nn.SmoothL1Loss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    visualizer = Visualizer()
    
    EPOCHS = 15
    print("🚀 开始训练 PI-SE-Mamba (Physics-Informed SE-Mamba)...")
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        for x, stats, y_d_log, y_m in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
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
        
        # 验证阶段
        val_mae = evaluate_model(model, val_loader)
        avg_train_loss = train_loss / len(train_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_train_loss)
        history['val_mae'].append(val_mae)
        
        print(f"[Epoch {epoch+1:02d}] Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f} km")
        scheduler.step(avg_train_loss)
    
    # 最终评估与可视化
    print("\n" + "="*60)
    print("🏁 训练完成，生成论文级可视化图表...")
    print("="*60)
    
    mae, preds, trues, range_errors, attention = evaluate_model(model, val_loader, return_details=True)
    
    print(f"\n📊 最终性能指标:")
    print(f"   Overall MAE: {mae:.4f} km")
    for range_name, errors in range_errors.items():
        print(f"   {range_name}: {np.mean(errors):.4f} km")
    
    print("\n🎨 生成可视化图表...")
    visualizer.plot_training_curves(history)
    visualizer.plot_predictions(trues, preds, mae, range_errors)
    visualizer.plot_attention(attention)
    
    ablation_results = run_ablation_study(model, val_loader)
    visualizer.plot_ablation(ablation_results)
    
    torch.save(model.state_dict(), 'agent_model_semamba.pth')
    print("\n💾 模型已保存: agent_model_semamba.pth")
    print("✨ 所有图表已生成完毕！")

if __name__ == "__main__":
    train_model()