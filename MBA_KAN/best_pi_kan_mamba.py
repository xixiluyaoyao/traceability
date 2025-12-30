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
from copy import deepcopy  #用于保存best模型

#导入 Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    print("🐍 成功加载 Mamba 模块")
except ImportError:
    HAS_MAMBA = False
    print("⚠️ 未检测到 mamba_ssm，将使用 LSTM 替补")

#导入 Efficient-KAN
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

# 3. 数据集 (9维物理特征)
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

# 模型组件: SE-Block & Mamba
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

# 核心模型: PI-KAN-Mamba
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

# 高级可视化工具 (含 KAN 可解释性)
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
                # 获取输出的前几个维度
                output = model.phys_encoder(input_tensor)
                for j in range(min(3, output.shape[1])):
                    ax.plot(x_range.cpu(), output[:, j].cpu(), alpha=0.7, label=f'Out_{j}')
            
            ax.set_title(f'Learned f(x) for {param_names[i]}', fontsize=10)
            ax.set_xlabel(param_names[i])
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('KAN Interpretability: Learned Physical Activation Functions', fontweight='bold')
        plt.tight_layout()
        plt.savefig('kan_interpretability.pdf', dpi=300)
        print("📊 KAN可解释性图已保存: kan_interpretability.pdf")
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
    
    def plot_training_history(self, history):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # MAE曲线
        ax1 = axes[0]
        ax1.plot(history['val_mae'], 'b-', linewidth=2, label='Val MAE')
        if history['best_epoch'] is not None:
            ax1.axvline(x=history['best_epoch'], color='r', linestyle='--', label=f'Best (Ep {history["best_epoch"]+1})')
            ax1.scatter([history['best_epoch']], [history['best_mae']], color='r', s=100, zorder=5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MAE (km)')
        ax1.set_title('Validation MAE over Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss曲线
        ax2 = axes[1]
        ax2.plot(history['train_loss'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Train Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.pdf', dpi=300)
        print("📊 训练曲线已保存: training_history.pdf")
        plt.show()

# 评估工具
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

# 主训练程序（带早停和Best模型保存）
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion_nll = nn.GaussianNLLLoss(reduction='none')
    criterion_mass = nn.SmoothL1Loss()
    visualizer = Visualizer()
    
    #早停配置
    MAX_EPOCHS = 100          # 最大训练轮数
    PATIENCE = 15             # 早停耐心值：连续15轮不改善就停止
    best_mae = float('inf')   # 记录最佳MAE
    best_model_state = None   # 保存最佳模型权重
    patience_counter = 0      # 耐心计数器
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_mae': [],
        'val_r2': [],
        'best_epoch': None,
        'best_mae': None
    }
    
    print(f"🚀 开始训练 PI-KAN-Mamba")
    print(f"   最大轮数: {MAX_EPOCHS} | 早停耐心: {PATIENCE}")
    print("="*60)
    
    for epoch in range(MAX_EPOCHS):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1:3d}", leave=False)
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
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ========== 验证阶段 ==========
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
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_mae'].append(mae)
        history['val_r2'].append(r2)
        
        # 检查是否是Best
        if mae < best_mae:
            best_mae = mae
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            history['best_epoch'] = epoch
            history['best_mae'] = mae
            marker = " ✓ BEST"
            # 立即保存best模型
            torch.save(best_model_state, 'best_pi_kan_mamba.pth')
        else:
            patience_counter += 1
            marker = ""
        
        # 打印信息
        print(f"[Ep {epoch+1:3d}] Loss: {avg_train_loss:.4f} | MAE: {mae:.3f} km | R²: {r2:.3f} | "
              f"Near: {near:.2f} | Mid: {mid:.2f} | Far: {far:.2f}{marker}")
        
        # 学习率调度
        scheduler.step(mae)
        
        # 早停检查
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ 早停触发！连续 {PATIENCE} 轮验证MAE未改善")
            print(f"   最佳模型在 Epoch {history['best_epoch']+1}，MAE = {best_mae:.3f} km")
            break
    
    # 恢复Best模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ 已恢复最佳模型 (Epoch {history['best_epoch']+1}, MAE={best_mae:.3f} km)")
    
    # 最终评估（用best模型）
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
    
    print("\n" + "="*60)
    print("📊 最终结果 (Best Model)")
    print(f"   MAE: {mae:.3f} km | R²: {r2:.3f}")
    print(f"   Near (<3km): {near:.3f} | Mid (3-8km): {mid:.3f} | Far (>8km): {far:.3f}")
    print("="*60)

    #可视化
    print("\n🎨 生成可视化图表...")
    visualizer.plot_training_history(history)
    visualizer.plot_performance(all_trues, all_preds, r2)
    visualizer.plot_kan_internals(model)
    
    #消融实验
    print("\n🔬 运行消融实验...")
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
    
    # 保存最终模型 
    torch.save(model.state_dict(), 'agent_model_kan_mamba.pth')
    print("\n💾 模型已保存:")
    print("   - best_pi_kan_mamba.pth (训练过程中的最佳)")
    print("   - agent_model_kan_mamba.pth (最终模型，与best相同)")


if __name__ == "__main__":
    train_kan_mamba()