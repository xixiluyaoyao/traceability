"""
完整消融实验 + Baseline对比 (带保存/加载功能)

使用方法:
  python ablation_experiment_v2.py           # 正常运行（自动加载已有模型）
  python ablation_experiment_v2.py --retrain # 强制重新训练所有模型
  python ablation_experiment_v2.py --plot    # 只画图，不训练

==========================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.metrics import r2_score
from copy import deepcopy
from tqdm import tqdm
import os
import pandas as pd


# 环境配置
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    print("🐍 Mamba 已加载")
except ImportError:
    HAS_MAMBA = False
    print("⚠️ 无 mamba_ssm，Mamba模型将使用LSTM替代")

try:
    from efficient_kan import KAN
    HAS_KAN = True
    print("🕸️ KAN 已加载")
except ImportError:
    HAS_KAN = False
    print("⚠️ 无 efficient_kan，KAN将使用MLP替代")
    class KAN(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.net = nn.Sequential()
            for i in range(len(layers)-1):
                self.net.add_module(f'lin{i}', nn.Linear(layers[i], layers[i+1]))
                if i < len(layers)-2:
                    self.net.add_module(f'act{i}', nn.SiLU())
        def forward(self, x):
            return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

# 保存/加载配置
SAVE_DIR = 'saved_models'
RESULTS_FILE = 'experiment_results.npz'

def get_model_path(name):
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
    return os.path.join(SAVE_DIR, f'{safe_name}.pth')

def get_history_path(name):
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
    return os.path.join(SAVE_DIR, f'{safe_name}_history.npz')

def save_model(model, name, history=None):
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = get_model_path(name)
    torch.save(model.state_dict(), model_path)
    if history:
        hist_path = get_history_path(name)
        np.savez(hist_path, train_loss=history['train_loss'], val_mae=history['val_mae'])
    print(f"   💾 已保存: {model_path}")

def load_model(model, name):
    model_path = get_model_path(name)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        return True
    return False

def load_history(name):
    hist_path = get_history_path(name)
    if os.path.exists(hist_path):
        data = np.load(hist_path)
        return {'train_loss': data['train_loss'].tolist(), 'val_mae': data['val_mae'].tolist()}
    return None

def save_all_results(results, histories):
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_dict = {}
    for name, res in results.items():
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        save_dict[f'{safe_name}_mae'] = res['mae']
        save_dict[f'{safe_name}_preds'] = res['preds']
        save_dict[f'{safe_name}_trues'] = res['trues']
    for name, hist in histories.items():
        if hist:
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
            save_dict[f'{safe_name}_hist_loss'] = hist['train_loss']
            save_dict[f'{safe_name}_hist_mae'] = hist['val_mae']
    save_dict['model_names'] = np.array(list(results.keys()), dtype=object)
    np.savez(os.path.join(SAVE_DIR, RESULTS_FILE), **save_dict)
    print(f"\n💾 所有结果已保存到: {os.path.join(SAVE_DIR, RESULTS_FILE)}")

def load_all_results():
    results_path = os.path.join(SAVE_DIR, RESULTS_FILE)
    if not os.path.exists(results_path):
        return None, None
    data = np.load(results_path, allow_pickle=True)
    model_names = data['model_names'].tolist()
    results = {}
    histories = {}
    for name in model_names:
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        results[name] = {
            'mae': float(data[f'{safe_name}_mae']),
            'preds': data[f'{safe_name}_preds'],
            'trues': data[f'{safe_name}_trues']
        }
        hist_loss_key = f'{safe_name}_hist_loss'
        if hist_loss_key in data:
            histories[name] = {
                'train_loss': data[hist_loss_key].tolist(),
                'val_mae': data[f'{safe_name}_hist_mae'].tolist()
            }
        else:
            histories[name] = None
    return results, histories


# 数据集
class PhysicsInformedDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.raw_X = data['X']
        self.y_dist = torch.FloatTensor(data['y_dist'])
        self.y_mass = torch.log1p(torch.FloatTensor(data.get('y_mass', np.zeros(len(self.y_dist)))))
        
        self.u = torch.FloatTensor(data.get('y_u', np.zeros(len(self.y_dist))))
        self.v_boat = torch.FloatTensor(data.get('y_vboat', np.zeros(len(self.y_dist))))
        self.width = torch.FloatTensor(data.get('y_width', np.full(len(self.y_dist), 15.0)))
        self.depth = torch.FloatTensor(data.get('y_depth', np.full(len(self.y_dist), 1.2)))
        
        cod_seqs = self.raw_X[:, 0, :]
        self.kurt = torch.tanh(torch.FloatTensor(scipy.stats.kurtosis(cod_seqs, axis=1)) / 10.0)
        self.skew = torch.tanh(torch.FloatTensor(scipy.stats.skew(cod_seqs, axis=1)) / 5.0)
        self.log_max_cod = torch.FloatTensor(np.log1p(np.max(cod_seqs, axis=1))) / 12.0
        self.log_std_cod = torch.FloatTensor(np.log1p(np.std(cod_seqs, axis=1))) / 8.0
        print(f"📂 加载 {len(self)} 样本")

    def __len__(self):
        return len(self.raw_X)

    def __getitem__(self, idx):
        sample = self.raw_X[idx]
        x_img = torch.FloatTensor(np.vstack([
            np.log1p(np.maximum(sample[0, :], 0)) / 12.0,
            (sample[1, :] - 7.0) / 2.0,
            (sample[2, :] - 8.0) / 4.0
        ]))
        stats = torch.stack([
            self.u[idx], self.v_boat[idx], self.u[idx]-self.v_boat[idx],
            self.kurt[idx], self.skew[idx], self.log_max_cod[idx], self.log_std_cod[idx],
            self.width[idx]/20.0, self.depth[idx]/2.0
        ])
        return x_img, stats, torch.log10(self.y_dist[idx]), self.y_mass[idx]

# 模型组件
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1)

class SEBlockSeq(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, l, c = x.size()
        y = self.avg_pool(x.permute(0,2,1)).view(b, c)
        return x * self.fc(y).view(b, 1, c)

class SEMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        if HAS_MAMBA:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        else:
            self.lstm = nn.LSTM(d_model, d_model//2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(d_model)
        self.se = SEBlockSeq(d_model)
    
    def forward(self, x):
        res = x
        x = self.norm(x)
        if HAS_MAMBA:
            x = self.mamba(x)
        else:
            x, _ = self.lstm(x)
        return self.se(x) + res

# 所有模型定义
class MambaKAN(nn.Module):
    """完整模型: Mamba + KAN + Physics"""
    def __init__(self, d_model=64):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.mamba_layers = nn.Sequential(
            SEMambaBlock(d_model), SEMambaBlock(d_model), SEMambaBlock(d_model)
        )
        self.phys_encoder = KAN([9, 32, 32])
        self.head = nn.Sequential(
            nn.Linear(d_model+32, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3)
        )
    def forward(self, x, stats):
        x = self.input_proj(x.permute(0, 2, 1))
        x = self.mamba_layers(x)
        seq_feat = x.mean(dim=1)
        phys_feat = self.phys_encoder(stats)
        return self.head(torch.cat([seq_feat, phys_feat], dim=1))

class MambaMLP(nn.Module):
    """消融: KAN → MLP"""
    def __init__(self, d_model=64):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.mamba_layers = nn.Sequential(
            SEMambaBlock(d_model), SEMambaBlock(d_model), SEMambaBlock(d_model)
        )
        self.phys_encoder = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 32))
        self.head = nn.Sequential(
            nn.Linear(d_model+32, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3)
        )
    def forward(self, x, stats):
        x = self.input_proj(x.permute(0, 2, 1))
        x = self.mamba_layers(x)
        seq_feat = x.mean(dim=1)
        phys_feat = self.phys_encoder(stats)
        return self.head(torch.cat([seq_feat, phys_feat], dim=1))

class CNNKAN(nn.Module):
    """消融: Mamba → CNN"""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            SEBlock1D(32, reduction=4), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            SEBlock1D(64, reduction=8), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.phys_encoder = KAN([9, 32, 32])
        self.head = nn.Sequential(
            nn.Linear(64+32, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3)
        )
    def forward(self, x, stats):
        cnn_feat = self.cnn(x).squeeze(-1)
        phys_feat = self.phys_encoder(stats)
        return self.head(torch.cat([cnn_feat, phys_feat], dim=1))

class CNNMLP(nn.Module):
    """旧模型: CNN + MLP"""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            SEBlock1D(32, reduction=4), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            SEBlock1D(64, reduction=8), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.phys_encoder = nn.Sequential(
            nn.Linear(9, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(128+32, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3)
        )
    def forward(self, x, stats):
        cnn_feat = self.cnn(x).squeeze(-1)
        phys_feat = self.phys_encoder(stats)
        return self.head(torch.cat([cnn_feat, phys_feat], dim=1))

class MambaKANNoPhys(nn.Module):
    """消融: 无物理特征"""
    def __init__(self, d_model=64):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.mamba_layers = nn.Sequential(
            SEMambaBlock(d_model), SEMambaBlock(d_model), SEMambaBlock(d_model)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3)
        )
    def forward(self, x, stats):
        x = self.input_proj(x.permute(0, 2, 1))
        x = self.mamba_layers(x)
        seq_feat = x.mean(dim=1)
        return self.head(seq_feat)

class PureLSTM(nn.Module):
    """Baseline: 纯LSTM"""
    def __init__(self, d_model=64):
        super().__init__()
        self.lstm = nn.LSTM(3, d_model, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(d_model*2, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 3))
    def forward(self, x, stats):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.head(x.mean(dim=1))

class PureTransformer(nn.Module):
    """Baseline: 纯Transformer"""
    def __init__(self, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 3))
    def forward(self, x, stats):
        x = self.input_proj(x.permute(0, 2, 1))
        x = self.transformer(x)
        return self.head(x.mean(dim=1))

# 训练器
class Trainer:
    def __init__(self, model, name):
        self.model = model.to(device)
        self.name = name
        self.best_mae = float('inf')
        self.best_state = None
        self.history = {'train_loss': [], 'val_mae': []}
    
    def train(self, train_loader, val_loader, max_epochs=50, patience=12, lr=1e-3):
        print(f"\n{'='*50}")
        print(f"🚀 训练: {self.name}")
        print(f"{'='*50}")
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion_nll = nn.GaussianNLLLoss(reduction='none')
        criterion_mass = nn.SmoothL1Loss()
        
        patience_counter = 0
        
        for epoch in range(max_epochs):
            self.model.train()
            total_loss = 0
            for x, stats, y_d_log, y_m in train_loader:
                x, stats = x.to(device), stats.to(device)
                y_d_log, y_m = y_d_log.to(device), y_m.to(device)
                
                optimizer.zero_grad()
                pred = self.model(x, stats)
                dist_mu, dist_log_var, mass_mu = pred[:, 0], pred[:, 1], pred[:, 2]
                
                weights = 1.0 + 2.0 * torch.exp(-0.5 * torch.pow(10, y_d_log))
                loss = torch.mean(criterion_nll(dist_mu, y_d_log, torch.exp(dist_log_var)) * weights) * 10.0
                loss += criterion_mass(mass_mu, y_m) * 0.5
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            val_mae = self.evaluate(val_loader)
            self.history['train_loss'].append(total_loss / len(train_loader))
            self.history['val_mae'].append(val_mae)
            
            scheduler.step(val_mae)
            
            if val_mae < self.best_mae:
                self.best_mae = val_mae
                self.best_state = deepcopy(self.model.state_dict())
                patience_counter = 0
                marker = " ✓"
            else:
                patience_counter += 1
                marker = ""
            
            if (epoch + 1) % 5 == 0 or marker:
                print(f"  Ep {epoch+1:3d} | MAE: {val_mae:.3f} km{marker}")
            
            if patience_counter >= patience:
                print(f"  ⏹️ 早停 @ Ep {epoch+1}")
                break
        
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        print(f"  ✅ Best MAE: {self.best_mae:.3f} km")
        return self.history
    
    def evaluate(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, stats, y_d_log, _ in loader:
                x, stats = x.to(device), stats.to(device)
                pred = self.model(x, stats)[:, 0]
                preds.extend(torch.pow(10, pred).cpu().numpy())
                trues.extend(torch.pow(10, y_d_log).numpy())
        return np.mean(np.abs(np.array(preds) - np.array(trues)))
    
    def get_predictions(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, stats, y_d_log, _ in loader:
                x, stats = x.to(device), stats.to(device)
                pred = self.model(x, stats)[:, 0]
                preds.extend(torch.pow(10, pred).cpu().numpy())
                trues.extend(torch.pow(10, y_d_log).numpy())
        return np.array(preds), np.array(trues)


# 可视化
def plot_results(results, histories):
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'Ours (Mamba+KAN)': '#e74c3c', 'Mamba+MLP': '#3498db', 'CNN+KAN': '#2ecc71',
        'CNN+MLP (Previous)': '#9b59b6', 'No Physics': '#f39c12',
        'Pure LSTM': '#7f8c8d', 'Pure Transformer': '#34495e'
    }
    
    fig = plt.figure(figsize=(20, 16))
    
    # 图1: 训练曲线
    ax1 = fig.add_subplot(2, 3, 1)
    for name, hist in histories.items():
        if hist and len(hist['val_mae']) > 0:
            ax1.plot(hist['val_mae'], label=name, color=colors.get(name, 'gray'), linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation MAE (km)', fontsize=12)
    ax1.set_title('(a) Training Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_ylim(bottom=0)
    
    # 图2: 柱状图
    ax2 = fig.add_subplot(2, 3, 2)
    names = list(results.keys())
    maes = [results[n]['mae'] for n in names]
    bars = ax2.bar(range(len(names)), maes, color=[colors.get(n, 'gray') for n in names], edgecolor='black', linewidth=1.2)
    for bar, mae in zip(bars, maes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{mae:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
    ax2.set_ylabel('MAE (km)', fontsize=12)
    ax2.set_title('(b) Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(maes) * 1.2)
    
    # 图3: 散点图
    ax3 = fig.add_subplot(2, 3, 3)
    ours_name = 'Ours (Mamba+KAN)'
    baseline_name = 'CNN+MLP (Previous)'
    if ours_name in results and baseline_name in results:
        trues = results[ours_name]['trues']
        idx = np.random.choice(len(trues), min(500, len(trues)), replace=False)
        ax3.scatter(trues[idx], results[baseline_name]['preds'][idx], c=colors[baseline_name], alpha=0.4, s=20, label=baseline_name)
        ax3.scatter(trues[idx], results[ours_name]['preds'][idx], c=colors[ours_name], alpha=0.6, s=25, label=ours_name)
        ax3.plot([0, 12], [0, 12], 'k--', lw=2, label='Ideal')
        ax3.set_xlabel('Ground Truth (km)', fontsize=12)
        ax3.set_ylabel('Predicted (km)', fontsize=12)
        ax3.set_title('(c) Prediction Scatter', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.set_xlim(0, 12); ax3.set_ylim(0, 12)
    
    # 图4: 热力图
    ax4 = fig.add_subplot(2, 3, 4)
    bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]
    bin_labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12']
    heatmap_data = []
    for name in names:
        trues = results[name]['trues']
        preds = results[name]['preds']
        row = []
        for lo, hi in bins:
            mask = (trues >= lo) & (trues < hi)
            row.append(np.mean(np.abs(preds[mask] - trues[mask])) if mask.sum() > 0 else 0)
        heatmap_data.append(row)
    heatmap_df = pd.DataFrame(heatmap_data, index=[n.replace(' ', '\n') for n in names], columns=bin_labels)
    sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax4, cbar_kws={'label': 'MAE (km)'}, linewidths=0.5)
    ax4.set_xlabel('Distance Range (km)', fontsize=12)
    ax4.set_title('(d) MAE by Distance Range', fontsize=14, fontweight='bold')
    
    # 图5: 误差曲线
    ax5 = fig.add_subplot(2, 3, 5)
    for name in names:
        errors = np.sort(np.abs(results[name]['preds'] - results[name]['trues']))
        percentiles = np.linspace(0, 100, len(errors))
        lw = 3 if 'Ours' in name else 1.5
        ax5.plot(percentiles, errors, color=colors.get(name, 'gray'), linewidth=lw, label=name, alpha=0.8)
    ax5.set_xlabel('Sample Percentile (%)', fontsize=12)
    ax5.set_ylabel('Absolute Error (km)', fontsize=12)
    ax5.set_title('(e) Sorted Error Distribution', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=8, loc='upper left')
    ax5.set_xlim(0, 100); ax5.set_ylim(0, 6)
    
    # 图6: 消融柱状图
    ax6 = fig.add_subplot(2, 3, 6)
    ablation_models = ['Ours (Mamba+KAN)', 'Mamba+MLP', 'CNN+KAN', 'CNN+MLP (Previous)', 'No Physics']
    ablation_names = [n for n in ablation_models if n in results]
    ablation_maes = [results[n]['mae'] for n in ablation_names]
    x = np.arange(len(ablation_names))
    bars = ax6.barh(x, ablation_maes, color=[colors.get(n, 'gray') for n in ablation_names], edgecolor='black', linewidth=1.2, height=0.6)
    ours_mae = results.get('Ours (Mamba+KAN)', {}).get('mae', 1.0)
    for i, (bar, mae, name) in enumerate(zip(bars, ablation_maes, ablation_names)):
        if 'Ours' not in name:
            diff = ((mae - ours_mae) / ours_mae) * 100
            ax6.text(mae + 0.05, bar.get_y() + bar.get_height()/2, f'{mae:.2f} (+{diff:.0f}%)', va='center', fontsize=10)
        else:
            ax6.text(mae + 0.05, bar.get_y() + bar.get_height()/2, f'{mae:.2f} (Ours)', va='center', fontsize=10, fontweight='bold')
    ax6.set_yticks(x)
    ax6.set_yticklabels([n.replace(' ', '\n') for n in ablation_names], fontsize=9)
    ax6.set_xlabel('MAE (km)', fontsize=12)
    ax6.set_title('(f) Ablation Study', fontsize=14, fontweight='bold')
    ax6.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('ablation_and_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('ablation_and_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 已保存: ablation_and_comparison.pdf/png")
    plt.show()


def print_summary_table(results):
    print("\n" + "="*70)
    print("📊 实验结果汇总")
    print("="*70)
    print(f"{'Model':<25} {'MAE (km)':<12} {'R²':<10} {'Near (<3km)':<12} {'Far (>8km)':<12}")
    print("-"*70)
    for name, res in results.items():
        trues, preds = res['trues'], res['preds']
        mae = res['mae']
        r2 = r2_score(trues, preds)
        near_mask = trues < 3.0
        far_mask = trues > 8.0
        mae_near = np.mean(np.abs(preds[near_mask] - trues[near_mask])) if near_mask.sum() > 0 else 0
        mae_far = np.mean(np.abs(preds[far_mask] - trues[far_mask])) if far_mask.sum() > 0 else 0
        print(f"{name:<25} {mae:<12.3f} {r2:<10.3f} {mae_near:<12.3f} {mae_far:<12.3f}")
    print("="*70)

# 主程序
def main(force_retrain=False, only_plot=False):
    """
    主函数
    参数:
        force_retrain: 强制重新训练所有模型
        only_plot: 只画图，不训练
    """
    
    if only_plot:
        print("📊 只画图模式：加载已保存的结果...")
        results, histories = load_all_results()
        if results is None:
            print("❌ 未找到保存的结果，请先运行训练")
            return None
        print(f"✅ 加载了 {len(results)} 个模型的结果")
        print_summary_table(results)
        plot_results(results, histories)
        try:
            from comparison_figures import generate_from_results
            generate_from_results(results)
        except ImportError:
            print("⚠️ 未找到 comparison_figures.py，跳过单独对比图")
        return results
    
    if not os.path.exists('ultimate_dataset_v3.npz'):
        print("❌ 找不到数据集"); return
    
    ds = PhysicsInformedDataset('ultimate_dataset_v3.npz')
    n = len(ds)
    train_size, val_size = int(0.7*n), int(0.15*n)
    test_size = n - train_size - val_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        ds, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    print(f"📊 数据: Train {train_size} | Val {val_size} | Test {test_size}")
    
    model_classes = {
        'Ours (Mamba+KAN)': MambaKAN,
        'Mamba+MLP': MambaMLP,
        'CNN+KAN': CNNKAN,
        'CNN+MLP (Previous)': CNNMLP,
        'No Physics': MambaKANNoPhys,
        'Pure LSTM': PureLSTM,
        'Pure Transformer': PureTransformer,
    }
    
    results = {}
    histories = {}
    
    for name, ModelClass in model_classes.items():
        print(f"\n{'='*50}")
        print(f"📦 处理模型: {name}")
        print(f"{'='*50}")
        
        model = ModelClass()
        model_loaded = False
        
        if not force_retrain:
            if name == 'Ours (Mamba+KAN)' and os.path.exists('best_pi_kan_mamba.pth'):
                try:
                    model.load_state_dict(torch.load('best_pi_kan_mamba.pth', map_location=device))
                    model_loaded = True
                    print(f"   ✅ 从 best_pi_kan_mamba.pth 加载")
                except Exception as e:
                    print(f"   ⚠️ 加载失败: {e}")
            
            if not model_loaded and load_model(model, name):
                model_loaded = True
                print(f"   ✅ 从 {get_model_path(name)} 加载")
        
        model = model.to(device)
        
        if model_loaded and not force_retrain:
            trainer = Trainer(model, name)
            preds, trues = trainer.get_predictions(test_loader)
            mae = np.mean(np.abs(preds - trues))
            results[name] = {'mae': mae, 'preds': preds, 'trues': trues}
            histories[name] = load_history(name)
            print(f"   📊 Test MAE: {mae:.3f} km")
        else:
            trainer = Trainer(model, name)
            hist = trainer.train(train_loader, val_loader, max_epochs=50, patience=12)
            histories[name] = hist
            preds, trues = trainer.get_predictions(test_loader)
            mae = np.mean(np.abs(preds - trues))
            results[name] = {'mae': mae, 'preds': preds, 'trues': trues}
            save_model(trainer.model, name, hist)
    
    save_all_results(results, histories)
    print_summary_table(results)
    plot_results(results, histories)
    
    try:
        from comparison_figures import generate_from_results
        generate_from_results(results)
    except ImportError:
        print("⚠️ 未找到 comparison_figures.py，跳过单独对比图")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='消融实验和Baseline对比')
    parser.add_argument('--retrain', action='store_true', help='强制重新训练所有模型')
    parser.add_argument('--plot', action='store_true', help='只画图，不训练')
    args = parser.parse_args()
    
    main(force_retrain=args.retrain, only_plot=args.plot)