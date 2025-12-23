import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import scipy.stats

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training Device: {device}")


# ==========================================
# 1. 数据集: 物理参数分离版
# ==========================================
class PhysicsInformedDataset(Dataset):
    def __init__(self, npz_path):
        try:
            data = np.load(npz_path)
            print(f"📂 成功加载数据集: {npz_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 找不到数据集: {npz_path}")

        self.raw_X = data['X']
        self.y_dist = torch.FloatTensor(data['y_dist'])

        # 处理 Mass
        if 'y_mass' in data:
            self.y_mass = torch.log1p(torch.FloatTensor(data['y_mass']))
        else:
            self.y_mass = torch.zeros_like(self.y_dist)

        # 加载物理参数
        if 'y_u' in data and 'y_vboat' in data:
            self.u = torch.FloatTensor(data['y_u'])
            self.v_boat = torch.FloatTensor(data['y_vboat'])
            self.width = torch.FloatTensor(data['y_width'])
            self.depth = torch.FloatTensor(data['y_depth'])
            print("✅ 物理参数 (U, Boat, Width, Depth) 已全部加载")
        else:
            print("⚠️ 未找到物理参数，使用默认值")
            self.u = torch.zeros_like(self.y_dist)
            self.v_boat = torch.zeros_like(self.y_dist)
            self.width = torch.full_like(self.y_dist, 10.0)  # 默认值
            self.depth = torch.full_like(self.y_dist, 1.0)

        # 预计算统计特征
        print("⚡ 预计算波形统计特征...")
        cod_seqs = self.raw_X[:, 0, :]
        self.kurt = torch.FloatTensor(scipy.stats.kurtosis(cod_seqs, axis=1))
        self.skew = torch.FloatTensor(scipy.stats.skew(cod_seqs, axis=1))
        # Tanh 压缩防止数值爆炸
        self.kurt = torch.tanh(self.kurt / 10.0)
        self.skew = torch.tanh(self.skew / 5.0)

        max_vals = np.max(cod_seqs, axis=1)
        std_vals = np.std(cod_seqs, axis=1)
        # Log 变换
        self.log_max_cod = torch.FloatTensor(np.log1p(max_vals)) / 12.0  # 适配 v4 数据
        self.log_std_cod = torch.FloatTensor(np.log1p(std_vals)) / 8.0

    def __len__(self):
        return len(self.raw_X)

    def __getitem__(self, idx):
        sample = self.raw_X[idx].copy()

        # --- 1. 物理参数 ---
        u_val = self.u[idx]
        v_boat_val = self.v_boat[idx]
        v_rel = u_val - v_boat_val  # 相对速度

        # --- 2. 图像通道 (仅水质, 去掉速度通道) ---
        cod_raw = sample[0, :]
        # Global Log Normalization
        cod_norm = np.log1p(np.maximum(cod_raw, 0)) / 12.0

        ph_norm = (sample[1, :] - 7.0) / 2.0
        do_norm = (sample[2, :] - 8.0) / 4.0

        # 只堆叠 3 个通道 [COD, pH, DO]
        # 速度信息通过 stats 传入，不再干扰 CNN 视线
        x_img = torch.FloatTensor(np.vstack([cod_norm, ph_norm, do_norm])).float()

        # --- 3. 物理上下文 (Physics Context) ---
        stats = torch.stack([
            u_val,
            v_boat_val,
            torch.tensor(v_rel, dtype=torch.float),  # 关键: 显式传入相对速度
            self.kurt[idx],
            self.skew[idx],
            self.log_max_cod[idx],
            self.log_std_cod[idx],
            self.width[idx] / 20.0,  # 归一化处理
            self.depth[idx] / 2.0
        ]).float()

        # --- 4. 标签 (Log Space) ---
        target_dist_log = torch.log10(self.y_dist[idx])

        return x_img, stats, target_dist_log, self.y_mass[idx]


# ==========================================
# 2. 模型: PI-Attentive Net (SE-Block + Physics Fusion)
# ==========================================
class SEBlock(nn.Module):
    """ Squeeze-and-Excitation Attention Block """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class PI_Attentive_MoE(nn.Module):
    def __init__(self):
        super().__init__()

        # Branch 1: CNN + Attention (提取波形特征)
        # Input: [Batch, 3, 30]
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            SEBlock(32, reduction=4),  # Attention!
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            SEBlock(64, reduction=8),  # Attention!
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )  # Output: [Batch, 128, 1]

        # Branch 2: Physics Encoder (处理流速等参数)
        # Input: 7 dims
        self.phys_encoder = nn.Sequential(
            nn.Linear(9, 32),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Fusion Head
        # 128 (CNN) + 32 (Physics) = 160
        self.head = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            # [修改] 输出改为 3 维:
            # 0: Log_Dist_Mu (距离预测)
            # 1: Log_Dist_Sigma (距离不确定度)
            # 2: Log_Mass_Mu (源强预测 - 辅助任务)
            nn.Linear(64, 3)
        )

    def forward(self, x, stats):
        # 1. Image Branch
        cnn_feat = self.cnn(x)
        cnn_feat = cnn_feat.view(cnn_feat.size(0), -1)  # Flatten -> [B, 128]

        # 2. Physics Branch
        phys_feat = self.phys_encoder(stats)  # -> [B, 32]

        # 3. Injection
        combined = torch.cat([cnn_feat, phys_feat], dim=1)
        return self.head(combined)


# ==========================================
# 3. 诊断函数: 敏感度测试
# ==========================================
def run_sensitivity_test(model, val_loader):
    """
    诊断模型是否真的在看波峰：
    随机取一个近场样本，把它的波峰强行抹平，看预测距离是否变大。
    """
    model.eval()
    print("\n🔍 [诊断] 正在进行波峰敏感度测试 (Peak Sensitivity Test)...")

    # 找一个近场样本 (<1.0km)
    target_sample = None
    target_stats = None
    target_dist = None

    for x, stats, y_d_log, _ in val_loader:
        real_dist = torch.pow(10, y_d_log)
        mask = real_dist < 1.0
        if mask.any():
            idx = torch.where(mask)[0][0]
            target_sample = x[idx:idx + 1].clone().to(device)
            target_stats = stats[idx:idx + 1].clone().to(device)
            target_dist = real_dist[idx].item()
            break

    if target_sample is None:
        print("⚠️ 验证集中没找到近场样本，跳过测试。")
        return

    # 1. 原始预测
    with torch.no_grad():
        pred_orig = model(target_sample, target_stats)
        dist_orig = torch.pow(10, pred_orig[0, 0]).item()

    # 2. 抹平波峰 (把 COD 通道置为 0)
    modified_sample = target_sample.clone()
    modified_sample[:, 0, :] = 0.0  # Kill the peak!

    # 也要把 stats 里的 max_cod 抹掉，不然模型会从 stats 里偷看
    modified_stats = target_stats.clone()
    modified_stats[:, 5] = 0.0  # log_max_cod = 0
    modified_stats[:, 6] = 0.0  # log_std_cod = 0

    with torch.no_grad():
        pred_mod = model(modified_sample, modified_stats)
        dist_mod = torch.pow(10, pred_mod[0, 0]).item()

    print(f"   样本真实距离: {target_dist:.2f} km")
    print(f"   [1] 原始预测: {dist_orig:.2f} km")
    print(f"   [2] 抹平波峰后: {dist_mod:.2f} km")

    change = dist_mod - dist_orig
    if change > 2.0:
        print("✅ 诊断通过：模型通过波峰判断距离 (抹平波峰导致预测距离剧增)")
    else:
        print("❌ 诊断警告：模型对波峰不敏感！可能依然在猜概率。")
    print("-" * 50)


# ==========================================
# 4. 主训练循环
# ==========================================
def train_pi_attentive():
    if os.path.exists('ultimate_dataset_v3.npz'):
        data_path = 'ultimate_dataset_v3.npz'
        print(f"📂 使用数据集: {data_path}")
    else:
        print("❌ 错误: 找不到数据集，请先运行 generate_micro_data.py")
        return

    BATCH_SIZE = 64
    EPOCHS = 15

    ds = PhysicsInformedDataset(data_path)
    train_len = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, len(ds) - train_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = PI_Attentive_MoE().to(device)

    # ⚠️ 关键: reduction='none' 允许我们手动加权
    criterion_nll = nn.GaussianNLLLoss(reduction='none')
    criterion_mass = nn.SmoothL1Loss(reduction='mean')   # Mass 用回归

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("🚀 开始训练 PI-Attentive Net (带加权Loss)...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}/{EPOCHS}", leave=False)

        for x, stats, y_d_log, y_m in loop:
            x, stats = x.to(device), stats.to(device)
            y_d_log, y_m = y_d_log.to(device), y_m.to(device)

            optimizer.zero_grad()
            pred = model(x, stats)
            dist_mu = pred[:, 0]  # 距离均值
            dist_log_var = pred[:, 1]  # 距离方差的log
            mass_mu = pred[:, 2]  # 源强预测

            # 1. 距离 Loss (NLL + 加权)
            dist_var = torch.exp(dist_log_var)
            with torch.no_grad():
                true_dist_km = torch.pow(10, y_d_log)
                weights = 1.0 + 2.0 * torch.exp(-0.5 * true_dist_km)  # 近场加权

            raw_loss_d = criterion_nll(dist_mu, y_d_log, dist_var)
            loss_d = torch.mean(raw_loss_d * weights)

            # 2. 源强 Loss (辅助任务)
            # 这会强迫模型去理解"现在的浓度高是因为距离近，还是因为源强 大"
            loss_m = criterion_mass(mass_mu, y_m)

            # 3. 总 Loss
            # 给 Mass 任务 0.5 的权重，让它辅助主任务
            loss = loss_d * 10.0 + loss_m * 0.5

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # === 验证 & 诊断 ===
        model.eval()
        val_loss = 0
        all_preds, all_trues = [], []
        errors = {'near': [], 'mid': [], 'far': []}

        with torch.no_grad():
            for x, stats, y_d_log, y_m in val_loader:
                x, stats = x.to(device), stats.to(device)
                y_d_log = y_d_log.to(device)

                pred = model(x, stats)
                mu = pred[:, 0]

                # 计算验证 Loss (不加权，看原始表现)
                val_loss += nn.SmoothL1Loss()(pred[:, 0], y_d_log).item()

                pred_km = torch.pow(10, mu).cpu().numpy()
                true_km = torch.pow(10, y_d_log).cpu().numpy()

                all_preds.extend(pred_km)
                all_trues.extend(true_km)

                abs_err = np.abs(pred_km - true_km)
                for i, d in enumerate(true_km):
                    if d < 3.0:
                        errors['near'].append(abs_err[i])
                    elif d < 8.0:
                        errors['mid'].append(abs_err[i])
                    else:
                        errors['far'].append(abs_err[i])

        # 打印指标
        mae_near = np.mean(errors['near']) if errors['near'] else 0
        mae_mid = np.mean(errors['mid']) if errors['mid'] else 0
        mae_far = np.mean(errors['far']) if errors['far'] else 0
        total_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_trues)))

        print(f"Ep {epoch + 1}: Val Loss={val_loss / len(val_loader):.4f} | Avg MAE={total_mae:.2f}km")
        print(f"      📍 Near: {mae_near:.2f} | Mid: {mae_mid:.2f} | Far: {mae_far:.2f}")

        # 运行敏感度诊断
        if (epoch + 1) % 5 == 0:
            run_sensitivity_test(model, val_loader)

        scheduler.step(val_loss)

    # === 最终画图 ===
    print("\n📊 生成最终结果图...")
    all_trues = np.array(all_trues)
    all_preds = np.array(all_preds)

    plt.figure(figsize=(14, 6))

    # 散点图
    plt.subplot(1, 2, 1)
    plt.scatter(all_trues, all_preds, alpha=0.5, s=10, c='teal', label='Predictions')
    plt.plot([0, 12], [0, 12], 'k--', lw=2)
    plt.title(f"Probabilistic Regression (MAE={total_mae:.2f}km)")
    plt.xlabel("True Distance (km)")
    plt.ylabel("Predicted Distance (km)")
    plt.grid(True, alpha=0.3)

    # 诊断图: 距离 vs 误差
    # 我们希望看到近场(左边)的误差很低，而不是很高
    plt.subplot(1, 2, 2)
    abs_errors = np.abs(all_preds - all_trues)
    plt.scatter(all_trues, abs_errors, alpha=0.5, s=10, c='crimson')
    plt.hlines(0.5, 0, 12, colors='k', linestyles='dashed', label='0.5km Error')
    plt.title("Diagnosis: Distance vs. Absolute Error")
    plt.xlabel("True Distance (km)")
    plt.ylabel("Absolute Error (km)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return model


if __name__ == "__main__":
    model = train_pi_attentive()
    torch.save(model.state_dict(), 'agent_model_final.pth')
    print("💾 模型已保存: agent_model_final.pth")