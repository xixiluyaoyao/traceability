import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os

#定义 ResNet-1D 残差块 (网络升级)
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果输入输出通道不一致，用 1x1 卷积调整 shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 核心：残差连接
        out = self.relu(out)
        return out


class DeepWaterAE(nn.Module):
    """深层残差自编码器"""

    def __init__(self, n_features=4):
        super().__init__()

        # --- Encoder (特征提取器) ---
        self.encoder = nn.Sequential(
            # 初始升维
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # 残差块堆叠 (越深特征越抽象)
            ResBlock1D(32, 64),
            nn.MaxPool1d(2),  # 降采样 15 -> 7

            ResBlock1D(64, 128),
            nn.MaxPool1d(2),  # 降采样 7 -> 3

            ResBlock1D(128, 256)
            # 最终 Latent 特征形状: [Batch, 256, 3]
        )

        # Decoder (重建器)
        self.decoder = nn.Sequential(
            # 镜像上采样
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # 3 -> 6
            ResBlock1D(256, 128),

            nn.Upsample(scale_factor=2.5, mode='linear', align_corners=False),  # 6 -> 15 (大约)
            ResBlock1D(128, 64),

            ResBlock1D(64, 32),

            # 输出层
            nn.Conv1d(32, n_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 确保输入输出长度一致 (处理上采样的舍入误差)
        target_len = x.shape[2]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # 强制插值回原始长度 (万能保险)
        if decoded.shape[2] != target_len:
            decoded = torch.nn.functional.interpolate(decoded, size=target_len, mode='linear', align_corners=False)

        return decoded

#数据加载
def load_and_clean_data(clean_path, dirty_path, seq_len=15):
    data_chunks = []

    # 加载干净数据 (取 30%)
    if os.path.exists(clean_path):
        print(f" 加载干净背景: {clean_path} ...")
        clean = np.load(clean_path)
        # 随机采样 30%
        n_sample = int(len(clean) * 0.3)
        indices = np.random.choice(len(clean), n_sample, replace=False)
        data_chunks.append(clean[indices])
        print(f"   -> 采样 {n_sample} 条 (30%)")

    # 加载污染数据 (取 50% - 重点关注)
    if os.path.exists(dirty_path):
        print(f" 加载污染数据: {dirty_path} ...")
        dirty = np.load(dirty_path)
        # 随机采样 50% (Over-sampling)
        n_sample = int(len(dirty) * 0.5)
        indices = np.random.choice(len(dirty), n_sample, replace=False)
        data_chunks.append(dirty[indices])
        print(f"   -> 采样 {n_sample} 条 (50%) - 强化训练！")

    if not data_chunks:
        raise ValueError("没有找到 .npy 数据文件！")

    # 合并
    all_data = np.vstack(data_chunks)

    # NaN 清洗与数值稳定
    print(" 正在清洗数据 (NaN Check)...")
    # 替换 Inf 为 NaN，然后删除含 NaN 的行
    all_data = np.nan_to_num(all_data, nan=np.nan, posinf=np.nan, neginf=np.nan)
    mask = ~np.isnan(all_data).any(axis=1)
    all_data = all_data[mask]
    print(f"   -> 清洗后剩余有效数据: {len(all_data)} 条")

    # 标准化 (防止数值过大导致梯度爆炸)
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)

    # 构建时间序列
    # 由于不做时序预测，我们这里简单地把每15行切成一段
    # 为了速度，直接 reshape (可能会丢掉最后一点尾数)
    n_seq = len(all_data) // seq_len
    X = all_data[:n_seq * seq_len].reshape(n_seq, seq_len, -1)  # [N, 15, 4]
    X = X.transpose(0, 2, 1)  # [N, 4, 15]

    return X

# 3. 训练主循环 (含 GPU 加速与 NaN 防护)
def main():
    # 配置
    CLEAN_PATH = 'clean_background.npy'
    DIRTY_PATH = 'dirty_background.npy'
    BATCH_SIZE = 1024  # 显存够大就往大了调，加速训练
    LR = 1e-4  # 稍微调低一点，防止 NaN
    EPOCHS = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" 训练设备: {device}")
    if device.type == 'cuda':
        print(f" GPU: {torch.cuda.get_device_name(0)}")

    #准备数据
    try:
        X_train = load_and_clean_data(CLEAN_PATH, DIRTY_PATH)
    except Exception as e:
        print(f" 数据加载失败: {e}")
        return

    # 转 Tensor
    dataset = TensorDataset(torch.FloatTensor(X_train))
    # num_workers=4 和 pin_memory=True 是 GPU 加速的关键
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True)

    #初始化模型
    model = DeepWaterAE(n_features=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print(f" 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(" 开始训练...")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        valid_batches = 0

        for batch_idx, batch in enumerate(loader):
            x = batch[0].to(device, non_blocking=True)  # 异步传输

            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)

            # NaN 防护
            if torch.isnan(loss):
                print(f"⚠️ Warning: Loss is NaN at Epoch {epoch}, Batch {batch_idx}!")
                print("   可能原因: 数据异常或学习率过高。跳过此 Batch。")
                continue

            loss.backward()

            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1

        avg_loss = total_loss / max(1, valid_batches)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.6f}")

    # 保存权重
    # 只保存 Encoder 部分，给下游任务用
    torch.save(model.encoder.state_dict(), 'resnet_encoder.pth')
    print(" 预训练完成！ResNet Encoder 已保存至 'resnet_encoder.pth'")


if __name__ == "__main__":
    # Windows 下多进程 DataLoader 需要保护
    import multiprocessing

    multiprocessing.freeze_support()
    main()