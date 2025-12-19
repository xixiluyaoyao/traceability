import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


#ResBlock1D
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

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
        out += residual
        out = self.relu(out)
        return out


#自注意力机制 (Self-Attention)
class SelfAttention(nn.Module):
    """
    通过关注全局特征来解决中远场波形模糊的问题
    """

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放系数

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width = x.size()

        # 投影到 Query, Key, Value
        proj_query = self.query(x).view(batch_size, -1, width).permute(0, 2, 1)  # B x W x C'
        proj_key = self.key(x).view(batch_size, -1, width)  # B x C' x W

        # 计算注意力图 (Attention Map)
        energy = torch.bmm(proj_query, proj_key)  # B x W x W
        attention = self.softmax(energy)

        # 加权求和
        proj_value = self.value(x).view(batch_size, -1, width)  # B x C x W
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x W

        out = out.view(batch_size, C, width)

        # 残差连接：原始特征 + 注意力特征
        out = self.gamma * out + x
        return out


# 溯源网络
class SpillAdaptedNet(nn.Module):
    def __init__(self, n_features=5, seq_len=15, engineered_dim=44):
        super().__init__()

        #Encoder (ResNet)
        self.encoder_stem = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.encoder_layers = nn.Sequential(
            ResBlock1D(32, 64),
            nn.MaxPool1d(2),
            ResBlock1D(64, 128),
            nn.MaxPool1d(2),
            ResBlock1D(128, 256)
        )

        # 自注意力层，加在 ResNet 之后
        # 帮助模型在时序维度上寻找关联，区分中场和远场的细微差别
        self.attention = SelfAttention(256)

        self.flatten_dim = 256 * 3

        # 偷排特征提取器
        self.spill_feature_extractor = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 工程特征处理
        self.engineered_processor = nn.Sequential(
            nn.Linear(engineered_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 24),
            nn.ReLU()
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(64 + 24, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 48),
            nn.ReLU()
        )

        # 预测分支
        self.distance_predictor = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.source_predictor = nn.Sequential(
            nn.Linear(48 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.bucket_classifier = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        self.register_buffer('dist_min', torch.tensor(0.05))
        self.register_buffer('dist_max', torch.tensor(10.0))

    def forward(self, temporal, engineered):
        x = temporal.transpose(1, 2)

        x = self.encoder_stem(x)
        x = self.encoder_layers(x)

        #通过注意力机制增强特征
        x = self.attention(x)

        x = x.view(x.size(0), -1)

        spill_features = self.spill_feature_extractor(x)
        eng_features = self.engineered_processor(engineered)

        fused = self.fusion(torch.cat([spill_features, eng_features], dim=1))

        dist_raw = self.distance_predictor(fused)
        distance_pred = self.dist_min + (self.dist_max - self.dist_min) * dist_raw

        source_input = torch.cat([fused, distance_pred], dim=1)
        source_pred = self.source_predictor(source_input)

        bucket_logits = self.bucket_classifier(fused)

        return source_pred, distance_pred, bucket_logits


# 权重加载
def load_pretrained_weights(model, pretrained_path='resnet_encoder.pth'):
    if not os.path.exists(pretrained_path):
        print("⚠️ 未找到预训练权重，将使用随机初始化训练。")
        return model

    print(f" 正在加载预训练权重: {pretrained_path} ...")
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()

    loaded_count = 0
    # 注意：SelfAttention 层的参数是新加的，预训练里没有，这没关系，它会随机初始化并跟随后续训练

    for key, val in pretrained_dict.items():
        if key == 'encoder.0.weight':
            target_key = 'encoder_stem.0.weight'
            if target_key in model_dict:
                with torch.no_grad():
                    model_dict[target_key][:, :3, :] = val[:, :3, :]
                loaded_count += 1
                continue
        elif key == 'encoder.0.bias':
            target_key = 'encoder_stem.0.bias'
            if target_key in model_dict: model_dict[target_key] = val; loaded_count += 1; continue
        elif key == 'encoder.1.weight':
            target_key = 'encoder_stem.1.weight'
            if target_key in model_dict: model_dict[target_key] = val; loaded_count += 1; continue
        elif key == 'encoder.1.bias':
            target_key = 'encoder_stem.1.bias'
            if target_key in model_dict: model_dict[target_key] = val; loaded_count += 1; continue

        layer_map = {'3.': '0.', '5.': '2.', '7.': '4.'}
        new_key = key.replace('encoder.', '')
        for old_idx, new_idx in layer_map.items():
            if new_key.startswith(old_idx):
                target_key = 'encoder_layers.' + new_key.replace(old_idx, new_idx)
                if target_key in model_dict and model_dict[target_key].shape == val.shape:
                    model_dict[target_key] = val
                    loaded_count += 1
                break

    model.load_state_dict(model_dict)
    print(f" 成功加载 {loaded_count} 个 ResNet 层参数。Attention 层将从头训练。")
    return model

# Loss 函数
class SpillAdaptedLoss(nn.Module):
    def __init__(self, alpha_dist=0.4, alpha_source=0.4, alpha_class=0.2):
        super().__init__()
        self.alpha_dist = alpha_dist
        self.alpha_source = alpha_source
        self.alpha_class = alpha_class

    def forward(self, source_pred, dist_pred, bucket_logits,
                source_true, dist_true, bucket_true):

        dist_loss = F.huber_loss(dist_pred.squeeze(), dist_true, delta=0.1)

        #使用温和的 sqrt 权重，既照顾近场也不完全放弃远场
        dist_weights = 1.0 / torch.sqrt(dist_true + 0.1)

        # 源强度 Loss
        source_loss = (dist_weights * F.huber_loss(
            source_pred.squeeze(), source_true, delta=0.3, reduction='none'
        )).mean()

        # 分类 Loss
        ce_loss = F.cross_entropy(bucket_logits, bucket_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** 2 * ce_loss).mean()

        # 一致性 Loss
        bucket_centers = torch.tensor([0.25, 1.5, 4.0, 7.5], device=bucket_logits.device)
        expected_dist = bucket_centers[bucket_true]
        consistency_loss = F.mse_loss(dist_pred.squeeze(), expected_dist)

        total_loss = ((self.alpha_dist + 0.1) * dist_loss +
                      self.alpha_source * source_loss +
                      self.alpha_class * focal_loss)

        return total_loss, {
            'dist_loss': dist_loss.item(),
            'source_loss': source_loss.item(),
            'class_loss': focal_loss.item(),
            'consistency_loss': consistency_loss.item()
        }


def create_spill_adapted_model(n_features=5, seq_len=15, engineered_dim=44):
    model = SpillAdaptedNet(n_features, seq_len, engineered_dim)
    model = load_pretrained_weights(model, 'resnet_encoder.pth')
    print(f" 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model

def train_spill_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    """训练偷排适配模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 3, epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    criterion = SpillAdaptedLoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = []

        for batch in train_loader:
            temporal, engineered, source_true, dist_true, bucket_true = [
                x.to(device) for x in batch
            ]

            optimizer.zero_grad()
            source_pred, dist_pred, bucket_logits = model(temporal, engineered)

            loss, loss_dict = criterion(
                source_pred, dist_pred, bucket_logits,
                source_true, dist_true, bucket_true
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        # 验证阶段
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                temporal, engineered, source_true, dist_true, bucket_true = [
                    x.to(device) for x in batch
                ]

                source_pred, dist_pred, bucket_logits = model(temporal, engineered)
                loss, _ = criterion(
                    source_pred, dist_pred, bucket_logits,
                    source_true, dist_true, bucket_true
                )
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_spill_model.pth')

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    return model