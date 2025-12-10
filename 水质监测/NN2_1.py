import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpillAdaptedNet(nn.Module):
    """针对偷排场景的神经网络"""

    def __init__(self, n_features=5, seq_len=15, engineered_dim=44):
        super().__init__()

        # 1. 时序特征提取 - 专门捕获偷排的瞬时特征
        self.temporal_conv = nn.Sequential(
            # 多尺度卷积捕获不同的衰减模式
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),# 小尺度捕获局部波动
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),# 中尺度捕获中等范围的衰减趋势
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=7, padding=3),# 大尺度捕获整体衰减模式
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)# 全局平均池化
        )

        # 2. 偷排特征提取器
        self.spill_feature_extractor = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 32),
            nn.ReLU()
        )

        # 3. 工程特征处理
        self.engineered_processor = nn.Sequential(
            nn.Linear(engineered_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 24),
            nn.ReLU()
        )

        # 4. 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(32 + 24, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 48),
            nn.ReLU()
        )

        # 5. 距离预测分支 - 强化距离回溯能力
        self.distance_predictor = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            #Sigmoid限制输出范围,避免预测出负距离或超大距离
            nn.Sigmoid()  # 输出0-1，后续映射到实际距离范围
        )

        # 6. 源强度预测分支 - 考虑距离信息
        self.source_predictor = nn.Sequential(
            nn.Linear(48 + 1, 40),  # 48特征 + 1距离
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(40, 24),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(24, 1)
        )

        # 7. 4桶分类器
        self.bucket_classifier = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 4) # 输出从3调整到4
        )

        # 距离映射参数
        self.register_buffer('dist_min', torch.tensor(0.05)) #
        self.register_buffer('dist_max', torch.tensor(10.0)) # 10.0km

    def forward(self, temporal, engineered):
        # 时序特征提取
        temporal_conv = self.temporal_conv(temporal.transpose(1, 2)).squeeze(-1)
        spill_features = self.spill_feature_extractor(temporal_conv)

        # 工程特征处理
        eng_features = self.engineered_processor(engineered)

        # 特征融合
        fused = self.fusion(torch.cat([spill_features, eng_features], dim=1))

        # 距离预测
        dist_raw = self.distance_predictor(fused)
        distance_pred = self.dist_min + (self.dist_max - self.dist_min) * dist_raw

        # 源强度预测（利用距离信息）
        #距离越远,衰减越多,反推源强度需要更大的"放大系数"，让模型学习: 源强度 = 测量浓度 × 距离相关的放大因子
        source_input = torch.cat([fused, distance_pred], dim=1)
        source_pred = self.source_predictor(source_input)

        # 分类预测
        bucket_logits = self.bucket_classifier(fused)

        return source_pred, distance_pred, bucket_logits


class SpillAdaptedLoss(nn.Module):
    """偷排场景损失函数"""

    def __init__(self, alpha_dist=0.4, alpha_source=0.4, alpha_class=0.2):
        super().__init__()
        self.alpha_dist = alpha_dist
        self.alpha_source = alpha_source
        self.alpha_class = alpha_class

    def forward(self, source_pred, dist_pred, bucket_logits,
                source_true, dist_true, bucket_true):
        # 1. 距离损失 - 使用Huber loss，对偷排的距离预测更稳健
        #Huber对离群点不敏感(比MSE鲁棒)
        dist_loss = F.huber_loss(dist_pred.squeeze(), dist_true, delta=0.1)#误差<0.1km用L2,>0.1km用L1

        # 2. 源强度损失 - 考虑距离权重
        # 距离越近，源强度预测应该越准确
        dist_weights = 1.0 / (dist_true + 0.1)  # 距离越近权重越大
        source_loss = (dist_weights * F.huber_loss(
            source_pred.squeeze(), source_true, delta=0.3, reduction='none'
        )).mean()

        # 3. 分类损失 - Focal loss处理类别不平衡
        ce_loss = F.cross_entropy(bucket_logits, bucket_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** 2 * ce_loss #置信度高的样本权重降低，难分样本权重更大
        class_loss = focal_loss.mean()

        # 4. 一致性损失 - 确保距离预测与分类一致
        # 匹配4桶分类中心：0.25km, 1.5km, 4.0km, 7.5km
        bucket_centers = torch.tensor([0.25, 1.5, 4.0, 7.5], device=bucket_logits.device)
        expected_dist = bucket_centers[bucket_true]
        consistency_loss = F.mse_loss(dist_pred.squeeze(), expected_dist)

        total_loss = ((self.alpha_dist + 0.1) * dist_loss +  # 将距离权重提高 0.1
                      self.alpha_source * source_loss +
                      self.alpha_class * class_loss)

        return total_loss, {
            'dist_loss': dist_loss.item(),
            'source_loss': source_loss.item(),
            'class_loss': class_loss.item(),
            'consistency_loss': consistency_loss.item()
        }


def create_spill_adapted_model(n_features=5, seq_len=15, engineered_dim=44):
    """创建偷排适配模型"""
    model = SpillAdaptedNet(n_features, seq_len, engineered_dim)
    print(f"偷排适配模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


# 训练函数
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


# 测试函数
def test_spill_model(model, test_loader):
    """测试偷排模型性能，并按距离分桶展示结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_source_pred, all_source_true = [], []
    all_dist_pred, all_dist_true = [], []
    all_bucket_pred, all_bucket_true = [], []

    with torch.no_grad():
        for batch in test_loader:
            temporal, engineered, source_true, dist_true, bucket_true = [
                x.to(device) for x in batch
            ]

            source_pred, dist_pred, bucket_logits = model(temporal, engineered)
            bucket_pred = bucket_logits.argmax(dim=1)

            all_source_pred.extend(source_pred.cpu().numpy())
            all_source_true.extend(source_true.cpu().numpy())
            all_dist_pred.extend(dist_pred.cpu().numpy())
            all_dist_true.extend(dist_true.cpu().numpy())
            all_bucket_pred.extend(bucket_pred.cpu().numpy())
            all_bucket_true.extend(bucket_true.cpu().numpy())

    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

    # 将列表转换为 NumPy 数组
    all_dist_pred = np.array(all_dist_pred).flatten()
    all_dist_true = np.array(all_dist_true).flatten()
    all_bucket_true = np.array(all_bucket_true)
    all_bucket_pred = np.array(all_bucket_pred)
    all_source_true = np.array(all_source_true).flatten()
    all_source_pred = np.array(all_source_pred).flatten()

    # 计算整体指标
    source_r2 = r2_score(all_source_true, all_source_pred)
    source_mae = mean_absolute_error(all_source_true, all_source_pred)
    dist_r2 = r2_score(all_dist_true, all_dist_pred)
    dist_mae = mean_absolute_error(all_dist_true, all_dist_pred)
    bucket_acc = accuracy_score(all_bucket_true, all_bucket_pred)

    print(f"偷排模型整体测试结果:")
    print(f"源强度 - R²: {source_r2:.4f}, MAE: {source_mae:.2f}")
    print(f"距离   - R²: {dist_r2:.4f}, MAE: {dist_mae:.4f} km")
    print(f"分类   - 准确率: {bucket_acc:.3f}")
    print("-" * 40)
    print("分桶性能细分 (按真实距离分桶):")

    # 距离分桶定义 (与 train2_1_1.py 中的 create_buckets 一致)
    bucket_names = ["0: 近场 (0-0.5km)", "1: 中场 (0.5-2.5km)", "2: 远场 (2.5-5.5km)", "3: 超远场 (>5.5km)"]

    bucket_metrics = {}

    for i in range(4):
        mask = (all_bucket_true == i)

        if np.sum(mask) == 0:
            bucket_metrics[bucket_names[i]] = {"count": 0, "dist_mae": np.nan, "class_acc": np.nan}
            continue

        # 距离 MAE
        mae = mean_absolute_error(all_dist_true[mask], all_dist_pred[mask])

        # 分类准确率 (评估该样本是否被正确分到其真实距离桶)
        acc = accuracy_score(all_bucket_true[mask], all_bucket_pred[mask])

        bucket_metrics[bucket_names[i]] = {
            "count": int(np.sum(mask)),
            "dist_mae": mae,
            "class_acc": acc
        }

        print(f"  {bucket_names[i]} (N={int(np.sum(mask))}):")
        print(f"    距离 MAE: {mae:.4f} km")
        print(f"    分类准确率: {acc:.3f}")

    return {
        'source_r2': source_r2, 'source_mae': source_mae,
        'dist_r2': dist_r2, 'dist_mae': dist_mae,
        'bucket_acc': bucket_acc,
        'bucket_metrics': bucket_metrics
    }


if __name__ == "__main__":
    # 测试模型创建
    model = create_spill_adapted_model()

    # 测试前向传播
    batch_size = 32
    temporal = torch.randn(batch_size, 15, 5)
    engineered = torch.randn(batch_size, 44)

    source_pred, dist_pred, bucket_logits = model(temporal, engineered)
    print(f"输出形状: 源强度{source_pred.shape}, 距离{dist_pred.shape}, 分类{bucket_logits.shape}")