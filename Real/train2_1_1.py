import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import multiprocessing
# 引用模型定义
from NN2_1 import create_spill_adapted_model,train_spill_model


def extract_features(X):
    """
    提取工程特征
    X shape: [N, Seq_Len, N_Features]
    """
    N, L, F = X.shape
    cod_seq = X[:, :, 0]
    ph_seq = X[:, :, 1]
    do_seq = X[:, :, 2]
    vel_seq = X[:, :, 3]

    cod_max = np.max(cod_seq, axis=1)
    cod_mean = np.mean(cod_seq, axis=1)
    cod_std = np.std(cod_seq, axis=1)
    ph_mean = np.mean(ph_seq, axis=1)
    do_mean = np.mean(do_seq, axis=1)
    vel_mean = np.mean(vel_seq, axis=1)

    seq_index = np.arange(L)
    cod_slope = np.zeros(N)
    for i in range(N):
        try:
            cod_slope[i] = np.polyfit(seq_index, cod_seq[i, :], 1)[0]
        except:
            cod_slope[i] = 0

    cod_range = cod_max - np.min(cod_seq, axis=1)
    feature_list = [cod_max, cod_mean, cod_std, cod_slope, cod_range, ph_mean, do_mean, vel_mean]
    all_features = np.hstack([f.reshape(-1, 1) for f in feature_list])
    return all_features


def load_spill_data():
    """加载数据"""
    print(" 正在加载数据集 truck_spill_dataset.npz ...")
    data = np.load('truck_spill_dataset.npz')

    # 堆叠输入特征
    X = np.stack([data['cod_sequences'], data['ph_sequences'],
                  data['do_sequences'], data['velocity_sequences'],
                  data['missing_masks']], axis=-1)
    y_raw = data['targets_raw']
    river_features_raw = data['river_features']
    return X, y_raw, river_features_raw


def create_buckets(distances):
    """创建4桶分类"""
    bins = [0.50, 2.50, 5.50]
    bucket_labels = np.digitize(distances, bins)
    return bucket_labels


def prepare_data():
    """准备数据与预处理"""
    X, y_raw, river_features_raw = load_spill_data()

    # 特征工程
    print(" 正在提取工程特征...")
    seq_eng_features = extract_features(X)
    key_features = np.hstack([seq_eng_features, river_features_raw])

    # Padding
    N, current_dim = key_features.shape
    target_dim = 44
    if current_dim < target_dim:
        padding = np.zeros((N, target_dim - current_dim))
        eng_features = np.hstack([key_features, padding])
    else:
        eng_features = key_features[:, :target_dim]

    bucket_labels = create_buckets(y_raw[:, 1])

    # 划分数据集 (Stratified split 保证各桶比例一致)
    X_train, X_test, eng_train, eng_test, y_train, y_test, b_train, b_test = train_test_split(
        X, eng_features, y_raw, bucket_labels, test_size=0.2, random_state=42, stratify=bucket_labels)
    X_train, X_val, eng_train, eng_val, y_train, y_val, b_train, b_val = train_test_split(
        X_train, eng_train, y_train, b_train, test_size=0.2, random_state=42, stratify=b_train)

    # 序列特征标准化
    for i in range(4):  # 对 COD, pH, DO, Vel 进行标准化
        scaler = StandardScaler()
        X_train[:, :, i] = scaler.fit_transform(X_train[:, :, i])
        X_val[:, :, i] = scaler.transform(X_val[:, :, i])
        X_test[:, :, i] = scaler.transform(X_test[:, :, i])

    # 工程特征标准化
    eng_scaler = StandardScaler()
    eng_train = eng_scaler.fit_transform(eng_train)
    eng_val = eng_scaler.transform(eng_val)
    eng_test = eng_scaler.transform(eng_test)

    # 【关键】源浓度 Log 变换 + 标准化
    source_scaler = StandardScaler()
    # 使用 log1p 平滑长尾分布
    y_src_train = source_scaler.fit_transform(np.log1p(y_train[:, 0:1])).flatten()
    y_src_val = source_scaler.transform(np.log1p(y_val[:, 0:1])).flatten()
    y_src_test = source_scaler.transform(np.log1p(y_test[:, 0:1])).flatten()

    print(" 数据预处理完成。")
    return {
        'train': (X_train, eng_train, y_src_train, y_train[:, 1], b_train),
        'val': (X_val, eng_val, y_src_val, y_val[:, 1], b_val),
        'test': (X_test, eng_test, y_src_test, y_test[:, 1], b_test),
        'source_scaler': source_scaler  # 返回 scaler 用于后续反归一化
    }


def create_data_loaders(data_dict, batch_size=256):  # 增大 BatchSize 利用 GPU
    """
    创建数据加载器
    """
    # 1. 加权采样器 (解决类别不平衡)
    train_labels = data_dict['train'][4]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    def make_loader(data_tuple, sampler=None, shuffle=False):
        # 统一转为 Tensor
        # data_tuple: (X, eng, y_src, y_dist, y_bucket)
        # 注意: X 需要 transpose 为 [N, Features, Seq_Len] 以适配 Conv1d
        temporal_tensor = torch.FloatTensor(data_tuple[0])  # [N, 15, 5]
        # 在这里不做 transpose，因为模型内部 forward 第一步做了，或者可以在这里做
        # 保持与 NN2_1.py 输入一致: forward(temporal, ...) -> temporal.transpose(1, 2)

        tensors = [
            temporal_tensor,
            torch.FloatTensor(data_tuple[1]),  # eng
            torch.FloatTensor(data_tuple[2]),  # source
            torch.FloatTensor(data_tuple[3]),  # dist
            torch.LongTensor(data_tuple[4])  # bucket
        ]
        # Windows 下多进程 num_workers 设置
        # 如果是 Windows，建议设为 0 或 2；Linux 可以设为 4 或 8
        import platform
        workers = 0 if platform.system() == 'Windows' else 4

        return DataLoader(
            TensorDataset(*tensors),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=workers,  # 多进程加载
            pin_memory=True,  # 锁页内存，加速 CPU->GPU 传输
            persistent_workers=(workers > 0)  # 保持进程活跃
        )

    return {
        'train': make_loader(data_dict['train'], sampler=sampler),
        'val': make_loader(data_dict['val'], shuffle=False),
        'test': make_loader(data_dict['test'], shuffle=False)
    }


# 重写测试函数以支持反归一化
def test_spill_model_advanced(model, test_loader, source_scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_src_pred, all_src_true = [], []
    all_dist_pred, all_dist_true = [], []
    all_bucket_pred, all_bucket_true = [], []

    print("🧪 开始测试评估...")
    with torch.no_grad():
        for batch in test_loader:
            temporal, engineered, source_true, dist_true, bucket_true = [x.to(device) for x in batch]

            s_pred, d_pred, b_logits = model(temporal, engineered)
            b_pred = b_logits.argmax(dim=1)

            all_src_pred.extend(s_pred.cpu().numpy())
            all_src_true.extend(source_true.cpu().numpy())
            all_dist_pred.extend(d_pred.cpu().numpy())
            all_dist_true.extend(dist_true.cpu().numpy())
            all_bucket_pred.extend(b_pred.cpu().numpy())
            all_bucket_true.extend(bucket_true.cpu().numpy())

    # 转 Numpy
    all_src_pred = np.array(all_src_pred).flatten()
    all_src_true = np.array(all_src_true).flatten()
    all_dist_pred = np.array(all_dist_pred).flatten()
    all_dist_true = np.array(all_dist_true).flatten()
    all_bucket_true = np.array(all_bucket_true)
    all_bucket_pred = np.array(all_bucket_pred)

    # 反归一化源强度
    if source_scaler:
        print(" 正在对源强度进行反归一化 (Log -> Real mg/L)...")
        # nverse Standard Scaler
        all_src_pred = source_scaler.inverse_transform(all_src_pred.reshape(-1, 1)).flatten()
        all_src_true = source_scaler.inverse_transform(all_src_true.reshape(-1, 1)).flatten()
        # Inverse Log (expm1)
        # 防止溢出，做个 clip
        all_src_pred = np.expm1(np.clip(all_src_pred, 0, 15))  # 限制最大值防止 inf
        all_src_true = np.expm1(np.clip(all_src_true, 0, 15))

    # 计算指标
    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

    src_r2 = r2_score(all_src_true, all_src_pred)
    src_mae = mean_absolute_error(all_src_true, all_src_pred)
    dist_r2 = r2_score(all_dist_true, all_dist_pred)
    dist_mae = mean_absolute_error(all_dist_true, all_dist_pred)
    acc = accuracy_score(all_bucket_true, all_bucket_pred)

    print(f"\n 最终测试结果:")
    print(f"   源强度 R²: {src_r2:.4f} | MAE: {src_mae:.2f} mg/L")
    print(f"   距离   R²: {dist_r2:.4f} | MAE: {dist_mae:.4f} km")
    print(f"   分类 Acc : {acc:.4f}")

    # 分桶详细报告
    bucket_names = ["近场(0-0.5)", "中场(0.5-2.5)", "远场(2.5-5.5)", "超远(>5.5)"]
    for i in range(4):
        mask = (all_bucket_true == i)
        if np.sum(mask) > 0:
            d_mae = mean_absolute_error(all_dist_true[mask], all_dist_pred[mask])
            c_acc = accuracy_score(all_bucket_true[mask], all_bucket_pred[mask])
            print(f"  {bucket_names[i]}: Dist MAE={d_mae:.3f}km, Class Acc={c_acc:.3f}")

    return {'src_mae': src_mae, 'dist_mae': dist_mae}


def main():
    # 打印设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" 运行设备: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    data_dict = prepare_data()
    loaders = create_data_loaders(data_dict, batch_size=256)  # 增大Batch

    print(f"训练集: {len(data_dict['train'][0])} 样本")

    # 创建模型 (会自动加载 resnet_encoder.pth)
    model = create_spill_adapted_model()

    # 训练 (建议 Epoch 不用太多，因为数据量大了)
    trained_model = train_spill_model(
        model, loaders['train'], loaders['val'],
        epochs=50,  # 50轮
        lr=1e-3
    )

    print("\n 正在加载最佳模型权重进行测试...")
    trained_model.load_state_dict(torch.load('best_spill_model.pth'))

    # 测试 (传入 scaler)
    test_spill_model_advanced(trained_model, loaders['test'], source_scaler=data_dict['source_scaler'])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()