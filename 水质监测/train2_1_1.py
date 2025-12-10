import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from NN2_1 import create_spill_adapted_model, train_spill_model, test_spill_model
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def extract_features(X):
    """
    提取工程特征 (X shape: [N, Seq_Len, N_Features])
    Features: COD_max, COD_mean, COD_std, COD_slope (关键特征), pH_mean, DO_mean, Velocity_mean, ...
    """
    N, L, F = X.shape

    # 提取时间序列数据: X[:,:,0] = COD, X[:,:,1] = pH, X[:,:,2] = DO, X[:,:,3] = Velocity
    cod_seq = X[:, :, 0]
    ph_seq = X[:, :, 1]
    do_seq = X[:, :, 2]
    vel_seq = X[:, :, 3]

    # 1. 统计特征
    cod_max = np.max(cod_seq, axis=1)
    cod_mean = np.mean(cod_seq, axis=1)
    cod_std = np.std(cod_seq, axis=1)
    ph_mean = np.mean(ph_seq, axis=1)
    do_mean = np.mean(do_seq, axis=1)
    vel_mean = np.mean(vel_seq, axis=1)

    # 2. 衰减和梯度特征 (对距离预测至关重要)
    # 线性拟合的斜率，用于表示沿河道的衰减率
    seq_index = np.arange(L)
    cod_slope = np.zeros(N)
    for i in range(N):
        # 拟合 y=ax+b, a即为斜率。斜率的绝对值越大，越可能靠近污染源
        cod_slope[i] = np.polyfit(seq_index, cod_seq[i, :], 1)[0]

    # 3. 极值差 (反映瞬时污染强度)
    cod_range = cod_max - np.min(cod_seq, axis=1)

    # 构建特征列表
    feature_list = [cod_max, cod_mean, cod_std, cod_slope, cod_range, ph_mean, do_mean, vel_mean]

    # 堆叠特征 (共 8 维)
    all_features = np.hstack([f.reshape(-1, 1) for f in feature_list])

    return all_features  # 返回 8 维关键特征


def load_spill_data():
    """加载偷排数据 """
    data = np.load('truck_spill_dataset.npz')

    X = np.stack([data['cod_sequences'], data['ph_sequences'],
                  data['do_sequences'], data['velocity_sequences'],
                  data['missing_masks']], axis=-1)
    y_raw = data['targets_raw']

    # 加载河流特征
    river_features_raw = data['river_features']

    return X, y_raw, river_features_raw  # 返回河流特征


def create_buckets(distances):
    """创建4桶分类，以适应10km的溯源范围"""
    # Near: < 0.5 km (0)
    # Mid: 0.5 - 2.5 km (1)
    # Far: 2.5 - 5.5 km (2)
    # Ultra-Far: > 5.5 km (3)
    bins = [0.50, 2.50, 5.50] # 3个分界线将数据分成4个桶
    bucket_labels = np.digitize(distances, bins)
    return bucket_labels


def prepare_data():
    """准备训练数据"""
    X, y_raw, river_features_raw = load_spill_data()

    # 1. 提取序列统计特征 (8 维)
    seq_eng_features = extract_features(X)

    # 2. 合并序列特征和河流特征 (8 + 4 = 12 维)
    key_features = np.hstack([seq_eng_features, river_features_raw])

    # 3. 填充至模型要求的 engineered_dim=44
    N, current_dim = key_features.shape
    target_dim = 44
    if current_dim < target_dim:
        padding_zeros = np.zeros((N, target_dim - current_dim))
        eng_features = np.hstack([key_features, padding_zeros])
    else:
        eng_features = key_features[:, :target_dim]  # 如果超出了则截断

    # 创建桶标签
    bucket_labels = create_buckets(y_raw[:, 1])

    # 数据划分
    X_train, X_test, eng_train, eng_test, y_train, y_test, b_train, b_test = train_test_split(
        X, eng_features, y_raw, bucket_labels, test_size=0.2, random_state=42, stratify=bucket_labels)
    X_train, X_val, eng_train, eng_val, y_train, y_val, b_train, b_val = train_test_split(
        X_train, eng_train, y_train, b_train, test_size=0.2, random_state=42, stratify=b_train)

    # 标准化
    for i in range(4):
        scaler = StandardScaler()
        X_train[:, :, i] = scaler.fit_transform(X_train[:, :, i])
        X_val[:, :, i] = scaler.transform(X_val[:, :, i])
        X_test[:, :, i] = scaler.transform(X_test[:, :, i])

    eng_scaler = StandardScaler()
    eng_train = eng_scaler.fit_transform(eng_train)
    eng_val = eng_scaler.transform(eng_val)
    eng_test = eng_scaler.transform(eng_test)

    # 源浓度标准化
    source_scaler = StandardScaler()
    y_src_train = source_scaler.fit_transform(np.log1p(y_train[:, 0:1])).flatten()
    y_src_val = source_scaler.transform(np.log1p(y_val[:, 0:1])).flatten()
    y_src_test = source_scaler.transform(np.log1p(y_test[:, 0:1])).flatten()

    return {
        'train': (X_train, eng_train, y_src_train, y_train[:, 1], b_train),
        'val': (X_val, eng_val, y_src_val, y_val[:, 1], b_val),
        'test': (X_test, eng_test, y_src_test, y_test[:, 1], b_test),
        'source_scaler': source_scaler
    }


def create_data_loaders(data_dict, batch_size=64):
    """创建数据加载器
    近场样本少(因为阈值严格)，不平衡会导致模型偏向预测远场，所以用WeightedRandomSampler对近点做更多采样
    """

    # 1. 计算训练集权重
    train_labels = data_dict['train'][4]
    class_counts = np.bincount(train_labels)

    # 避免除以零，计算逆频权重
    class_weights = 1.0 / np.maximum(class_counts, 1)

    # 为每个训练样本分配其对应的类别权重
    sample_weights = class_weights[train_labels]

    # 2. 创建 Weighted Random Sampler
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),  # 仍以原始训练集大小进行采样
        replacement=True
    )

    def make_loader(data_tuple, sampler=None, shuffle=False):
        tensors = [torch.FloatTensor(x) for x in data_tuple[:2]] + [torch.FloatTensor(data_tuple[2]),
                                                                    torch.FloatTensor(data_tuple[3]),
                                                                    torch.LongTensor(data_tuple[4])]
        # 注意：使用 sampler 时，不能设置 shuffle=True
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle
        )

    # 3. 创建 DataLoader
    return {
        'train': make_loader(data_dict['train'], sampler=sampler),  # 训练集使用 sampler
        'val': make_loader(data_dict['val'], shuffle=False),
        'test': make_loader(data_dict['test'], shuffle=False)
    }


def main():
    print("开始训练偷排适配神经网络...")

    # 准备数据
    data_dict = prepare_data()
    loaders = create_data_loaders(data_dict)

    print(f"训练样本: {len(data_dict['train'][0])}")
    print(f"验证样本: {len(data_dict['val'][0])}")
    print(f"测试样本: {len(data_dict['test'][0])}")

    # 创建模型
    model = create_spill_adapted_model()

    # 训练模型
    trained_model = train_spill_model(
        model, loaders['train'], loaders['val'],
        epochs=100, lr=1e-3
    )

    # 测试模型
    results = test_spill_model(trained_model, loaders['test'])

    print("训练完成！")
    return results


if __name__ == "__main__":
    main()