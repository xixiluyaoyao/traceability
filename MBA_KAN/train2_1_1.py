import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ 关键修改：直接从 NN2_1 导入所有核心组件
# 确保你的 NN2_1.py 就在旁边
from NN2_1 import create_spill_adapted_model, train_spill_model


# ==========================================
# 1. 特征工程 (波形特征提取)
# ==========================================
def extract_features(X):
    """
    输入 X: [N, 15, 4] (COD, pH, DO, Vel) -> 输出: [N, 8]
    针对无人船模式，斜率(Slope)特征尤为重要，因为它指示了离源头的方向
    """
    N, L, F = X.shape
    cod_seq, ph_seq, do_seq, vel_seq = X[:, :, 0], X[:, :, 1], X[:, :, 2], X[:, :, 3]

    # 基础统计特征
    cod_max = np.max(cod_seq, axis=1)
    cod_mean = np.mean(cod_seq, axis=1)
    cod_std = np.std(cod_seq, axis=1)
    ph_mean = np.mean(ph_seq, axis=1)
    do_mean = np.mean(do_seq, axis=1)
    vel_mean = np.mean(vel_seq, axis=1)

    # 形状特征 (斜率) - 在空间模式下，斜率代表浓度梯度
    seq_index = np.arange(L)
    cod_slope = np.zeros(N)
    for i in range(N):
        try:
            # 简单的线性拟合获取梯度
            cod_slope[i] = np.polyfit(seq_index, cod_seq[i, :], 1)[0]
        except:
            cod_slope[i] = 0.0

    cod_range = cod_max - np.min(cod_seq, axis=1)

    feature_list = [cod_max, cod_mean, cod_std, cod_slope, cod_range, ph_mean, do_mean, vel_mean]
    return np.hstack([f.reshape(-1, 1) for f in feature_list])


# ==========================================
# 2. 数据加载与预处理
# ==========================================
def prepare_data():
    # 这里读取的是上一轮生成的“无人船”数据
    path = 'boat_survey_long_dataset.npz'

    # 兼容性处理：如果没生成新数据，还是读旧的试试
    if not os.path.exists(path):
        print(f"⚠️ 未找到 {path}，尝试读取旧版 truck_spill_dataset.npz...")
        path = 'truck_spill_dataset.npz'

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到数据集！请先运行 generate_boat.py 或 generate2_2.py")

    print(f"📦 加载数据集: {path} ...")
    data = np.load(path)
    X_raw = data['sequences']  # [N, 15, 4]
    y_raw = data['targets']  # [Source, Distance]

    # 维度检查 (确保是 [N, 15, 4])
    if X_raw.shape[1] == 4:  # 如果是 [N, 4, 15]
        X = X_raw.transpose(0, 2, 1)
    else:
        X = X_raw

    print("🔧 提取波形工程特征...")
    eng_features = extract_features(X)

    # Padding 到 44维 (为了兼容模型默认设置)
    target_dim = 44
    if eng_features.shape[1] < target_dim:
        padding = np.zeros((len(X), target_dim - eng_features.shape[1]))
        eng_features = np.hstack([eng_features, padding])

    # 数据划分
    distances = y_raw[:, 1]
    # 根据距离分桶，用于分层采样 (0-2km, 2-10km, 10-30km)
    # 船测模式下，我们更关注近场梯度，所以分桶阈值设小一点
    bucket_labels = np.digitize(distances, [2.0, 10.0, 30.0])

    # Split: Train(70%) / Val(15%) / Test(15%)
    X_train, X_test, eng_train, eng_test, y_train, y_test, b_train, b_test = train_test_split(
        X, eng_features, y_raw, bucket_labels, test_size=0.15, random_state=42, stratify=bucket_labels)
    X_train, X_val, eng_train, eng_val, y_train, y_val, b_train, b_val = train_test_split(
        X_train, eng_train, y_train, b_train, test_size=0.15, random_state=42, stratify=b_train)

    # 标准化 (Scaling)
    for i in range(4):  # 4个通道分别标准化
        scaler = StandardScaler()
        X_train[:, :, i] = scaler.fit_transform(X_train[:, :, i])
        X_val[:, :, i] = scaler.transform(X_val[:, :, i])
        X_test[:, :, i] = scaler.transform(X_test[:, :, i])

    eng_scaler = StandardScaler()
    eng_train = eng_scaler.fit_transform(eng_train)
    eng_val = eng_scaler.transform(eng_val)
    eng_test = eng_scaler.transform(eng_test)

    # 目标值 Log 变换
    source_scaler = StandardScaler()
    y_src_train = source_scaler.fit_transform(np.log1p(y_train[:, 0:1])).flatten()
    y_src_val = source_scaler.transform(np.log1p(y_val[:, 0:1])).flatten()
    y_src_test = source_scaler.transform(np.log1p(y_test[:, 0:1])).flatten()

    print(f"✅ 数据准备就绪: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return {
        'train': (X_train, eng_train, y_src_train, y_train[:, 1], b_train),
        'val': (X_val, eng_val, y_src_val, y_val[:, 1], b_val),
        'test': (X_test, eng_test, y_src_test, y_test[:, 1], b_test),
        'source_scaler': source_scaler
    }


# ==========================================
# 3. 数据加载器 (补全缺失部分)
# ==========================================
def create_loaders(data_dict, batch_size=256):
    train_data = data_dict['train']
    # 加权采样 (解决样本不均衡)
    # np.bincount 可能遇到空的桶导致长度不对，这里加个简单的容错
    buckets = train_data[4]
    if len(buckets) == 0:
        raise ValueError("训练集为空，无法创建加载器！")

    counts = np.bincount(buckets)
    # 防止除以0
    class_weights = 1.0 / np.maximum(counts, 1)
    # 生成每个样本的权重
    weights = class_weights[buckets]

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # 【修正点】这里参数名改成了 sampler，与下面调用时保持一致
    def make_loader(d, sampler=None, shuffle=False):
        # d 是一个 tuple: (X, eng, y_src, y_dist, y_bucket)
        tensors = [
            torch.FloatTensor(d[0]),
            torch.FloatTensor(d[1]),
            torch.FloatTensor(d[2]),
            torch.FloatTensor(d[3]),
            torch.LongTensor(d[4])
        ]

        # Windows下多进程设为0，Linux可设为4
        import platform
        workers = 0 if platform.system() == 'Windows' else 4

        return DataLoader(
            TensorDataset(*tensors),
            batch_size=batch_size,
            sampler=sampler,  # 这里匹配参数名
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True
        )

    return {
        # 这里调用时用了 sampler=sampler，所以上面的定义必须是 sampler=None
        'train': make_loader(data_dict['train'], sampler=sampler, shuffle=False),
        'val': make_loader(data_dict['val']),
        'test': make_loader(data_dict['test'])
    }


# ==========================================
# 4. 改进版评估函数 (分段统计)
# ==========================================
def evaluate_model_segmented(model, loader, source_scaler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    src_true, src_pred = [], []
    dist_true, dist_pred = [], []

    print("🧪 正在进行分段评估...")
    with torch.no_grad():
        for batch in loader:
            x, eng, y_src, y_dist, _ = [t.to(device) for t in batch]
            s_out, d_out, _ = model(x, eng)

            src_pred.extend(s_out.cpu().numpy().flatten())
            src_true.extend(y_src.cpu().numpy().flatten())
            dist_pred.extend(d_out.cpu().numpy().flatten())
            dist_true.extend(y_dist.cpu().numpy().flatten())

    # 反归一化
    src_pred_real = np.expm1(source_scaler.inverse_transform(np.array(src_pred).reshape(-1, 1)).flatten())
    src_true_real = np.expm1(source_scaler.inverse_transform(np.array(src_true).reshape(-1, 1)).flatten())
    dist_pred = np.array(dist_pred)
    dist_true = np.array(dist_true)

    # 防止负数
    src_pred_real = np.maximum(src_pred_real, 0)
    dist_pred = np.maximum(dist_pred, 0)

    # --- 全局指标 ---
    from sklearn.metrics import mean_absolute_error, r2_score
    print("\n====== 🌍 全局测试报告 ======")
    print(
        f"源强 MAE: {mean_absolute_error(src_true_real, src_pred_real):.2f} mg/L (R2: {r2_score(src_true_real, src_pred_real):.3f})")
    print(f"距离 MAE: {mean_absolute_error(dist_true, dist_pred):.2f} km   (R2: {r2_score(dist_true, dist_pred):.3f})")

    # --- 分段指标 ---
    print("\n====== 📏 分距离段评估 ======")
    bins = [0, 2, 10, 30, 100]
    labels = ["近场 (0-2km)", "中场 (2-10km)", "远场 (10-30km)", "超远 (>30km)"]

    indices = np.digitize(dist_true, bins)
    for i in range(1, len(bins)):
        mask = (indices == i)
        if np.sum(mask) > 0:
            d_mae = mean_absolute_error(dist_true[mask], dist_pred[mask])
            d_r2 = r2_score(dist_true[mask], dist_pred[mask])
            print(f"[{labels[i - 1]:<13}] 样本: {np.sum(mask):<5} | 距离MAE: {d_mae:.2f} km | R2: {d_r2:.2f}")
        else:
            print(f"[{labels[i - 1]:<13}] 无样本")


# ==========================================
# 5. 主程序
# ==========================================
def main():
    # 1. 准备数据
    try:
        data = prepare_data()
    except Exception as e:
        print(e);
        return

    loaders = create_loaders(data)

    # 2. 创建模型
    # n_features=4 (COD, pH, DO, Vel)
    model = create_spill_adapted_model(n_features=4, engineered_dim=44)

    # ⚠️【关键】这里不调用 load_pretrained_weights，直接从头训练
    print("ℹ️ 提示: 本次训练从零开始 (From Scratch)，不加载旧的预训练权重。")

    # 3. 训练
    print("\n🚀 开始训练 (无人船空间巡测版)...")
    # 先跑 30 个 Epoch 看看效果
    trained_model = train_spill_model(model, loaders['train'], loaders['val'], epochs=30, lr=1e-3)

    # 4. 详细评估
    evaluate_model_segmented(trained_model, loaders['test'], data['source_scaler'])


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()