import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os

# 确保可以导入当前目录下的模块
sys.path.append(os.getcwd())

from generate2_1 import TruckSpillSimulator
from NN2_1 import create_spill_adapted_model
# 直接导入训练代码中的函数，确保逻辑绝对一致
from train2_1_1 import extract_features, load_spill_data, create_buckets

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def prepare_test_data_from_dataset():
    """
    从数据集中加载测试数据，并使用训练集的参数进行标准化。
    这是画出真实预测误差图的关键。
    """
    print("正在从数据集加载测试数据并恢复 Scaler...")
    # 1. 加载原始数据
    X, y_raw, river_features_raw = load_spill_data()

    # 2. 提取特征 (与训练时一致)
    seq_eng_features = extract_features(X)
    key_features = np.hstack([seq_eng_features, river_features_raw])

    # 填充维度至 44
    N, current_dim = key_features.shape
    target_dim = 44
    if current_dim < target_dim:
        padding_zeros = np.zeros((N, target_dim - current_dim))
        eng_features = np.hstack([key_features, padding_zeros])
    else:
        eng_features = key_features[:, :target_dim]

    # 3. 数据划分 (使用与训练相同的种子 random_state=42)
    bucket_labels = create_buckets(y_raw[:, 1])

    # 第一次划分：分离出测试集 (Test Set)
    # y_raw[:, 1] 是距离
    X_train_val, X_test, eng_train_val, eng_test, y_train_val, y_test, b_train_val, b_test = train_test_split(
        X, eng_features, y_raw, bucket_labels, test_size=0.2, random_state=42, stratify=bucket_labels)

    # 第二次划分：分离出训练集 (Train Set) 用于拟合 Scaler
    # 我们不需要验证集，只需要训练集来 fit
    X_train, _, eng_train, _, _, _, _, _ = train_test_split(
        X_train_val, eng_train_val, y_train_val, b_train_val, test_size=0.2, random_state=42, stratify=b_train_val)

    # 4. 在训练集上拟合 Scaler
    feature_scalers = {}
    for i in range(4):  # COD, pH, DO, Velocity
        scaler = StandardScaler()
        scaler.fit(X_train[:, :, i])
        feature_scalers[i] = scaler

    eng_scaler = StandardScaler()
    eng_scaler.fit(eng_train)

    # 5. 使用训练集的参数转换测试集
    X_test_norm = X_test.copy()
    for i in range(4):
        X_test_norm[:, :, i] = feature_scalers[i].transform(X_test[:, :, i])

    eng_test_norm = eng_scaler.transform(eng_test)

    # 返回归一化后的测试数据和真实的距离标签
    return X_test_norm, eng_test_norm, y_test[:, 1]


def plot_cod_curves_strict():
    """
    绘制符合物理逻辑的 COD 衰减曲线。
    逻辑：流量 Q = U * B * Depth
    流量越大 -> 稀释倍数越大 (范围 15-40) -> 初始浓度越低。
    """
    simulator = TruckSpillSimulator()

    source_conc = 20000
    background = 5.0

    # 定义三组不同河道工况
    scenarios_config = [
        {
            'name': '小型河道 (窄/浅/慢)',
            'params': {'U': 0.3, 'B': 20, 'depth': 2.0, 'T': 20},
            'desc': 'B=20m, H=2m, U=0.3'
        },
        {
            'name': '中型河道 (基准)',
            'params': {'U': 0.5, 'B': 40, 'depth': 3.0, 'T': 20},
            'desc': 'B=40m, H=3m, U=0.5'
        },
        {
            'name': '大型河道 (宽/深/快)',
            'params': {'U': 0.8, 'B': 60, 'depth': 4.0, 'T': 20},
            'desc': 'B=60m, H=4m, U=0.8'
        }
    ]

    # 计算流量范围以进行映射
    qs = []
    for s in scenarios_config:
        p = s['params']
        qs.append(p['U'] * p['B'] * p['depth'])
    min_q, max_q = min(qs), max(qs)

    # 映射函数：流量 Q -> 稀释倍数 [15, 40]
    def get_dilution(q):
        if max_q == min_q: return 25
        return 15 + (q - min_q) / (max_q - min_q) * (40 - 15)

    plt.figure(figsize=(10, 6))

    for s in scenarios_config:
        p = s['params']
        Q = p['U'] * p['B'] * p['depth']
        dilution = get_dilution(Q)

        # 计算参数
        k20 = 0.4
        k_T = k20 * (1.047 ** (p['T'] - 20))
        C0 = source_conc / dilution + background

        scenario_data = {
            'U': p['U'], 'B': p['B'], 'depth': p['depth'], 'T': p['T'],
            'k_T': k_T, 'C0': C0, 'background_cod': background,
            'source_concentration': source_conc
        }

        # 求解
        x, c = simulator.solve_simple_transport(scenario_data, L_max=7000, dx=50)

        label = f"{s['name']}\n{s['desc']} (稀释{dilution:.1f}倍)"
        plt.plot(x / 1000.0, c, linewidth=2, label=label)

    plt.xlabel('距离源头 (km)', fontsize=12)
    plt.ylabel('COD 浓度 (mg/L)', fontsize=12)
    plt.title('不同河道条件下的COD衰减曲线对比', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 7)
    plt.tight_layout()
    # plt.savefig('cod_curves.png') # 如果是在本地运行
    plt.show()


def plot_prediction_errors_from_dataset():
    """
    使用测试集数据绘制预测误差棒。
    """
    # 1. 准备数据
    try:
        X_test, eng_test, true_distances = prepare_test_data_from_dataset()
    except Exception as e:
        print(f"准备数据时出错: {e}")
        return

    # 2. 加载模型
    model = create_spill_adapted_model()
    model_path = 'best_spill_model.pth'

    if not os.path.exists(model_path):
        print(f"❌ 未找到模型文件 {model_path}，无法绘图")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    model.eval()

    # 3. 进行预测
    t_tensor = torch.FloatTensor(X_test)
    e_tensor = torch.FloatTensor(eng_test)

    with torch.no_grad():
        _, dist_pred, _ = model(t_tensor, e_tensor)
        predicted_distances = dist_pred.numpy().flatten()

    # 4. 分桶统计误差
    # 将真实距离分为若干个区间 (0-0.5, 0.5-1.0, ...)
    max_dist = np.ceil(true_distances.max())
    bins = np.arange(0, max_dist + 0.5, 0.5)

    bin_centers = []
    bin_means = []
    bin_stds = []

    print("正在计算误差分布...")
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        # 找到落在该距离区间的所有测试样本
        mask = (true_distances >= low) & (true_distances < high)

        if np.sum(mask) > 5:  # 至少有5个样本才统计，避免偶然性
            preds_in_bin = predicted_distances[mask]

            mean_pred = np.mean(preds_in_bin)
            std_pred = np.std(preds_in_bin)  # 标准差即为误差棒长度

            bin_centers.append((low + high) / 2)
            bin_means.append(mean_pred)
            bin_stds.append(std_pred)

    # 5. 绘图
    plt.figure(figsize=(10, 6))

    # 理想预测线 y=x
    plt.plot([0, max_dist], [0, max_dist], 'k--', alpha=0.5, label='理想预测 (y=x)')

    # 绘制误差棒
    # x轴: 真实距离中心, y轴: 预测距离均值, 误差棒: 预测标准差
    plt.errorbar(bin_centers, bin_means, yerr=bin_stds,
                 fmt='o-', capsize=5, capthick=1, ecolor='red', color='blue',
                 label='模型预测 (均值 ± 标准差)')

    mae = np.mean(np.abs(predicted_distances - true_distances))

    plt.xlabel('真实距离 (km)', fontsize=12)
    plt.ylabel('预测距离 (km)', fontsize=12)
    plt.title(f'模型溯源误差分析 (基于测试集数据)\n整体 MAE = {mae:.3f} km', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_dist)
    plt.ylim(0, max_dist)

    plt.tight_layout()
    # plt.savefig('prediction_errors.png') # 如果是在本地运行
    plt.show()


def main():
    print("=" * 60)
    print("开始生成可视化图表 (最终修正版)...")
    print("1. 物理规律图: 流量与稀释倍数严格关联")
    print("2. 误差分析图: 使用真实测试集数据")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    plot_cod_curves_strict()
    plot_prediction_errors_from_dataset()


if __name__ == "__main__":
    main()