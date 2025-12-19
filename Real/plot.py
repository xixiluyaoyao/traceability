import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import warnings

# 忽略字体警告
warnings.filterwarnings("ignore")

# 引用模型代码
from NN2_1 import create_spill_adapted_model
from train2_1_1 import prepare_data, create_data_loaders

def configure_plotting_style():
    plt.style.use('seaborn-v0_8-whitegrid')

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    # 字体大小设置
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.linewidth'] = 1.5

def get_test_predictions():
    """获取预测结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" 正在加载模型 (忽略下方的'从头训练'提示，稍后会覆盖权重)...")

    # 1. 准备数据
    data_dict = prepare_data()
    loaders = create_data_loaders(data_dict, batch_size=512)
    test_loader = loaders['test']

    # 加载模型
    # 这里会打印 "Attention 层将从头训练"，这是正常的初始化日志
    model = create_spill_adapted_model()

    #加载训练好的最佳权重 (覆盖掉上面的初始化权重)
    model_path = 'best_spill_model.pth'
    print(f" 正在加载最佳权重: {model_path} ...")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.to(device)
    model.eval()

    #推理
    dist_preds, dist_trues = [], []
    print(" 正在对测试集进行推理...")
    with torch.no_grad():
        for batch in test_loader:
            temporal, engineered, source_true, dist_true, bucket_true = [x.to(device) for x in batch]
            _, dist_pred, _ = model(temporal, engineered)
            dist_preds.extend(dist_pred.cpu().numpy().flatten())
            dist_trues.extend(dist_true.cpu().numpy().flatten())

    return np.array(dist_trues), np.array(dist_preds)


def plot_combined_error_trend(y_true, y_pred):
    """
    Figure 1: 预测一致性与不确定性演化
    """
    print(" 正在绘制 Figure 1 (预测散点与误差趋势)...")

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # 分桶计算误差棒
    bins = np.linspace(0, 10, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    df = pd.DataFrame({'True': y_true, 'Pred': y_pred})
    df['Bin'] = pd.cut(df['True'], bins=bins, labels=bin_centers)

    stats = df.groupby('Bin', observed=True)['Pred'].agg(['mean', 'std'])
    valid_centers = stats.index.astype(float)
    valid_means = stats['mean']
    valid_stds = stats['std']

    fig, ax = plt.subplots(figsize=(10, 8))

    # 背景散点
    ax.scatter(y_true, y_pred, alpha=0.15, s=10, color='#8c9eff', label='测试样本', rasterized=True)

    # 对角线
    ax.plot([0, 10], [0, 10], 'k--', linewidth=1.5, alpha=0.6, label='理想预测线')

    # 误差棒折线
    ax.errorbar(valid_centers, valid_means, yerr=valid_stds,
                fmt='o-', color='#d32f2f', ecolor='#ef5350',
                elinewidth=2, capsize=4, markersize=6, linewidth=2,
                label='预测均值 ± 1倍标准差')

    # 标签
    ax.set_xlabel('真实距离 (km)', fontsize=16, fontweight='bold')
    ax.set_ylabel('预测距离 (km)', fontsize=16, fontweight='bold')
    ax.set_title(f'模型预测一致性与不确定性演化\n($R^2={r2:.3f}$, $MAE={mae:.3f}$ km)', fontsize=18)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.9)

    # 区域标注
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=5.5, color='gray', linestyle=':', alpha=0.5)

    trans = ax.get_xaxis_transform()
    ax.text(0.25, 0.02, '近场', transform=trans, ha='center', color='gray', fontsize=12, fontweight='bold')
    ax.text(1.5, 0.02, '中场区域', transform=trans, ha='center', color='gray', fontsize=12, fontweight='bold')
    ax.text(4.0, 0.02, '远场区域', transform=trans, ha='center', color='gray', fontsize=12, fontweight='bold')
    ax.text(7.75, 0.02, '超远场', transform=trans, ha='center', color='gray', fontsize=12, fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig('Figure1_Trend_CN.png', dpi=300)
    print(" Figure 1 保存成功: Figure1_Trend.png")


def plot_ablation_study():
    """
    Figure 2: 消融实验对比
    """
    print(" 正在绘制 Figure 2 (消融实验)...")

    # 数据
    models = ['基准模型\n(ResNet, 5w数据)', '物理增强\n(深化方程, 5w数据)', '本文模型 (SOTA)\n(Attention + 20w数据)']

    # MAE 数据
    mae_scores = [0.6189, 0.4984, 0.4622]
    mid_scores = [0.2722, 0.3280, 0.2590]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width / 2, mae_scores, width, label='整体平均误差 (MAE)', color='#a8dadc')
    rects2 = ax.bar(x + width / 2, mid_scores, width, label='中场区域误差 (MAE)', color='#1d3557')

    ax.set_ylabel('平均绝对误差 (km)', fontsize=14, fontweight='bold')
    ax.set_title('消融实验：物理模型深化与数据规模的影响', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')

    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=11)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=11, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('Figure2_Ablation_CN.png', dpi=300)
    print(" Figure 2 保存成功: Figure2_Ablation.png")


def main():
    # 配置字体
    configure_plotting_style()

    try:
        y_true, y_pred = get_test_predictions()
        plot_combined_error_trend(y_true, y_pred)
    except Exception as e:
        print(f"无法推理: {e}")
        print("提示: 请确保目录下有 best_spill_model.pth 和 truck_spill_dataset.npz")

    plot_ablation_study()
    print("\n 所有中文图表绘制完成！")


if __name__ == "__main__":
    main()