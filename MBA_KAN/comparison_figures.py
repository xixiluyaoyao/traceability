"""
论文级对比图：Ours vs LSTM / Transformer
生成两张精美的对比图
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
import os

# 设置绘图风格
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
})


def load_results():
    """
    加载实验结果
    """
    # 尝试加载之前实验保存的结果
    if os.path.exists('experiment_results.npz'):
        data = np.load('experiment_results.npz', allow_pickle=True)
        return data['results'].item()
    
    # 如果没有，创建模拟数据结构（运行ablation_experiment.py后会有真实数据）
    print("⚠️ 未找到实验结果文件，请先运行 ablation_experiment.py")
    print("   或者将此代码放在实验代码后面一起运行")
    return None


def create_ours_vs_lstm_figure(ours_preds, ours_trues, lstm_preds, lstm_trues, save_path='ours_vs_lstm.pdf'):
    """
    创建 Ours vs LSTM 的对比图
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], 
                           hspace=0.28, wspace=0.25)
    
    # 计算指标
    ours_mae = np.mean(np.abs(ours_preds - ours_trues))
    lstm_mae = np.mean(np.abs(lstm_preds - lstm_trues))
    ours_errors = np.abs(ours_preds - ours_trues)
    lstm_errors = np.abs(lstm_preds - lstm_trues)
    
    # 颜色方案
    ours_color = '#E74C3C'  # 红色
    lstm_color = '#7F8C8D'  # 灰色
    
    # ========== (a) 散点图对比 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 采样点
    n_samples = min(600, len(ours_trues))
    idx = np.random.choice(len(ours_trues), n_samples, replace=False)
    
    # 先画LSTM（灰色背景）
    ax1.scatter(lstm_trues[idx], lstm_preds[idx], c=lstm_color, alpha=0.35, 
               s=30, label=f'LSTM (MAE={lstm_mae:.2f}km)', edgecolors='none')
    # 再画Ours（红色前景）
    ax1.scatter(ours_trues[idx], ours_preds[idx], c=ours_color, alpha=0.7, 
               s=35, label=f'Ours (MAE={ours_mae:.2f}km)', edgecolors='white', linewidth=0.3)
    
    # 理想线
    ax1.plot([0, 12], [0, 12], 'k--', lw=2, alpha=0.6, label='Ideal')
    
    ax1.set_xlabel('Ground Truth Distance (km)', fontweight='bold')
    ax1.set_ylabel('Predicted Distance (km)', fontweight='bold')
    ax1.set_title('(a) Prediction Scatter Comparison', fontweight='bold', pad=10)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 添加LSTM标注
    lstm_mean = np.mean(lstm_preds)
    ax1.axhline(y=lstm_mean, color=lstm_color, linestyle=':', alpha=0.8, lw=1.5)
    ax1.annotate('Higher prediction variance', xy=(9, lstm_mean), 
                xytext=(9.5, lstm_mean + 2),
                fontsize=9, color='#555', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    
    # ========== (b) 误差分布曲线 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 排序误差
    ours_sorted = np.sort(ours_errors)
    lstm_sorted = np.sort(lstm_errors)
    percentiles = np.linspace(0, 100, len(ours_sorted))
    
    ax2.fill_between(percentiles, lstm_sorted, ours_sorted, 
                     where=(lstm_sorted > ours_sorted),
                     color=ours_color, alpha=0.15, label='Our Advantage')
    ax2.plot(percentiles, lstm_sorted, color=lstm_color, linewidth=2.5, 
            linestyle='--', label='LSTM', alpha=0.8)
    ax2.plot(percentiles, ours_sorted, color=ours_color, linewidth=3, 
            label='Ours (PI-KAN-Mamba)')
    
    ax2.set_xlabel('Sample Percentile (%)', fontweight='bold')
    ax2.set_ylabel('Absolute Error (km)', fontweight='bold')
    ax2.set_title('(b) Sorted Error Distribution', fontweight='bold', pad=10)
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 6)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 标注关键点
    p90_ours = ours_sorted[int(len(ours_sorted)*0.9)]
    p90_lstm = lstm_sorted[int(len(lstm_sorted)*0.9)]
    ax2.annotate(f'90th: {p90_ours:.1f}km', xy=(90, p90_ours), 
                xytext=(75, p90_ours-0.8), fontsize=9, color=ours_color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=ours_color, lw=1.2))
    ax2.annotate(f'90th: {p90_lstm:.1f}km', xy=(90, p90_lstm), 
                xytext=(75, p90_lstm+0.5), fontsize=9, color=lstm_color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=lstm_color, lw=1.2))
    
    # ========== (c) 分距离段MAE柱状图 ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]
    bin_labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12']
    
    ours_bin_mae = []
    lstm_bin_mae = []
    for lo, hi in bins:
        mask = (ours_trues >= lo) & (ours_trues < hi)
        ours_bin_mae.append(np.mean(ours_errors[mask]) if mask.sum() > 0 else 0)
        lstm_bin_mae.append(np.mean(lstm_errors[mask]) if mask.sum() > 0 else 0)
    
    x = np.arange(len(bins))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, lstm_bin_mae, width, label='LSTM', 
                    color=lstm_color, alpha=0.7, edgecolor='black', linewidth=1)
    bars2 = ax3.bar(x + width/2, ours_bin_mae, width, label='Ours', 
                    color=ours_color, alpha=0.85, edgecolor='black', linewidth=1)
    
    # 在柱子上标注改进百分比
    for i, (l, o) in enumerate(zip(lstm_bin_mae, ours_bin_mae)):
        if l > 0:
            improve = (l - o) / l * 100
            ax3.annotate(f'-{improve:.0f}%', xy=(x[i] + width/2, o), 
                        xytext=(x[i] + width/2, o + 0.15),
                        ha='center', fontsize=8, color='#27ae60', fontweight='bold')
    
    ax3.set_xlabel('Distance Range (km)', fontweight='bold')
    ax3.set_ylabel('MAE (km)', fontweight='bold')
    ax3.set_title('(c) MAE by Distance Range', fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(bin_labels)
    ax3.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # ========== (d) 误差箱线图 ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 按距离分组的误差
    ours_grouped = [ours_errors[(ours_trues >= lo) & (ours_trues < hi)] 
                   for lo, hi in bins]
    lstm_grouped = [lstm_errors[(lstm_trues >= lo) & (lstm_trues < hi)] 
                   for lo, hi in bins]
    
    positions_lstm = np.arange(len(bins)) * 2
    positions_ours = positions_lstm + 0.7
    
    bp1 = ax4.boxplot(lstm_grouped, positions=positions_lstm, widths=0.5,
                      patch_artist=True, showfliers=False)
    bp2 = ax4.boxplot(ours_grouped, positions=positions_ours, widths=0.5,
                      patch_artist=True, showfliers=False)
    
    # 设置颜色
    for patch in bp1['boxes']:
        patch.set_facecolor(lstm_color)
        patch.set_alpha(0.6)
    for patch in bp2['boxes']:
        patch.set_facecolor(ours_color)
        patch.set_alpha(0.8)
    
    ax4.set_xlabel('Distance Range (km)', fontweight='bold')
    ax4.set_ylabel('Absolute Error (km)', fontweight='bold')
    ax4.set_title('(d) Error Distribution by Range', fontweight='bold', pad=10)
    ax4.set_xticks(positions_lstm + 0.35)
    ax4.set_xticklabels(bin_labels)
    ax4.legend([bp1['boxes'][0], bp2['boxes'][0]], ['LSTM', 'Ours'], 
              loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.set_ylim(0, 8)
    
    # 添加总标题
    fig.suptitle('PI-KAN-Mamba vs. LSTM Baseline', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f"📊 已保存: {save_path}")
    plt.show()


def create_ours_vs_transformer_figure(ours_preds, ours_trues, tf_preds, tf_trues, 
                                       save_path='ours_vs_transformer.pdf'):
    """
    创建 Ours vs Transformer 的对比图
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], 
                           hspace=0.28, wspace=0.25)
    
    # 计算指标
    ours_mae = np.mean(np.abs(ours_preds - ours_trues))
    tf_mae = np.mean(np.abs(tf_preds - tf_trues))
    ours_errors = np.abs(ours_preds - ours_trues)
    tf_errors = np.abs(tf_preds - tf_trues)
    
    # 颜色方案
    ours_color = '#E74C3C'  # 红色
    tf_color = '#9B59B6'    # 紫色
    
    # ========== (a) 散点图对比 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_samples = min(600, len(ours_trues))
    idx = np.random.choice(len(ours_trues), n_samples, replace=False)
    
    # 先画Transformer（紫色背景）
    ax1.scatter(tf_trues[idx], tf_preds[idx], c=tf_color, alpha=0.35, 
               s=30, label=f'Transformer (MAE={tf_mae:.2f}km)', edgecolors='none')
    # 再画Ours（红色前景）
    ax1.scatter(ours_trues[idx], ours_preds[idx], c=ours_color, alpha=0.7, 
               s=35, label=f'Ours (MAE={ours_mae:.2f}km)', edgecolors='white', linewidth=0.3)
    
    ax1.plot([0, 12], [0, 12], 'k--', lw=2, alpha=0.6, label='Ideal')
    
    ax1.set_xlabel('Ground Truth Distance (km)', fontweight='bold')
    ax1.set_ylabel('Predicted Distance (km)', fontweight='bold')
    ax1.set_title('(a) Prediction Scatter Comparison', fontweight='bold', pad=10)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 标注Transformer的发散特性
    ax1.annotate('Transformer:\nHigher variance\nat all ranges', 
                xy=(3, 7), xytext=(1, 9),
                fontsize=9, color='#8e44ad', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # ========== (b) 误差分布曲线 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    ours_sorted = np.sort(ours_errors)
    tf_sorted = np.sort(tf_errors)
    percentiles = np.linspace(0, 100, len(ours_sorted))
    
    ax2.fill_between(percentiles, tf_sorted, ours_sorted, 
                     where=(tf_sorted > ours_sorted),
                     color=ours_color, alpha=0.15, label='Our Advantage')
    ax2.plot(percentiles, tf_sorted, color=tf_color, linewidth=2.5, 
            linestyle='--', label='Transformer', alpha=0.8)
    ax2.plot(percentiles, ours_sorted, color=ours_color, linewidth=3, 
            label='Ours (PI-KAN-Mamba)')
    
    ax2.set_xlabel('Sample Percentile (%)', fontweight='bold')
    ax2.set_ylabel('Absolute Error (km)', fontweight='bold')
    ax2.set_title('(b) Sorted Error Distribution', fontweight='bold', pad=10)
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 8)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 标注差距
    mid_idx = len(percentiles) // 2
    gap = tf_sorted[mid_idx] - ours_sorted[mid_idx]
    ax2.annotate(f'Median gap:\n{gap:.1f}km', xy=(50, (tf_sorted[mid_idx] + ours_sorted[mid_idx])/2), 
                xytext=(60, tf_sorted[mid_idx] + 1), fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    
    # ========== (c) 分距离段MAE柱状图 ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]
    bin_labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12']
    
    ours_bin_mae = []
    tf_bin_mae = []
    for lo, hi in bins:
        mask = (ours_trues >= lo) & (ours_trues < hi)
        ours_bin_mae.append(np.mean(ours_errors[mask]) if mask.sum() > 0 else 0)
        tf_bin_mae.append(np.mean(tf_errors[mask]) if mask.sum() > 0 else 0)
    
    x = np.arange(len(bins))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, tf_bin_mae, width, label='Transformer', 
                    color=tf_color, alpha=0.7, edgecolor='black', linewidth=1)
    bars2 = ax3.bar(x + width/2, ours_bin_mae, width, label='Ours', 
                    color=ours_color, alpha=0.85, edgecolor='black', linewidth=1)
    
    for i, (t, o) in enumerate(zip(tf_bin_mae, ours_bin_mae)):
        if t > 0:
            improve = (t - o) / t * 100
            ax3.annotate(f'-{improve:.0f}%', xy=(x[i] + width/2, o), 
                        xytext=(x[i] + width/2, o + 0.2),
                        ha='center', fontsize=8, color='#27ae60', fontweight='bold')
    
    ax3.set_xlabel('Distance Range (km)', fontweight='bold')
    ax3.set_ylabel('MAE (km)', fontweight='bold')
    ax3.set_title('(c) MAE by Distance Range', fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(bin_labels)
    ax3.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # ========== (d) 性能雷达图 / 综合对比 ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 计算多个指标
    from sklearn.metrics import r2_score
    
    metrics = ['MAE\n(km)', 'R²', 'Near\n(<3km)', 'Mid\n(3-8km)', 'Far\n(>8km)']
    
    # Ours指标
    ours_r2 = r2_score(ours_trues, ours_preds)
    near_mask = ours_trues < 3
    mid_mask = (ours_trues >= 3) & (ours_trues < 8)
    far_mask = ours_trues >= 8
    ours_near = np.mean(ours_errors[near_mask])
    ours_mid = np.mean(ours_errors[mid_mask])
    ours_far = np.mean(ours_errors[far_mask])
    
    # Transformer指标
    tf_r2 = r2_score(tf_trues, tf_preds)
    tf_near = np.mean(tf_errors[near_mask])
    tf_mid = np.mean(tf_errors[mid_mask])
    tf_far = np.mean(tf_errors[far_mask])
    
    # 柱状图对比
    x = np.arange(5)
    width = 0.35
    
    # 注意：R²越高越好，其他越低越好，需要统一方向
    # 这里直接展示原始值
    ours_vals = [ours_mae, ours_r2, ours_near, ours_mid, ours_far]
    tf_vals = [tf_mae, tf_r2, tf_near, tf_mid, tf_far]
    
    ax4.barh(x - width/2, tf_vals, width, label='Transformer', 
            color=tf_color, alpha=0.7, edgecolor='black')
    ax4.barh(x + width/2, ours_vals, width, label='Ours', 
            color=ours_color, alpha=0.85, edgecolor='black')
    
    # 标注数值
    for i, (t, o) in enumerate(zip(tf_vals, ours_vals)):
        ax4.text(t + 0.1, x[i] - width/2, f'{t:.2f}', va='center', fontsize=9, color=tf_color)
        ax4.text(o + 0.1, x[i] + width/2, f'{o:.2f}', va='center', fontsize=9, color=ours_color, fontweight='bold')
    
    ax4.set_yticks(x)
    ax4.set_yticklabels(metrics)
    ax4.set_xlabel('Value', fontweight='bold')
    ax4.set_title('(d) Comprehensive Metrics Comparison', fontweight='bold', pad=10)
    ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax4.invert_yaxis()
    
    # 添加总标题
    fig.suptitle('PI-KAN-Mamba vs. Transformer Baseline', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f"📊 已保存: {save_path}")
    plt.show()


def main():
    """
    主函数：生成对比图
    需要在运行完 ablation_experiment.py 后运行，
    或者将结果数据传入
    """
    
    print("="*60)
    print("📊 生成对比图")
    print("="*60)
    
    # ========== 方法1: 从全局变量获取（如果和ablation一起运行）==========
    # 如果你把这段代码加到 ablation_experiment.py 的 main() 末尾，
    # 可以直接使用 results 变量
    
    # ========== 方法2: 手动输入你的实验结果 ==========
    # 根据你的实验结果填入（从ablation实验的输出复制）
    
    print("\n请确保已运行 ablation_experiment.py 并记录了结果")
    print("现在使用模拟数据生成示例图...")
    
    # 模拟数据（替换成你的真实数据）
    np.random.seed(42)
    n_samples = 2000
    
    # 真实距离（均匀分布）
    trues = np.random.uniform(0.3, 12, n_samples)
    
    # Ours预测（最好）
    ours_preds = trues + np.random.normal(0, 0.8, n_samples) * (1 + trues/15)
    ours_preds = np.clip(ours_preds, 0.1, 15)
    
    # LSTM预测（有mode collapse倾向）
    lstm_preds = trues * 0.6 + np.mean(trues) * 0.4 + np.random.normal(0, 1.5, n_samples)
    lstm_preds = np.clip(lstm_preds, 0.1, 15)
    
    # Transformer预测（方差大）
    tf_preds = trues + np.random.normal(0, 1.8, n_samples) * (1 + trues/10)
    tf_preds = np.clip(tf_preds, 0.1, 15)
    
    # 调整以匹配你的实验MAE
    # Ours: ~1.27, LSTM: ~2.12, Transformer: ~2.65
    
    print(f"\n模拟数据 MAE:")
    print(f"  Ours: {np.mean(np.abs(ours_preds - trues)):.2f} km")
    print(f"  LSTM: {np.mean(np.abs(lstm_preds - trues)):.2f} km")
    print(f"  Transformer: {np.mean(np.abs(tf_preds - trues)):.2f} km")
    
    # 生成图表
    print("\n" + "="*60)
    create_ours_vs_lstm_figure(ours_preds, trues, lstm_preds, trues)
    
    print("\n" + "="*60)
    create_ours_vs_transformer_figure(ours_preds, trues, tf_preds, trues)
    
    print("\n✅ 完成！生成了两张对比图：")
    print("   - ours_vs_lstm.pdf/png")
    print("   - ours_vs_transformer.pdf/png")


def generate_from_results(results):
    """
    从ablation实验的results字典生成图表
    在ablation_experiment.py的main()末尾调用：
    
    from comparison_figures import generate_from_results
    generate_from_results(results)
    """
    ours = results.get('Ours (Mamba+KAN)', {})
    lstm = results.get('Pure LSTM', {})
    tf = results.get('Pure Transformer', {})
    
    if ours and lstm:
        create_ours_vs_lstm_figure(
            ours['preds'], ours['trues'],
            lstm['preds'], lstm['trues']
        )
    
    if ours and tf:
        create_ours_vs_transformer_figure(
            ours['preds'], ours['trues'],
            tf['preds'], tf['trues']
        )


if __name__ == "__main__":
    main()