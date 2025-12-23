import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import zoom
from train2_1_1 import prepare_data, create_data_loaders
from NN2_1 import create_spill_adapted_model

# ================= 配置学术风格 =================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 公式字体
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5


class GradCAM:
    """ (保持逻辑不变，只负责计算) """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, temporal, engineered):
        self.model.zero_grad()
        source_pred, dist_pred, bucket_logits = self.model(temporal, engineered)
        target = dist_pred
        target.backward(retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=2)
        activations = self.activations
        for i in range(activations.shape[1]):
            activations[:, i, :] *= pooled_gradients[:, i].unsqueeze(-1)

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap.cpu().detach().numpy(), dist_pred.item()


def visualize_gradcam_beautiful():
    # 1. 准备环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}...")

    data_dict = prepare_data()
    loaders = create_data_loaders(data_dict, batch_size=1)

    model = create_spill_adapted_model()
    # ⚠️ 请确保 best_spill_model.pth 文件存在
    model.load_state_dict(torch.load('best_spill_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # 目标层：最后一层卷积
    target_layer = model.encoder_layers[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    # 2. 筛选样本 (为了展示效果，我们选最具代表性的)
    # 我们希望涵盖: 近场(0), 中场(1), 远场(2), 超远(3)
    target_buckets = [0, 1, 2, 3]
    samples = {}

    print("Searching for representative samples...")
    for batch in loaders['test']:
        temporal, eng, s, d, b = [x.to(device) for x in batch]
        b_idx = b.item()

        if b_idx in target_buckets and b_idx not in samples:
            temporal.requires_grad = True
            heatmap, pred_dist = grad_cam(temporal, eng)

            # 数据拉伸与平滑
            input_len = 15
            zoom_factor = input_len / len(heatmap)
            heatmap_resized = zoom(heatmap, zoom_factor, order=1)
            # 归一化到 0-1
            heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (
                        heatmap_resized.max() - heatmap_resized.min() + 1e-8)

            samples[b_idx] = {
                'wave': temporal[0, :, 0].detach().cpu().numpy(),  # COD
                'heatmap': heatmap_resized,
                'true_dist': d.item(),
                'pred_dist': pred_dist
            }
        if len(samples) >= 4:
            break

    # 3. 🎨 开始绘制优美图表
    print("Plotting publication-ready figures...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.flatten()

    labels = [
        "Near Field (<0.5km)",
        "Mid Field (0.5-2.5km)",
        "Far Field (2.5-5.5km)",
        "Ultra-Far Field (>5.5km)"
    ]

    for i in range(4):
        if i not in samples: continue
        ax = axes[i]
        data = samples[i]
        wave = data['wave']
        heatmap = data['heatmap']

        # --- A. 绘制热力背景 (Imshow 方法) ---
        # extent=[x_min, x_max, y_min, y_max] 用于将热力图铺满背景
        # 使用 'Reds' 色谱，alpha=0.4 保持通透
        im = ax.imshow(
            heatmap[np.newaxis, :],
            extent=[0, len(wave) - 1, wave.min() - 0.2, wave.max() + 0.2],
            cmap='Reds',
            aspect='auto',
            alpha=0.5,
            vmin=0, vmax=1,
            interpolation='bilinear'  # 丝滑插值
        )

        # --- B. 绘制波形曲线 ---
        # 颜色：科研蓝 (#004488)，线宽：2.5，带一点透明度让网格透出来
        ax.plot(wave, color='#004488', linewidth=2.5, label='Signal (COD)', zorder=10)

        # --- C. 美化细节 ---
        # 标题：使用 LaTeX 加粗，显示真实与预测值
        title_str = f"Range: {labels[i]}\n" + \
                    r"$\mathbf{D_{true}}$: " + f"{data['true_dist']:.2f}km | " + \
                    r"$\mathbf{D_{pred}}$: " + f"{data['pred_dist']:.2f}km"
        ax.set_title(title_str, fontsize=14, loc='left')

        # 坐标轴标签
        if i >= 2:  # 只在最下面两张图显示 X 轴标签
            ax.set_xlabel("Time Step (Seq)", fontsize=12, fontweight='bold')

        if i % 2 == 0:  # 只在左边两张图显示 Y 轴标签
            ax.set_ylabel("Norm. Intensity ($\sigma$)", fontsize=12, fontweight='bold')

        # 简化边框 (只保留左边和下边)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        # 网格：灰色虚线，置于底层
        ax.grid(True, linestyle=':', alpha=0.6, color='gray')

        # 限制 Y 轴范围，留出一点空白
        ax.set_ylim(wave.min() - 0.2, wave.max() + 0.2)
        ax.set_xlim(0, 14)

    # 添加统一的 Colorbar (放在底部或右侧)
    # 这里我们放一个独立的 Colorbar 表示 "Model Attention"
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.06, shrink=0.6)
    cbar.set_label('Model Attention Weight (Grad-CAM)', fontsize=13, fontweight='bold')
    cbar.outline.set_visible(False)

    # 保存图片
    plt.savefig('paper_vis_beautiful.png', dpi=300, bbox_inches='tight')
    print("✅ 美化图表已保存: paper_vis_beautiful.png")


if __name__ == "__main__":
    visualize_gradcam_beautiful()