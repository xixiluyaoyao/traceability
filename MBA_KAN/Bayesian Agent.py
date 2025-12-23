import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

# 引入你的模块 (请确保文件名对应)
from train_moe_micro import PI_Attentive_MoE  # 模型定义
from generate_micro_data import UltimateMicroSimulator  # 物理引擎

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RealBayesianAgent:
    def __init__(self, river_len=12.0):
        self.river_len = river_len
        self.grid = np.linspace(0, river_len, 400)
        self.belief = np.ones_like(self.grid) / len(self.grid)

    def update(self, pred_mu, pred_sigma, current_pos):
        """
        pred_mu: 线性距离 (km)
        pred_sigma: 线性标准差 (km)
        """
        # 1. 转换坐标: 假设源头在上游 (Source = Current - Dist ? 或者是 Current + Dist，取决于你坐标系定义)
        # 这里假设 0是上游源头，12是下游。船在 current_pos。
        # 所以 Pred Source = Current_Pos - Pred_Dist (向上游找)
        # 或者如果 0是下游，12是上游，那就是 + Pred_Dist。
        # 让我们沿用之前的逻辑：0=上游。
        # 如果船在 10km，预测源头在 8.5km 远，那源头坐标 = 10 - 8.5 = 1.5km
        pred_loc = current_pos - pred_mu

        # 2. 计算似然 (Likelihood)
        likelihood = norm.pdf(self.grid, loc=pred_loc, scale=pred_sigma)

        # 3. 贝叶斯更新
        self.belief = self.belief * likelihood
        self.belief /= (np.sum(self.belief) + 1e-12)  # 归一化

        return self.grid[np.argmax(self.belief)]


class VirtualRiverEnvironment:
    def __init__(self, true_source_loc, mass_kg):
        self.true_source = true_source_loc
        self.mass = mass_kg
        self.sim = UltimateMicroSimulator()

        # 物理参数
        self.U = 0.4
        self.width = 15.0
        self.depth = 1.2
        self.Q = self.width * self.depth * self.U
        self.D = 0.15 * self.U * self.width

        self.t_axis_h, self.bg_matrix = self.sim._get_background_segment()
        self.t_axis_s = self.t_axis_h * 3600.0

    def measure_at(self, boat_pos_km):
        # 计算船离源头多远
        dist_km = np.abs(boat_pos_km - self.true_source)
        dist_m = dist_km * 1000.0

        # 1. 生成波形
        mass_mg = self.mass * 1e6
        # 为了简单，假设船速 v_boat=0 (悬停测量)
        v_boat = 0.0

        # 找到波峰时刻
        temp_curve = self.sim.solve_ade_moving(self.t_axis_s, dist_m, v_boat, mass_mg, self.U, self.Q, self.D, k=0.1)
        peak_idx = np.argmax(temp_curve)

        # 采样 30 个点
        start = max(0, peak_idx - 15)
        end = start + 30
        if end > len(self.t_axis_s): start = 0; end = 30  # 兜底

        t_win = self.t_axis_s[start:end]
        pollutant = self.sim.solve_ade_moving(t_win, dist_m, v_boat, mass_mg, self.U, self.Q, self.D, k=0.1)

        # 2. 合成特征 (COD, pH, DO)
        bg = self.bg_matrix[start:end].copy()
        bg[:, 0] += pollutant

        # 3. 预处理 (必须和训练时完全一致!)
        cod_norm = np.log1p(np.maximum(bg[:, 0], 0)) / 12.0
        ph_norm = (bg[:, 1] - 7.0) / 2.0
        do_norm = (bg[:, 2] - 8.0) / 4.0

        x_img = np.vstack([cod_norm, ph_norm, do_norm])
        x_tensor = torch.FloatTensor(x_img).unsqueeze(0).to(device)  # [1, 3, 30]

        # 4. 统计特征
        from scipy.stats import kurtosis, skew
        k_val = np.tanh(kurtosis(bg[:, 0]) / 10.0)
        s_val = np.tanh(skew(bg[:, 0]) / 5.0)
        log_max = np.log1p(np.max(bg[:, 0])) / 12.0
        log_std = np.log1p(np.std(bg[:, 0])) / 8.0
        v_rel = self.U - v_boat

        stats_vec = np.array([self.U, v_boat, v_rel, k_val, s_val, log_max, log_std,self.width / 20.0,
            self.depth / 2.0])
        stats_tensor = torch.FloatTensor(stats_vec).unsqueeze(0).to(device)

        return x_tensor, stats_tensor


def run_real_mission():
    print("🤖 加载模型权重: agent_model_final.pth ...")
    model = PI_Attentive_MoE().to(device)
    try:
        model.load_state_dict(torch.load('agent_model_final.pth', map_location=device))
    except:
        print("❌ 错误：找不到 agent_model_final.pth，请先运行训练脚本！")
        return
    model.eval()

    # 设定真实环境：源头在 1.5km，源强 60kg (随机值)
    TRUE_SOURCE = 1.5
    TRUE_MASS = 60.0
    print(f"🌊 环境设定: 真实源头={TRUE_SOURCE}km, 真实源强={TRUE_MASS}kg")

    env = VirtualRiverEnvironment(TRUE_SOURCE, TRUE_MASS)
    agent = RealBayesianAgent(river_len=12.0)

    # 路径规划
    path = [10.0, 5.0, 2.5]

    plt.figure(figsize=(10, 8))

    for i, pos in enumerate(path):
        print(f"\n--- Step {i + 1}: Boat @ {pos}km ---")

        # 1. 获取真实测量数据
        x, stats = env.measure_at(pos)

        # 2. 模型推理
        with torch.no_grad():
            pred = model(x, stats)
            # pred: [log_dist_mu, log_dist_logvar, log_mass]
            log_dist_mu = pred[0, 0].item()
            log_dist_logvar = pred[0, 1].item()

            # 还原
            dist_mu = 10 ** log_dist_mu
            # 注意: log_var 是 log(sigma^2) -> sigma = sqrt(exp(log_var))
            # 这里的 sigma 是 Log 空间的 sigma。
            # 为了贝叶斯更新，我们需要线性空间的近似 Sigma。
            # 近似: Sigma_linear ≈ Dist_linear * ln(10) * Sigma_log
            sigma_log = np.sqrt(np.exp(log_dist_logvar))
            sigma_linear = dist_mu * np.log(10) * sigma_log

            # 加上一个底噪，防止 sigma 太小导致数值问题
            sigma_linear = max(sigma_linear, 0.1)

        print(f"🧠 模型预测: 距离 {dist_mu:.2f}km ± {sigma_linear:.2f}km")

        # 3. 贝叶斯更新
        guess = agent.update(dist_mu, sigma_linear, pos)

        # 4. 画图
        plt.subplot(3, 1, i + 1)
        plt.plot(agent.grid, agent.belief, 'b-', lw=2, label='Belief')
        plt.axvline(TRUE_SOURCE, color='r', linestyle='--', label='True Source')
        plt.scatter([pos], [0], color='k', marker='^', s=100, label='Boat')
        plt.title(f"Step {i + 1} @ {pos}km | Pred: {dist_mu:.2f}km | Agent Guess: {guess:.2f}km")
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_real_mission()