import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import pickle
from tqdm import tqdm


class RobustUSVSimulator:
    """
    [科研版 v3.0] 面向无人船溯源的鲁棒性数据生成器
    - 集成真实水质背景 (Station Time Series)
    - 物理增强的ADE扩散模型 (Physics-Informed)
    - 模拟非理想传感器特性 (Sensor Noise & Failure)
    """

    def __init__(self, pkl_path='station_time_series.pkl', rng_seed=42):
        self.generated_data = []
        np.random.seed(rng_seed)
        self.pkl_path = pkl_path

        # 1. 加载真实背景库
        if os.path.exists(self.pkl_path):
            print(f"📚 正在加载真实背景库: {self.pkl_path} ...")
            with open(self.pkl_path, 'rb') as f:
                self.station_data = pickle.load(f)
            self.station_names = list(self.station_data.keys())
            print(f"✅ 加载成功！可用站点数: {len(self.station_names)}")
        else:
            print(f"❌ 警告: 未找到 {self.pkl_path}")
            print("   请先运行 '数据重组脚本.py' 生成背景库，否则将使用纯随机噪声背景。")
            self.station_names = []

    def _get_background_segment(self, duration_hours=24, dt_minutes=1):
        """
        从真实数据中提取一段背景，并插值到 1分钟/点 的高分辨率
        """
        # 如果没有真实数据，生成纯随机噪声 (Fallback)
        if not self.station_names:
            t_target = np.arange(0, duration_hours, dt_minutes / 60.0)
            # 模拟: COD=4.0, pH=7.5, DO=8.0, Temp=20.0
            base = np.array([4.0, 7.5, 8.0, 20.0])
            noise = np.random.normal(0, [0.5, 0.1, 0.5, 1.0], size=(len(t_target), 4))
            return t_target, base + noise

        # 尝试提取真实片段
        for _ in range(10):
            try:
                name = np.random.choice(self.station_names)
                data = self.station_data[name]  # [Time, COD, pH, DO, Temp]

                # 需要的点数 (原数据约15-30min一个点，需保证原数据够长)
                min_points = int(duration_hours * 2) + 5
                if len(data) < min_points: continue

                # 随机截取
                max_idx = len(data) - min_points
                start_idx = np.random.randint(0, max_idx)
                segment = data[start_idx: start_idx + min_points]

                # 时间处理与去重
                t_objs = pd.to_datetime(segment[:, 0])
                t_relative = (t_objs - t_objs[0]).total_seconds() / 3600.0
                values = segment[:, 1:].astype(float)

                # 简单清洗 NaN
                if np.isnan(values).any():
                    col_mean = np.nanmean(values, axis=0)
                    inds = np.where(np.isnan(values))
                    values[inds] = np.take(col_mean, inds[1])

                # 排序与去重 (防止 interp1d 报错)
                _, unique_indices = np.unique(t_relative, return_index=True)
                if len(unique_indices) < 5: continue

                t_relative = t_relative[unique_indices]
                values = values[unique_indices]

                # 线性插值到目标分辨率 (例如 1分钟)
                f_interp = interp1d(t_relative, values, axis=0, kind='linear', fill_value="extrapolate")
                t_target = np.arange(0, duration_hours, dt_minutes / 60.0)
                bg_interpolated = f_interp(t_target)

                # 物理约束: COD, DO 不能小于 0.1
                bg_interpolated[:, 0] = np.maximum(bg_interpolated[:, 0], 0.1)
                bg_interpolated[:, 2] = np.maximum(bg_interpolated[:, 2], 0.1)

                return t_target, bg_interpolated

            except Exception:
                continue

        # 如果多次失败，返回默认噪声
        return self._get_background_segment(duration_hours, dt_minutes)

    @staticmethod
    def _apply_sensor_imperfections(sequences, drop_prob=0.05, outlier_prob=0.01):
        """
        [关键] 模拟非理想传感器数据 (Robustness)
        """
        cod, ph, do, vel = sequences
        seq_len = len(cod)

        # 1. 模拟丢包 (Dropouts): 某段时间传感器读数为0或卡死
        if np.random.random() < 0.15:  # 15% 概率出现丢包
            drop_len = np.random.randint(2, 6)  # 丢 2-6 分钟
            if seq_len > drop_len:
                start = np.random.randint(0, seq_len - drop_len)
                # 模拟传感器输出归零
                cod[start: start + drop_len] = 0.01

                # 2. 模拟毛刺 (Outliers): 水草缠绕、气泡等
        # 随机选择 1% 的点变成异常值
        mask_outlier = np.random.rand(seq_len) < outlier_prob
        noise_spike = np.random.choice([20.0, -10.0], size=mask_outlier.sum())
        cod[mask_outlier] += noise_spike

        # 3. 模拟量程饱和 (Saturation): 假设仪器上限在 80-150 之间波动
        saturation_limit = np.random.uniform(80, 150)
        cod = np.clip(cod, a_min=None, a_max=saturation_limit)

        # 4. 模拟高斯白噪声 (仪器底噪)
        cod += np.random.normal(0, 0.1, seq_len)
        ph += np.random.normal(0, 0.02, seq_len)
        do += np.random.normal(0, 0.05, seq_len)

        return [cod, ph, do, vel]

    def solve_ade(self, t_hours, distance_m, mass_mg, U, Q, D, k):
        """一维 ADE 解析解"""
        A = Q / U
        if distance_m < 10: distance_m = 10.0

        # 转换单位
        t_seconds = t_hours * 3600.0
        t_seconds[t_seconds < 1.0] = 1.0  # 防止除零
        k_s = k / 86400.0

        term1 = mass_mg / (A * np.sqrt(4 * np.pi * D * t_seconds))
        exponent = -((distance_m - U * t_seconds) ** 2) / (4 * D * t_seconds) - k_s * t_seconds
        return term1 * np.exp(exponent)

    def generate_dataset(self, n_samples=5000, obs_window_min=30):
        """主生成循环 (修正版：强制高信噪比)"""
        self.generated_data = []
        print(f"🚀 开始生成数据集 (修正版 - 保证波形可见)...")

        pbar = tqdm(total=n_samples)

        while len(self.generated_data) < n_samples:
            # 1. 获取背景
            try:
                t_axis, bg_matrix = self._get_background_segment(duration_hours=24, dt_minutes=1)
            except:
                continue

            # 计算背景噪声水平 (标准差)
            bg_cod_base = np.mean(bg_matrix[:, 0])
            bg_noise_std = np.std(bg_matrix[:, 0])
            # 防止背景太干净导致除零，设置最小噪声基准
            bg_noise_std = max(bg_noise_std, 0.1)

            # 2. 随机水力参数
            U = np.random.uniform(0.3, 1.2)
            U_dynamic = U + np.sin(np.linspace(0, 10, len(t_axis))) * 0.05

            width = np.random.uniform(20, 100)
            depth = np.random.uniform(1.5, 5.0)
            Q = width * depth * U

            # 扩散系数 D (稍微减小一点上限，防止稀释太快)
            D = np.random.uniform(0.3, 1.0) * U * width

            # 3. 污染源设定 (大幅提升源强上限，保证远场能看到)
            # 旧范围: 5000~60000 -> 新范围: 20000~200000 mg/L
            source_conc = np.random.uniform(20000, 200000)
            # 旧体积: 10~80 -> 新体积: 50~150 m3
            vol_m3 = np.random.uniform(50, 150)
            mass_mg = source_conc * vol_m3 * 1000

            dist_km = np.random.uniform(2.0, 50.0)
            dist_m = dist_km * 1000

            # 4. 计算 ADE
            temp = np.mean(bg_matrix[:, 3])
            k_val = 0.2 * (1.047 ** (temp - 20))  # 稍微降低衰减系数 k，让污染物存活更久

            pollutant_curve = self.solve_ade(t_axis, dist_m, mass_mg, U, Q, D, k_val)

            peak_val = np.max(pollutant_curve)

            # =======================================================
            # 🔥 核心修正：信噪比检查 (SNR Check)
            # =======================================================
            # 只有当污染峰值 显著高于 背景噪声 (例如 5倍标准差) 时才保留
            # 并且峰值绝对浓度至少要有 2.0 mg/L (防止数值太小)
            if peak_val < 2.0 or peak_val < bg_noise_std * 5.0:
                continue  # 信号太弱，重开

            # 5. 确定有效观测区间
            # 阈值设为峰值的 5%，或者是背景噪声的 2倍，取大者
            # 这样保证我们在波形的“山脚下”也能截取到数据，而不是只在山顶
            threshold = max(peak_val * 0.05, bg_noise_std * 2.0)

            valid_indices = np.where(pollutant_curve > threshold)[0]
            if len(valid_indices) < obs_window_min:
                continue

            plume_start = valid_indices[0]
            plume_end = valid_indices[-1]
            plume_duration = plume_end - plume_start

            # 扩大采样范围，允许智能体看到从 "刚起步" 到 "拖尾结束"
            safe_start = max(0, plume_start - 20)
            safe_end = min(len(t_axis) - obs_window_min, plume_end + 60)

            if safe_end <= safe_start: continue

            agent_start_idx = np.random.randint(safe_start, safe_end)
            agent_end_idx = agent_start_idx + obs_window_min

            # 6. 数据合成
            cod_clean = bg_matrix[:, 0] + pollutant_curve

            obs_cod = cod_clean[agent_start_idx: agent_end_idx].copy()
            obs_ph = bg_matrix[agent_start_idx: agent_end_idx, 1].copy()
            obs_do = bg_matrix[agent_start_idx: agent_end_idx, 2].copy()
            obs_vel = U_dynamic[agent_start_idx: agent_end_idx].copy()

            # 7. 应用传感器非理想条件
            [obs_cod, obs_ph, obs_do, obs_vel] = self._apply_sensor_imperfections(
                [obs_cod, obs_ph, obs_do, obs_vel]
            )

            # 计算相对位置
            relative_pos = (agent_start_idx - plume_start) / (plume_duration + 1e-6)

            features = np.vstack([obs_cod, obs_ph, obs_do, obs_vel])

            labels = {
                'distance_km': dist_km,
                'source_mass_kg': mass_mg / 1e6,
                'relative_position': relative_pos,
                'river_width': width
            }

            self.generated_data.append({'features': features, 'labels': labels})
            pbar.update(1)

        pbar.close()
        return self.generated_data

    def save_dataset(self, filename='robust_usv_dataset'):
        """格式化并保存为 NPZ"""
        sequences = []
        targets_dist = []
        targets_mass = []
        targets_pos = []

        for sample in self.generated_data:
            sequences.append(sample['features'])
            l = sample['labels']
            targets_dist.append(l['distance_km'])
            targets_mass.append(l['source_mass_kg'])
            targets_pos.append(l['relative_position'])

        # 转换为 Numpy 数组
        # X: (N, 4, Window_Len)
        X = np.array(sequences)
        # y: 多任务标签
        y_dist = np.array(targets_dist)
        y_mass = np.array(targets_mass)
        y_pos = np.array(targets_pos)

        save_path = f'{filename}.npz'
        np.savez_compressed(
            save_path,
            X=X,
            y_dist=y_dist,
            y_mass=y_mass,
            y_pos=y_pos
        )
        print(f"\n✨ 数据集生成完毕！已保存至: {save_path}")
        print(f"   样本形状 (X): {X.shape}")
        print(f"   包含标签: 距离 (y_dist), 质量 (y_mass), 相对位置 (y_pos)")


def main():
    # 使用示例
    try:
        # 实例化模拟器
        # 确保当前目录下有 'station_time_series.pkl'，如果没有会自动使用随机噪声
        sim = RobustUSVSimulator(pkl_path='station_time_series.pkl')

        # 生成 5000 条样本用于测试 (正式跑建议 10万+)
        # 窗口设为 30 分钟 (无人船在该点停留 30 分钟)
        sim.generate_dataset(n_samples=150000, obs_window_min=30)

        # 保存
        sim.save_dataset('train_dataset_v3')

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()