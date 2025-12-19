import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from scipy.ndimage import gaussian_filter1d

class TruckSpillSimulator:
    """
    基于真实水文背景的油罐车偷排模拟器 (Data-Driven Simulator)
    集成鲁棒性测试：随机背景波动因子
    """

    def __init__(self, rng_seed=42):
        self.scenarios = []
        self.generated_data = []
        np.random.seed(rng_seed)

        # 加载真实数据背景库
        # 优先读取清洗好的 clean_background.npy (I-III类水)
        bg_file = 'clean_background.npy'

        if os.path.exists(bg_file):
            self.real_backgrounds = np.load(bg_file)
            print(f"已加载真实背景库 ({bg_file}), 共 {len(self.real_backgrounds)} 条数据")
            print("   (将在生成时引入 0.7~1.5倍 的随机波动，以模拟转换系数的不确定性)")
            # 数据格式: [cod, ph, do, temp]
        else:
            print(f"未找到 {bg_file}，将降级使用纯随机模拟背景。")
            self.real_backgrounds = None

    def sample_spill_parameters(self, n_scenarios=1000):
        """采样基本参数 (融合真实数据 + 随机波动)"""
        scenarios = []
        for i in range(n_scenarios):
            # 河道几何参数 (随机生成)
            U = np.random.uniform(0.3, 0.8)  # 流速
            B = np.random.uniform(20, 60)  # 河宽
            depth = np.random.uniform(2.0, 4.0)  # 水深

            # 背景水质参数
            if self.real_backgrounds is not None:
                # 随机抽取一行真实数据
                idx = np.random.randint(len(self.real_backgrounds))
                # clean_background.npy 里已经是 (Mn * 3.0) 了
                bg_cod_base, bg_ph, bg_do, bg_temp = self.real_backgrounds[idx]

                # 引入随机波动因子
                # 模拟 CODcr/CODMn 经验公式的不确定性
                # 0.7 ~ 1.5 的波动意味着实际倍率在 2.1 ~ 4.5 之间变化
                # 这能强有力地证明模型对背景估算误差的鲁棒性
                uncertainty_factor = np.random.uniform(0.7, 1.5)
                background_cod = bg_cod_base * uncertainty_factor

                # 使用真实的温度 (如果有效)，否则随机
                T = bg_temp if bg_temp > 0 else np.random.uniform(15, 25)

                # 记录真实的 pH 和 DO 基准
                base_ph = bg_ph
                base_do = bg_do
            else:
                # 降级方案：纯随机
                background_cod = np.random.uniform(3.0, 6.0)
                T = np.random.uniform(15, 25)
                base_ph = 7.5
                base_do = 8.0

            # 计算降解系数 k (受温度 T 影响)
            k20 = np.random.uniform(0.2, 0.6)
            k_T = k20 * (1.047 ** (T - 20))

            # 偷排参数
            total_volume = np.random.uniform(30, 80)
            avg_concentration = np.random.uniform(12000, 25000)

            # 初始稀释
            dilution_factor = np.random.uniform(15, 40)
            C0 = avg_concentration / dilution_factor + background_cod

            scenarios.append({
                'U': U, 'B': B, 'depth': depth, 'T': T, 'k_T': k_T,
                'C0': C0, 'background_cod': background_cod,
                'source_concentration': avg_concentration,
                'base_ph': base_ph,
                'base_do': base_do
            })

        self.scenarios = scenarios
        return scenarios

    def solve_simple_transport(self, scenario, L_max=10000, dx=50):
        """
        传输模型：引入湍流和死水区效应
        """
        x = np.arange(0, L_max + dx, dx)
        U = scenario['U']
        k = scenario['k_T'] / 86400
        C0 = scenario['C0']
        background = scenario['background_cod']

        # 基础 ADR 解析解 (理想曲线)
        concentrations_ideal = np.zeros_like(x)
        for i, distance in enumerate(x):
            if distance == 0:
                concentrations_ideal[i] = C0
            else:
                travel_time = distance / U
                bio_decay = np.exp(-k * travel_time)
                # 距离衰减 (模拟横向/纵向扩散导致的稀释)
                # 让远场衰减稍微慢一点，模拟长距离输移的持续性
                distance_decay = np.exp(-distance / 2000)

                # 额外的经验衰减
                if distance > 3000:
                    extra_decay = np.exp(-(distance - 3000) / 3000)  # 放宽衰减
                else:
                    extra_decay = 1.0

                total_attenuation = bio_decay * distance_decay * extra_decay
                C_local = background + (C0 - background) * total_attenuation
                concentrations_ideal[i] = C_local


        #物理增强模块 (Physics Augmentation)
        # A. 模拟死水区 (Dead Zone / Transient Storage)
        # 物理现象：污染物进入死水区，导致峰值过后的浓度下降变慢（长拖尾）
        # 实现：找到峰值，人为抬高峰值后半段的曲线
        # 在一维稳态假设下，浓度随距离下降。我们模拟一种"滞后释放"
        concentrations_storage = concentrations_ideal.copy()
        # 简单的数值模拟：当前的浓度不仅仅取决于当前距离，还受到上游高浓度的"拖累"
        # 使用一个卷积平滑或指数加权移动平均来实现"拖尾"
        alpha = np.random.uniform(0.1, 0.3)  # 滞留系数
        for i in range(1, len(x)):
            # 下一个点的浓度 = 理想浓度 + 上一个点残留的一点点 (死水释放)
            concentrations_storage[i] = concentrations_ideal[i] * (1 - alpha) + concentrations_storage[i - 1] * alpha

        # 防止死水区效应导致浓度高于源头或过高
        concentrations_storage = np.maximum(concentrations_ideal, concentrations_storage)

        # B. 模拟湍流扰动 (Turbulence)
        # 物理现象：河道不是光滑管道，有涡旋导致浓度忽高忽低
        # 实现：有色噪声 (Colored Noise) - 平滑的随机波动

        # 生成白噪声
        white_noise = np.random.normal(0, 1, size=len(x))
        # 高斯滤波使其平滑，变成"波浪状"的湍流 (sigma控制涡旋的大小)
        eddy_scale = np.random.uniform(2, 5)  # 涡旋尺度
        turbulence = gaussian_filter1d(white_noise, sigma=eddy_scale)

        # 归一化并缩放湍流强度 ( 10% ~ 30% 的波动)
        turbulence_intensity = np.random.uniform(0.1, 0.3)
        # 湍流是乘性噪声 (浓度高的地方波动大)
        concentrations_final = concentrations_storage * (1 + turbulence * turbulence_intensity)

        # 3. 传感器测量噪声 (Measurement Noise)
        # 叠加高频白噪声
        noise_magnitude = 0.05 * (C0 - background) + 0.02 * background
        sensor_noise = np.random.normal(0, noise_magnitude, len(concentrations_final))

        concentrations_final = np.maximum(concentrations_final + sensor_noise, background * 0.5)

        return x, concentrations_final

    def select_segments(self, scenario, x_full, c_full):
        """测段选择 """
        background = scenario['background_cod']
        segments = []

        distance_ranges = [
            (80, 500), (600, 2000), (2500, 5000), (5500, 9000)
        ]

        for start_min, start_max in distance_ranges:
            for attempt in range(5):
                s_start = np.random.uniform(start_min, start_max)
                s_end = s_start + np.random.uniform(200, 400)

                if s_end > x_full[-1]: continue

                mask = (x_full >= s_start) & (x_full <= s_end)
                c_segment = c_full[mask]

                if len(c_segment) < 8: continue

                segment_max = np.max(c_segment)
                # 使用相对增量来判断，避免高背景导致的误判
                conc_increase = segment_max - background

                # 动态阈值：要求测段内至少有一定的污染增量 ( > 2 mg/L 或 背景的20%)
                if conc_increase < max(2.0, background * 0.2):
                    continue

                # 测量数据生成逻辑
                f_cod = interp1d(x_full, c_full)
                N_points = 15
                x_segment = np.linspace(s_start, s_end, N_points)
                c_true_points = f_cod(x_segment)

                measurements = []
                U_river = scenario['U']

                base_ph_val = scenario.get('base_ph', 7.5)
                base_do_val = scenario.get('base_do', 8.0)

                # 计算污染程度用于 pH/DO 联动
                max_pollute_conc = scenario['C0'] - background
                if max_pollute_conc < 0.1: max_pollute_conc = 0.1

                for i in range(N_points):
                    # COD 测量
                    cod_error = np.random.normal(0, 0.05 * c_true_points[i])
                    cod_measured = c_true_points[i] + cod_error

                    # 流速测量
                    vel_measured = U_river + np.random.normal(0, 0.05 * U_river)

                    # 污染联动比例
                    current_conc_inc = max(0, cod_measured - background)
                    cod_ratio = current_conc_inc / max_pollute_conc

                    # pH、DO 生成
                    # 基础波动 (真实背景也会动)
                    ph_noise = np.random.normal(0, 0.05)
                    do_noise = np.random.normal(0, 0.1)

                    #污染响应 (COD越高，pH/DO越低)
                    ph_drop = 0.5 * cod_ratio * np.random.uniform(0.8, 1.2)
                    do_drop = 2.5 * cod_ratio * np.random.uniform(0.8, 1.2)

                    ph_measured = base_ph_val + ph_noise - ph_drop
                    do_measured = base_do_val + do_noise - do_drop

                    measurements.append({
                        'cod': max(0.1, cod_measured),
                        'ph': max(4.0, min(10.0, ph_measured)),
                        'do': max(0.0, min(14.0, do_measured)),
                        'velocity': vel_measured
                    })

                # 缺失值
                missing_indicators = np.ones(N_points)
                if np.random.rand() < 0.6:
                    num_missing = np.random.randint(1, 4)
                    missing_indices = np.random.choice(N_points, num_missing, replace=False)
                    missing_indicators[missing_indices] = 0.0

                segment_center_distance = (s_start + s_end) / 2.0 / 1000.0

                segments.append({
                    'cod_sequence': [m['cod'] for m in measurements],
                    'ph_sequence': [m['ph'] for m in measurements],
                    'do_sequence': [m['do'] for m in measurements],
                    'velocity_sequence': [m['velocity'] for m in measurements],
                    'missing_indicators': missing_indicators.tolist(),
                    'center_distance_km': segment_center_distance,
                    'river_U': U_river,
                    'river_B': scenario['B'],
                    'river_T': scenario['T'],
                    'background_cod': background,
                })
                break
        return segments

    def generate_spill_dataset(self, n_samples=3000):
        """生成数据集"""
        scenarios = self.sample_spill_parameters(n_samples)
        generated_data = []

        for scenario in scenarios:
            if len(generated_data) >= n_samples: break
            try:
                x_full, c_full = self.solve_simple_transport(scenario)
                segments = self.select_segments(scenario, x_full, c_full)
                for segment_data in segments:
                    if len(generated_data) >= n_samples: break
                    labels = {
                        'source_concentration_mg_L': scenario['source_concentration'],
                        'distance_to_segment_km': segment_data['center_distance_km'],
                        'inlet_concentration_mg_L': scenario['C0'],
                        'background_concentration_mg_L': scenario['background_cod']
                    }
                    generated_data.append({'features': segment_data, 'labels': labels})
            except Exception:
                continue

        self.generated_data = generated_data
        print(f"生成有效样本: {len(generated_data)}")
        return generated_data

    def format_for_nn(self, target_length=15):
        """格式化数据 """
        sequences = []
        targets = []
        river_features = []

        for sample in self.generated_data:
            features = sample['features']
            labels = sample['labels']

            # 序列数据提取
            seqs = [
                features['cod_sequence'],
                features['ph_sequence'],
                features['do_sequence'],
                features['velocity_sequence'],
                features['missing_indicators']
            ]

            standardized_seqs = []
            for seq in seqs:
                if len(seq) == target_length:
                    std_seq = seq
                elif len(seq) < target_length:
                    x_old = np.linspace(0, 1, len(seq))
                    x_new = np.linspace(0, 1, target_length)
                    f = interp1d(x_old, seq, kind='linear', fill_value='extrapolate')
                    std_seq = f(x_new).tolist()
                else:
                    indices = np.linspace(0, len(seq) - 1, target_length, dtype=int)
                    std_seq = [seq[i] for i in indices]
                standardized_seqs.append(std_seq)

            sequences.append(standardized_seqs)
            river_features.append(
                [features['river_U'], features['river_B'], features['river_T'], features['background_cod']])
            targets.append([labels['source_concentration_mg_L'], labels['distance_to_segment_km'],
                            labels['inlet_concentration_mg_L'], labels['background_concentration_mg_L']])

        sequences_array = np.array(sequences).transpose(0, 2, 1)
        dataset = {
            'cod_sequences': sequences_array[:, :, 0],
            'ph_sequences': sequences_array[:, :, 1],
            'do_sequences': sequences_array[:, :, 2],
            'velocity_sequences': sequences_array[:, :, 3],
            'missing_masks': sequences_array[:, :, 4],
            'targets_raw': np.array(targets),
            'river_features': np.array(river_features)
        }
        return dataset

    def save_dataset(self, dataset, filename='truck_spill_dataset'):
        np.savez_compressed(f'{filename}.npz', **dataset)
        print(f"数据集已保存: {filename}.npz")

    def quick_validation(self):
        if len(self.generated_data) == 0: return
        distances = [s['labels']['distance_to_segment_km'] for s in self.generated_data]
        cod_sequences = [np.mean(s['features']['cod_sequence']) for s in self.generated_data]
        correlation = np.corrcoef(distances, cod_sequences)[0, 1]
        print(f"样本数: {len(self.generated_data)}")
        print(f"距离范围: {min(distances):.3f} - {max(distances):.3f} km")
        print(f"距离与COD相关性: {correlation:.3f}")


def main():
    simulator = TruckSpillSimulator()
    # 增加样本量以覆盖更多随机情况
    generated_data = simulator.generate_spill_dataset(n_samples=200000)

    if len(generated_data) < 1000:
        print("样本数量不足")
        return

    simulator.quick_validation()
    dataset = simulator.format_for_nn()
    simulator.save_dataset(dataset, 'truck_spill_dataset')
    print("完成！基于真实背景的增强型偷排数据集已生成。")


if __name__ == "__main__":
    main()