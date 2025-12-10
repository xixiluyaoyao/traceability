import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class TruckSpillSimulator:
    """简化的油罐车偷排模拟器"""

    def __init__(self, rng_seed=42): # 增加随机种子确保结果可复现
        self.scenarios = []
        self.generated_data = []
        np.random.seed(rng_seed) # 设定随机种子

    def sample_spill_parameters(self, n_scenarios=1000):
        """采样基本参数"""
        scenarios = []
        for i in range(n_scenarios):
            # 基本河道参数
            U = np.random.uniform(0.3, 0.8) # 流速 0.3-0.8 m/s
            B = np.random.uniform(20, 60) # 河宽 20-60 m
            depth = np.random.uniform(2.0, 4.0) # 水深 2-4 m

            # 衰减参数
            T = np.random.uniform(15, 25) # 水温 15-25°C
            k20 = np.random.uniform(0.2, 0.6) # 20°C时的降解系数
            k_T = k20 * (1.047 ** (T - 20)) # 温度修正公式（经验公式,水温每升高1°C,生化反应速率增加约4.7%）

            # 偷排参数 - 高浓度短时间
            total_volume = np.random.uniform(30, 80)  # 两车总体积（30-80 m³）
            avg_concentration = np.random.uniform(12000, 25000)  # 源头浓度 12000-25000 mg/L

            background_cod = np.random.uniform(3.0, 6.0) #背景COD值

            # 简单的初始稀释（ 油罐车偷排是高浓度短时间排放,初始就会被河水快速稀释。）
            Q_river = U * B * depth
            dilution_factor = np.random.uniform(15, 40)  # 15-40倍稀释
            C0 = avg_concentration / dilution_factor + background_cod

            scenarios.append({
                'U': U, 'B': B, 'depth': depth, 'T': T, 'k_T': k_T,
                'C0': C0, 'background_cod': background_cod,
                'source_concentration': avg_concentration
            })

        self.scenarios = scenarios
        return scenarios

    def solve_simple_transport(self, scenario, L_max=10000, dx=50): # L_max=10000, dx=50
        """简化的传输模型 - 重点是距离衰减 """
        x = np.arange(0, L_max + dx, dx)
        U = scenario['U']
        k = scenario['k_T'] / 86400
        C0 = scenario['C0']
        background = scenario['background_cod']

        concentrations = np.zeros_like(x)

        for i, distance in enumerate(x):
            if distance == 0:
                concentrations[i] = C0
            else:
                travel_time = distance / U # 污染物到达该点的时间

                # 核心：简单但有效的距离衰减
                # 1. 生化降解
                bio_decay = np.exp(-k * travel_time) # 1500m特征距离

                # 2. 关键的距离衰减 - 确保强负相关（模拟横向扩散和纵向混合）
                distance_decay = np.exp(-distance / 1500)  # 调整为1500m特征衰减距离，延长溯源范围

                # 3. 额外的远距离衰减 (调整衰减起始点和特征距离)，防止污染物影响过远(物理上合理)
                if distance > 3000: # 从3000m开始额外衰减
                    extra_decay = np.exp(-(distance - 3000) / 2000)
                else:
                    extra_decay = 1.0

                # 综合衰减
                total_attenuation = bio_decay * distance_decay * extra_decay
                C_local = background + (C0 - background) * total_attenuation
                concentrations[i] = C_local

        # 引入更大的噪声，模拟近场的高波动性（解决近桶准确率低的问题）
        noise_magnitude = 0.05 * (C0 - background) # 从0.01 调整到 0.05
        noise = np.random.normal(0, noise_magnitude, len(concentrations))
        concentrations = np.maximum(concentrations + noise, background)

        return x, concentrations

    def select_segments(self, scenario, x_full, c_full):
        """简单的测段选择 """
        background = scenario['background_cod']
        segments = []

        # 尝试在不同距离生成多个测段
        #用于模拟实际工作场景:工作人员可能在不同距离测量，让模型学会在各个距离范围内溯源
        distance_ranges = [
            (80, 500),      # Near (0.08 - 0.5 km)
            (600, 2000),    # Mid (0.6 - 2.0 km)
            (2500, 5000),   # Far (2.5 - 5.0 km)
            (5500, 9000)    # Ultra-Far (5.5 - 9.0 km) - 增加长距离样本
        ]

        for start_min, start_max in distance_ranges:
            for attempt in range(5):  # 增加尝试次数，以确保找到更多有效段
                s_start = np.random.uniform(start_min, start_max)
                s_end = s_start + np.random.uniform(200, 400)

                if s_end > x_full[-1]:
                    continue

                mask = (x_full >= s_start) & (x_full <= s_end)
                c_segment = c_full[mask]

                if len(c_segment) < 8:
                    continue

                segment_max = np.max(c_segment)
                segment_mean = np.mean(c_segment)
                max_bg_ratio = segment_max / background

                # 距离越远标准越宽松
                distance_km = s_start / 1000.0
                if distance_km <= 0.5: # 近场需要浓度>背景2倍
                    threshold = 2.0
                elif distance_km <= 2.5: # 中场1.4倍
                    threshold = 1.4
                elif distance_km <= 5.5: # 远场1.15倍
                    threshold = 1.15
                else:
                    threshold = 1.05 # 超远场只需要略高于背景值

                if max_bg_ratio >= threshold:
                    # 测量数据生成逻辑

                    f_cod = interp1d(x_full, c_full)
                    N_points = 15  # 假设测量点数量为15
                    x_segment = np.linspace(s_start, s_end, N_points)
                    c_true_points = f_cod(x_segment)

                    measurements = []

                    # 获取河流基本参数
                    U_river = scenario['U']
                    B_river = scenario['B']
                    T_river = scenario['T']

                    for i in range(N_points):
                        # 引入COD测量误差 (例如：±5% 误差)
                        cod_error = np.random.normal(0, 0.05 * c_true_points[i])
                        cod_measured = c_true_points[i] + cod_error

                        # 流速: 基于真实流速 U + 随机扰动
                        vel_measured = U_river + np.random.normal(0, 0.05 * U_river)

                        #COD高 → 有机物多 → 微生物耗氧 → DO下降
                        #有机酸积累 → pH下降
                        # pH/DO: 基于COD比值模拟污染影响
                        cod_ratio = (cod_measured - background) / (scenario['C0'] - background + 1e-6)
                        ph_base = 7.5 - 0.2 * np.random.rand()
                        ph_measured = ph_base - 0.5 * max(0, cod_ratio * np.random.rand())

                        do_sat = 9.0  # 饱和溶解氧
                        do_base = do_sat * np.random.uniform(0.8, 1.0)
                        do_measured = do_base - 3.0 * max(0, cod_ratio * np.random.rand())

                        measurements.append({
                            'cod': max(background, cod_measured),
                            'ph': max(6.0, ph_measured),
                            'do': max(4.0, do_measured),
                            'velocity': vel_measured
                        })

                    # 缺失数据模拟
                    missing_indicators = np.ones(N_points)
                    if np.random.rand() < 0.6:  # 60%概率丢失数据
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
                        # 加入河流特征作为额外的'特征'
                        'river_U': U_river,
                        'river_B': B_river,
                        'river_T': T_river,
                        'background_cod': background,
                    })
                    break

        return segments

    def generate_spill_dataset(self, n_samples=3000):
        """生成数据集"""
        scenarios = self.sample_spill_parameters(n_samples)
        generated_data = []

        for scenario in scenarios:
            if len(generated_data) >= n_samples:
                break

            try:
                x_full, c_full = self.solve_simple_transport(scenario)
                segments = self.select_segments(scenario, x_full, c_full)

                for segment_data in segments:
                    if len(generated_data) >= n_samples:
                        break

                    labels = {
                        'source_concentration_mg_L': scenario['source_concentration'],
                        'distance_to_segment_km': segment_data['center_distance_km'],
                        'inlet_concentration_mg_L': scenario['C0'],
                        'background_concentration_mg_L': scenario['background_cod']
                    }

                    sample = {
                        'features': segment_data,
                        'labels': labels
                    }

                    generated_data.append(sample)

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

            seqs = [
                features['cod_sequence'],
                features['ph_sequence'],
                features['do_sequence'],
                features['velocity_sequence'],
                features['missing_indicators']
            ]

            standardized_seqs = []
            # 插值/采样逻辑
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

            # 收集河流特征
            river_features.append([
                features['river_U'],
                features['river_B'],
                features['river_T'],
                features['background_cod']
            ])

            targets.append([
                labels['source_concentration_mg_L'],
                labels['distance_to_segment_km'],
                labels['inlet_concentration_mg_L'],
                labels['background_concentration_mg_L']
            ])

        sequences_array = np.array(sequences).transpose(0, 2, 1)
        targets_array = np.array(targets)
        river_features_array = np.array(river_features)

        #数据格式输出
        dataset = {
            'cod_sequences': sequences_array[:, :, 0],# 15个COD测量点
            'ph_sequences': sequences_array[:, :, 1],
            'do_sequences': sequences_array[:, :, 2],
            'velocity_sequences': sequences_array[:, :, 3],
            'missing_masks': sequences_array[:, :, 4],# 1=有效, 0=缺失
            'targets_raw': targets_array,# [源浓度, 距离, 入流浓度, 背景浓度]
            'river_features': river_features_array  # [流速U, 河宽B, 水温T, 背景COD]
        }

        return dataset

    def save_dataset(self, dataset, filename='truck_spill_dataset'):
        """保存数据集"""
        np.savez_compressed(f'{filename}.npz', **dataset)
        print(f"数据集已保存: {filename}.npz")

    def quick_validation(self):
        """验证数据质量"""
        if len(self.generated_data) == 0:
            return

        distances = [s['labels']['distance_to_segment_km'] for s in self.generated_data]
        cod_sequences = [np.mean(s['features']['cod_sequence']) for s in self.generated_data]

        correlation = np.corrcoef(distances, cod_sequences)[0, 1]

        near_count = sum(1 for d in distances if d <= 0.3)
        mid_count = sum(1 for d in distances if 0.3 < d <= 1.0)
        far_count = sum(1 for d in distances if d > 1.0)

        print(f"样本数: {len(self.generated_data)}")
        print(f"距离范围: {min(distances):.3f} - {max(distances):.3f} km")
        print(f"距离与COD相关性: {correlation:.3f} (目标: <-0.5)")
        print(f"3桶分布: Near={near_count}, Mid={mid_count}, Far={far_count}")

        if correlation <= -0.5:
            print("✅ 距离与COD相关性良好")
        else:
            print("⚠️ 距离与COD相关性过弱")


def main():
    """主函数"""
    simulator = TruckSpillSimulator()

    generated_data = simulator.generate_spill_dataset(n_samples=3000)

    if len(generated_data) < 1000:
        print("样本数量不足")
        return

    simulator.quick_validation()
    dataset = simulator.format_for_nn()
    simulator.save_dataset(dataset, 'truck_spill_dataset')

    print("完成！油罐车偷排数据集")


if __name__ == "__main__":
    main()