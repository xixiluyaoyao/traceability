import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os


def build_station_time_series(index_file='snapshot_index_enhanced.csv', output_file='station_time_series.pkl'):
    """
    将“时间切片”数据转换为“站点时序”数据 (修复版：增加鲁棒性)
    """
    if not os.path.exists(index_file):
        print(f"❌ 错误：找不到文件 {index_file}")
        return

    print("1. 读取索引文件 (启用低内存模式防止警告)...")
    # low_memory=False 解决 DtypeWarning
    # dtype=str 强制所有列先按字符串读取，防止混合类型报错
    df = pd.read_csv(index_file, low_memory=False, dtype=str)

    total_rows = len(df)
    print(f"   原始数据行数: {total_rows}")

    print("2. 清洗时间与数值格式...")

    # 【关键修复】 errors='coerce'：如果遇到 "2023-" 这种烂数据，自动变成 NaT 而不是报错崩溃
    df['full_time'] = pd.to_datetime(df['full_time'], errors='coerce')

    # 丢弃时间解析失败的行
    time_na_count = df['full_time'].isna().sum()
    if time_na_count > 0:
        print(f"   ⚠️ 发现 {time_na_count} 条时间格式错误的记录，已剔除 (如 '2023-' 等)。")
        df = df.dropna(subset=['full_time'])

    # 按时间排序
    df = df.sort_values(by='full_time')

    # 强制将数值列转换为数字，非数字变为 NaN
    numeric_cols = ['cod_mn', 'ph', 'do', 'temp']
    for col in numeric_cols:
        # errors='coerce'：遇到 "I类"、"-" 等非数字字符，转为 NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("3. 开始重组站点序列...")

    # 按站点分组
    grouped = df.groupby('station_name')

    valid_stations = {}
    min_length = 50  # 放宽限制，只要有50个点就算有效（约8天数据）

    # 定义需要的列
    cols = ['full_time', 'cod_mn', 'ph', 'do', 'temp']

    for station, group in tqdm(grouped, desc="Processing Stations"):
        # 再次按时间排序，确保万无一失
        group = group.sort_values('full_time')

        # 简单的插值填充 (处理 NaN)
        # 限制插值方向，防止开头结尾全是填充
        # 对于水质数据，线性插值比较合理
        group_numeric = group[numeric_cols].interpolate(method='linear', limit_direction='both', limit=3)

        # 将处理好的数值列放回 group
        for col in numeric_cols:
            group[col] = group_numeric[col]

        # 再次清洗：如果插值后还有 NaN (说明整列缺失太多)，则丢弃该站点或切片
        if group[numeric_cols].isna().any().any():
            # 简单策略：直接丢弃含有 NaN 的行，看看剩多少
            group = group.dropna(subset=numeric_cols)

        if len(group) < min_length:
            continue

        # 存入字典: [Timestamp, COD, pH, DO, Temp]
        # 注意：这里存的是 DataFrame 的 values，Timestamp 对象保留在第一列
        valid_stations[station] = group[cols].values

    print(f"4. 重组完成！有效站点数: {len(valid_stations)}")
    print(f"   正在保存至 {output_file} ...")

    with open(output_file, 'wb') as f:
        pickle.dump(valid_stations, f)

    print(f"✅ 完成。文件已保存，大小约 {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    build_station_time_series()