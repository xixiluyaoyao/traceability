import os
import pandas as pd
from tqdm import tqdm
import csv
import re


def build_enhanced_index(root_path, output_csv='snapshot_index_enhanced.csv'):
    """
    扫描水质监测快照文件，构建增强索引（包含水质参数）
    """
    # 预定义核心列的关键字
    targets = {
        'province': '省份',
        'basin': '流域',
        'station': '断面名称',
        'time': '监测时间',
        # 新增：水质参数
        'cod_mn': '高锰酸盐指数(mg/L)',
        'ph': 'pH(无量纲)',
        'do': '溶解氧(mg/L)',
        'temp': '水温(℃)',
        'water_quality': '水质类别'
    }

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f_out:
        writer = csv.writer(f_out)
        # 扩展表头
        writer.writerow([
            'file_path', 'full_time', 'province', 'basin', 'station_name',
            'cod_mn', 'ph', 'do', 'temp', 'water_quality'
        ])

        # 1. 递归获取所有文件
        all_files = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.lower().endswith('.csv'):
                    all_files.append(os.path.join(root, file))

        print(f"🚀 准备处理 {len(all_files)} 个快照文件...")

        success_count = 0
        skip_count = 0
        error_count = 0

        for file_path in tqdm(all_files):
            try:
                # 2. 从文件名中提取年份
                year_match = re.search(r'data-(\d{4})', os.path.basename(file_path))
                year_prefix = year_match.group(1) if year_match else "2024"

                # 3. 使用 encoding='utf-8-sig' 自动去除BOM
                df_head = pd.read_csv(file_path, encoding='utf-8-sig', nrows=1)
                cols = df_head.columns.tolist()

                # 映射实际列名
                mapping = {}
                for key, val in targets.items():
                    # 精确匹配或包含匹配
                    exact_match = [c for c in cols if c.strip() == val]
                    if exact_match:
                        mapping[key] = exact_match[0]
                    else:
                        # 退化到包含匹配
                        contain_match = [c for c in cols if val in str(c)]
                        if contain_match:
                            mapping[key] = contain_match[0]

                # 至少需要基本的4列（省份、流域、站点、时间）
                required_keys = ['province', 'basin', 'station', 'time']
                if not all(k in mapping for k in required_keys):
                    skip_count += 1
                    continue

                # 4. 读取所有可用的列
                df = pd.read_csv(file_path, encoding='utf-8-sig', usecols=list(mapping.values()))

                # 过滤掉站点情况为"维护"的行
                if '站点情况' in df.columns:
                    df = df[df['站点情况'] != '维护']

                # 5. 提取时间
                time_col = mapping['time']
                valid_times = df[time_col].dropna()
                valid_times = valid_times[~valid_times.astype(str).str.contains('--', na=False)]

                if valid_times.empty:
                    skip_count += 1
                    continue

                raw_time = valid_times.iloc[0]
                full_time = f"{year_prefix}-{raw_time.strip()}"

                # 6. 批量写入（包含水质参数）
                rows_written = 0
                for _, row in df.iterrows():
                    station_name = row[mapping['station']]

                    # 跳过无效站点
                    if pd.isna(station_name) or station_name.strip() == '':
                        continue

                    # 跳过时间无效的行
                    row_time = row[mapping['time']]
                    if pd.isna(row_time) or '--' in str(row_time):
                        continue

                    # 提取水质参数（如果存在）
                    def safe_get(key):
                        """安全获取参数值"""
                        if key not in mapping:
                            return ''
                        val = row[mapping[key]]
                        # 跳过无效值
                        if pd.isna(val) or str(val) in ['*', '--', '']:
                            return ''
                        return val

                    writer.writerow([
                        file_path,
                        full_time,
                        row[mapping['province']],
                        row[mapping['basin']],
                        station_name,
                        safe_get('cod_mn'),
                        safe_get('ph'),
                        safe_get('do'),
                        safe_get('temp'),
                        safe_get('water_quality')
                    ])
                    rows_written += 1

                if rows_written > 0:
                    success_count += 1

            except Exception as e:
                error_count += 1
                # print(f"❌ 处理失败: {os.path.basename(file_path)} - {e}")
                continue

    print(f"\n✨ 增强索引构建完成！")
    print(f"   成功处理: {success_count} 个文件")
    print(f"   跳过文件: {skip_count} 个")
    print(f"   错误文件: {error_count} 个")
    print(f"   结果已存入: {output_csv}")

    # 统计数据质量
    print(f"\n📊 数据质量统计:")
    df_index = pd.read_csv(output_csv)
    print(f"   总记录数: {len(df_index):,}")
    print(f"   唯一站点: {df_index['station_name'].nunique()}")

    # 统计各参数的有效率
    for col in ['cod_mn', 'ph', 'do', 'temp']:
        valid_count = df_index[col].replace('', pd.NA).notna().sum()
        rate = valid_count / len(df_index) * 100
        print(f"   {col} 有效率: {rate:.1f}% ({valid_count:,}/{len(df_index):,})")


def main():
    DATA_ROOT = "2023年4月-2025年4月"

    # 检查路径是否存在
    if not os.path.exists(DATA_ROOT):
        print(f"❌ 错误：找不到路径 {DATA_ROOT}")
        print(f"   当前工作目录: {os.getcwd()}")
        return

    build_enhanced_index(DATA_ROOT)


if __name__ == "__main__":
    main()