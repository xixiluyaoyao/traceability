import os
import pandas as pd
import numpy as np
import warnings

# 忽略 pandas 的一些切片警告
warnings.filterwarnings('ignore')


def is_clean_water(quality_str):
    """
    判断水质是否属于 I-III 类（干净背景）。
    适配全角/半角罗马数字 (Ⅰ, Ⅱ, Ⅲ, I, II, III) 及中文 (一, 二, 三)
    """
    if pd.isna(quality_str):
        return False

    q = str(quality_str).strip().upper()

    # 定义“干净”的白名单
    clean_tags = [
        'I', 'II', 'III',
        'Ⅰ', 'Ⅱ', 'Ⅲ',
        '1', '2', '3',
        '一', '二', '三'
    ]

    return q in clean_tags


def process_and_split_data(root_folder_path, cod_factor=3.0):
    """
    遍历文件夹，提取数据，并分割为 Clean / Dirty / All 三个数据集
    """
    clean_data_list = []
    dirty_data_list = []

    file_count = 0
    total_rows = 0

    # 定义 CSV 列名与模型所需特征的映射关系
    # CSV列名 -> 目标列名
    col_map = {
        '高锰酸盐指数(mg/L)': 'cod',  # 稍后乘以 3.0
        'pH(无量纲)': 'ph',
        '溶解氧(mg/L)': 'do',
        '水温(℃)': 'temp',
        '水质类别': 'quality'
    }

    print(f" 开始扫描文件夹: {root_folder_path}")
    print(f" 转换系数: COD_Cr ≈ {cod_factor} × 高锰酸盐指数")

    #递归遍历所有子文件夹
    for root, dirs, files in os.walk(root_folder_path):
        for file in files:
            # 只处理 csv 文件
            if not file.lower().endswith('.csv'):
                continue

            file_path = os.path.join(root, file)
            file_count += 1

            if file_count % 100 == 0:
                print(f"   已处理 {file_count} 个文件...", end='\r')

            try:
                # 尝试读取 CSV (处理常见的编码问题)
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='gb18030')
                    except:
                        continue  # 实在读不了就跳过

                # 检查关键列是否存在
                if '高锰酸盐指数(mg/L)' not in df.columns:
                    continue

                # 提取需要的列
                available_cols = [c for c in col_map.keys() if c in df.columns]
                df_sub = df[available_cols].rename(columns=col_map)

                # 数据清洗

                # 1. 处理 '*' 和其他非数值字符，强制转为 numeric
                numeric_cols = ['cod', 'ph', 'do', 'temp']
                for col in numeric_cols:
                    if col in df_sub.columns:
                        df_sub[col] = pd.to_numeric(df_sub[col], errors='coerce')

                # 2. 只有当关键参数 (COD, pH, DO) 都不为空时才保留
                # 温度如果没有，可以暂时给个默认值，或者也剔除
                if 'temp' not in df_sub.columns:
                    df_sub['temp'] = 20.0  # 默认补全

                # 删除包含 NaN 的行
                df_sub.dropna(subset=['cod', 'ph', 'do'], inplace=True)

                if df_sub.empty:
                    continue

                # 3. 核心转换: 高锰酸盐 -> COD
                df_sub['cod'] = df_sub['cod'] * cod_factor

                # 数据分流 (Clean vs Dirty)

                if 'quality' in df_sub.columns:
                    # 标记每一行是否干净
                    is_clean_mask = df_sub['quality'].apply(is_clean_water)

                    # 提取特征矩阵: [COD, pH, DO, Temp]
                    # 注意：必须保持这个顺序，与 generate2_1.py 里的读取顺序一致
                    feature_cols = ['cod', 'ph', 'do', 'temp']

                    # 分流
                    clean_part = df_sub.loc[is_clean_mask, feature_cols].values
                    dirty_part = df_sub.loc[~is_clean_mask, feature_cols].values

                    if len(clean_part) > 0:
                        clean_data_list.append(clean_part)
                    if len(dirty_part) > 0:
                        dirty_data_list.append(dirty_part)

                    total_rows += len(df_sub)
                else:
                    # 如果没有水质类别，默认算作 Clean 或者跳过，这里假设作为 Dirty 保险一点
                    # 或者可以根据 COD 值硬性判断
                    pass

            except Exception as e:
                # print(f"Error in {file}: {e}")
                pass

    # 3. 汇总与保存
    print(f"\n 扫描完成! 共处理 {file_count} 个文件。")

    if not clean_data_list and not dirty_data_list:
        print(" 未提取到任何有效数据，请检查路径或文件内容。")
        return

    # 合并 list -> numpy array
    clean_matrix = np.vstack(clean_data_list) if clean_data_list else np.empty((0, 4))
    dirty_matrix = np.vstack(dirty_data_list) if dirty_data_list else np.empty((0, 4))
    all_matrix = np.vstack([clean_matrix, dirty_matrix]) if (
                len(clean_matrix) > 0 or len(dirty_matrix) > 0) else np.empty((0, 4))

    # 打印统计信息
    print("-" * 50)
    print(f" 数据集统计报告:")
    print(f"   1. 全量数据 (All):   {all_matrix.shape[0]} 条 -> 用于【自监督预训练】")
    print(f"   2. 干净背景 (Clean): {clean_matrix.shape[0]} 条 -> 用于【训练溯源模型背景】 (I-III类)")
    print(f"   3. 污染数据 (Dirty): {dirty_matrix.shape[0]} 条 -> 用于【鲁棒性测试/困难样本】 (IV-劣V类)")
    print("-" * 50)

    if len(all_matrix) > 0:
        print(f" 真实参数分布 (用于校准模拟器):")
        print(f"   COD均值: {np.mean(all_matrix[:, 0]):.2f} (方差: {np.var(all_matrix[:, 0]):.2f})")
        print(f"   pH 均值: {np.mean(all_matrix[:, 1]):.2f}")
        print(f"   DO 均值: {np.mean(all_matrix[:, 2]):.2f}")

        # 计算协方差矩阵 (COD vs DO)
        if len(all_matrix) > 100:
            corr = np.corrcoef(all_matrix[:, 0], all_matrix[:, 2])[0, 1]
            print(f"   COD与DO相关系数: {corr:.4f} (负值代表符合物理规律)")
    print("-" * 50)

    # 保存文件
    np.save('clean_background.npy', clean_matrix)
    np.save('dirty_background.npy', dirty_matrix)
    np.save('all_data_matrix.npy', all_matrix)

    print(f" 文件已保存:")
    print(f"   - clean_background.npy")
    print(f"   - dirty_background.npy")
    print(f"   - all_data_matrix.npy")


if __name__ == "__main__":
    # 在这里修改文件夹路径
    DATA_ROOT = r"2023年4月-2025年4月"

    # 检查路径是否存在
    if not os.path.exists(DATA_ROOT):
        print(f"错误: 找不到路径 {DATA_ROOT}")
    else:
        process_and_split_data(DATA_ROOT)