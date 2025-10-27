import json
import numpy as np
from tqdm import tqdm

def analyze_json_dataset(dataset_path, length_threshold=940):
    """
    分析指令微调JSON数据集，统计输入字段的token数量，并计算超长样本的占比。

    参数:
    dataset_path (str): JSON数据集文件的路径。
    length_threshold (int): 定义“超长样本”的token数量阈值。
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{dataset_path}'")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 '{dataset_path}' 不是一个有效的JSON文件。")
        return

    if not data or not isinstance(data, list):
        print("错误：JSON文件内容为空或格式不正确（应为一个列表）。")
        return

    token_counts = []
    print(f"正在分析数据集中的 {len(data)} 条目...")

    for entry in tqdm(data, desc="正在处理条目"):
        if "output" in entry and isinstance(entry["output"], str):
            # 通过空格分割来计算token的数量
            tokens = entry["output"].split()
            token_counts.append(len(tokens))
        else:
            print(f"警告：跳过一个格式不正确的条目: {entry}")

    if not token_counts:
        print("数据集中没有找到有效的'output'字段进行分析。")
        return

    # 将列表转换为numpy数组以便进行高效计算
    token_counts_np = np.array(token_counts)

    # --- 核心统计计算 ---
    total_samples = len(token_counts_np)
    max_tokens = np.max(token_counts_np)
    min_tokens = np.min(token_counts_np)
    avg_tokens = np.mean(token_counts_np)
    
    # --- 新增的统计逻辑 ---
    # 计算超过阈值的样本数量
    count_over_threshold = np.sum(token_counts_np > length_threshold)
    
    # 计算占比
    percentage_over_threshold = (count_over_threshold / total_samples) * 100 if total_samples > 0 else 0

    print("\n--- 数据集分析结果 ---")
    print(f"总样本数: {total_samples}")
    print(f"输入token最大长度: {max_tokens}")
    print(f"输入token最小长度: {min_tokens}")
    print(f"输入token平均长度: {avg_tokens:.2f}")
    print("----------------------")
    print(f"长度超过 {length_threshold} 个token的样本数: {count_over_threshold}")
    print(f"占比: {percentage_over_threshold:.2f}%")
    print("----------------------")


if __name__ == '__main__':
    # 请确保这个路径是您实际的JSON文件路径
    json_file_path = '/mnt/nvme_share/srt30/APCodec-Reproduction/exp_gem3/valid_data.json'
    
    # 你可以在这里修改你关心的长度阈值
    threshold = 256
    
    analyze_json_dataset(json_file_path, length_threshold=threshold)