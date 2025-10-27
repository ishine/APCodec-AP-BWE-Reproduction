import json
from collections import Counter

path = "/mnt/nvme_share/srt30/APCodec-Reproduction/exp_do/train_data.json"
cnt = Counter()

try:
    with open(path) as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"错误：在路径 {path} 未找到文件。")
    print("请确保文件路径正确后再运行。")
    data = []
except json.JSONDecodeError:
    print(f"错误：文件 {path} 不是一个有效的 JSON 文件。")
    data = []

if data:
    for item in data:
        if "token" in item and isinstance(item["token"], str):
            for t in item["token"].split():
                try:
                    cnt[int(t)] += 1
                except ValueError:
                    pass
        else:
            pass

    if not cnt:
        print("文件中未找到有效的 token。")
    else:
        total = sum(cnt.values())
        most_common_512 = cnt.most_common(512) 

        print("总 token 数:", total)
        print("top1024 常见 token:", most_common_512) 
        print("\n--- Top 1024 Token 分布统计 ---")

        count_range_1 = 0
        count_range_2 = 0
        count_range_3 = 0

        for token, freq in most_common_512:
            if token < 1024:
                count_range_1 += 1
            elif 1024 <= token < 2048:
                count_range_2 += 1
            elif 2048 <= token < 3072:
                count_range_3 += 1

        print(f"在频率最高的前 {len(most_common_512)} 个 token 中：")
        print(f"  - 小于 1024 的 token 数量: {count_range_1}")
        print(f"  - 在 1024 到 2047 之间 (含) 的 token 数量: {count_range_2}")
        print(f"  - 在 2048 到 3071 之间 (含) 的 token 数量: {count_range_3}")

else:
    if not 'data' in locals():
        pass 
    else:
        print("未从文件中加载任何数据或数据为空。")
