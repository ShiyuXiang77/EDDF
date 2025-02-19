#对于文件夹下的json文件，读取scores列表的第一个值，如果第一个大于0.51，num（初始化为0）+1，最后计算num/scores的个数的值，每一个json元素作为csv文件的一行，记录文件名，大于0.51的个数，num/scores(ASR)的值
import os
import json
import csv

def compute_asr(input_folder, output_csv):
    """计算每个 JSON 文件的 ASR，并计算整个文件夹的总 ASR，最终写入 CSV"""
    if not os.path.exists(input_folder):
        print(f"Error: 文件夹 {input_folder} 不存在")
        return

    results = []
    total_num = 0  # 累计所有文件的大于0.6的个数
    total_scores = 0  # 累计所有 scores 的数量

    for filename in os.listdir(input_folder):
        if not filename.endswith(".json"):
            continue  # 只处理 JSON 文件

        file_path = os.path.join(input_folder, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error reading {filename}: {e}")
            continue

        # 计算当前文件的 ASR
        file_num = 0
        file_total_scores = 0

        for item in data:
            scores = item.get("scores", [])
            if isinstance(scores, list) and len(scores) > 0:
                file_total_scores += 1
                if scores[0] > 0.6:
                    file_num += 1

        # 计算单个文件的 ASR
        file_asr = round(file_num / file_total_scores, 4) if file_total_scores > 0 else 0.0
        results.append([filename, file_num, file_total_scores, file_asr])

        # 更新总计数
        total_num += file_num
        total_scores += file_total_scores

    # 计算整个文件夹的 ASR
    final_asr = round(total_num / total_scores, 4) if total_scores > 0 else 0.0

    # 写入 CSV 文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["文件名", "大于0.5的个数", "总数", "ASR"])
        writer.writerows(results)  # 写入单个文件的 ASR

        # 额外写入整文件夹的 ASR 结果
        writer.writerow(["总计", total_num, total_scores, final_asr])

    print(f"Results saved to {output_csv}")
    print(f"整体 ASR: {final_asr}")


# 运行
input_folder = "/mnt/workspace/our_work/ThirdVersion/result/qwen/topk=5/threshold=0.5/gte_Qwen2-1.5B-instruct/attack/single"
output_csv = "/mnt/workspace/our_work/ThirdVersion/result/qwen/topk=5/threshold=0.5/gte_Qwen2-1.5B-instruct/attack/single/no_second_analysis.csv"

compute_asr(input_folder, output_csv)