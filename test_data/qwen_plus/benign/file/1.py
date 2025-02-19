import os
import json

def load_json(file_path):
    """加载 JSON 文件，确保非空、无 null 元素，并返回列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):  # 确保 JSON 是列表结构
                raise ValueError(f"Invalid JSON format in {file_path}, expected a list.")
            
            # 过滤掉 None（null）元素
            return [item for item in data if item is not None]
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading {file_path}: {e}")
        return []  # 返回空列表，避免 NoneType 访问错误

def save_json(file_path, data):
    """保存 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def filter_json_files(folder1, folder2, output_folder):
    """
    遍历 folder1 和 folder2 中相同名称的 JSON 文件：
    - 仅保留 folder2 中 `query` 在 folder1 中出现的元素
    - 结果保存到 output_folder
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder1):
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        output_path = os.path.join(output_folder, filename)

        if not filename.endswith(".json") or not os.path.exists(file2_path):
            continue  # 只处理 JSON 文件，并确保 folder2 中有相同文件

        json1 = load_json(file1_path)
        json2 = load_json(file2_path)

        if not json1 or not json2:  # 避免空数据
            print(f"Skipping {filename} due to empty or invalid JSON.")
            continue

        # 生成 json1 中的 query 集合
        valid_queries = {list(item.values())[0] for item in json1 if item }

        # 过滤 json2，仅保留 json1 中存在的 query
        filtered_json2 = [item for item in json2 if item.get("query") in valid_queries]

        save_json(output_path, filtered_json2)
        print(f"Processed: {filename}")

# 运行
folder1 = "/mnt/workspace/our_work/ThirdVersion/test_data/qwen_plus/benign/file"
folder2 = "/mnt/workspace/our_work/dataset-25-1-16/wildjailbreak_final/defence/benign_test/benign_result/Self-Reminder/json1"
output_folder = "/mnt/workspace/our_work/dataset-25-1-16/wildjailbreak_final/defence/benign_test/benign_result/Self-Reminder/json2"
filter_json_files(folder1, folder2, output_folder)
