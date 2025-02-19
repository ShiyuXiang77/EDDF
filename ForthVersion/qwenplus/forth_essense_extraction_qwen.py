import os
import json
import re
import subprocess
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from prompts import prompt_pattern
from utils import append_to_json, filter_json
import requests
from openai import OpenAI

def run_deepseek(prompt: str) -> str:
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return completion.choices[0].message.content


# for i in range(len(df_test_pattern)):
def process_item(item: Dict, error_path: str) -> Dict:
    """处理单个数据项的函数"""
    if "pattern" in item:
        return item
    # prompt = list(item.values())[0]
    prompt = item["adversarial"]
    formatted_prompt = prompt_pattern.format(prompt=prompt)
    try:
        result = run_deepseek(formatted_prompt)
        # match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
        # if match:
        #     parsed_result = match.group(1)
        # if isinstance(parsed_result, str):
        #     parsed_result = json.loads(parsed_result)
        parsed_result = json.loads(result)
        item["components"] = parsed_result.get("components", [])
        item["pattern"] = parsed_result.get("pattern", "")
        return item
    except Exception as e:
        print(f"发生了其他意外错误: {e}")
        data1 = {
            "prompt": prompt,
            "error": f"pattern 解析失败: {e}"
        }
        append_to_json(error_path, data1)
        return item

def process_dataset(
        dataset_path: str,
        error_path: str,
        max_workers: int = None,
        start_index: int = 0,
        end_index: int = None
):
    """并行处理数据集，支持起始和结束索引"""
    # 读取数据
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 根据起始和结束索引切片
    if end_index is None:
        end_index = len(data)

    # 切片处理的数据
    data_slice = data[start_index:end_index]

    # 使用进程池并行处理
    futures_map = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务并记录原始索引
        for local_idx, item in enumerate(data_slice):
            future = executor.submit(process_item, item, error_path)
            futures_map[future] = local_idx

        # 处理已完成的 future
        processed_count = 0
        for future in as_completed(futures_map.keys()):
            local_idx = futures_map[future]
            try:
                result = future.result()
                data[start_index + local_idx] = result
                print(f"Processed item {start_index + local_idx}")
                processed_count += 1
            except Exception as e:
                print(f"Error processing item {start_index + local_idx}: {e}")

            # 定期保存
            if processed_count % 20 == 0:
                with open(dataset_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
                print(f"Saved progress at item {start_index + local_idx + 1}")

    # 最终保存
    with open(dataset_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main():
    max_workers = 6   # 预留一个CPU核心

    folder_path = '/mnt/workspace/our_work/dataset-25-1-16/wildjailbreak_final/essence/qwen/train'
    error_path1='/mnt/workspace/our_work/dataset-25-1-16/wildjailbreak_final/essence/qwen/error'

    # 遍历文件夹中的所有 JSON 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            dataset_path_test = os.path.join(folder_path, filename)
            error_path = os.path.join(error_path1, f"{filename.split('.')[0]}.json")

            print(f"正在处理文件: {dataset_path_test}")

            # 调用 process_dataset 函数处理每个文件
            process_dataset(
                dataset_path=dataset_path_test,
                error_path=error_path,
                max_workers=max_workers,
                start_index=0# 起始索引
            )

            print(f"{dataset_path_test} 已完成")

if __name__ == '__main__':
    main()




