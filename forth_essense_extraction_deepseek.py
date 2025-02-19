import os
import json
import re
import subprocess
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from prompts import prompt_pattern
from utils import append_to_json, filter_json
import requests


def run_deepseek(prompt: str) -> str:
    """使用subprocess运行deepseek模型，增加重试机制"""
    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "stop": ["null"],
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
    }

    headers = {
        "Authorization": "Bearer ",
        "Content-Type": "application/json"
    }

    try:
        response = requests.request("POST", url, json=payload, headers=headers)
        # 打印状态码和原始响应
        # print(f"Status Code: {response.status_code}")
        # print(f"Response Text: {response.text}")

        # 确保请求成功
        response.raise_for_status()

        result = json.loads(response.text)
        if 'choices' not in result:
            print(f"API返回内容异常: {result}")
            raise KeyError("API response missing 'choices' key")

        content = result['choices'][0]['message']['content']
        return content

    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        print(f"Raw response: {response.text}")
        raise
    except KeyError as e:
        print(f"响应格式错误: {e}")
        raise


# for i in range(len(df_test_pattern)):
def process_item(item: Dict, error_path: str) -> Dict:
    """处理单个数据项的函数"""
    if "pattern" in item:
        return item
    prompt = list(item.values())[0]
    # prompt = item["adversarial"]
    formatted_prompt = prompt_pattern.format(prompt=prompt)
    try:
        result = run_deepseek(formatted_prompt)
        match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
        if match:
            parsed_result = match.group(1)
        if isinstance(parsed_result, str):
            parsed_result = json.loads(parsed_result)
        item["components"] = parsed_result.get("components", [])
        item["pattern"] = parsed_result.get("pattern", "")
        return item
    except Exception as e:
        print(f"发生了其他意外错误: {e}")
        data1 = {
            "index": item["idx"],
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
    max_workers = 2   # 预留一个CPU核心

    folder_path = '../dataset-25-1-16/wildjailbreak_final/file'
    error_path1='../dataset-25-1-16/wildjailbreak_final/error'

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




