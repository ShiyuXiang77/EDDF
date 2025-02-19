import os
from openai import OpenAI
import subprocess
from prompts import prompt_user
from utils import append_to_json, filter_json, read_json
import json
import re
import os
import json
import re
import subprocess
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from prompts import prompt_pattern
from utils import append_to_json, filter_json
import requests

# # dataset_path_test = "/mnt/workspace/our_work/ThirdVersion/test_data/benign/Cipher_benign_100_prompt_our_work_test.json"
# dataset_path_test = "/mnt/workspace/our_work/ThirdVersion/test_data/benign/Cipher_benign_100_prompt_our_work_test.json"
# # save_path_test = "../dataset-25-1-16/defence/generated_data/multiLang_pattern.json"
# error_path = "/mnt/workspace/our_work/ThirdVersion/test_data/multiLangCipher_benign_error.json"
# def run_deepseek(prompt):
#     try:
#         # 使用 subprocess 运行 ollama run mistral 命令
#         result = subprocess.run(
#             ["ollama", "run", "deepseek-r1:14b"],
#             input=prompt,
#             capture_output=True,
#             text=True,
#             check=True,
#             encoding='utf-8'
#         )
#         return result.stdout.strip()
#     except subprocess.CalledProcessError as e:
#         print(f"Error running: {e}")
#         return "Error"
# with open(dataset_path_test, 'r', encoding='utf-8') as f:
#     data = json.load(f)
# # for i in range(len(df_test_pattern)):
# for i in range(0,len(data)):
#     # 对于每一个元素，提取Adversial和最后是query的prompt的pattern
#     item = data[i]
#     # if "components"不在该元素的Key中
#     if "pattern" in item and item["pattern"]:
#         continue
#     else:
#         #如果 prompt在Key中，则提取prompt
#         # if "query_nest_prompt" in item:
#         #     prompt = item["query_nest_prompt"]
#         for key in list(item.keys()):
#             if "prompt" in key:
#                 prompt = item[key]
#                 formatted_prompt = prompt_user.format(prompt=prompt)
#                 result=run_deepseek(formatted_prompt)
#                 result = re.split(r"</think>", result, maxsplit=1)[-1]
#                 parsed_result = result.strip()  # 去除两端的空白符
#                 parsed_result = filter_json(parsed_result)
#                 try:
#                     if isinstance(parsed_result, str):
#                         parsed_result = json.loads(parsed_result)
#                     # item["components"] = parsed_result.get("components", [])
#                     item['result']=parsed_result
#                     item["pattern"] = parsed_result.get("pattern", "")
#                     print(i)
#                 except Exception as e:
#                     print(parsed_result )
#                     print(i)
#                     print(f"发生了其他意外错误: {e}")
#                     data1 = {
#                         "index": i,
#                         "prompt": prompt,
#                         "result":parsed_result,
#                         "error": f"pattern 解析失败: {e}"
#                     }
#                     append_to_json(error_path, data1)
# #多少条之前保存一下
#     if i % 10 == 0 and i != 0:  # i != 0 防止第一次保存
#         with open(dataset_path_test, 'w', encoding='utf-8') as file:
#             json.dump(data, file, ensure_ascii=False, indent=4)
#         print(f"Saved at iteration {i}")
# with open(dataset_path_test, 'w', encoding='utf-8') as file:
#     json.dump(data, file, ensure_ascii=False, indent=4)

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

def process_item(item: Dict, error_path: str) -> Dict:
    """处理单个数据项的函数"""
    parsed_result=None
    try:
        # 检查是否已经处理过
        if "pattern" in item and item["pattern"]:
            return item
        # prompt = item.get("adversarial", "")
        prompt = list(item.values())[0]
        formatted_prompt =prompt_user.format(prompt=prompt)
        result = run_deepseek(formatted_prompt)
        try:
            # 尝试直接解析 result
            parsed_result = json.loads(result)
        except json.JSONDecodeError:
            # 如果直接解析失败，从包含 ```json 的部分提取 JSON 字符串
            match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if match:
                parsed_result = match.group(1)
            if isinstance(parsed_result, str):
                parsed_result = json.loads(parsed_result)
        item["result"] = parsed_result
        # item["components"] = parsed_result.get("components", [])
        item["pattern"] = parsed_result.get("pattern", "")
        return item
    except Exception as e:
        error_data = {
            "prompt": prompt,
            "error": f"Processing failed: {str(e)}"
        }
        if result is not None:
            error_data["result"] = result
            print(result)
        print(f"发生了其他意外错误: {e}")
        append_to_json(error_path, error_data)
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
            if processed_count % 10 == 0:
                with open(dataset_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
                print(f"Saved progress at item {start_index + local_idx + 1}")


    # 最终保存
    with open(dataset_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    # # 自动检测CPU数量并设置workers数量
    import multiprocessing
    max_workers = 5   # 预留一个CPU核心

    folder_path = '/mnt/workspace/our_work/ThirdVersion/test_data/deepseek_14b/attack/mix/file'
    error_path1='/mnt/workspace/our_work/ThirdVersion/test_data/deepseek_14b/attack/mix/error'

    # 遍历文件夹中的所有 JSON 文件S
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
                start_index=0 # 起始索引

            )

            print(f"{dataset_path_test} 已完成")
 
if __name__ == "__main__":
    main()

