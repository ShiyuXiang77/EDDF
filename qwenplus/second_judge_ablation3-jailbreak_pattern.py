import json
import shutil
import pandas as pd
from config import Config
from utils import append_to_json, read_json
from prompts import second_judge,LLM_judge,LLM_judge1
from openai import OpenAI
import os
from prompts import prompt_pattern
from utils import append_to_json, filter_json, read_json
import json
import re
import subprocess
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from prompts import prompt_pattern
from utils import append_to_json, filter_json
from utils import filter_json

threshold = 0.5
import requests

def run_deepseek(prompt: str) -> str:
    """使用subprocess运行deepseek模型，增加重试机制"""
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
        temperature=0.9,
    )
    return completion.choices[0].message.content

def process_item(item: Dict, error_path: str) -> Dict:
    """处理单个数据项的函数"""
    # parsed_result=None
    result=None
    try:
        prompt_final_test = list(item.values())[0]
        # 检查是否已经处理过
        if item.get('judge') == True or "similar_prompt" not in item :
            # item["judge"]=False
            return item
        # prompt_final_test = list(item.values())[0]
        # pattern=item["pattern"]
        # prompt = item.get("adversarial", "")
        similar_scores=item["scores"]
        similar_prompts=item['similar_prompt']
        similar_patterns=item['similar_pattern']
        #创建副本，避免被修改
        similar_prompts = similar_prompts.copy()
        similar_patterns = similar_patterns.copy()
        if similar_scores[0] > threshold:
            formatted_prompt = LLM_judge1.format(prompt_user=prompt_final_test)
            result = run_deepseek(formatted_prompt)
            # if result == "Error":
            #     raise ValueError("Model query failed")
            # result = re.split(r"</think>", result, maxsplit=1)[-1]
            parsed_result = json.loads(result)
            is_harmful = parsed_result["is_harmful"] if "is_harmful" in parsed_result else False
            reasoning = parsed_result["reasoning"] if "reasoning" in parsed_result else ""
            item['second result'] = result
            item['judge']=True
            item["is harmful"] = is_harmful
            item["reasoning"] = reasoning
            item["second judge"] = False
            return item
        else:
            # 遍历 similar_scores，将大于阈值的 corresponding similar_prompt 和 similar_pattern 置为空
            # for a in range(len(similar_scores)):
            #     if similar_scores[a] > threshold:
            #         similar_prompts[a] = ""
            #         similar_patterns[a] = ""
            formatted_prompt = second_judge.format(prompt_user=prompt_final_test,query_essence=" ",p1=similar_prompts[0],e1=similar_patterns[0],p2=similar_prompts[1],e2=similar_patterns[1],p3=similar_prompts[2],e3=similar_patterns[2],p4=similar_prompts[3],e4=similar_patterns[3],p5=similar_prompts[4],e5=similar_patterns[4])
            result = run_deepseek(formatted_prompt)
            # if result == "Error":
            #     raise ValueError("Model query failed")
            # result = re.split(r"</think>", result, maxsplit=1)[-1]
            parsed_result = json.loads(result)
            is_harmful = parsed_result["is_harmful"] if "is_harmful" in parsed_result else False
            reasoning = parsed_result["reasoning"] if "reasoning" in parsed_result else ""
            item['second result'] = result
            item['judge']=True
            item["is harmful"] = is_harmful
            item["reasoning"] = reasoning
            item["second judge"] = True
            return item
    except Exception as e:
        error_data = {
            "prompt": prompt_final_test,
            "error": f"Processing failed: {str(e)}"
        }
        if result is not None:
            error_data["result"] = result
            print(result)
        print(f"发生了其他意外错误: {e}")
        if  "Input data may contain inappropriate content" in str(e):
            item['judge']=True
            item["is harmful"]=True
            item["reasoning"]="Input data may contain inappropriate content"
            item["second judge"]=True
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
    max_workers = 3   # 预留一个CPU核心

    # folder_path = './test_data/benign/file'
    # error_path1='./test_data/benign/second error'
    folder_path = '/mnt/workspace/our_work/ThirdVersion/result/ablation3-jailbreak_pattern/attack/single/1'
    error_path1='/mnt/workspace/our_work/ThirdVersion/result/ablation3-jailbreak_pattern/attack/mix/error'
    # result_path1="./ThirdVersion/benign/single attack/result"
    # 遍历文件夹中的所有 JSON 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            dataset_path_test = os.path.join(folder_path, filename)
            error_path = os.path.join(error_path1, f"{filename.split('.')[0]}.json")
            # result_path = os.path.join(result_path1, f"{filename.split('.')[0]}_result.json")
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

