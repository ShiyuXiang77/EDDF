# utils.py
import json
import os
import re
def append_to_json(file_path, data):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)

    with open(file_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    existing_data.append(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


import re


def filter_json(parsed_result):
    # 使用正则表达式去除开头的 "```json" 部分，并提取花括号内的 JSON 内容
    match = re.match(r'```json\s*(\{.*\})\s*', parsed_result, re.DOTALL)

    # 如果匹配成功
    if match:
        # 提取 JSON 内容
        json_content = match.group(1).strip()

        # 清理掉不需要的控制字符（例如不可打印字符）
        json_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_content)

        return json_content.strip()
    parsed_result=re.sub(r'[\x00-\x1f\x7f-\x9f]', '', parsed_result)
    # 如果没有匹配到有效的 JSON 格式，返回原始内容
    return parsed_result.strip()
