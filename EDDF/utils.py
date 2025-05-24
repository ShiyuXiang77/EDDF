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
def filter_json(result):
    try:
        # 尝试直接解析 result
        parsed_result = json.loads(result)
    except json.JSONDecodeError:
        # 如果直接解析失败，从包含 ```json 的部分提取 JSON 字符串
        try:
            match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if match:
                parsed_result = match.group(1)
            if isinstance(parsed_result, str):
                parsed_result = json.loads(parsed_result)
        except:
            match = result.strip()
            match = re.sub(r'^```|```$', '', match.strip(), flags=re.MULTILINE)
            match = re.sub(r',\s*([}\]])', r'\1', match)
            parsed_result = json.loads(match)
        return parsed_result