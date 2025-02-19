from openai import OpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from prompts import judge_prompt
from utils import append_to_json, filter_json, read_json
import json
dataset_path_test = "../dataset-25-1-16/wildjailbreak_final/train-attack_shuffled.json"
# save_path_test = "../dataset-25-1-16/defence/generated_data/multiLang_pattern.json"
error_path = "../dataset-25-1-16/wildjailbreak_final/train-attack_error.json"
with open(dataset_path_test, 'r', encoding='utf-8') as f:
    data = json.load(f)

for i in range():
    item=data[i]
    if "judge" in item:
       continue
    else:
        prompt = next((value for key, value in item.items() if "prompt" in key), None)
        components = str(item['components'])
        pattern = str(item['pattern'])
        formatted_prompt = judge_prompt.format(jailbreak_prompt=prompt, components=components, pattern=pattern)
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key="",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': formatted_prompt}],
            temperature=0,
        )
        reply_content = completion.choices[0].message.content
        item["judge"] = reply_content
