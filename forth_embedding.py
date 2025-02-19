import pandas as pd
import json
from vectorstore import VectorStore
# save_path_train = "../dataset-25-1-16/wildjailbreak_final/jailbreak-classification_jailbreak_full.json"

vectorstore = VectorStore()
# vectorstore.clear_data()
# train_data = pd.read_json(save_path_train)
# with open(save_path_train, 'r', encoding='utf-8') as f:
#     train_data= json.load(f)
# patterns=[]
# prompts=[]
# for item in train_data:
#     # 判断 'pattern' 键存在且其值不为空（这里判断非空字符串，你也可以根据需要调整条件）
#     if 'pattern' in item and item['pattern']:
#         patterns.append(item['pattern'])
#         # 查找第一个包含 "prompt" 的键对应的值
#         prompt = next((value for key, value in item.items() if "prompt" in key), None)
#         prompts.append(prompt)
# # 创建 metadatas 列表
# metadatas = [{"prompt": line} for line in prompts]
# #清空数据集 需要时操作

# vectorstore.add_documents(patterns, metadatas)
# #出现metadatas为空可能是因为内存不够
# print(vectorstore.vectorstore._collection.peek(5))

# folder_path = '/mnt/workspace/our_work/ThirdVersion/dataset-25-1-16/wildjailbreak_final/train'
# error_path1='/mnt/workspace/our_work/ThirdVersion/ThirdVersion/test_data/benign/error'
import os
    # 遍历文件夹中的所有 JSON 文件
folder_path = '/mnt/workspace/our_work/dataset-25-1-16/wildjailbreak_final/essence/qwen/file'
# folder_path = '/mnt/workspace/our_work/dataset-25-1-16/wildjailbreak_final/file'
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        dataset_path_test = os.path.join(folder_path, filename)
        with open(dataset_path_test, 'r', encoding='utf-8') as f:
            train_data= json.load(f)
        patterns=[]
        prompts=[]
        for item in train_data:
            # 判断 'pattern' 键存在且其值不为空（这里判断非空字符串，你也可以根据需要调整条件）
            if 'pattern' in item and item['pattern']:
                patterns.append(item['pattern'])
                # 查找第一个包含 "prompt" 的键对应的值
                prompt = next((value for key, value in item.items() if "prompt" in key), None)
                # prompt=item["adversarial"]
                prompts.append(prompt)
        # 创建 metadatas 列表
        metadatas = [{"prompt": line} for line in prompts]
        vectorstore.add_documents(patterns, metadatas)

        print(f"{dataset_path_test} 已完成")
print(vectorstore.vectorstore._collection.peek(5))