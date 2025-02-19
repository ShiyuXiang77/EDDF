import json
import shutil
import pandas as pd
from vectorstore import VectorStore
from config import Config
from embedding import get_embedding_model
from utils import filter_json
import re
import os

def process_json_files(input_folder: str,k,output_folder):
    """
    处理文件夹中的所有 JSON 文件
    
    Args:
        input_folder (str): 包含 JSON 文件的文件夹路径
    """
    # 确保文件夹存在
    if not os.path.exists(input_folder):
        print(f"文件夹不存在: {input_folder}")
        return
    #如果output_folder不存在，自己建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化 VectorStore 和 embedding 模型
    vectorstore = VectorStore()
    model_name = Config.EMBEDDING_MODEL_NAME
    embedding_model = get_embedding_model(model_name)

    # 遍历文件夹中的所有 JSON 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            output_path=os.path.join(output_folder,filename)
            try:
                # 读取 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 标记是否有修改
                # file_modified = False
                num=0
                # 遍历每个数据项
                for item in data:
                    if item.get("similar_prompt"):
                        continue
                    # 检查 pattern 是否存在且非空
                        # 获取 prompt（假设第一个值是 prompt）
                    prompt = list(item.values())[0]
                    # pattern = item['pattern']
                    
                    # 生成查询嵌入
                    # query_embedding = embedding_model.embed_query(pattern)
                    
                    # 相似性搜索
                    results = vectorstore.similarity_search(prompt,k)
                    num=num+1
                    print(num)
                    # 处理结果
                    similar_prompts = []
                    # similar_patterns = []
                    scores = []
                    
                    for result, score in results:
                        similar_prompts.append(result.page_content)
                        # similar_prompt = result.metadata.get("prompt", "")
                        # similar_prompts.append(similar_prompt)
                        scores.append(score)
                    
                    # 更新数据项
                    item['scores'] = scores
                    item['similar_prompt'] = similar_prompts
                        
                # 如果文件有修改，则保存
                # if file_modified:
                with open(output_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
                    print(f"已处理并保存: {output_path}")
                
            except json.JSONDecodeError:
                print(f"无法解析 JSON 文件: {filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {e}")

def main():
    # 指定要处理的文件夹路径
    # input_folder = '/mnt/workspace/our_work/ThirdVersion/result/qwen/topk=5/threshold=0.5/attack/mix'
    # output_folder="/mnt/workspace/our_work/ThirdVersion/result/qwen/topk=5/threshold=0.5/attack/mix"
    # input_folder = '/mnt/workspace/our_work/ThirdVersion/test_data/qwen_plus/attack/single/file'
    # output_folder="/mnt/workspace/our_work/ThirdVersion/result/qwen/topk=5/threshold=0.5/MiniLM-L6-v2"
    input_folder = '/mnt/workspace/our_work/ThirdVersion/test_data/only_prompt/benign/file'
    output_folder="/mnt/workspace/our_work/ThirdVersion/result/no_pattern/benign/file"
    k=10
    # 调用处理函数
    process_json_files(input_folder,k,output_folder)

if __name__ == "__main__":
    main()
    