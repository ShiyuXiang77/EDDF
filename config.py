# config.py
from dotenv import load_dotenv, find_dotenv
import os

# 加载环境变量
_ = load_dotenv(find_dotenv())
# 设置环境变量 API_KEY
os.environ["OPENAI_API_KEY"] =''
os.environ["OPENAI_API_BASE"] = 'https://api.deepseek.com'
os.environ["GOOGLE_API_KEY"]=''
# 配置文件：存储需要的配置信息
class Config:
    # LLM配置
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    # Chroma 配置
    # PERSIST_DIRECTORY = "./chroma/MiniLM-L6-v2"
    PERSIST_DIRECTORY="./chroma/gte_Qwen2-1.5B-instruct"
    PERSIST_DIRECTORY="./chroma/qwen-gte"
    # PERSIST_DIRECTORY="./chroma/no_pattern"
    # EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_NAME="gte_Qwen2-1.5B-instruct"

