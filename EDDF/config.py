# config.py
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] =''
os.environ["OPENAI_API_BASE"] = ''
os.environ["GOOGLE_API_KEY"]=''
class Config:
    # LLM配置
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    model_name = ""
    # PERSIST_DIRECTORY = "./chroma/MiniLM-L6-v2"
    PERSIST_DIRECTORY="./chroma/gte_Qwen2-1.5B-instruct"
    # EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_NAME="gte_Qwen2-1.5B-instruct"

