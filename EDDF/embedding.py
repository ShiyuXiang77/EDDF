# embedding.py
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding  # 防止除以零
    return embedding / norm

class BaseEmbedding(Embeddings):
    def embed_documents(self, texts):
        raise NotImplementedError

    def embed_query(self, query):
        raise NotImplementedError

# 对于Jina模型的封装
class JinaEmbedding(BaseEmbedding):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer("../../jina-embeddings-v3", trust_remote_code=True)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, task="retrieval.passage", prompt_name="retrieval.passage", convert_to_numpy=True)
        return [normalize_embedding(embedding).tolist() for embedding in embeddings]

    def embed_query(self, query):
        query_embedding = self.model.encode([query], task="retrieval.query", prompt_name="retrieval.query", convert_to_numpy=True)
        return normalize_embedding(query_embedding).tolist()
class BertEmbedding(BaseEmbedding):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

class MiniLMEmbedding(BertEmbedding):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return [normalize_embedding(embedding).tolist() for embedding in embeddings]
        # return [embedding.tolist() for embedding in embeddings]
    def embed_query(self, query):
        query_embedding = self.model.encode(query)
        return normalize_embedding(query_embedding).tolist()
        # return query_embedding.tolist()
class GteQwenEmbedding(BertEmbedding):
    def __init__(self):
        from modelscope import snapshot_download
        model_dir = snapshot_download('iic/gte_Qwen2-1.5B-instruct', cache_dir='/mnt/workspace/model/model/modelscope')
        self.model = SentenceTransformer(model_dir, trust_remote_code=True)
        # In case you want to reduce the maximum length:
        # self.model.max_seq_length = 8192
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        # return [normalize_embedding(embedding).tolist() for embedding in embeddings]
        return [embedding.tolist() for embedding in embeddings]
    def embed_query(self, query):
        query_embedding = self.model.encode(query,prompt_name="query")
        # return normalize_embedding(query_embedding).tolist()
        return query_embedding.tolist()

def get_embedding_model(model_name):
    if model_name == 'jina-embeddings-v3':
        return JinaEmbedding(model_name)
    elif model_name == 'bert-base-nli-mean-tokens':
        return BertEmbedding(model_name)
    elif model_name == 'all-MiniLM-L6-v2':
        return MiniLMEmbedding()
    elif model_name=="gte_Qwen2-1.5B-instruct":
        return  GteQwenEmbedding()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
