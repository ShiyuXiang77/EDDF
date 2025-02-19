# vectorstore.py
import shutil
from langchain_chroma import Chroma
from embedding import get_embedding_model  # 从 embedding.py 导入获取Embedding模型的函数
from config import Config

class VectorStore:
    def __init__(self):
        """
        构造函数，根据传入的模型名称选择对应的嵌入模型

        :param persist_directory: Chroma数据库的存储目录
        :param model_name: 使用的嵌入模型名称，如 'jina-embeddings-v3', 'bert-base-nli-mean-tokens' 等
        """
        self.persist_directory = './MiniLM-L6-v2'
        self.model_name =Config.EMBEDDING_MODEL_NAME
        self.embedding_function = get_embedding_model(self.model_name)  # 使用传入的模型名称获取嵌入类实例
        self.vectorstore = Chroma(
                            persist_directory=self.persist_directory,
                            embedding_function=self.embedding_function,
                            collection_metadata={"hnsw":"cosine"}
                            )

    def add_documents(self, texts, metadatas,batch_size=10):
        """
        向向量数据库中添加文档

        :param texts: 文本列表
        :param metadatas: 元数据列表
        """
        if len(texts) != len(metadatas):
            raise ValueError("文本列表和元数据列表的长度不一致！")
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            # 向向量数据库添加当前批次的文档
            self.vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)

            print(f"成功添加 {len(batch_texts)} 篇文档到数据库")
        print(f"成功添加 {len(texts)} 篇文档到数据库")

    # def similarity_search(self, query_embedding, k=5):
    #     """
    #     通过查询嵌入进行相似度检索

    #     :param query_embedding: 查询的嵌入向量
    #     :param k: 返回的最相似文档数量
    #     """
    #     return self.vectorstore.similarity_search_by_vector_with_relevance_scores(embedding=query_embedding, k=k)
    def similarity_search(self, query, k=5):
        """
        通过查询嵌入进行相似度检索

        :param query_embedding: 查询的嵌入向量
        :param k: 返回的最相似文档数量
        """
        return self.vectorstore.similarity_search_with_score(query=query,k=k)
    def clear_data(self):
        """
        清空当前向量数据库的数据，删除数据库文件夹及其内容
        """
        try:
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            print(f"数据库目录 '{self.persist_directory}' 已被清空")
        except Exception as e:
            print(f"清空数据库时发生错误: {e}")

