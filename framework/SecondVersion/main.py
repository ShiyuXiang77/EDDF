from spark_llm import initialize_spark_llm
from data_loader import fetch_content_from_url, append_to_json, load_attack_knowledge
from embedding import TfidfEmbeddings
from prompts import chat_prompt1, chat_prompt2
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 初始化 LLM 和嵌入模型
llm = initialize_spark_llm()
embeddings = TfidfEmbeddings()
# 加载向量数据库
persist_directory = "chroma_db_split"
embeddings = TfidfEmbeddings()
vectorstore = Chroma(persist_directory, embeddings)

# test
# query = "What is the backdoor attack method described in the paper?"
# # 获取查询向量
# query_vector = embeddings.embed_query(query)
# results = vectorstore.similarity_search(query, k=1)
# # 获取所有文档的向量
# document_vectors = [embeddings.embed_documents([doc.page_content])[0] for doc in results]
# # 计算查询与文档之间的余弦相似度
# similarities = cosine_similarity([query_vector], document_vectors).flatten()
# # 打印查询结果及其相似度分数
# print("查询结果：")
# for i, result in enumerate(results):
#     print(f"Result {i+1}:")
#     print(f"Content: {result.page_content}")
#     print(f"Similarity Score: {similarities[i]}")  # 输出计算的相似度分数")

# 用户输入
# 导入attack prompt
df = pd.read_parquet('../dataset/0000.parquet')
for index, row in df.iterrows():
    content = row['prompt']
    messages = chat_prompt2.format_messages(content=content)
    result = llm.invoke(messages)
    output_parser = StrOutputParser()
    parsed_result = output_parser.invoke(result)
    #响应结果也要存储
    query_vector = embeddings.embed_query(parsed_result)
    #相似度检测
    results = vectorstore.similarity_search(parsed_result, k=1)
    document_vectors = [embeddings.embed_documents([doc.page_content])[0] for doc in results]
    similarities = cosine_similarity([query_vector], document_vectors).flatten()
    print(results)
    print(similarities)
