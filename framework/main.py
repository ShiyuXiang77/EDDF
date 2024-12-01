from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
import pandas as pd
import json
from langchain import LLMChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#星火认知大模型Spark Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
#星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
SPARKAI_APP_ID = 'a54ddac8'
SPARKAI_API_SECRET = 'YjM2ZmE5NDcyNjA0MjIyNGViNDI5NTI0'
SPARKAI_API_KEY = 'a9cadac070a5ec1ccd54a110497e2686'
#星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_DOMAIN = 'generalv3.5'

spark = ChatSparkLLM(
    spark_api_url=SPARKAI_URL,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN,
    streaming=False,
)

#通过url导入攻击知识
with open('arxiv_attack_paper_url.txt', 'r', encoding='utf-8') as file:
    urls = file.readlines()  # 读取所有行
    for url in urls:
        url = url.strip()
        #LLM对攻击知识进行抽取
        messages = [ChatMessage(
            role="user",
            content=f"""
                    执行以下操作：
                    1. 访问下面的URL，阅读论文，并提取论文中描述的攻击方式。
                    2. 总结攻击方式的关键特征和步骤。 
                    3. 给出攻击场景
                    4. 给出可能的恶意prompt 
                    url:
                    ```{url}```
                    """
        )]
        handler = ChunkPrintHandler()
        a = spark.generate([messages], callbacks=[handler])
        #将a拼接到json文件attck_knowledge.json中,格式为{"url": “xxx" attack_knowledge"："xxx"} JSON格式保存的有问题，需要修改一下
        with open('attack_knowledge.json', 'a', encoding='utf-8') as file:
            file.write(f'{{"url": "{url}",\n'
                       f'"attack_knowledge": "{a.generations}"}}\n')
#rag构建数据库，这个部分也需要修改一下
# 从文件读取攻击知识
def load_attack_knowledge(file_path):
    attack_knowledge_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            attack_knowledge_list.append(json.loads(line))
    return attack_knowledge_list

attack_knowledge_list = load_attack_knowledge('attack_knowledge.json')
attack_knowledge = [item['attack_knowledge'] for item in attack_knowledge_list]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(attack_knowledge)
faiss_index = FAISS.from_embeddings(OpenAIEmbeddings(), tfidf_matrix)

#导入attack prompt
df = pd.read_parquet('dataset/0000.parquet')
for index, row in df.iterrows():
    text = row['prompt']
    messages = [ChatMessage(
        role="user",
        content=f"""
            执行以下操作：
            1. 阅读下面的文本，并提取关键词。
            2. 

            text:
            ```{text}```
            """
    )]
    handler = ChunkPrintHandler()
    user_response= spark.generate([messages], callbacks=[handler])
    user_tfidf = vectorizer.transform([user_response])
    # 计算与attack_knowledge的相似度
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    max_similarity = max(cosine_similarities)

    # 设定阈值进行有害性判断
    threshold = 0.7  # 设定一个相似度阈值
    if max_similarity > threshold:
        print(f"提示: 用户输入的内容可能有害。最大相似度: {max_similarity}")
    else:
        print(f"提示: 用户输入的内容无害。最大相似度: {max_similarity}")




