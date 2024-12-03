from langchain.embeddings.base import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from data_loader import  load_attack_knowledge
class TfidfEmbeddings(Embeddings):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def embed_documents(self, texts):
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray().tolist()

    def embed_query(self, query):
        query_vector = self.vectorizer.transform([query])
        return query_vector.toarray().flatten().tolist()


embeddings = TfidfEmbeddings()

# 分块文本
file_path = 'attack_knowledge.json'
attack_knowledge_list = load_attack_knowledge(file_path)

# 提取 "attack_knowledge" 字段并合并为单一字符串
combined_text = "\n\n".join(item['attack_knowledge'] for item in attack_knowledge_list)

# 使用 RecursiveCharacterTextSplitter 分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) #需要修改分块方式
split_docs = text_splitter.split_text(combined_text)
# 构建向量数据库
# 初始化自定义嵌入器
embeddings = TfidfEmbeddings()

# 将分块后的数据转换为 LangChain 文档格式
documents = [Document(page_content=chunk) for chunk in split_docs]

# 使用分块后的文档构建向量数据库
persist_directory = "chroma_db_split"
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)