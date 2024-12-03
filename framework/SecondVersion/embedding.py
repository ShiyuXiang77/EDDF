from langchain.embeddings.base import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfEmbeddings(Embeddings):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def embed_documents(self, texts):
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray().tolist()

    def embed_query(self, query):
        query_vector = self.vectorizer.transform([query])
        return query_vector.toarray().flatten().tolist()
