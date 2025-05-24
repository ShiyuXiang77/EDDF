# vectorstore.py
import shutil
from langchain_chroma import Chroma
from embedding import get_embedding_model  # Import the function to get the embedding model from embedding.py
from config import Config

class VectorStore:
    def __init__(self):
        """
        Constructor that selects the corresponding embedding model based on the given model name.

        :param persist_directory: Directory to store the Chroma database
        :param model_name: Name of the embedding model to use, such as 'jina-embeddings-v3', 'bert-base-nli-mean-tokens', etc.
        """
        self.persist_directory = Config.PERSIST_DIRECTORY
        self.model_name = Config.EMBEDDING_MODEL_NAME
        self.embedding_function = get_embedding_model(self.model_name)  # Get the embedding model instance based on the model name
        self.vectorstore = Chroma(
                            persist_directory=self.persist_directory,
                            embedding_function=self.embedding_function,
                            collection_metadata={"hnsw": "cosine"}
                            )

    def add_documents(self, texts, metadatas, batch_size=10):
        """
        Add documents to the vector database.

        :param texts: List of text documents
        :param metadatas: List of corresponding metadata
        """
        if len(texts) != len(metadatas):
            raise ValueError("The lengths of the text list and metadata list do not match!")
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            # Add the current batch of documents to the vector database
            self.vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)

            print(f"Successfully added {len(batch_texts)} documents to the database")
        print(f"Successfully added {len(texts)} documents to the database")

    def similarity_search(self, query, k=5):
        """
        Perform similarity search based on a query.

        :param query: The query text
        :param k: Number of most similar documents to return
        """
        return self.vectorstore.similarity_search_with_score(query=query, k=k)

    def clear_data(self):
        """
        Clear the current data in the vector database by deleting the database directory and its contents.
        """
        try:
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            print(f"Database directory '{self.persist_directory}' has been cleared")
        except Exception as e:
            print(f"An error occurred while clearing the database: {e}")


if __name__ == '__main__':
    import pandas as pd
    import json
    vectorstore = VectorStore()
    import os
    # Traverse all JSON files in the folder
    folder_path = ' '
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            dataset_path_test = os.path.join(folder_path, filename)
            with open(dataset_path_test, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            patterns = []
            prompts = []
            for item in train_data:
                # Check if the 'pattern' key exists and is not empty
                if 'pattern' in item and item['pattern']:
                    patterns.append(item['pattern'])
                    # Find the first value associated with a key containing "prompt"
                    prompt = next((value for key, value in item.items() if "prompt" in key), None)
                    # prompt=item["adversarial"]
                    prompts.append(prompt)
            # Create the metadatas list
            metadatas = [{"prompt": line} for line in prompts]
            # Clear the dataset if needed (commented out here)
            vectorstore.add_documents(patterns, metadatas)
            print(f"{dataset_path_test} completed")
