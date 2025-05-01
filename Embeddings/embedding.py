import os
import numpy as np
import weaviate
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChunkUploader:
    def __init__(self, 
                 text_file: str,
                 embedding_model_name: str = "intfloat/e5-base-v2",
                 weaviate_url: str = "http://localhost:8080",
                 class_name: str = "Documents"):
        self.text_file = text_file
        self.class_name = class_name
        self.embed_model = SentenceTransformer(embedding_model_name)
        self.client = weaviate.Client(url=weaviate_url)

    def load_text(self) -> str:
        with open(self.text_file, "r", encoding="utf-8") as f:
            return f.read()


    def split_text_semantic(self, text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ".", "!", "?", "\n", " "],
            length_function=len
        )
        return splitter.split_text(text)


    def create_weaviate_schema(self):
        self.client.schema.delete_all()
        class_obj = {
            "class": self.class_name,
            "vectorizer": "none",
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"]
                }
            ]
        }
        self.client.schema.create_class(class_obj)


    def insert_chunks(self, chunks: List[str]):
        embeddings = self.embed_model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype(np.float32)

        for chunk, embedding in zip(chunks, embeddings):
            self.client.data_object.create(
                data_object={"text": chunk},
                class_name=self.class_name,
                vector=embedding.tolist()
            )

    def run(self):
        print("📄 Reading and processing text file...")
        text = self.load_text()
        chunks = self.split_text_semantic(text)

        print("🧱 Setting up Weaviate schema...")
        self.create_weaviate_schema()

        print("📤 Uploading chunks to Weaviate...")
        self.insert_chunks(chunks)
        print("✅ All chunks inserted into Weaviate!")


if __name__ == "__main__":
    uploader = ChunkUploader("/home/shtlp_0047/Rag_ollama_2/Rag_Evaluation/data/modern_history_of_india.txt")
    uploader.run()