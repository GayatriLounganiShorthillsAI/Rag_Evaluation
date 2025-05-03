import os
import numpy as np
import weaviate
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/embedder.log"),
        logging.StreamHandler()
    ]
)

class ChunkUploader:
    def __init__(self,
                 text_file: str,
                 embedding_model_name: str = "intfloat/e5-base-v2",
                 weaviate_url: str = "http://localhost:8080",
                 class_name: str = "Documents"):
        self.text_file = text_file
        self.class_name = class_name
        self.weaviate_url = weaviate_url

        try:
            self.client = weaviate.Client(url=weaviate_url)
            logging.info(f"Connected to Weaviate at {weaviate_url}")
        except Exception as e:
            logging.error(f"Failed to connect to Weaviate: {e}")
            raise

        try:
            self.embed_model = SentenceTransformer(embedding_model_name)
            logging.info(f"Loaded embedding model: {embedding_model_name}")
        except Exception as e:
            logging.error(f"Error loading embedding model: {e}")
            raise



    def load_text(self) -> str:
        if not os.path.exists(self.text_file):
            logging.error(f"Text file not found: {self.text_file}")
            raise FileNotFoundError(f"{self.text_file} does not exist")

        try:
            with open(self.text_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to read file: {e}")
            raise



    def split_text_semantic(self, text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ".", "!", "?", "\n", " "],
            length_function=len
        )
        chunks = splitter.split_text(text)
        logging.info(f"Text split into {len(chunks)} chunks")
        return chunks


    def create_weaviate_schema(self):
        try:
            self.client.schema.delete_all()
            logging.info("Old schema deleted (if any)")

            class_obj = {
                "class": self.class_name,
                "vectorizer": "none",
                "properties": [
                    {"name": "text", "dataType": ["text"]}
                ]
            }
            self.client.schema.create_class(class_obj)
            logging.info(f"Schema created with class: {self.class_name}")
        except Exception as e:
            logging.error(f"Failed to create schema in Weaviate: {e}")
            raise


    def insert_chunks(self, chunks: List[str]):
        try:
            embeddings = self.embed_model.encode(chunks, show_progress_bar=True)
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            raise

        embeddings = np.array(embeddings).astype(np.float32)

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                self.client.data_object.create(
                    data_object={"text": chunk},
                    class_name=self.class_name,
                    vector=embedding.tolist()
                )
                logging.debug(f"Inserted chunk {idx+1}/{len(chunks)}")
            except Exception as e:
                logging.warning(f"Failed to insert chunk {idx+1}: {e}")

        logging.info("All chunks uploaded (some may have failed, check logs)")


    def run(self):
        try:
            logging.info(" Starting upload process...")
            text = self.load_text()
            chunks = self.split_text_semantic(text)

            self.create_weaviate_schema()
            self.insert_chunks(chunks)
            logging.info(" Upload completed successfully!")
        except Exception as e:
            logging.error(f" Process failed: {e}")


if __name__ == "__main__":
    uploader = ChunkUploader("/home/shtlp_0047/Rag_ollama_2/Rag_Evaluation/data/modern_history_of_india.txt")
    uploader.run()