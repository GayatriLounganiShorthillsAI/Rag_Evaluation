import os
import requests
import pandas as pd
import logging
from time import sleep
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/ques_gen.log"),
        logging.StreamHandler()
    ]
)

# Text loading component
class TextLoader:
    @staticmethod
    def load(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                logging.info(f"Loaded text from {path}")
                return f.read()
        except Exception as e:
            logging.error(f"Error reading file {path}: {e}")
            raise


# Text splitting component
class TextSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ".", "!", "?", "\n", " "],
            length_function=len
        )

    def split(self, text: str) -> List[str]:
        try:
            chunks = self.splitter.split_text(text)
            logging.info(f"Split text into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logging.error(f"Error splitting text: {e}")
            raise


# LLaMA model client
class LLaMAClient:
    def __init__(self, model: str, url: str):
        self.model = model
        self.url = url
        logging.info(f"LLaMAClient initialized with model: {model} and URL: {url}")

    def query(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logging.error(f"Error querying LLaMA: {e}")
            return ""


# QA generation component
class QAGenerator:
    def __init__(self, llama_client: LLaMAClient):
        self.llama = llama_client

    def _build_prompt(self, chunk: str) -> str:
        return f"""
Based only on the following historical text, generate 2 simple factual questions and their short, correct answers. 
Output format strictly like:
Question: <your question>
Answer: <your answer>

Historical text:
{chunk}
"""

    def generate(self, chunks: List[str], total: int = 500) -> pd.DataFrame:
        questions, answers = [], []
        total = min(total, len(chunks))
        logging.info(f"Generating Q&A for {total} chunks")

        for i, chunk in enumerate(chunks[:total]):
            prompt = self._build_prompt(chunk)
            response = self.llama.query(prompt)

            if not response:
                continue

            lines = response.split("\n")
            q, a = "", ""
            for line in lines:
                if "Question:" in line:
                    if q and a:
                        questions.append(q)
                        answers.append(a)
                        q, a = "", ""
                    q = line.split(":", 1)[1].strip()
                elif "Answer:" in line:
                    a = line.split(":", 1)[1].strip()

            if q and a:
                questions.append(q)
                answers.append(a)

            percent = int(((i + 1) / total) * 100)
            print(f"Progress: {percent}% ({i+1}/{total})", end='\r')
            sleep(0.2)

        logging.info("Q&A generation complete")
        return pd.DataFrame({"Question": questions, "Answer": answers})


# CSV saving component
class DataSaver:
    @staticmethod
    def to_csv(df: pd.DataFrame, path: str):
        try:
            df.to_csv(path, index=False)
            logging.info(f"Saved {len(df)} Q&A pairs to {path}")
        except Exception as e:
            logging.error(f"Error saving CSV to {path}: {e}")
            raise


# Entrypoint
if __name__ == "__main__":
    try:
        # Constants
        TEXT_PATH = "data/modern_history_of_india.txt"
        CSV_OUTPUT = "data/qa_dataset_1000.csv"
        MODEL = os.getenv("OLLAMA_MODEL", "llama3")
        URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

        # Step-by-step pipeline
        raw_text = TextLoader.load(TEXT_PATH)
        chunks = TextSplitter().split(raw_text)
        llama_client = LLaMAClient(model=MODEL, url=URL)
        qa_generator = QAGenerator(llama_client)
        df = qa_generator.generate(chunks)
        DataSaver.to_csv(df, CSV_OUTPUT)

    except Exception as e:
        logging.critical(f"Fatal error: {e}")
