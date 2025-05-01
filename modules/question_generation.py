import os
import requests
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from time import sleep
from typing import List

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

class LLaMAQuestionGenerator:
    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url

    def load_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def split_text_semantic(self, text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ".", "!", "?", "\n", " "],
            length_function=len
        )
        return splitter.split_text(text)

    def query_llama(self, prompt: str) -> str:
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
            return f"Error: {e}"

    def generate_qa_pairs(self, chunks: List[str], total: int = 500) -> pd.DataFrame:
        questions, answers = [], []
        total = min(total, len(chunks))
        print("\U0001F680 Generating Q&A...")

        for i, chunk in enumerate(chunks[:total]):
            prompt = f"""
Based only on the following historical text, generate 2 simple factual questions and their short, correct answers. 
Output format strictly like:
Question: <your question>
Answer: <your answer>

Historical text:
{chunk}
"""
            response_text = self.query_llama(prompt)

            print("\n--- LLaMA Response ---")
            print(response_text)
            print("----------------------\n")


            output_lines = response_text.strip().split("\n")

            current_question, current_answer = "", ""
            for line in output_lines:
                if "Question:" in line:
                    if current_question and current_answer:
                        questions.append(current_question)
                        answers.append(current_answer)
                        current_question, current_answer = "", ""
                    current_question = line.split(":", 1)[1].strip()
                elif "Answer:" in line:
                    current_answer = line.split(":", 1)[1].strip()

            if current_question and current_answer:
                questions.append(current_question)
                answers.append(current_answer)

            percent = int(((i + 1) / total) * 100)
            print(f"Progress: {percent}% ({i+1}/{total})", end='\r')
            sleep(0.2)

        print("\n✅ Q&A generation complete!")
        return pd.DataFrame({"Question": questions, "Answer": answers})

    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} Q&A pairs to {output_path}")


if __name__ == "__main__":
    generator = LLaMAQuestionGenerator()
    raw_text = generator.load_text("data/modern_history_of_india.txt")
    chunks = generator.split_text_semantic(raw_text)
    qa_df = generator.generate_qa_pairs(chunks)
    generator.save_to_csv(qa_df, "data/qa_dataset_1000.csv")