import os
import json
import numpy as np
import requests
import logging
import weaviate
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st
from io import StringIO


# Load environment variables from .env
load_dotenv()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)


# Use environment variables
HISTORY_FILE = os.getenv("HISTORY_FILE", "data/history.json")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")



class WeaviateRetriever:
    def __init__(self, weaviate_url: str = "http://localhost:8080", embedding_model: str = "intfloat/e5-base-v2"):
        try: 
            self.client = weaviate.Client(url=weaviate_url)
            self.embed_model = SentenceTransformer(embedding_model)
        except Exception as e:
            logging.error(f"Error initializing WeaviateRetriever: {e}")
            raise



    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        try:
            query_embedding = self.embed_model.encode([query])
            query_embedding = np.array(query_embedding).astype(np.float32)
            response = self.client.query.get("Documents", ["text"])\
                .with_near_vector({"vector": query_embedding[0]})\
                .with_limit(top_k).do()
            return [item["text"] for item in response["data"]["Get"]["Documents"]]

        except Exception as e:
            logging.error(f"Retrieval failed: {e}")
            return []



class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            logging.error(f"Failed to initialize Reranker: {e}")
            raise


    def rerank(self, query: str, chunks: List[str], top_k: int = 3) -> List[str]:
        if not chunks:
            logging.warning("No chunks to rerank.")
            return []
        try:
            pairs = [(query, chunk) for chunk in chunks]
            scores = self.model.predict(pairs)
            reranked = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
            return reranked[:top_k]
        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            return []
    



class LlamaQuerier:
    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url

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
            logging.error(f"Failed to query model: {e}")
            return f"Error querying {self.model}: {e}"



class QAApp:
    def __init__(self):
        self.retriever = WeaviateRetriever()
        self.reranker = Reranker()
        self.llm = LlamaQuerier()
        self.history = self.load_history()


    def load_history(self):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Could not load JSON file {HISTORY_FILE}: {e}")
            return []



    def save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save JSON file {HISTORY_FILE}: {e}")



    def run(self):
            st.title(f"RAG Q&A ({OLLAMA_MODEL.capitalize()})")

            if "history" not in st.session_state:
                st.session_state.history = self.history

            st.sidebar.title("📜 Chat History")

            question = st.text_input("Ask your question about History:")

            if question:
                try:
                    chunks = self.retriever.retrieve(question, top_k=10)
                    context = self.reranker.rerank(question, chunks, top_k=5)
                    prompt = self._format_prompt(question, context)
                    answer = self.llm.query(prompt)

                    st.session_state.history.append({"question": question, "answer": answer})
                    self.history = st.session_state.history  # sync session state to self.history
                    self.save_history()

                    st.markdown("**Answer:**")
                    st.markdown(f"<div style='padding: 0.75em; background-color: #f4f4f4; border-radius: 6px;'>{answer}</div>", unsafe_allow_html=True)

                except Exception as e:
                    logging.error(f"Error during Q&A processing: {e}")
                    st.error("An error occurred while processing your question.")

            if st.session_state.history:
                for idx, entry in reversed(list(enumerate(st.session_state.history))):
                    with st.sidebar.expander(f"Q{idx+1}: {entry['question'][:60]}..."):
                        st.markdown(f"**Q:** {entry['question']}")
                        st.markdown(f"**A:** {entry['answer']}")


    def _format_prompt(self, question, context_chunks):
            context = "\n".join(context_chunks)
            return f"""
    You are a factual Q&A assistant based only on the provided historical context.

    Guidelines:
    
    - Use only the information from the given context to answer.
    - Answer Generic Questions which express gratitude or greetings like "Hii" , "Hello" , "Bye", "Thank you" accordingly without using retrieved context.
    - Do NOT infer, assume, or add any information not directly found in the context.
    - If the answer is not directly stated or clearly present in the context, respond exactly with: "Answer not found in the context."
    - Ignore loosely related content. Focus only on content that exactly matches or answers the question.
    - If the question is asking about a specific topic (e.g., 'Indus Valley Civilization'), do not include information about unrelated or predecessor topics unless explicitly asked.


    Context:
    {context}

    Question: {question}
    Answer:
    """


if __name__ == "__main__":
    app = QAApp()
    app.run()