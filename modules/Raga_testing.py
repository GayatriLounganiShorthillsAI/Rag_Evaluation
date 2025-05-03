

import os
import json
import re
import time
import logging
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

# ====================== CONFIGURATION ======================
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/rag_evaluation.log"),
        logging.StreamHandler()
    ]
)

class RAGEvaluator:
    def __init__(self, excel_path, output_xlsx, output_json, batch_size=15, sleep_seconds=60):
        self.excel_path = excel_path
        self.output_xlsx = output_xlsx
        self.output_json = output_json
        self.batch_size = batch_size
        self.sleep_seconds = sleep_seconds
        self.model = self.configure_model()

    def configure_model(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY not found in environment.")
            raise ValueError("Missing API Key.")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("models/gemini-2.0-flash")

    def build_prompt(self, question, answer, predicted_answer, context):
        return f"""
You are evaluating the quality of a predicted answer in a retrieval-augmented question-answering (RAG) system. Use the provided *context*, *question*, and *ground truth answer* to score the *predicted answer* on the following 4 criteria:

Return ONLY a Python-style list of 4 floats between 0 and 1 (inclusive), rounded to 5 decimal places, and in this order:

1. Faithfulness: To what extent is the predicted answer strictly supported by the provided context? (1 = completely supported, 0 = contradicts or unsupported)
2. Answer Relevancy: How well does the predicted answer fully and directly address the question? (1 = fully answers the question, 0 = irrelevant)
3. Context Precision: What proportion of the retrieved context is actually relevant to generating the answer? (1 = all context is useful, 0 = none of the context is relevant)
4. Context Recall: How completely does the provided context support the ground truth answer? (1 = context fully contains the ground truth answer, 0 = context lacks key information)

Important:
- Use only the *provided context* to judge Faithfulness and Context Recall.
- Do NOT include explanations, just return a Python-style list like this: [0.95521, 0.85312, 0.91230, 0.85428]

Question: {question}
Ground Truth Answer: {answer}
Predicted Answer: {predicted_answer}
Context: {context}
"""

    def extract_scores(self, response_text):
        match = re.search(r"\[\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*\]", response_text)
        return [float(match.group(i)) for i in range(1, 5)] if match else [None] * 4

    def evaluate_batch(self, df_slice, start_index):
        scores_list = []
        for idx, row in tqdm(df_slice.iterrows(), total=len(df_slice), desc=f"Evaluating rows {start_index}-{start_index+len(df_slice)-1}"):
            prompt = self.build_prompt(row["Question"], row["Answer"], row["Predicted_Answer"], row["Context"])
            try:
                response = self.model.generate_content(prompt)
                scores = self.extract_scores(response.text)
            except Exception as e:
                logging.error(f"Error at index {start_index + idx}: {e}")
                scores = [None] * 4
            logging.info(f"Row {start_index + idx} scores: {scores}")
            scores_list.append(scores)
        return scores_list

    def append_to_excel(self, df_slice):
        if os.path.exists(self.output_xlsx):
            existing_df = pd.read_excel(self.output_xlsx)
            combined_df = pd.concat([existing_df, df_slice], ignore_index=True)
        else:
            combined_df = df_slice
        combined_df.to_excel(self.output_xlsx, index=False)
        logging.info(f"Appended {len(df_slice)} rows to {self.output_xlsx}")

    def update_json_scores(self, new_scores):
        filtered_scores = [s for s in new_scores if None not in s]
        if not filtered_scores:
            logging.warning("No valid scores to update in JSON.")
            return

        new_count = len(filtered_scores)
        new_sums = [sum(s[i] for s in filtered_scores) for i in range(4)]

        old_sums, old_count = [0.0] * 4, 0
        if os.path.exists(self.output_json):
            with open(self.output_json, "r") as f:
                existing_avg = json.load(f)
            old_scores = [
                existing_avg["faithfulness"],
                existing_avg["answer_relevancy"],
                existing_avg["context_precision"],
                existing_avg["context_recall"],
            ]
            old_count = len(pd.read_excel(self.output_xlsx)) - new_count
            old_sums = [old_scores[i] * old_count for i in range(4)]

        total_count = new_count + old_count
        final_avg = {
            "faithfulness": round((old_sums[0] + new_sums[0]) / total_count, 4),
            "answer_relevancy": round((old_sums[1] + new_sums[1]) / total_count, 4),
            "context_precision": round((old_sums[2] + new_sums[2]) / total_count, 4),
            "context_recall": round((old_sums[3] + new_sums[3]) / total_count, 4),
        }

        with open(self.output_json, "w") as f:
            json.dump(final_avg, f, indent=2)
        logging.info(f"Updated JSON scores: {final_avg}")

    def run(self):
        df = pd.read_excel(self.excel_path)
        already_done = 0

        if os.path.exists(self.output_xlsx):
            already_done = len(pd.read_excel(self.output_xlsx))
        logging.info(f"Resuming from index {already_done}")

        for i in range(already_done, len(df), self.batch_size):
            df_slice = df.iloc[i:i + self.batch_size].copy()
            if df_slice.empty:
                break

            scores = self.evaluate_batch(df_slice, i)
            df_slice["evaluation_scores"] = scores

            self.append_to_excel(df_slice)
            self.update_json_scores(scores)

            logging.info(f"Sleeping for {self.sleep_seconds}s before next batch...\n")
            time.sleep(self.sleep_seconds)

# ====================== EXECUTION ======================
if __name__ == "__main__":
    evaluator = RAGEvaluator(
        excel_path="data/qa_with_predictions.xlsx",
        output_xlsx="data/qa_evaluated_raga_scores.xlsx",
        output_json="data/raga_testing_score.json",
        batch_size=10,
        sleep_seconds=60
    )
    evaluator.run()
