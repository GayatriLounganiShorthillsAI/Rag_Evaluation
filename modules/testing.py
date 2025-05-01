import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from typing import Dict

# Config
INPUT_FILE = "data/qa_with_predictions_part.xlsx"
OUTPUT_FILE = "data/qa_evaluated_scores_part.xlsx"
SUMMARY_FILE = "data/evaluation_summary_llama3.json"

# Metric Weights
METRIC_WEIGHTS = {
    "rouge_score": 0,
    "cosine_similarity": 0.4,
    "bert_score_f1": 0.6
}

class QAEvaluator:
    def __init__(self):
        self.similarity_model = SentenceTransformer("intfloat/e5-base-v2")
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def calculate_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        emb_gen = self.similarity_model.encode(generated)
        emb_ref = self.similarity_model.encode(reference)
        cosine_sim = np.dot(emb_gen, emb_ref) / (np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref))
        rouge_score = self.rouge.score(reference, generated)['rougeL'].fmeasure
        _, _, bert_f1 = bert_score([generated], [reference], lang="en", model_type="bert-base-uncased")
        final_score = (
            METRIC_WEIGHTS["rouge_score"] * rouge_score +
            METRIC_WEIGHTS["cosine_similarity"] * float(cosine_sim) +
            METRIC_WEIGHTS["bert_score_f1"] * bert_f1.mean().item()
        )
        return {
            "rouge_score": rouge_score,
            "cosine_similarity": float(cosine_sim),
            "bert_score_f1": bert_f1.mean().item(),
            "final_score": final_score
        }

    def calculate_grade(self, score: float) -> str:
        if score >= 0.90: return "A (Excellent)"
        elif score >= 0.80: return "B (Good)"
        elif score >= 0.70: return "C (Average)"
        elif score >= 0.60: return "D (Below Average)"
        else: return "F (Poor)"

    def evaluate_excel(self, input_path: str, output_path: str, summary_path: str) -> None:
        df = pd.read_excel(input_path)

        if not all(col in df.columns for col in ["Question", "Answer", "Predicted_Answer"]):
            raise ValueError("Excel file must contain 'Question', 'Answer', and 'Predicted_Answer' columns.")

        rouge_scores = []
        cosine_scores = []
        bert_f1_scores = []
        final_scores = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            reference = str(row["Answer"])
            prediction = str(row["predicted_answer"])

            metrics = self.calculate_metrics(prediction, reference)
            rouge_scores.append(metrics["rouge_score"])
            cosine_scores.append(metrics["cosine_similarity"])
            bert_f1_scores.append(metrics["bert_score_f1"])
            final_scores.append(metrics["final_score"])

        df["rouge_score"] = rouge_scores
        df["cosine_similarity"] = cosine_scores
        df["bert_score_f1"] = bert_f1_scores
        df["final_score"] = final_scores

        df.to_excel(output_path, index=False)
        print(f"✅ Scores saved to {output_path}")

        summary = {
            "rouge_score": np.mean(rouge_scores),
            "cosine_similarity": np.mean(cosine_scores),
            "bert_score_f1": np.mean(bert_f1_scores),
            "final_score": np.mean(final_scores),
        }
        summary["grade"] = self.calculate_grade(summary["final_score"])

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"📊 Summary saved to {summary_path}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    evaluator = QAEvaluator()
    evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)

