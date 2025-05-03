import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from typing import Dict, List
import logging 

# Load environment variables
load_dotenv()


# Constants
INPUT_FILE = os.getenv("INPUT_FILE", "data/qa_with_predictions.xlsx")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/qa_evaluated_bert_scores.xlsx")
SUMMARY_FILE = os.getenv("SUMMARY_FILE", "data/bert_testing_score.json")


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/bert_evaluation.log"),
        logging.StreamHandler()
    ]
)



METRIC_WEIGHTS = {
    "rouge_score": 0,
    "cosine_similarity": 0.4,
    "bert_score_f1": 0.6
}


class QAEvaluator:
    def __init__(self):
        self.similarity_model = SentenceTransformer("intfloat/e5-base-v2")
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        logging.info("QAEvaluator initialized with E5 and ROUGE scorers.")


    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def _get_bert_f1(self, generated: str, reference: str) -> float:
        _, _, bert_f1 = bert_score([generated], [reference], lang="en", model_type="bert-base-uncased")
        return bert_f1.mean().item()

    def _get_rouge_score(self, generated: str, reference: str) -> float:
        return self.rouge.score(reference, generated)['rougeL'].fmeasure

    def calculate_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        emb_gen = self.similarity_model.encode(generated)
        emb_ref = self.similarity_model.encode(reference)

        cosine_sim = self._cosine_similarity(emb_gen, emb_ref)
        rouge = self._get_rouge_score(generated, reference)
        bert_f1 = self._get_bert_f1(generated, reference)

        final_score = (
            METRIC_WEIGHTS["rouge_score"] * rouge +
            METRIC_WEIGHTS["cosine_similarity"] * cosine_sim +
            METRIC_WEIGHTS["bert_score_f1"] * bert_f1
        )

        logging.debug(f"Metrics -> Cosine: {cosine_sim:.4f}, ROUGE: {rouge:.4f}, BERT-F1: {bert_f1:.4f}, Final: {final_score:.4f}")

        return {
            "rouge_score": rouge,
            "cosine_similarity": cosine_sim,
            "bert_score_f1": bert_f1,
            "final_score": final_score
        }

    def calculate_grade(self, score: float) -> str:
        if score >= 0.90:
            return "A (Excellent)"
        elif score >= 0.80:
            return "B (Good)"
        elif score >= 0.70:
            return "C (Average)"
        elif score >= 0.60:
            return "D (Below Average)"
        return "F (Poor)"

    def evaluate_excel(self, input_path: str, output_path: str, summary_path: str) -> None:
        logging.info(f"Reading input Excel file: {input_path}")
        df = pd.read_excel(input_path)


        required_cols = {"Question", "Answer", "Predicted_Answer"}
        if not required_cols.issubset(df.columns):
            logging.error(f"Excel file missing required columns: {required_cols}")
            raise ValueError(f"Excel file must contain columns: {required_cols}")

        scores = {
            "rouge_score": [],
            "cosine_similarity": [],
            "bert_score_f1": [],
            "final_score": []
        }


        logging.info("Starting evaluation of predictions...")

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            reference = str(row["Answer"])
            prediction = str(row["Predicted_Answer"])

            try:
                metrics = self.calculate_metrics(prediction, reference)
                for key in scores:
                    scores[key].append(metrics[key])
            except Exception as e:
                logging.warning(f"Error evaluating row {index}: {e}")
                for key in scores:
                    scores[key].append(0.0)

        for metric, values in scores.items():
            df[metric] = values

        df.to_excel(output_path, index=False)
        logging.info(f"Evaluation results saved to: {output_path}")

        summary = {metric: float(np.mean(values)) for metric, values in scores.items()}
        summary["grade"] = self.calculate_grade(summary["final_score"])

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        logging.info(f"Summary saved to: {summary_path}")
        logging.info(f"Summary:\n{json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    evaluator = QAEvaluator()
    evaluator.evaluate_excel(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)