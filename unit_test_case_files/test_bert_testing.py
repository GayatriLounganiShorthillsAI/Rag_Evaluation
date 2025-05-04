import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import os
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from modules.bert_testing import QAEvaluator


class TestQAEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = QAEvaluator()
        self.generated = "The capital of France is Paris."
        self.reference = "Paris is the capital city of France."

    @patch.object(QAEvaluator, '_cosine_similarity', return_value=0.85)
    @patch.object(QAEvaluator, '_get_rouge_score', return_value=0.7)
    @patch.object(QAEvaluator, '_get_bert_f1', return_value=0.88)
    def test_calculate_metrics(self, mock_bert, mock_rouge, mock_cosine):
        result = self.evaluator.calculate_metrics(self.generated, self.reference)
        expected_final_score = (
            0 * 0.7 + 
            0.4 * 0.85 + 
            0.6 * 0.88
        )

        self.assertAlmostEqual(result['final_score'], expected_final_score, places=4)
        self.assertEqual(result['cosine_similarity'], 0.85)
        self.assertEqual(result['bert_score_f1'], 0.88)
        self.assertEqual(result['rouge_score'], 0.7)

    def test_calculate_grade(self):
        self.assertEqual(self.evaluator.calculate_grade(0.91), "A (Excellent)")
        self.assertEqual(self.evaluator.calculate_grade(0.82), "B (Good)")
        self.assertEqual(self.evaluator.calculate_grade(0.75), "C (Average)")
        self.assertEqual(self.evaluator.calculate_grade(0.62), "D (Below Average)")
        self.assertEqual(self.evaluator.calculate_grade(0.5), "F (Poor)")

    @patch.object(QAEvaluator, 'calculate_metrics', return_value={
        "rouge_score": 0.6,
        "cosine_similarity": 0.7,
        "bert_score_f1": 0.9,
        "final_score": 0.81
    })
    def test_evaluate_excel(self, mock_metrics):
        # Create a dummy input Excel file
        dummy_data = {
            "Question": ["What is the capital of France?"],
            "Answer": ["Paris"],
            "Predicted_Answer": ["The capital is Paris."]
        }
        dummy_df = pd.DataFrame(dummy_data)
        input_path = "test_input.xlsx"
        output_path = "test_output.xlsx"
        summary_path = "test_summary.json"

        dummy_df.to_excel(input_path, index=False)

        try:
            self.evaluator.evaluate_excel(input_path, output_path, summary_path)

            # Check output Excel
            self.assertTrue(os.path.exists(output_path))
            evaluated_df = pd.read_excel(output_path)
            self.assertIn("final_score", evaluated_df.columns)

            # Check JSON summary
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r") as f:
                summary = json.load(f)
                self.assertEqual(summary["grade"], "B (Good)")
                self.assertAlmostEqual(summary["final_score"], 0.81, places=2)
        finally:
            # Clean up test files
            for path in [input_path, output_path, summary_path]:
                if os.path.exists(path):
                    os.remove(path)


if __name__ == '__main__':
    unittest.main()
