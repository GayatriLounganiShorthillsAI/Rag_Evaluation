import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.Raga_testing import RAGEvaluator

class TestRAGEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = RAGEvaluator(
            excel_path="test_data.xlsx",
            output_xlsx="output.xlsx",
            output_json="output.json",
            batch_size=2,
            sleep_seconds=0
        )

    def test_build_prompt_structure(self):
        question = "What is AI?"
        answer = "AI stands for Artificial Intelligence."
        predicted_answer = "Artificial Intelligence is AI."
        context = "Artificial Intelligence, abbreviated as AI, is..."
        prompt = self.evaluator.build_prompt(question, answer, predicted_answer, context)
        self.assertIn("Question:", prompt)
        self.assertIn("Predicted Answer:", prompt)
        self.assertIn("Context:", prompt)

    def test_extract_scores_valid(self):
        text = "[0.98765, 0.87654, 0.76543, 0.65432]"
        expected = [0.98765, 0.87654, 0.76543, 0.65432]
        result = self.evaluator.extract_scores(text)
        self.assertEqual(result, expected)

    def test_extract_scores_invalid(self):
        text = "No valid list here"
        expected = [None, None, None, None]
        result = self.evaluator.extract_scores(text)
        self.assertEqual(result, expected)

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "faithfulness": 0.8,
        "answer_relevancy": 0.9,
        "context_precision": 0.85,
        "context_recall": 0.75
    }))
    @patch("os.path.exists", return_value=True)
    @patch("pandas.read_excel", return_value=pd.DataFrame([1, 2, 3, 4, 5]))
    def test_update_json_scores(self, mock_read_excel, mock_exists, mock_file):
        new_scores = [
            [0.9, 0.95, 0.85, 0.8],
            [0.8, 0.9, 0.8, 0.7]
        ]

        # Step 1: Averages from new scores
        new_avg = {
            "faithfulness": (0.9 + 0.8) / 2,          # 0.85
            "answer_relevancy": (0.95 + 0.9) / 2,     # 0.925
            "context_precision": (0.85 + 0.8) / 2,    # 0.825
            "context_recall": (0.8 + 0.7) / 2         # 0.75
        }

        # Step 2: Blend with old values (from mock_open)
        expected_updated_data = {
        "faithfulness": 0.82,       # Corrected from 0.83
        "answer_relevancy": 0.91,
        "context_precision": 0.84,
        "context_recall": 0.75
        }

        with patch("json.dump") as mock_json_dump:
            self.evaluator.update_json_scores(new_scores)
            mock_json_dump.assert_called_once()
            args, kwargs = mock_json_dump.call_args
            actual_data = args[0]
            self.assertEqual(actual_data, expected_updated_data)

if __name__ == "__main__":
    unittest.main()
