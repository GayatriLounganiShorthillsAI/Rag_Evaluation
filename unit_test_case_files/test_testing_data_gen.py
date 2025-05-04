# test_data_generation.py
import unittest
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.testing_data_generation import RAGQuestionAnswering


class TestRAGQuestionAnswering(unittest.TestCase):

    @patch("modules.testing_data_generation.weaviate.Client")
    @patch("modules.testing_data_generation.SentenceTransformer")
    def setUp(self, mock_sentence_transformer, mock_weaviate_client):
        self.mock_embed_model = MagicMock()
        self.mock_embed_model.encode.return_value = [[0.1] * 384]  # dummy vector
        mock_sentence_transformer.return_value = self.mock_embed_model

        self.mock_weaviate = MagicMock()
        mock_weaviate_client.return_value = self.mock_weaviate
        self.qa = RAGQuestionAnswering()

    @patch("modules.testing_data_generation.requests.post")
    def test_query_llama_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Paris"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.qa.query_llama("What is the capital of France?")
        self.assertEqual(result, "Paris")
        mock_post.assert_called_once()

    def test_question_fetch(self):
        csv_content = "Question,Answer\nWhat is AI?,Artificial Intelligence\n"
        with patch("builtins.open", mock_open(read_data=csv_content)):
            results = list(self.qa.question_fetch("dummy.csv"))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["Question"], "What is AI?")
            self.assertEqual(results[0]["Answer"], "Artificial Intelligence")

    def test_prepare_workbook_create_new(self):
        with patch("modules.testing_data_generation.os.path.exists", return_value=False):
            wb, ws = self.qa.prepare_workbook("dummy.xlsx")
            self.assertEqual(ws.max_row, 1)
            self.assertEqual(ws.cell(row=1, column=1).value, "Question")

    def test_prepare_workbook_existing(self):
        with patch("modules.testing_data_generation.os.path.exists", return_value=True):
            with patch("modules.testing_data_generation.load_workbook") as mock_load:
                mock_ws = MagicMock()
                mock_wb = MagicMock()
                mock_wb.active = mock_ws
                mock_load.return_value = mock_wb
                wb, ws = self.qa.prepare_workbook("dummy.xlsx")
                self.assertEqual(wb, mock_wb)
                self.assertEqual(ws, mock_ws)

    def test_search_weaviate(self):
        self.mock_weaviate.query.get.return_value.with_near_vector.return_value.with_limit.return_value.do.return_value = {
            "data": {
                "Get": {
                    "Documents": [{"text": "Sample context"}]
                }
            }
        }
        result = self.qa.search_weaviate("Test query")
        self.assertEqual(result, ["Sample context"])


if __name__ == "__main__":
    unittest.main()
