import unittest
from unittest.mock import patch, MagicMock, mock_open
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from main import WeaviateRetriever, Reranker, LlamaQuerier, QAApp


class TestWeaviateRetriever(unittest.TestCase):

    @patch("main.weaviate.Client")
    @patch("main.SentenceTransformer")
    def test_retrieve_success(self, mock_model, mock_client):
        mock_model().encode.return_value = [[0.1] * 384]
        mock_query = MagicMock()
        mock_query.get().with_near_vector().with_limit().do.return_value = {
            "data": {
                "Get": {
                    "Documents": [{"text": "Doc 1"}, {"text": "Doc 2"}]
                }
            }
        }
        mock_client().query = mock_query
        retriever = WeaviateRetriever()
        result = retriever.retrieve("test query")
        self.assertEqual(result, ["Doc 1", "Doc 2"])

    @patch("main.weaviate.Client", side_effect=Exception("Connection Error"))
    def test_init_failure(self, mock_client):
        with self.assertRaises(Exception):
            WeaviateRetriever()


class TestReranker(unittest.TestCase):

    @patch("main.CrossEncoder")
    def test_rerank_success(self, mock_encoder):
        mock_encoder().predict.return_value = [0.5, 0.8, 0.2]
        reranker = Reranker()
        result = reranker.rerank("query", ["a", "b", "c"], top_k=2)
        self.assertEqual(result, ["b", "a"])

    @patch("main.CrossEncoder")
    def test_rerank_empty_chunks(self, mock_encoder):
        reranker = Reranker()
        result = reranker.rerank("query", [])
        self.assertEqual(result, [])


class TestLlamaQuerier(unittest.TestCase):

    @patch("main.requests.post")
    def test_query_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": "Test answer"}
        querier = LlamaQuerier()
        result = querier.query("What is Python?")
        self.assertEqual(result, "Test answer")

    @patch("main.requests.post", side_effect=Exception("Connection Error"))
    def test_query_failure(self, mock_post):
        querier = LlamaQuerier()
        result = querier.query("Hello?")
        self.assertTrue("Error querying" in result)


class TestQAApp(unittest.TestCase):

    @patch("main.open", new_callable=mock_open, read_data='[{"question": "Q", "answer": "A"}]')
    def test_load_history_success(self, mock_file):
        app = QAApp()
        self.assertEqual(app.history, [{"question": "Q", "answer": "A"}])

    @patch("main.open", side_effect=FileNotFoundError())
    def test_load_history_file_not_found(self, mock_file):
        app = QAApp()
        self.assertEqual(app.history, [])

    @patch("main.open", new_callable=mock_open)
    def test_save_history_success(self, mock_file):
        app = QAApp()
        app.history = [{"question": "Q", "answer": "A"}]
        app.save_history()
        mock_file.assert_called_with("data/history.json", "w", encoding="utf-8")
        handle = mock_file()
        handle.write.assert_called()

    def test_format_prompt(self):
        app = QAApp()
        prompt = app._format_prompt("Who built Taj Mahal?", ["Taj Mahal was built by Shah Jahan."])
        self.assertIn("Taj Mahal was built by Shah Jahan.", prompt)
        self.assertIn("Who built Taj Mahal?", prompt)


if __name__ == '__main__':
    unittest.main()


