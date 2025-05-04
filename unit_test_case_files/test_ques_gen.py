# test_question_generation.py
import unittest
import os
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.question_generation import (
    TextLoader,
    TextSplitter,
    LLaMAClient,
    QAGenerator,
    DataSaver
)

class TestTextLoader(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data="Sample text content.")
    def test_load(self, mock_file):
        content = TextLoader.load("dummy_path.txt")
        self.assertEqual(content, "Sample text content.")
        mock_file.assert_called_with("dummy_path.txt", "r", encoding="utf-8")


class TestTextSplitter(unittest.TestCase):
    def test_split(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        splitter = TextSplitter(chunk_size=30, chunk_overlap=5)
        chunks = splitter.split(text)
        self.assertTrue(len(chunks) > 0)
        self.assertIsInstance(chunks[0], str)


class TestLLaMAClient(unittest.TestCase):
    @patch("modules.question_generation.requests.post")
    def test_query_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Question: What is AI?\nAnswer: Artificial Intelligence"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = LLaMAClient(model="llama3", url="http://localhost:11434/api/generate")
        result = client.query("Sample prompt")
        self.assertIn("Question:", result)


class TestQAGenerator(unittest.TestCase):
    def setUp(self):
        mock_client = MagicMock()
        mock_client.query.return_value = (
            "Question: What is the capital of France?\n"
            "Answer: Paris\n"
            "Question: What is 2+2?\n"
            "Answer: 4"
        )
        self.qa_generator = QAGenerator(mock_client)

    def test_generate(self):
        chunks = ["Some historical text 1", "Some historical text 2"]
        df = self.qa_generator.generate(chunks, total=1)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Question", df.columns)
        self.assertIn("Answer", df.columns)
        self.assertGreater(len(df), 0)


class TestDataSaver(unittest.TestCase):
    @patch("pandas.DataFrame.to_csv")
    def test_to_csv(self, mock_to_csv):
        df = pd.DataFrame({"Question": ["What?"], "Answer": ["This."]})
        DataSaver.to_csv(df, "dummy_output.csv")
        mock_to_csv.assert_called_once_with("dummy_output.csv", index=False)


if __name__ == "__main__":
    unittest.main()
