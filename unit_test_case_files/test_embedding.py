import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.embedding import ChunkUploader

class TestChunkUploader(unittest.TestCase):

    @patch("modules.embedding.SentenceTransformer")
    @patch("builtins.open", new_callable=mock_open, read_data="This is a test sentence. Another sentence.")
    @patch("os.path.exists", return_value=True)
    def test_load_text(self, mock_exists, mock_open_file, mock_sentence_transformer):
        mock_sentence_transformer.return_value = MagicMock()
        uploader = ChunkUploader("dummy_path.txt")
        result = uploader.load_text()
        self.assertEqual(result, "This is a test sentence. Another sentence.")

    @patch("modules.embedding.SentenceTransformer")
    def test_split_text_semantic(self, mock_sentence_transformer):
        mock_sentence_transformer.return_value = MagicMock()
        uploader = ChunkUploader("dummy.txt")
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = uploader.split_text_semantic(text, chunk_size=40, chunk_overlap=0)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

    @patch("modules.embedding.SentenceTransformer")
    @patch("weaviate.Client")
    def test_create_weaviate_schema(self, mock_client_class, mock_sentence_transformer):
        mock_sentence_transformer.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        uploader = ChunkUploader("dummy.txt")
        uploader.create_weaviate_schema()

        mock_client.schema.delete_all.assert_called_once()
        mock_client.schema.create_class.assert_called_once()

    @patch("modules.embedding.SentenceTransformer")
    @patch("weaviate.Client")
    def test_insert_chunks(self, mock_weaviate_client, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]] * 2
        mock_sentence_transformer.return_value = mock_model

        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        uploader = ChunkUploader("dummy.txt")
        chunks = ["chunk 1", "chunk 2"]
        uploader.insert_chunks(chunks)

        self.assertEqual(mock_client.data_object.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
