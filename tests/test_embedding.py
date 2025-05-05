import os
import pytest
import numpy as np
import logging
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Import the class from modules/embedding.py
from modules.embedding import ChunkUploader

# Configure test-specific logger
log_file = Path(__file__).parent / "logs" / "test_embedding.log"
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

class TestChunkUploader:
    @pytest.fixture
    def mock_weaviate_client(self):
        with patch("modules.embedding.weaviate.Client") as mock_client:
            mock_instance = MagicMock()
            # Set up required methods/attributes
            mock_instance.schema = MagicMock()
            mock_instance.data_object = MagicMock()
            mock_client.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch("modules.embedding.SentenceTransformer") as mock_st:
            mock_instance = MagicMock()
            mock_st.return_value = mock_instance
            yield mock_instance
    
    def test_init(self, mock_weaviate_client, mock_sentence_transformer, setup_logging):
        logger.info("Testing ChunkUploader initialization")
        
        # Test successful initialization
        uploader = ChunkUploader(
            text_file="test.txt",
            embedding_model_name="test-model",
            weaviate_url="http://test-url",
            class_name="TestClass"
        )
        
        assert uploader.text_file == "test.txt"
        assert uploader.class_name == "TestClass"
        assert uploader.weaviate_url == "http://test-url"
        assert uploader.client == mock_weaviate_client
        assert uploader.embed_model == mock_sentence_transformer
        
        # Test with weaviate connection error
        with patch("modules.embedding.weaviate.Client", side_effect=Exception("Connection error")):
            with pytest.raises(Exception):
                ChunkUploader("test.txt")
        
        # Test with embedding model error
        with patch("modules.embedding.SentenceTransformer", side_effect=Exception("Model error")):
            with pytest.raises(Exception):
                ChunkUploader("test.txt")
        
        logger.info("ChunkUploader initialization test completed")
    
    def test_load_text(self, mock_weaviate_client, mock_sentence_transformer, setup_logging):
        logger.info("Testing ChunkUploader.load_text method")
        
        uploader = ChunkUploader("test.txt")
        
        # Test with existing file
        test_content = "This is test content"
        with patch("os.path.exists") as mock_exists, \
             patch("builtins.open", mock_open(read_data=test_content)):
            
            mock_exists.return_value = True
            content = uploader.load_text()
            
            assert content == test_content
        
        # Test with non-existent file
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False
            with pytest.raises(FileNotFoundError):
                uploader.load_text()
        
        # Test with file error
        with patch("os.path.exists") as mock_exists, \
             patch("builtins.open", side_effect=Exception("File error")):
            
            mock_exists.return_value = True
            with pytest.raises(Exception):
                uploader.load_text()
        
        logger.info("ChunkUploader.load_text test completed")
    
    def test_split_text_semantic(self, mock_weaviate_client, mock_sentence_transformer, setup_logging):
        logger.info("Testing ChunkUploader.split_text_semantic method")
        
        uploader = ChunkUploader("test.txt")
        
        with patch("modules.embedding.RecursiveCharacterTextSplitter") as mock_splitter_class:
            # Setup mock splitter instance
            mock_splitter_instance = MagicMock()
            mock_splitter_instance.split_text.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
            mock_splitter_class.return_value = mock_splitter_instance
            
            chunks = uploader.split_text_semantic("Sample text for splitting", chunk_size=500, chunk_overlap=50)
            
            assert chunks == ["Chunk 1", "Chunk 2", "Chunk 3"]
            mock_splitter_instance.split_text.assert_called_once_with("Sample text for splitting")
            
            # Check the splitter was created with correct parameters
            mock_splitter_class.assert_called_once()
            call_args = mock_splitter_class.call_args[1]
            assert call_args["chunk_size"] == 500
            assert call_args["chunk_overlap"] == 50
            assert "separators" in call_args
            assert call_args["length_function"] == len
        
        logger.info("ChunkUploader.split_text_semantic test completed")
    
    def test_create_weaviate_schema(self, mock_weaviate_client, mock_sentence_transformer, setup_logging):
        logger.info("Testing ChunkUploader.create_weaviate_schema method")
        
        uploader = ChunkUploader("test.txt", class_name="TestClass")
        
        # Test successful schema creation
        uploader.create_weaviate_schema()
        
        # Verify method calls
        mock_weaviate_client.schema.delete_all.assert_called_once()
        mock_weaviate_client.schema.create_class.assert_called_once()
        
        # Check class object structure
        class_obj = mock_weaviate_client.schema.create_class.call_args[0][0]
        assert class_obj["class"] == "TestClass"
        assert class_obj["vectorizer"] == "none"
        assert len(class_obj["properties"]) == 1
        assert class_obj["properties"][0]["name"] == "text"
        
        # Test with exception
        mock_weaviate_client.schema.delete_all.side_effect = Exception("Schema error")
        with pytest.raises(Exception):
            uploader.create_weaviate_schema()
        
        logger.info("ChunkUploader.create_weaviate_schema test completed")
    
    def test_insert_chunks(self, mock_weaviate_client, mock_sentence_transformer, setup_logging):
        logger.info("Testing ChunkUploader.insert_chunks method")
        
        uploader = ChunkUploader("test.txt", class_name="TestClass")
        
        # Mock embedding computation
        mock_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        mock_sentence_transformer.encode.return_value = mock_embeddings
        
        # Test successful insertion
        chunks = ["Chunk 1", "Chunk 2"]
        uploader.insert_chunks(chunks)
        
        # Verify embedding generation
        mock_sentence_transformer.encode.assert_called_once_with(chunks, show_progress_bar=True)
        
        # Verify insertions
        assert mock_weaviate_client.data_object.create.call_count == 2
        
        # Check first insertion
        first_call = mock_weaviate_client.data_object.create.call_args_list[0]
        assert first_call[1]["data_object"] == {"text": "Chunk 1"}
        assert first_call[1]["class_name"] == "TestClass"
        # Use pytest.approx for floating point comparisons
        np.testing.assert_array_almost_equal(first_call[1]["vector"], [0.1, 0.2, 0.3])
        
        # Check second insertion
        second_call = mock_weaviate_client.data_object.create.call_args_list[1]
        assert second_call[1]["data_object"] == {"text": "Chunk 2"}
        assert second_call[1]["class_name"] == "TestClass"
        np.testing.assert_array_almost_equal(second_call[1]["vector"], [0.4, 0.5, 0.6])
        
        # Test with embedding error
        mock_sentence_transformer.encode.side_effect = Exception("Embedding error")
        with pytest.raises(Exception):
            uploader.insert_chunks(chunks)
        
        # Test with insertion error on second chunk
        mock_sentence_transformer.encode.side_effect = None
        mock_weaviate_client.data_object.create.side_effect = [None, Exception("Insert error")]
        
        # Should handle the exception and continue
        uploader.insert_chunks(chunks)
        
        logger.info("ChunkUploader.insert_chunks test completed")
    
    def test_run(self, mock_weaviate_client, mock_sentence_transformer, setup_logging):
        logger.info("Testing ChunkUploader.run method")
        
        uploader = ChunkUploader("test.txt")
        
        # Mock all component methods
        with patch.object(uploader, "load_text") as mock_load, \
             patch.object(uploader, "split_text_semantic") as mock_split, \
             patch.object(uploader, "create_weaviate_schema") as mock_schema, \
             patch.object(uploader, "insert_chunks") as mock_insert:
            
            # Setup return values
            mock_load.return_value = "Test text content"
            mock_split.return_value = ["Chunk 1", "Chunk 2"]
            
            # Run the method
            uploader.run()
            
            # Verify the method calls
            mock_load.assert_called_once()
            mock_split.assert_called_once_with("Test text content")
            mock_schema.assert_called_once()
            mock_insert.assert_called_once_with(["Chunk 1", "Chunk 2"])
        
        # Test with an error
        with patch.object(uploader, "load_text", side_effect=Exception("Process error")):
            uploader.run()  # Should handle the exception
        
        logger.info("ChunkUploader.run test completed") 