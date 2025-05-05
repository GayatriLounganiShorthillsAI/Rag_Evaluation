import os
import pytest
import logging
import json
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Import the classes from main.py
from main import WeaviateRetriever, Reranker, LlamaQuerier, QAApp

# Configure test-specific logger
log_file = Path(__file__).parent / "logs" / "test_main.log"
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

class TestWeaviateRetriever:
    @pytest.fixture
    def mock_weaviate_client(self):
        with patch("main.weaviate.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_embed_model(self):
        with patch("main.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            yield mock_instance
    
    def test_init(self, mock_weaviate_client, mock_embed_model, setup_logging):
        logger.info("Testing WeaviateRetriever initialization")
        
        # Test successful initialization
        retriever = WeaviateRetriever(weaviate_url="http://testurl:8080", embedding_model="test-model")
        assert retriever.client == mock_weaviate_client
        assert retriever.embed_model == mock_embed_model
        
        # Test exception handling - we need to make the constructor throw an exception
        with patch("main.weaviate.Client", side_effect=Exception("Connection error")):
            with pytest.raises(Exception):
                WeaviateRetriever(weaviate_url="http://testurl:8080", embedding_model="test-model")
        
        logger.info("WeaviateRetriever initialization test completed")
    
    def test_retrieve(self, mock_weaviate_client, mock_embed_model, setup_logging):
        logger.info("Testing WeaviateRetriever.retrieve method")
        
        # Setup mock returns
        mock_embed_model.encode.return_value = [[0.1, 0.2, 0.3]]
        
        mock_query = MagicMock()
        mock_with_near_vector = MagicMock()
        mock_with_limit = MagicMock()
        mock_do = MagicMock()
        
        mock_query.get.return_value = mock_with_near_vector
        mock_with_near_vector.with_near_vector.return_value = mock_with_limit
        mock_with_limit.with_limit.return_value = mock_do
        mock_do.do.return_value = {
            "data": {
                "Get": {
                    "Documents": [
                        {"text": "Result 1"},
                        {"text": "Result 2"}
                    ]
                }
            }
        }
        
        mock_weaviate_client.query = mock_query
        
        # Test successful retrieval
        retriever = WeaviateRetriever()
        results = retriever.retrieve("test query", top_k=2)
        
        assert len(results) == 2
        assert results[0] == "Result 1"
        assert results[1] == "Result 2"
        
        # Test exception handling
        mock_embed_model.encode.side_effect = Exception("Encoding error")
        results = retriever.retrieve("failing query")
        assert results == []
        
        logger.info("WeaviateRetriever.retrieve test completed")

class TestReranker:
    @pytest.fixture
    def mock_cross_encoder(self):
        with patch("main.CrossEncoder") as mock_encoder:
            mock_instance = MagicMock()
            mock_encoder.return_value = mock_instance
            yield mock_instance
    
    def test_init(self, mock_cross_encoder, setup_logging):
        logger.info("Testing Reranker initialization")
        
        # Test successful initialization
        reranker = Reranker(model_name="test-model")
        assert reranker.model == mock_cross_encoder
        
        # Test exception handling - we need to make the constructor throw an exception
        with patch("main.CrossEncoder", side_effect=Exception("Model load error")):
            with pytest.raises(Exception):
                Reranker(model_name="test-model")
        
        logger.info("Reranker initialization test completed")
    
    def test_rerank(self, mock_cross_encoder, setup_logging):
        logger.info("Testing Reranker.rerank method")
        
        # Setup mock returns
        mock_cross_encoder.predict.return_value = [0.9, 0.1, 0.5]
        
        # Test successful reranking
        reranker = Reranker()
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        results = reranker.rerank("test query", chunks, top_k=2)
        
        assert len(results) == 2
        assert results[0] == "Chunk 1"  # Should be first due to highest score
        
        # Test empty chunks
        results = reranker.rerank("test query", [])
        assert results == []
        
        # Test exception handling
        mock_cross_encoder.predict.side_effect = Exception("Prediction error")
        results = reranker.rerank("test query", chunks)
        assert results == []
        
        logger.info("Reranker.rerank test completed")

class TestLlamaQuerier:
    @pytest.fixture
    def mock_requests(self):
        with patch("main.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Test response"}
            mock_post.return_value = mock_response
            yield mock_post, mock_response
    
    def test_init(self, setup_logging):
        logger.info("Testing LlamaQuerier initialization")
        
        querier = LlamaQuerier(model="test-model", url="http://test-url")
        assert querier.model == "test-model"
        assert querier.url == "http://test-url"
        
        logger.info("LlamaQuerier initialization test completed")
    
    def test_query(self, mock_requests, setup_logging):
        logger.info("Testing LlamaQuerier.query method")
        
        mock_post, mock_response = mock_requests
        
        # Test successful query
        querier = LlamaQuerier()
        response = querier.query("Test prompt")
        assert response == "Test response"
        
        # Test error handling
        mock_post.side_effect = Exception("Request error")
        response = querier.query("Test prompt")
        assert "Error querying" in response
        
        logger.info("LlamaQuerier.query test completed")

class TestQAApp:
    @pytest.fixture
    def mock_components(self):
        with patch("main.WeaviateRetriever") as mock_retriever, \
             patch("main.Reranker") as mock_reranker, \
             patch("main.LlamaQuerier") as mock_llm:
            
            mock_retriever_instance = MagicMock()
            mock_reranker_instance = MagicMock()
            mock_llm_instance = MagicMock()
            
            mock_retriever.return_value = mock_retriever_instance
            mock_reranker.return_value = mock_reranker_instance
            mock_llm.return_value = mock_llm_instance
            
            yield mock_retriever_instance, mock_reranker_instance, mock_llm_instance
    
    @pytest.fixture
    def mock_json_file(self, tmp_path):
        # Create a temporary JSON file for testing
        history_file = tmp_path / "history.json"
        history_data = [
            {"question": "Test question 1", "answer": "Test answer 1"},
            {"question": "Test question 2", "answer": "Test answer 2"}
        ]
        
        with open(history_file, "w") as f:
            json.dump(history_data, f)
        
        # Mock the HISTORY_FILE environment variable
        with patch.dict("os.environ", {"HISTORY_FILE": str(history_file)}):
            yield history_file, history_data
    
    def test_init_and_load_history(self, mock_components, mock_json_file, setup_logging):
        logger.info("Testing QAApp initialization and load_history")
        
        history_file, history_data = mock_json_file
        
        # Test successful initialization
        with patch.object(QAApp, "load_history", return_value=history_data) as mock_load:
            app = QAApp()
            assert app.retriever == mock_components[0]
            assert app.reranker == mock_components[1]
            assert app.llm == mock_components[2]
            assert app.history == history_data
            mock_load.assert_called_once()
        
        # Test with missing history file - need to mock load_history again
        os.remove(history_file)
        with patch.object(QAApp, "load_history", return_value=[]) as mock_load:
            app = QAApp()
            assert app.history == []
            mock_load.assert_called_once()
        
        logger.info("QAApp initialization and load_history test completed")
    
    def test_save_history(self, mock_components, mock_json_file, setup_logging):
        logger.info("Testing QAApp.save_history method")
        
        history_file, _ = mock_json_file
        
        # Create a directory for the history file if it doesn't exist
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        # Patch the HISTORY_FILE constant directly in the main module
        with patch("main.HISTORY_FILE", str(history_file)):
            # Create app and patch its load_history to return empty list
            with patch.object(QAApp, "load_history", return_value=[]):
                app = QAApp()
                app.history = [{"question": "New Q", "answer": "New A"}]
                
                # Test save_history method with mock_open
                # But capture what's written to verify
                written_data = ""
                
                def side_effect(data):
                    nonlocal written_data
                    written_data += data
                
                mock_file = mock_open()
                mock_file.return_value.write.side_effect = side_effect
                
                with patch("builtins.open", mock_file):
                    app.save_history()
                    
                    # Verify the file was opened correctly
                    mock_file.assert_called_once_with(str(history_file), "w", encoding="utf-8")
                    
                    # Verify something was written - json.dump makes multiple writes
                    assert mock_file().write.call_count > 0
                    
                    # Verify the content matches what we expect
                    import json
                    expected_json = json.dumps([{"question": "New Q", "answer": "New A"}], indent=2)
                    assert written_data.strip() == expected_json.strip()
            
            # Test error handling by making the file read-only
            os.chmod(history_file, 0o444)  # Make file read-only
            with patch.object(QAApp, "load_history", return_value=[]):
                app = QAApp()
                app.history = [{"question": "New Q", "answer": "New A"}]
                # Should handle exception and not raise it
                app.save_history()
                
            # Restore permissions for cleanup
            os.chmod(history_file, 0o644)
        
        logger.info("QAApp.save_history test completed")
    
    def test_format_prompt(self, mock_components, setup_logging):
        logger.info("Testing QAApp._format_prompt method")
        
        app = QAApp()
        context_chunks = ["Context 1", "Context 2"]
        formatted = app._format_prompt("Test question", context_chunks)
        
        assert "Test question" in formatted
        assert "Context 1\nContext 2" in formatted
        
        logger.info("QAApp._format_prompt test completed") 