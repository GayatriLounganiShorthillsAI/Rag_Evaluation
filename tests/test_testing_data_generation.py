import os
import csv
import pytest
import logging
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Import the class from modules/testing_data_generation.py
from modules.testing_data_generation import RAGQuestionAnswering

# Configure test-specific logger
log_file = Path(__file__).parent / "logs" / "test_testing_data_generation.log"
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

class TestRAGQuestionAnswering:
    @pytest.fixture
    def mock_weaviate_client(self):
        with patch("modules.testing_data_generation.weaviate.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            yield mock_client, mock_instance
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch("modules.testing_data_generation.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            yield mock_model, mock_instance
    
    @pytest.fixture
    def mock_requests(self):
        with patch("modules.testing_data_generation.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Test response"}
            mock_post.return_value = mock_response
            yield mock_post, mock_response
    
    def test_init(self, mock_weaviate_client, mock_sentence_transformer, setup_logging):
        logger.info("Testing RAGQuestionAnswering initialization")
        
        # Get the mock constructors and instances
        mock_client_constructor, mock_client_instance = mock_weaviate_client
        mock_model_constructor, mock_model_instance = mock_sentence_transformer
        
        # Test successful initialization
        rag_qa = RAGQuestionAnswering()
        
        assert rag_qa.client == mock_client_instance
        assert rag_qa.embed_model == mock_model_instance
        
        # Test with exception in weaviate client
        # Save original side effect
        original_client_side_effect = mock_client_constructor.side_effect
        try:
            mock_client_constructor.side_effect = Exception("Connection error")
            with pytest.raises(Exception):
                RAGQuestionAnswering()
        finally:
            # Restore original side effect
            mock_client_constructor.side_effect = original_client_side_effect
        
        logger.info("RAGQuestionAnswering initialization test completed")
    
    def test_search_weaviate(self, mock_weaviate_client, mock_sentence_transformer, setup_logging):
        logger.info("Testing RAGQuestionAnswering.search_weaviate method")
        
        # Get the mock instances
        _, mock_client_instance = mock_weaviate_client
        _, mock_model_instance = mock_sentence_transformer
        
        # Setup mock returns for the encoder
        # Use a list with a numpy array for the encoder return
        mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Create a new instance of RAGQuestionAnswering to avoid cross-test issues
        rag_qa = RAGQuestionAnswering()
        
        # Replace the client with our mock for better control
        rag_qa.client = mock_client_instance
        
        # Setup the mock query chain correctly
        mock_query = MagicMock()
        mock_client_instance.query = mock_query
        
        mock_get = MagicMock()
        mock_query.get.return_value = mock_get
        
        mock_with_near_vector = MagicMock()
        mock_get.with_near_vector.return_value = mock_with_near_vector
        
        mock_with_limit = MagicMock()
        mock_with_near_vector.with_limit.return_value = mock_with_limit
        
        # This is the result that will be returned when do() is called
        mock_result = {
            "data": {
                "Get": {
                    "Documents": [
                        {"text": "Result 1"},
                        {"text": "Result 2"}
                    ]
                }
            }
        }
        mock_with_limit.do.return_value = mock_result
        
        # Test successful search
        results = rag_qa.search_weaviate("test query", top_k=2)
        
        # Verify results
        assert len(results) == 2
        assert results[0] == "Result 1"
        assert results[1] == "Result 2"
        
        # Verify the query chain calls
        mock_query.get.assert_called_once()
        mock_get.with_near_vector.assert_called_once()
        mock_with_near_vector.with_limit.assert_called_once_with(2)
        mock_with_limit.do.assert_called_once()
        
        # Test embedding exception
        mock_model_instance.encode.side_effect = Exception("Encoding error")
        results = rag_qa.search_weaviate("test query")
        assert results == []
        
        logger.info("RAGQuestionAnswering.search_weaviate test completed")
    
    def test_query_llama(self, mock_weaviate_client, mock_sentence_transformer, mock_requests, setup_logging):
        logger.info("Testing RAGQuestionAnswering.query_llama method")
        
        mock_post, mock_response = mock_requests
        
        # Test successful query
        rag_qa = RAGQuestionAnswering()
        response = rag_qa.query_llama("Test prompt")
        
        assert response == "Test response"
        mock_post.assert_called_once()
        
        # Verify request payload
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "llama3"
        assert payload["prompt"] == "Test prompt"
        assert payload["stream"] is False
        
        # Test with HTTP error
        mock_post.side_effect = Exception("HTTP error")
        response = rag_qa.query_llama("Test prompt")
        assert "Error querying" in response
        
        logger.info("RAGQuestionAnswering.query_llama test completed")
    
    def test_question_fetch(self, mock_weaviate_client, mock_sentence_transformer, setup_logging):
        logger.info("Testing RAGQuestionAnswering.question_fetch method")
        
        # Test with correctly formatted CSV file
        csv_content = "Question,Answer\nTest question 1,Test answer 1\nTest question 2,Test answer 2"
        
        with patch("builtins.open", mock_open(read_data=csv_content)):
            with patch("csv.reader") as mock_csv_reader:
                mock_csv_reader.return_value = iter([
                    ["Question", "Answer"],
                    ["Test question 1", "Test answer 1"],
                    ["Test question 2", "Test answer 2"]
                ])
                
                rag_qa = RAGQuestionAnswering()
                results = list(rag_qa.question_fetch("test.csv"))
                
                assert len(results) == 2
                assert results[0] == {"Question": "Test question 1", "Answer": "Test answer 1"}
                assert results[1] == {"Question": "Test question 2", "Answer": "Test answer 2"}
        
        # Test with file error
        with patch("builtins.open", side_effect=Exception("File error")):
            rag_qa = RAGQuestionAnswering()
            with pytest.raises(Exception):
                list(rag_qa.question_fetch("nonexistent.csv"))
        
        logger.info("RAGQuestionAnswering.question_fetch test completed")
    
    def test_prepare_workbook(self, mock_weaviate_client, mock_sentence_transformer, setup_logging):
        logger.info("Testing RAGQuestionAnswering.prepare_workbook method")
        
        # Test with existing file
        with patch("os.path.exists") as mock_exists, \
             patch("modules.testing_data_generation.load_workbook") as mock_load, \
             patch("modules.testing_data_generation.Workbook") as mock_new_wb:
            
            mock_exists.return_value = True
            mock_wb = MagicMock()
            mock_ws = MagicMock()
            mock_wb.active = mock_ws
            mock_load.return_value = mock_wb
            
            rag_qa = RAGQuestionAnswering()
            result_wb, result_ws = rag_qa.prepare_workbook("existing.xlsx")
            
            assert result_wb == mock_wb
            assert result_ws == mock_ws
            mock_load.assert_called_once_with("existing.xlsx")
            mock_new_wb.assert_not_called()
        
        # Test with new file
        with patch("os.path.exists") as mock_exists, \
             patch("modules.testing_data_generation.load_workbook") as mock_load, \
             patch("modules.testing_data_generation.Workbook") as mock_new_wb:
            
            mock_exists.return_value = False
            mock_wb = MagicMock()
            mock_ws = MagicMock()
            mock_wb.active = mock_ws
            mock_new_wb.return_value = mock_wb
            
            rag_qa = RAGQuestionAnswering()
            result_wb, result_ws = rag_qa.prepare_workbook("new.xlsx")
            
            assert result_wb == mock_wb
            assert result_ws == mock_ws
            mock_new_wb.assert_called_once()
            mock_ws.append.assert_called_once_with(
                ["Question", "Answer", "Predicted_Answer", "Context"]
            )
        
        logger.info("RAGQuestionAnswering.prepare_workbook test completed")
    
    def test_run(self, mock_weaviate_client, mock_sentence_transformer, mock_requests, setup_logging):
        logger.info("Testing RAGQuestionAnswering.run method")
        
        rag_qa = RAGQuestionAnswering()
        
        # Mock all the component methods
        with patch.object(rag_qa, "prepare_workbook") as mock_prepare, \
             patch.object(rag_qa, "question_fetch") as mock_fetch, \
             patch.object(rag_qa, "search_weaviate") as mock_search, \
             patch.object(rag_qa, "query_llama") as mock_query:
            
            # Setup return values
            mock_wb = MagicMock()
            mock_ws = MagicMock()
            mock_prepare.return_value = (mock_wb, mock_ws)
            
            mock_fetch.return_value = [
                {"Question": "Test question 1", "Answer": "Test answer 1"},
                {"Question": "Test question 2", "Answer": "Test answer 2"},
                {"Question": "", "Answer": "Empty question"}  # Should be skipped
            ]
            
            mock_search.return_value = ["Context chunk 1", "Context chunk 2"]
            mock_query.return_value = "Predicted answer"
            
            # Run the method
            rag_qa.run()
            
            # Verify method calls
            mock_prepare.assert_called_once()
            assert mock_fetch.call_count == 1
            assert mock_search.call_count == 2  # Called for each non-empty question
            assert mock_query.call_count == 2
            
            # Verify workbook handling
            assert mock_ws.append.call_count == 2
            mock_wb.save.assert_called_with(os.getenv("OUTPUT_XLSX", "data/qa_with_predictions.xlsx"))
        
        logger.info("RAGQuestionAnswering.run test completed") 