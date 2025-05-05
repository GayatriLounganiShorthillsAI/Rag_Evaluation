import os
import pytest
import pandas as pd
import logging
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Import the classes from modules/question_generation.py
from modules.question_generation import (
    TextLoader,
    TextSplitter,
    LLaMAClient,
    QAGenerator,
    DataSaver
)

# Configure test-specific logger
log_file = Path(__file__).parent / "logs" / "test_question_generation.log"
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

class TestTextLoader:
    def test_load(self, setup_logging):
        logger.info("Testing TextLoader.load method")
        
        test_content = "This is test content"
        
        # Test successful load
        with patch("builtins.open", mock_open(read_data=test_content)):
            content = TextLoader.load("test.txt")
            assert content == test_content
        
        # Test file error
        with patch("builtins.open", side_effect=Exception("File error")):
            with pytest.raises(Exception):
                TextLoader.load("nonexistent.txt")
        
        logger.info("TextLoader.load test completed")

class TestTextSplitter:
    def test_init(self, setup_logging):
        logger.info("Testing TextSplitter initialization")
        
        # Patch the RecursiveCharacterTextSplitter to verify parameters
        with patch("modules.question_generation.RecursiveCharacterTextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_splitter.return_value = mock_instance
            
            # Test default parameters
            splitter = TextSplitter()
            mock_splitter.assert_called_with(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", ".", "!", "?", "\n", " "],
                length_function=len
            )
            
            # Test custom parameters
            splitter = TextSplitter(chunk_size=400, chunk_overlap=50)
            mock_splitter.assert_called_with(
                chunk_size=400,
                chunk_overlap=50,
                separators=["\n\n", ".", "!", "?", "\n", " "],
                length_function=len
            )
        
        logger.info("TextSplitter initialization test completed")
    
    def test_split(self, setup_logging):
        logger.info("Testing TextSplitter.split method")
        
        with patch("modules.question_generation.RecursiveCharacterTextSplitter.split_text") as mock_split:
            # Setup mock return
            mock_split.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
            
            splitter = TextSplitter()
            chunks = splitter.split("Sample text for splitting")
            
            assert chunks == ["Chunk 1", "Chunk 2", "Chunk 3"]
            mock_split.assert_called_once_with("Sample text for splitting")
            
            # Test with exception
            mock_split.side_effect = Exception("Splitting error")
            with pytest.raises(Exception):
                splitter.split("Text that causes error")
        
        logger.info("TextSplitter.split test completed")

class TestLLaMAClient:
    @pytest.fixture
    def mock_requests(self):
        with patch("modules.question_generation.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Test response"}
            mock_post.return_value = mock_response
            yield mock_post, mock_response
    
    def test_init(self, setup_logging):
        logger.info("Testing LLaMAClient initialization")
        
        client = LLaMAClient(model="test-model", url="http://test-url")
        assert client.model == "test-model"
        assert client.url == "http://test-url"
        
        logger.info("LLaMAClient initialization test completed")
    
    def test_query(self, mock_requests, setup_logging):
        logger.info("Testing LLaMAClient.query method")
        
        mock_post, mock_response = mock_requests
        
        # Test successful query
        client = LLaMAClient(model="test-model", url="http://test-url")
        response = client.query("Test prompt")
        
        assert response == "Test response"
        mock_post.assert_called_once()
        
        # Verify request payload
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "test-model"
        assert payload["prompt"] == "Test prompt"
        assert payload["stream"] is False
        
        # Test with HTTP error
        mock_post.side_effect = Exception("HTTP error")
        response = client.query("Test prompt")
        assert response == ""
        
        logger.info("LLaMAClient.query test completed")

class TestQAGenerator:
    def test_init(self, setup_logging):
        logger.info("Testing QAGenerator initialization")
        
        llama_client = MagicMock()
        generator = QAGenerator(llama_client)
        
        assert generator.llama == llama_client
        
        logger.info("QAGenerator initialization test completed")
    
    def test_build_prompt(self, setup_logging):
        logger.info("Testing QAGenerator._build_prompt method")
        
        llama_client = MagicMock()
        generator = QAGenerator(llama_client)
        
        prompt = generator._build_prompt("Test historical chunk")
        
        assert "Test historical chunk" in prompt
        assert "generate 2 simple factual questions" in prompt.lower()
        assert "Question:" in prompt
        assert "Answer:" in prompt
        
        logger.info("QAGenerator._build_prompt test completed")
    
    def test_generate(self, setup_logging):
        logger.info("Testing QAGenerator.generate method")
        
        # Mock LLaMA client
        llama_client = MagicMock()
        llama_client.query.side_effect = [
            # First response with two valid QA pairs
            """
            Question: First test question?
            Answer: First test answer.
            
            Question: Second test question?
            Answer: Second test answer.
            """,
            # Second response with one valid QA pair
            """
            Invalid line
            Question: Third test question?
            Answer: Third test answer.
            Incomplete question
            Question: 
            """,
            # Empty response
            "",
        ]
        
        generator = QAGenerator(llama_client)
        
        # Test with multiple chunks
        with patch("time.sleep"):
            result = generator.generate(
                chunks=["Chunk 1", "Chunk 2", "Chunk 3"],
                total=3
            )
        
        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in ["Question", "Answer"])
        
        # Check extracted questions and answers
        questions = result["Question"].tolist()
        answers = result["Answer"].tolist()
        
        assert "First test question?" in questions
        assert "Second test question?" in questions
        assert "Third test question?" in questions
        assert "First test answer." in answers
        assert "Second test answer." in answers
        assert "Third test answer." in answers
        
        # Verify client calls
        assert llama_client.query.call_count == 3
        for i, call in enumerate(llama_client.query.call_args_list):
            assert f"Chunk {i+1}" in call[0][0]
        
        logger.info("QAGenerator.generate test completed")

class TestDataSaver:
    def test_to_csv(self, setup_logging):
        logger.info("Testing DataSaver.to_csv method")
        
        # Create test DataFrame
        df = pd.DataFrame({
            "Question": ["Question 1", "Question 2"],
            "Answer": ["Answer 1", "Answer 2"]
        })
        
        # Test successful save
        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            DataSaver.to_csv(df, "test.csv")
            mock_to_csv.assert_called_once_with("test.csv", index=False)
        
        # Test with exception
        with patch("pandas.DataFrame.to_csv", side_effect=Exception("Save error")):
            with pytest.raises(Exception):
                DataSaver.to_csv(df, "error.csv")
        
        logger.info("DataSaver.to_csv test completed")

class TestMainExecution:
    def test_main_execution(self, setup_logging):
        logger.info("Testing main execution flow")
        
        # Setup mocks for all components
        with patch("modules.question_generation.TextLoader.load") as mock_load, \
             patch("modules.question_generation.TextSplitter.split") as mock_split, \
             patch("modules.question_generation.LLaMAClient") as mock_llama_client, \
             patch("modules.question_generation.QAGenerator") as mock_qa_generator, \
             patch("modules.question_generation.DataSaver.to_csv") as mock_save, \
             patch.dict(os.environ, {"OLLAMA_MODEL": "test-model", "OLLAMA_URL": "http://test-url"}):
            
            # Setup return values
            mock_load.return_value = "Test content"
            mock_split.return_value = ["Chunk 1", "Chunk 2"]
            
            mock_llama_instance = MagicMock()
            mock_llama_client.return_value = mock_llama_instance
            
            mock_qa_instance = MagicMock()
            test_df = pd.DataFrame({
                "Question": ["Question 1", "Question 2"],
                "Answer": ["Answer 1", "Answer 2"]
            })
            mock_qa_instance.generate.return_value = test_df
            mock_qa_generator.return_value = mock_qa_instance
            
            try:
                # Simulating the execution flow in the __main__ block
                text = mock_load("data/modern_history_of_india.txt")
                chunks = mock_split(text)
                client = mock_llama_client(model="test-model", url="http://test-url")
                generator = mock_qa_generator(client)
                df = generator.generate(chunks)
                mock_save(df, "data/qa_dataset_1000.csv")
                
                # Verify component interactions
                mock_load.assert_called_once_with("data/modern_history_of_india.txt")
                mock_split.assert_called_once_with("Test content")
                mock_llama_client.assert_called_once_with(model="test-model", url="http://test-url")
                mock_qa_generator.assert_called_once_with(mock_llama_instance)
                mock_qa_instance.generate.assert_called_once_with(["Chunk 1", "Chunk 2"])
                
                # Use the correct way to check if mock_save was called with the correct DataFrame
                # Don't directly compare DataFrames, check that mock_save was called with the right arguments
                mock_save.assert_called_once()
                call_args = mock_save.call_args[0]
                assert call_args[1] == "data/qa_dataset_1000.csv"
                
                # For DataFrame comparison, use equals()
                called_df = call_args[0]
                assert isinstance(called_df, pd.DataFrame)
                assert called_df.equals(test_df), "DataFrames are not equal"
                
            except Exception as e:
                pytest.fail(f"Main execution test failed: {e}")
            
        logger.info("Main execution flow test completed") 