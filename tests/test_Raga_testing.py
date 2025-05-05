import os
import json
import pytest
import pandas as pd
import logging
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Import the class from modules/Raga_testing.py
from modules.Raga_testing import RAGEvaluator

# Configure test-specific logger
log_file = Path(__file__).parent / "logs" / "test_Raga_testing.log"
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

class TestRAGEvaluator:
    @pytest.fixture
    def mock_genai(self):
        with patch("modules.Raga_testing.genai") as mock:
            mock_generative_model = MagicMock()
            mock.GenerativeModel.return_value = mock_generative_model
            yield mock, mock_generative_model
    
    @pytest.fixture
    def mock_env(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            yield
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            "Question": ["Question 1", "Question 2"],
            "Answer": ["Answer 1", "Answer 2"],
            "Predicted_Answer": ["Predicted 1", "Predicted 2"],
            "Context": ["Context 1", "Context 2"]
        })
    
    def test_init(self, mock_genai, mock_env, setup_logging):
        logger.info("Testing RAGEvaluator initialization")
        
        # Test successful initialization
        evaluator = RAGEvaluator(
            excel_path="test.xlsx",
            output_xlsx="output.xlsx",
            output_json="output.json"
        )
        
        assert evaluator.excel_path == "test.xlsx"
        assert evaluator.output_xlsx == "output.xlsx"
        assert evaluator.output_json == "output.json"
        assert evaluator.batch_size == 15
        assert evaluator.sleep_seconds == 60
        
        # Test model configuration
        mock_genai[0].configure.assert_called_once_with(api_key="fake_key")
        mock_genai[0].GenerativeModel.assert_called_once_with("models/gemini-2.0-flash")
        
        # Test with missing API key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing API Key"):
                RAGEvaluator("test.xlsx", "output.xlsx", "output.json")
        
        logger.info("RAGEvaluator initialization test completed")
    
    def test_build_prompt(self, mock_genai, mock_env, setup_logging):
        logger.info("Testing RAGEvaluator.build_prompt method")
        
        evaluator = RAGEvaluator("test.xlsx", "output.xlsx", "output.json")
        prompt = evaluator.build_prompt(
            question="Test question",
            answer="Test answer",
            predicted_answer="Test predicted",
            context="Test context"
        )
        
        # Check key elements in the prompt
        assert "Test question" in prompt
        assert "Test answer" in prompt
        assert "Test predicted" in prompt
        assert "Test context" in prompt
        assert "Faithfulness" in prompt
        assert "Answer Relevancy" in prompt
        assert "Context Precision" in prompt
        assert "Context Recall" in prompt
        
        logger.info("RAGEvaluator.build_prompt test completed")
    
    def test_extract_scores(self, mock_genai, mock_env, setup_logging):
        logger.info("Testing RAGEvaluator.extract_scores method")
        
        evaluator = RAGEvaluator("test.xlsx", "output.xlsx", "output.json")
        
        # Test with valid response text
        scores = evaluator.extract_scores("[0.95521, 0.85312, 0.91230, 0.85428]")
        assert scores == [0.95521, 0.85312, 0.91230, 0.85428]
        
        # Test with spaces in the response
        scores = evaluator.extract_scores("  [ 0.95521 , 0.85312 , 0.91230 , 0.85428 ]  ")
        assert scores == [0.95521, 0.85312, 0.91230, 0.85428]
        
        # Test with invalid response
        scores = evaluator.extract_scores("No valid scores")
        assert scores == [None, None, None, None]
        
        logger.info("RAGEvaluator.extract_scores test completed")
    
    def test_evaluate_batch(self, mock_genai, mock_env, sample_data, setup_logging):
        logger.info("Testing RAGEvaluator.evaluate_batch method")
        
        evaluator = RAGEvaluator("test.xlsx", "output.xlsx", "output.json")
        
        # Mock the model generate_content response
        mock_response = MagicMock()
        mock_response.text = "[0.95, 0.85, 0.90, 0.80]"
        evaluator.model.generate_content.return_value = mock_response
        
        # Test successful evaluation
        scores = evaluator.evaluate_batch(sample_data, 0)
        
        assert len(scores) == 2
        assert scores[0] == [0.95, 0.85, 0.90, 0.80]
        assert scores[1] == [0.95, 0.85, 0.90, 0.80]
        
        # Verify generate_content was called with appropriate prompts
        assert evaluator.model.generate_content.call_count == 2
        
        # Test with exception
        evaluator.model.generate_content.side_effect = Exception("API Error")
        scores = evaluator.evaluate_batch(sample_data.iloc[:1], 0)
        assert scores[0] == [None, None, None, None]
        
        logger.info("RAGEvaluator.evaluate_batch test completed")
    
    def test_append_to_excel(self, mock_genai, mock_env, sample_data, setup_logging):
        logger.info("Testing RAGEvaluator.append_to_excel method")
        
        evaluator = RAGEvaluator("test.xlsx", "output.xlsx", "output.json")
        
        # Test with existing file
        with patch("os.path.exists") as mock_exists, \
             patch("pandas.read_excel") as mock_read_excel, \
             patch("pandas.DataFrame.to_excel") as mock_to_excel:
            
            mock_exists.return_value = True
            existing_df = pd.DataFrame({"A": [1, 2]})
            mock_read_excel.return_value = existing_df
            
            evaluator.append_to_excel(sample_data)
            
            mock_exists.assert_called_once_with("output.xlsx")
            mock_read_excel.assert_called_once_with("output.xlsx")
            
            # Check that concat was called with correct data
            args, kwargs = mock_to_excel.call_args
            assert kwargs["index"] is False
        
        # Test with non-existing file
        with patch("os.path.exists") as mock_exists, \
             patch("pandas.DataFrame.to_excel") as mock_to_excel:
            
            mock_exists.return_value = False
            
            evaluator.append_to_excel(sample_data)
            
            mock_exists.assert_called_once_with("output.xlsx")
            mock_to_excel.assert_called_once()
        
        logger.info("RAGEvaluator.append_to_excel test completed")
    
    def test_update_json_scores(self, mock_genai, mock_env, setup_logging):
        logger.info("Testing RAGEvaluator.update_json_scores method")
        
        evaluator = RAGEvaluator("test.xlsx", "output.xlsx", "output.json")
        
        # Mock file operations
        mock_file = mock_open()
        
        # Test with no valid scores
        with patch("builtins.open", mock_file):
            evaluator.update_json_scores([[None, None, None, None]])
            mock_file.assert_not_called()
        
        # Test with valid scores, no existing JSON
        with patch("os.path.exists") as mock_exists, \
             patch("json.dump") as mock_json_dump:
            
            mock_exists.return_value = False
            new_scores = [[0.9, 0.8, 0.7, 0.6], [0.8, 0.7, 0.6, 0.5]]
            
            evaluator.update_json_scores(new_scores)
            
            mock_exists.assert_called_once_with("output.json")
            
            # Check the JSON content that was passed to json.dump
            call_args = mock_json_dump.call_args[0]
            saved_data = call_args[0]
            
            assert saved_data["faithfulness"] == pytest.approx(0.85)
            assert saved_data["answer_relevancy"] == pytest.approx(0.75)
            assert saved_data["context_precision"] == pytest.approx(0.65)
            assert saved_data["context_recall"] == pytest.approx(0.55)
        
        # Test with existing JSON
        with patch("os.path.exists") as mock_exists, \
             patch("builtins.open", mock_open(read_data='{"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.6}')), \
             patch("pandas.read_excel") as mock_read_excel, \
             patch("json.dump") as mock_json_dump:
            
            mock_exists.return_value = True
            mock_read_excel.return_value = pd.DataFrame({"A": range(5)})  # 5 rows
            new_scores = [[0.8, 0.7, 0.6, 0.5]]  # 1 new score
            
            evaluator.update_json_scores(new_scores)
            
            # Should have calculated weighted average with 4 existing + 1 new
            mock_json_dump.assert_called_once()
            args = mock_json_dump.call_args[0]
            assert args[0]["faithfulness"] == pytest.approx((0.9 * 4 + 0.8) / 5)
        
        logger.info("RAGEvaluator.update_json_scores test completed")
    
    def test_run(self, mock_genai, mock_env, sample_data, setup_logging):
        logger.info("Testing RAGEvaluator.run method")
        
        evaluator = RAGEvaluator("test.xlsx", "output.xlsx", "output.json")
        
        with patch("pandas.read_excel") as mock_read_excel, \
             patch("os.path.exists") as mock_exists, \
             patch.object(evaluator, "evaluate_batch") as mock_evaluate, \
             patch.object(evaluator, "append_to_excel") as mock_append, \
             patch.object(evaluator, "update_json_scores") as mock_update, \
             patch("time.sleep") as mock_sleep:
            
            mock_read_excel.return_value = sample_data
            mock_exists.return_value = False
            mock_evaluate.return_value = [[0.9, 0.8, 0.7, 0.6], [0.8, 0.7, 0.6, 0.5]]
            
            evaluator.run()
            
            # Check that all methods were called correctly
            mock_read_excel.assert_called_once_with("test.xlsx")
            mock_exists.assert_called_once_with("output.xlsx")
            mock_evaluate.assert_called_once()
            mock_append.assert_called_once()
            mock_update.assert_called_once()
            mock_sleep.assert_called_once_with(60)
        
        # Test resuming from existing file
        with patch("pandas.read_excel") as mock_read_excel, \
             patch("os.path.exists") as mock_exists, \
             patch.object(evaluator, "evaluate_batch") as mock_evaluate, \
             patch.object(evaluator, "append_to_excel") as mock_append, \
             patch.object(evaluator, "update_json_scores") as mock_update, \
             patch("time.sleep") as mock_sleep:
            
            # Main data has 5 rows, already processed 2
            main_df = pd.DataFrame({"A": range(5)})
            existing_df = pd.DataFrame({"A": range(2)})
            
            mock_read_excel.side_effect = [main_df, existing_df]
            mock_exists.return_value = True
            
            evaluator.run()
            
            # Should process remaining 3 rows
            assert mock_read_excel.call_count == 2
            assert mock_evaluate.call_count == 1
            slice_arg = mock_evaluate.call_args[0][0]
            assert len(slice_arg) == 3  # Should be processing 3 rows
        
        logger.info("RAGEvaluator.run test completed") 