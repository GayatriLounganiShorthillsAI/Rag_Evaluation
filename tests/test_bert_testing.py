import os
import json
import pytest
import numpy as np
import pandas as pd
import logging
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import torch

# Import the class from modules/bert_testing.py
from modules.bert_testing import QAEvaluator

# Configure test-specific logger
log_file = Path(__file__).parent / "logs" / "test_bert_testing.log"
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

class TestQAEvaluator:
    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch("modules.bert_testing.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_rouge_scorer(self):
        with patch("modules.bert_testing.rouge_scorer.RougeScorer") as mock_rouge:
            mock_instance = MagicMock()
            mock_rouge.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_bert_score(self):
        with patch("modules.bert_testing.bert_score") as mock_bert:
            # Configure the mock to return the F1 scores as expected
            mock_bert.return_value = (None, None, torch.tensor([0.75]))
            yield mock_bert
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            "Question": ["Question 1", "Question 2"],
            "Answer": ["Answer 1", "Answer 2"],
            "Predicted_Answer": ["Predicted 1", "Predicted 2"]
        })
    
    def test_init(self, mock_sentence_transformer, mock_rouge_scorer, setup_logging):
        logger.info("Testing QAEvaluator initialization")
        
        # Get the mock instances
        mock_st_instance = mock_sentence_transformer
        mock_rouge_instance = mock_rouge_scorer
        
        # Create evaluator after mocks are set up
        evaluator = QAEvaluator()
        
        # Check that the models were initialized correctly
        assert evaluator.similarity_model == mock_st_instance
        assert evaluator.rouge == mock_rouge_instance
        
        logger.info("QAEvaluator initialization test completed")
    
    def test_cosine_similarity(self, mock_sentence_transformer, mock_rouge_scorer, setup_logging):
        logger.info("Testing QAEvaluator._cosine_similarity method")
        
        # Get the mock instances
        mock_st_instance = mock_sentence_transformer
        mock_rouge_instance = mock_rouge_scorer
        
        evaluator = QAEvaluator()
        
        # Test with simple vectors
        emb1 = np.array([1, 0, 0])
        emb2 = np.array([0, 1, 0])
        similarity = evaluator._cosine_similarity(emb1, emb2)
        assert similarity == 0.0
        
        emb1 = np.array([1, 1, 0])
        emb2 = np.array([1, 1, 0])
        similarity = evaluator._cosine_similarity(emb1, emb2)
        # Use pytest.approx for floating point comparisons
        assert similarity == pytest.approx(1.0)
        
        emb1 = np.array([1, 1, 0])
        emb2 = np.array([1, 0, 0])
        similarity = evaluator._cosine_similarity(emb1, emb2)
        assert similarity == pytest.approx(0.71, abs=0.01)
        
        logger.info("QAEvaluator._cosine_similarity test completed")
    
    def test_get_bert_f1(self, mock_sentence_transformer, mock_rouge_scorer, mock_bert_score, setup_logging):
        logger.info("Testing QAEvaluator._get_bert_f1 method")
        
        # Get the mock instances
        mock_st_instance = mock_sentence_transformer
        mock_rouge_instance = mock_rouge_scorer
        
        evaluator = QAEvaluator()
        
        # Test with mock bert_score
        bert_f1 = evaluator._get_bert_f1("generated text", "reference text")
        assert bert_f1 == 0.75
        
        mock_bert_score.assert_called_once_with(
            ["generated text"], 
            ["reference text"], 
            lang="en", 
            model_type="bert-base-uncased"
        )
        
        logger.info("QAEvaluator._get_bert_f1 test completed")
    
    def test_get_rouge_score(self, mock_sentence_transformer, mock_rouge_scorer, setup_logging):
        logger.info("Testing QAEvaluator._get_rouge_score method")
        
        # Get the mock instances
        mock_st_instance = mock_sentence_transformer
        mock_rouge_instance = mock_rouge_scorer
        
        # Setup mock rouge scorer
        mock_rouge_instance.score.return_value = {'rougeL': MagicMock(fmeasure=0.85)}
        
        evaluator = QAEvaluator()
        rouge_score = evaluator._get_rouge_score("generated text", "reference text")
        
        assert rouge_score == 0.85
        mock_rouge_instance.score.assert_called_once_with("reference text", "generated text")
        
        logger.info("QAEvaluator._get_rouge_score test completed")
    
    def test_calculate_metrics(self, mock_sentence_transformer, mock_rouge_scorer, mock_bert_score, setup_logging):
        logger.info("Testing QAEvaluator.calculate_metrics method")
        
        # Get the mock instances
        mock_st_instance = mock_sentence_transformer
        mock_rouge_instance = mock_rouge_scorer
        
        # Setup mock returns
        mock_st_instance.encode.side_effect = [
            np.array([0.5, 0.5, 0.5]),  # First call for generated
            np.array([0.8, 0.1, 0.1])   # Second call for reference
        ]
        mock_rouge_instance.score.return_value = {'rougeL': MagicMock(fmeasure=0.65)}
        
        # Patch the logging.debug to avoid format string issues with MagicMock
        with patch("logging.debug"):
            evaluator = QAEvaluator()
            
            # Monkey patch cosine_similarity to return a fixed value
            original_cosine = evaluator._cosine_similarity
            evaluator._cosine_similarity = MagicMock(return_value=0.7)
            
            # Patch the _get_bert_f1 method to return a fixed value
            with patch.object(evaluator, '_get_bert_f1', return_value=0.75):
                metrics = evaluator.calculate_metrics("generated text", "reference text")
            
            # Expected final score based on weights from the class
            expected_final = 0 * 0.65 + 0.4 * 0.7 + 0.6 * 0.75
            
            assert metrics["rouge_score"] == 0.65
            assert metrics["cosine_similarity"] == 0.7
            assert metrics["bert_score_f1"] == 0.75
            assert metrics["final_score"] == pytest.approx(expected_final)
            
            # Restore original method
            evaluator._cosine_similarity = original_cosine
        
        logger.info("QAEvaluator.calculate_metrics test completed")
    
    def test_calculate_grade(self, mock_sentence_transformer, mock_rouge_scorer, setup_logging):
        logger.info("Testing QAEvaluator.calculate_grade method")
        
        # Get the mock instances
        mock_st_instance = mock_sentence_transformer
        mock_rouge_instance = mock_rouge_scorer
        
        evaluator = QAEvaluator()
        
        assert evaluator.calculate_grade(0.95) == "A (Excellent)"
        assert evaluator.calculate_grade(0.85) == "B (Good)"
        assert evaluator.calculate_grade(0.75) == "C (Average)"
        assert evaluator.calculate_grade(0.65) == "D (Below Average)"
        assert evaluator.calculate_grade(0.55) == "F (Poor)"
        
        logger.info("QAEvaluator.calculate_grade test completed")
    
    def test_evaluate_excel(self, mock_sentence_transformer, mock_rouge_scorer, mock_bert_score, sample_data, setup_logging):
        logger.info("Testing QAEvaluator.evaluate_excel method")
        
        # Get the mock instances
        mock_st_instance = mock_sentence_transformer
        mock_rouge_instance = mock_rouge_scorer
        
        evaluator = QAEvaluator()
        
        # Mock all the methods used in evaluate_excel
        with patch.object(evaluator, 'calculate_metrics') as mock_calc_metrics, \
             patch("pandas.read_excel") as mock_read_excel, \
             patch("pandas.DataFrame.to_excel") as mock_to_excel, \
             patch("builtins.open", mock_open()), \
             patch("json.dump") as mock_json_dump:
            
            # Setup return values
            mock_read_excel.return_value = sample_data
            mock_calc_metrics.side_effect = [
                {
                    "rouge_score": 0.7,
                    "cosine_similarity": 0.8,
                    "bert_score_f1": 0.9,
                    "final_score": 0.85
                },
                {
                    "rouge_score": 0.6,
                    "cosine_similarity": 0.7,
                    "bert_score_f1": 0.8,
                    "final_score": 0.75
                }
            ]
            
            # Call the method
            evaluator.evaluate_excel("input.xlsx", "output.xlsx", "summary.json")
            
            # Verify the method calls
            mock_read_excel.assert_called_once_with("input.xlsx")
            mock_to_excel.assert_called_once_with("output.xlsx", index=False)
            mock_json_dump.assert_called_once()
            
            # Verify metric calculation calls
            assert mock_calc_metrics.call_count == 2
            mock_calc_metrics.assert_any_call("Predicted 1", "Answer 1")
            mock_calc_metrics.assert_any_call("Predicted 2", "Answer 2")
            
            # Verify summary content
            summary_data = mock_json_dump.call_args[0][0]
            assert summary_data["rouge_score"] == pytest.approx(0.65)  # Average of 0.7 and 0.6
            assert summary_data["cosine_similarity"] == pytest.approx(0.75)
            assert summary_data["bert_score_f1"] == pytest.approx(0.85)
            assert summary_data["final_score"] == pytest.approx(0.8)
            assert summary_data["grade"] == "B (Good)"
        
        # Test with missing columns
        with patch("pandas.read_excel") as mock_read_excel:
            mock_read_excel.return_value = pd.DataFrame({"Not_Required": [1, 2]})
            
            with pytest.raises(ValueError, match="Excel file must contain columns"):
                evaluator.evaluate_excel("bad_input.xlsx", "output.xlsx", "summary.json")
        
        # Test with evaluation error
        with patch("pandas.read_excel") as mock_read_excel, \
             patch.object(evaluator, 'calculate_metrics') as mock_calc_metrics, \
             patch("pandas.DataFrame.to_excel") as mock_to_excel, \
             patch("builtins.open", mock_open()), \
             patch("json.dump") as mock_json_dump:
            
            mock_read_excel.return_value = sample_data
            mock_calc_metrics.side_effect = Exception("Evaluation error")
            
            # Should handle the exception and continue
            evaluator.evaluate_excel("input.xlsx", "output.xlsx", "summary.json")
            
            # Verify the Excel was still saved with 0.0 for failures
            mock_to_excel.assert_called_once()
        
        logger.info("QAEvaluator.evaluate_excel test completed") 