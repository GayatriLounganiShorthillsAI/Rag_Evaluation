import os
import pytest
import logging
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Import the class from scrapper.py
from scrapper import WikipediaScraper

# Configure test-specific logger
log_file = Path(__file__).parent / "logs" / "test_scrapper.log"
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

class TestWikipediaScraper:
    @pytest.fixture
    def mock_makedirs(self):
        with patch("os.makedirs") as mock:
            yield mock
    
    @pytest.fixture
    def mock_requests(self):
        with patch("scrapper.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = """
            <html>
                <body>
                    <div class="mw-parser-output">
                        <p>Test paragraph 1 with enough text to pass the length check.</p>
                        <p>Test paragraph 2 with enough text to pass the length check.</p>
                        <div>Not a paragraph</div>
                        <p>Short</p>
                    </div>
                </body>
            </html>
            """
            mock_get.return_value = mock_response
            yield mock_get, mock_response
    
    def test_init(self, mock_makedirs, setup_logging):
        logger.info("Testing WikipediaScraper initialization")
        
        topics = ["Topic1", "Topic2"]
        scraper = WikipediaScraper(topics, save_dir="test_data", log_file="test_logs/scraper.log")
        
        assert scraper.topics == topics
        assert scraper.save_dir == "test_data"
        assert "User-Agent" in scraper.headers
        
        # Test directory creation
        mock_makedirs.assert_any_call("test_data", exist_ok=True)
        mock_makedirs.assert_any_call(os.path.dirname("test_logs/scraper.log"), exist_ok=True)
        
        logger.info("WikipediaScraper initialization test completed")
    
    def test_clean_text(self, setup_logging):
        logger.info("Testing WikipediaScraper.clean_text method")
        
        # Test with extra spaces
        text = "  Test   with   extra    spaces  "
        cleaned = WikipediaScraper.clean_text(text)
        assert cleaned == "Test with extra spaces"
        
        logger.info("WikipediaScraper.clean_text test completed")
    
    def test_scrape_page(self, mock_requests, setup_logging):
        logger.info("Testing WikipediaScraper.scrape_page method")
        
        mock_get, mock_response = mock_requests
        scraper = WikipediaScraper(["Test"])
        
        # Test successful scraping
        text = scraper.scrape_page("Test_Page")
        assert "Test paragraph 1" in text
        assert "Test paragraph 2" in text
        assert "Not a paragraph" not in text
        assert "Short" not in text  # Should be filtered due to length check
        
        # Test with bad status code
        mock_response.status_code = 404
        text = scraper.scrape_page("Bad_Page")
        assert text == ""
        
        # Test with missing content div
        mock_response.status_code = 200
        mock_response.text = "<html><body>No content div</body></html>"
        text = scraper.scrape_page("Missing_Div")
        assert text == ""
        
        # Test with request exception
        mock_get.side_effect = Exception("Connection error")
        text = scraper.scrape_page("Error_Page")
        assert text == ""
        
        logger.info("WikipediaScraper.scrape_page test completed")
    
    def test_save_to_file(self, setup_logging):
        logger.info("Testing WikipediaScraper.save_to_file method")
        
        scraper = WikipediaScraper(["Test"])
        
        # Test successful save
        m = mock_open()
        with patch("builtins.open", m):
            scraper.save_to_file("test_file.txt", "Test content")
        
        m.assert_called_once_with("test_file.txt", "w", encoding="utf-8")
        m().write.assert_called_once_with("Test content")
        
        # Test with exception
        with patch("builtins.open", side_effect=Exception("Write error")):
            scraper.save_to_file("error_file.txt", "Test content")
            # Should log error but not raise exception
        
        logger.info("WikipediaScraper.save_to_file test completed")
    
    def test_scrape_all(self, mock_requests, setup_logging):
        logger.info("Testing WikipediaScraper.scrape_all method")
        
        topics = ["Topic1", "Topic2"]
        scraper = WikipediaScraper(topics)
        
        with patch.object(scraper, "scrape_page") as mock_scrape:
            mock_scrape.side_effect = ["Content 1", "Content 2"]
            
            with patch.object(scraper, "save_to_file") as mock_save:
                scraper.scrape_all()
                
                # Check that scrape_page was called for each topic
                assert mock_scrape.call_count == 2
                mock_scrape.assert_any_call("Topic1")
                mock_scrape.assert_any_call("Topic2")
                
                # Check that save_to_file was called with combined content
                mock_save.assert_called_once()
                call_args = mock_save.call_args[0]
                assert call_args[0].endswith("modern_history_of_india.txt")
                assert "Topic1" in call_args[1]
                assert "Content 1" in call_args[1]
                assert "Topic2" in call_args[1]
                assert "Content 2" in call_args[1]
        
        logger.info("WikipediaScraper.scrape_all test completed") 