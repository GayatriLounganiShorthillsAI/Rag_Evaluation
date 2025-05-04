import unittest
from unittest.mock import patch, MagicMock, mock_open

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from scrapper import WikipediaScraper 
import os


class TestWikipediaScraper(unittest.TestCase):

    def setUp(self):
        self.scraper = WikipediaScraper(topics=["Test_Topic"], save_dir="data", log_file="logs/scraper.log")

    def test_initialization(self):
        self.assertEqual(self.scraper.topics, ["Test_Topic"])
        self.assertTrue(os.path.exists("data"))
        self.assertTrue(os.path.exists("logs"))

    def test_clean_text(self):
        messy_text = "  Hello   world!  This is   spaced. "
        clean = self.scraper.clean_text(messy_text)
        self.assertEqual(clean, "Hello world! This is spaced.")

    @patch("requests.get")
    def test_scrape_page_success(self, mock_get):
        html_content = """
        <html><body>
        <div class="mw-parser-output">
            <p>This is the first paragraph with enough length to be included.</p>
            <p>Short.</p>
            <p>This is another valid paragraph with more than 50 characters.</p>
        </div>
        </body></html>
        """
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = html_content

        result = self.scraper.scrape_page("Test_Topic")
        self.assertIn("This is the first paragraph", result)
        self.assertIn("This is another valid paragraph", result)
        self.assertNotIn("Short.", result)

    @patch("requests.get", side_effect=Exception("Connection error"))
    def test_scrape_page_failure(self, mock_get):
        result = self.scraper.scrape_page("Non_Existent_Topic")
        self.assertEqual(result, "")

    @patch("builtins.open", new_callable=mock_open)
    def test_save_to_file(self, mock_file):
        self.scraper.save_to_file("dummy_path.txt", "Test content")
        mock_file.assert_called_with("dummy_path.txt", "w", encoding="utf-8")
        handle = mock_file()
        handle.write.assert_called_once_with("Test content")

    @patch.object(WikipediaScraper, 'scrape_page', return_value="Dummy page content for test.")
    @patch.object(WikipediaScraper, 'save_to_file')
    def test_scrape_all(self, mock_save, mock_scrape):
        scraper = WikipediaScraper(topics=["Topic_One", "Topic_Two"], save_dir="data")
        scraper.scrape_all()
        self.assertEqual(mock_scrape.call_count, 2)
        mock_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
