import os
import sys
import pytest
import logging
from pathlib import Path

# Add the parent directory to sys.path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def setup_logging():
    """Set up logging for all tests"""
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "test_suite.log"),
            logging.StreamHandler()
        ]
    )
    return logging 