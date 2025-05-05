#!/usr/bin/env python3
import os
import sys
import pytest
from pathlib import Path

def main():
    """Run all tests and generate a report."""
    # Change to the tests directory
    tests_dir = Path(__file__).parent
    os.chdir(tests_dir)
    
    # Create logs directory if it doesn't exist
    logs_dir = tests_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Run pytest with verbosity
    pytest_args = [
        "-v",
        "--junitxml=test_results.xml",
        "--color=yes",
    ]
    
    print(f"Running tests from {tests_dir}")
    exit_code = pytest.main(pytest_args)
    
    # Report summary
    if exit_code == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 