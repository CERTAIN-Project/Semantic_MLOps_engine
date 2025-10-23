#!/usr/bin/env python3
"""
Test runner script for the certain_library test suite.
This script helps run and validate all tests following the OPSD patterns.
"""

import os
import sys
import subprocess
import pytest


def run_tests():
    """Run all tests with proper configuration."""

    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"Running tests from: {script_dir}")
    print("=" * 50)

    # Test commands to run
    test_commands = [
        {
            "name": "Basic test discovery",
            "cmd": ["pytest", "--collect-only", "tests/"],
            "description": "Check if all tests can be discovered",
        },
        {
            "name": "OPSD variables tests",
            "cmd": ["pytest", "tests/test_opsd_variables.py", "-v"],
            "description": "Test OPSD variables and data structures",
        },
        {
            "name": "Integration tests",
            "cmd": ["pytest", "tests/test_opsd_logging_integration.py", "-v"],
            "description": "Test library functions with OPSD patterns",
        },
        {
            "name": "Train monitor tests",
            "cmd": ["pytest", "tests/test_train_monitor.py", "-v"],
            "description": "Test training monitoring functions",
        },
        {
            "name": "Data analysis tests",
            "cmd": ["pytest", "tests/test_data_analysis.py", "-v"],
            "description": "Test data analysis functions",
        },
        {
            "name": "Resource monitor tests",
            "cmd": ["pytest", "tests/test_resource_monitor.py", "-v"],
            "description": "Test resource monitoring functions",
        },
        # {
        #     "name": "All tests with coverage",
        #     "cmd": [
        #         "pytest",
        #         "tests/",
        #         "--cov=certain_library",
        #         "--cov-report=term-missing",
        #     ],
        #     "description": "Run all tests with coverage report",
        # },
    ]

    # Run each test command
    for test_config in test_commands:
        print(f"\n{test_config['name']}")
        print("-" * len(test_config["name"]))
        print(f"Description: {test_config['description']}")
        print(f"Command: {' '.join(test_config['cmd'])}")

        try:
            result = subprocess.run(
                test_config["cmd"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                print("✅ PASSED")
                if result.stdout:
                    print("Output:", result.stdout[-500:])  # Last 500 chars
            else:
                print("❌ FAILED")
                print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
                print("STDERR:", result.stderr[-1000:])

        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT - Test took too long")
        except Exception as e:
            print(f"💥 ERROR - {str(e)}")

        print("-" * 50)


def check_test_structure():
    """Check if all required test files exist."""

    required_files = [
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_opsd_variables.py",
        "tests/test_opsd_logging_integration.py",
        "tests/test_train_monitor.py",
        "tests/test_data_analysis.py",
        "tests/test_resource_monitor.py",
    ]

    print("Checking test file structure...")
    print("=" * 40)

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} test files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("\n✅ All required test files found!")
        return True


def validate_imports():
    """Validate that all imports work correctly."""

    print("\nValidating imports...")
    print("=" * 30)

    try:
        import certain_library as cl

        print("✅ certain_library import successful")

        # Test key imports
        from certain_library.train_monitor.log_model import log_model_info
        from certain_library.data_analysis.log_dataset import log_dataset
        from certain_library.resource_monitor.resource import start_tracker

        print("✅ All key function imports successful")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


if __name__ == "__main__":
    print("Certain Library Test Runner")
    print("=" * 50)

    # Check structure first
    if not check_test_structure():
        print("\n❌ Test structure validation failed!")
        sys.exit(1)

    # Validate imports
    if not validate_imports():
        print("\n❌ Import validation failed!")
        sys.exit(1)

    # Run tests
    print("\n" + "=" * 50)
    print("RUNNING TESTS")
    print("=" * 50)

    run_tests()

    print("\n" + "=" * 50)
    print("Test run completed!")
    print("=" * 50)
