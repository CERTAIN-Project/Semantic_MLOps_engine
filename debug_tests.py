#!/usr/bin/env python3
"""
Debug script to help troubleshoot test issues.
"""

import sys
import os
import traceback


def debug_imports():
    """Debug import issues."""
    print("Debugging imports...")

    try:
        sys.path.insert(0, os.getcwd())
        print(f"Python path: {sys.path[:3]}...")

        import certain_library as cl

        print("✅ certain_library imported successfully")

        # Test each module
        modules_to_test = [
            "certain_library.train_monitor.log_model",
            "certain_library.data_analysis.log_dataset",
            "certain_library.resource_monitor.resource",
        ]

        for module_name in modules_to_test:
            try:
                __import__(module_name)
                print(f"✅ {module_name}")
            except Exception as e:
                print(f"❌ {module_name}: {e}")

    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()


def debug_test_discovery():
    """Debug test discovery issues."""
    print("\nDebugging test discovery...")

    import pytest

    # Try to collect tests
    try:
        exit_code = pytest.main(["--collect-only", "tests/", "-q"])
        if exit_code == 0:
            print("✅ Test discovery successful")
        else:
            print(f"❌ Test discovery failed with exit code: {exit_code}")
    except Exception as e:
        print(f"❌ Test discovery error: {e}")
        traceback.print_exc()


def run_single_test():
    """Run a single test for debugging."""
    print("\nRunning single test...")

    try:
        from tests.test_opsd_variables import OPSDTestVariables

        # Test creating sample data
        sample_data = OPSDTestVariables.create_sample_opsd_dataframe(10)
        print(f"✅ Created sample data with shape: {sample_data.shape}")
        print(f"✅ Columns: {list(sample_data.columns)}")

        # Test data split
        X_train, X_test, y_train, y_test, train_ts, test_ts = (
            OPSDTestVariables.get_train_test_split(sample_data)
        )
        print(f"✅ Data split successful: train={len(X_train)}, test={len(X_test)}")

    except Exception as e:
        print(f"❌ Single test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Certain Library Test Debugger")
    print("=" * 40)

    debug_imports()
    debug_test_discovery()
    run_single_test()

    print("\nDebugging completed!")
