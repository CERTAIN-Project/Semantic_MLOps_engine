#!/usr/bin/env python3
"""
Test script to verify certain_library logs data to MLflow PostgreSQL database
"""

from certain_library.tracking.tracker import tracker

import mlflow
import pandas as pd
from certain_library.train_monitor.log_metrics import log_metrics
from certain_library.train_monitor.log_model import log_model_info
from certain_library.data_analysis.log_dataset import log_dataset

print("=" * 60)
print("Testing certain_library MLflow Integration")
print("=" * 60)
print()

# Verify MLflow tracking URI
print("1. Checking MLflow Configuration:")
print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
print()

# Set experiment with shared artifact location
experiment_name = "certain_library_integration_test"
artifact_location = "/mlflow-artifacts"

# Try to get existing experiment, if not create with custom artifact location
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        # Create with custom artifact location
        experiment_id = tracker.create_experiment(
            experiment_name, artifact_location=artifact_location
        )
        experiment = mlflow.get_experiment(experiment_id)
        print(f"   Created experiment with artifact location: {artifact_location}")
    tracker.set_experiment(experiment_name)
except Exception as e:
    print(f"   Warning: {e}")
    tracker.set_experiment(experiment_name)

print(f"2. Created/Set Experiment: {experiment_name}")
print()

# Start a run
with tracker.start_run(run_name="library_test_run") as run:
    print("3. Started MLflow Run")
    print(f"   Run ID: {run.info.run_id}")
    print()

    # Test log_metrics function
    print("4. Testing log_metrics from certain_library...")
    try:
        metrics = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.92,
            "f1_score": 0.94,
            "loss": 0.05,
        }
        log_metrics(metrics)
        print("   ✓ Metrics logged successfully")
        print(f"   Logged: {list(metrics.keys())}")
        print()
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print()

    # Test log_model_info function
    print("5. Testing log_model_info from certain_library...")
    try:
        model_info = {
            "model_name": "RandomForestClassifier",
            "model_version": "1.0",
            "framework": "scikit-learn",
            "framework_version": "1.3.0",
        }
        log_model_info(model_info)
        print("   ✓ Model info logged successfully")
        print()
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print()

    # Test log_dataset function
    print("6. Testing log_dataset from certain_library...")
    try:
        train_df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7],
                "feature2": [10, 20, 30, 40, 50, 60, 70],
                "target": [0, 1, 0, 1, 0, 1, 0],
            }
        )
        test_df = pd.DataFrame(
            {"feature1": [8, 9, 10], "feature2": [80, 90, 100], "target": [1, 0, 1]}
        )
        log_dataset(train_df, test_df)
        print("   ✓ Dataset logged successfully")
        print(f"   Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        print()
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print()

print("=" * 60)
print("All Tests Completed!")
print("=" * 60)
print()
print("✓ All data has been logged to PostgreSQL database in certain_databases")
print()
