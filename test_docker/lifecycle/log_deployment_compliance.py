#!/usr/bin/env python3
"""
Deployment compliance logger — run this from your CI/CD pipeline (or manually)
after a model has been packaged and deployed.

Usage
-----
    python log_deployment_compliance.py \\
        --run-id     <mlflow_run_id>          \\
        --deploy-id  <your_deployment_id>     \\
        --model-id   <your_model_id>          \\
        --image      docker.io/myorg/mymodel  \\
        --version    v1.2                     \\
        --build-log  "All checks passed."     \\
        --build-status success

The script reopens the existing MLflow run (no new run is created) and appends
deployment-lifecycle artifacts:
  • model_packaging   → model_packaging table
  • build_testing     → build_and_integration_testing table
  • standards         → standards table
  • interface         → interfaces table
"""

import argparse
import os
import sys

import mlflow

from certain_library.compliance.log_deployment import (
    log_model_packaging,
    log_build_testing,
    log_standards,
    log_interface,
)


# ---------------------------------------------------------------------------
# Defaults — override via CLI args or environment variables
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Log deployment compliance data into an existing MLflow run."
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="MLflow run ID of the training run to attach compliance data to.",
    )
    p.add_argument(
        "--deploy-id",
        required=True,
        help="Unique deployment identifier (e.g. 'dep-energy-xgb-prod-20260429').",
    )
    p.add_argument(
        "--model-id",
        required=True,
        help="Model identifier (e.g. 'energy-load-xgb-v1').",
    )
    p.add_argument(
        "--image",
        default="docker.io/certain/energy-xgb",
        help="Docker image reference for the deployed model.",
    )
    p.add_argument(
        "--version",
        default="v1.0",
        help="Interface / model version string.",
    )
    p.add_argument(
        "--build-status",
        default="success",
        choices=["success", "failure", "warning"],
        help="Overall CI/CD build outcome.",
    )
    p.add_argument(
        "--build-log",
        default="Build and smoke tests passed.",
        help="Build log summary or path to log file.",
    )
    p.add_argument(
        "--tests-total", type=int, default=5, help="Total number of tests run."
    )
    p.add_argument(
        "--tests-passed", type=int, default=5, help="Number of tests that passed."
    )
    p.add_argument(
        "--tests-failed", type=int, default=0, help="Number of tests that failed."
    )
    p.add_argument(
        "--coverage", type=float, default=92.0, help="Test coverage percentage."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"📦 Attaching deployment compliance to run: {args.run_id}")
    print(f"   deployment_id : {args.deploy_id}")
    print(f"   model_id      : {args.model_id}")
    print()

    # Reopen the existing training run — no new run is created
    with mlflow.start_run(run_id=args.run_id):

        # ------------------------------------------------------------------ #
        # 1. Model packaging
        # ------------------------------------------------------------------ #
        log_model_packaging(
            deployment_id=args.deploy_id,
            model_id=args.model_id,
            packaging_format="mlflow_model",
            dependencies=[
                "xgboost",
                "scikit-learn",
                "pandas",
                "numpy",
            ],
            containerization_details={
                "base_image": "python:3.11-slim",
                "cpu": "2",
                "memory": "4GB",
                "registry": args.image,
            },
        )
        print("  ✅ log_model_packaging done")

        # ------------------------------------------------------------------ #
        # 2. Build & integration testing
        # ------------------------------------------------------------------ #
        log_build_testing(
            deployment_id=args.deploy_id,
            model_id=args.model_id,
            build_status=args.build_status,
            build_logs=args.build_log,
            test_type="integration",
            test_results={
                "total": args.tests_total,
                "passed": args.tests_passed,
                "failed": args.tests_failed,
                "skipped": args.tests_total - args.tests_passed - args.tests_failed,
                "coverage_pct": args.coverage,
            },
        )
        print("  ✅ log_build_testing done")

        # ------------------------------------------------------------------ #
        # 3. Applicable standards
        # ------------------------------------------------------------------ #
        log_standards(
            deployment_id=args.deploy_id,
            model_id=args.model_id,
            standards=[
                {
                    "name": "ISO/IEC 42001:2023",
                    "description": "AI management system standard.",
                    "version": "2023",
                },
                {
                    "name": "EU AI Act",
                    "description": "European Union regulation on artificial intelligence.",
                    "version": "2024",
                },
                {
                    "name": "ETSI EN 303 645",
                    "description": "Cyber security standard referenced for data pipeline security.",
                    "version": "v2.1.1",
                },
            ],
        )
        print("  ✅ log_standards done")

        # ------------------------------------------------------------------ #
        # 4. Interface specification
        # ------------------------------------------------------------------ #
        log_interface(
            deployment_id=args.deploy_id,
            model_id=args.model_id,
            interface_type="REST API",
            specifications=(
                "POST /predict — accepts JSON with feature array, "
                "returns predicted load in MW."
            ),
            version=args.version,
            documentation_link=f"https://wiki.example.com/api/{args.model_id}",
        )
        print("  ✅ log_interface done")

    print()
    print("🎉 Deployment compliance logged successfully.")
    print(f"   Artifacts attached to MLflow run: {args.run_id}")


if __name__ == "__main__":
    main()
