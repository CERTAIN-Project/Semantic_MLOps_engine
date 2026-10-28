#!/usr/bin/env python3
"""
Decommissioning compliance logger — run this when a deployed model is being
retired (manually, from a monitoring alert, or from a release pipeline).

Usage
-----
    python log_decommission_compliance.py \\
        --run-id        <mlflow_run_id>           \\
        --deploy-id     <your_deployment_id>      \\
        --model-id      <your_model_id>           \\
        --reason        "Replaced by v2 model."   \\
        --changed-by    "ml-team"

The script reopens the existing MLflow run (no new run is created) and appends:
  • decommissioning artifact  → decomissioning table
  • change_log artifact       → change_logs table  (records who retired the model)
"""

from certain_library.tracking.tracker import tracker

import argparse
import os
import time

import mlflow

from certain_library.compliance.log_deployment import log_decommissioning
from certain_library.compliance.log_governance import log_change


# ---------------------------------------------------------------------------
# Defaults — override via CLI args or environment variables
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Log decommissioning compliance data into an existing MLflow run."
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="MLflow run ID of the original training run.",
    )
    p.add_argument(
        "--deploy-id",
        required=True,
        help="Deployment identifier being decommissioned.",
    )
    p.add_argument(
        "--model-id",
        required=True,
        help="Model identifier being decommissioned.",
    )
    p.add_argument(
        "--reason",
        required=True,
        help="Reason for decommissioning (e.g. 'Replaced by v2 model.').",
    )
    p.add_argument(
        "--changed-by",
        default="ml-team",
        help="Name or identifier of the person/team performing decommissioning.",
    )
    p.add_argument(
        "--procedure-details",
        default=(
            "Model endpoint shut down. Weights archived to cold storage. "
            "Stakeholders notified via email."
        ),
        help="Free-text description of the decommissioning procedure.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tracker.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"🗑️  Attaching decommissioning record to run: {args.run_id}")
    print(f"   deployment_id : {args.deploy_id}")
    print(f"   model_id      : {args.model_id}")
    print(f"   reason        : {args.reason}")
    print()

    # Reopen the existing training run — no new run is created
    with tracker.start_run(run_id=args.run_id):

        # ------------------------------------------------------------------ #
        # 1. Decommissioning record
        # ------------------------------------------------------------------ #
        log_decommissioning(
            deployment_id=args.deploy_id,
            model_id=args.model_id,
            decommissioning_actions=[
                "disable model endpoint",
                "archive model weights to cold storage",
                "remove Docker image from registry",
                "notify stakeholders",
            ],
            reason=args.reason,
            procedure_details=args.procedure_details,
            decommissioning_date=float(time.time()),
        )
        print("  ✅ log_decommissioning done")

        # ------------------------------------------------------------------ #
        # 2. Change log — record who triggered the decommission and why
        # ------------------------------------------------------------------ #
        log_change(
            changes=[
                {
                    "change_description": (
                        f"Model {args.model_id} (deployment {args.deploy_id}) "
                        f"decommissioned. Reason: {args.reason}"
                    ),
                    "changed_by": args.changed_by,
                }
            ]
        )
        print("  ✅ log_change done")

    print()
    print("🎉 Decommissioning compliance logged successfully.")
    print(f"   Artifacts attached to MLflow run: {args.run_id}")


if __name__ == "__main__":
    main()
