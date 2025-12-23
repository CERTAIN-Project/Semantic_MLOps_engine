"""small fix of primary keys

Revision ID: 07de3d447599
Revises: 559f69430939
Create Date: 2025-12-23 13:19:13.640502

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "07de3d447599"
down_revision: Union[str, None] = "559f69430939"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # === drift_metrics ===
    op.drop_constraint("drift_metrics_pkey", "drift_metrics", type_="primary")
    op.create_primary_key(
        "drift_metrics_pkey",
        "drift_metrics",
        ["experiment_id", "deployment_id", "model_id"],
    )
    op.alter_column("drift_metrics", "value", existing_type=sa.INTEGER(), nullable=True)

    # === model_metrics ===
    # Old PK: run_id, key, model_id, step, stage, timestamp, is_NaN, value
    # New PK: run_id, key, model_id, step, stage
    op.drop_constraint("model_metrics_pkey", "model_metrics", type_="primary")
    op.create_primary_key(
        "model_metrics_pkey",
        "model_metrics",
        ["run_id", "key", "model_id", "step", "stage"],
    )

    # === resources ===
    # Old PK: run_id, key, model_id, value, stage, timestamp, step
    # New PK: run_id, model_id, key, step, stage
    op.drop_constraint("resources_pkey", "resources", type_="primary")
    op.create_primary_key(
        "resources_pkey",
        "resources",
        ["run_id", "model_id", "key", "step", "stage"],
    )

    # === weight_distribution ===
    # Old PK: run_id, layer_name, model_id, step, stage, timestamp, is_NaN, mean, std
    # New PK: run_id, model_id, step, stage
    op.drop_constraint(
        "weight_distribution_pkey", "weight_distribution", type_="primary"
    )
    op.create_primary_key(
        "weight_distribution_pkey",
        "weight_distribution",
        ["run_id", "model_id", "step", "stage"],
    )

    # === examples ===
    # Old PK: run_id, model_id, input, prediction, ground_truth, step, stage, timestamp
    # New PK: run_id, model_id, input, step, stage
    op.drop_constraint("examples_pkey", "examples", type_="primary")
    op.create_primary_key(
        "examples_pkey",
        "examples",
        ["run_id", "model_id", "input", "step", "stage"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    # === examples ===
    op.drop_constraint("examples_pkey", "examples", type_="primary")
    op.create_primary_key(
        "examples_pkey",
        "examples",
        [
            "run_id",
            "model_id",
            "input",
            "prediction",
            "ground_truth",
            "step",
            "stage",
            "timestamp",
        ],
    )

    # === weight_distribution ===
    op.drop_constraint(
        "weight_distribution_pkey", "weight_distribution", type_="primary"
    )
    op.create_primary_key(
        "weight_distribution_pkey",
        "weight_distribution",
        [
            "run_id",
            "layer_name",
            "model_id",
            "step",
            "stage",
            "timestamp",
            "is_NaN",
            "mean",
            "std",
        ],
    )

    # === resources ===
    op.drop_constraint("resources_pkey", "resources", type_="primary")
    op.create_primary_key(
        "resources_pkey",
        "resources",
        ["run_id", "key", "model_id", "value", "stage", "timestamp", "step"],
    )

    # === model_metrics ===
    op.drop_constraint("model_metrics_pkey", "model_metrics", type_="primary")
    op.create_primary_key(
        "model_metrics_pkey",
        "model_metrics",
        ["run_id", "key", "model_id", "step", "stage", "timestamp", "is_NaN", "value"],
    )

    # === drift_metrics ===
    op.alter_column(
        "drift_metrics", "value", existing_type=sa.INTEGER(), nullable=False
    )
    op.drop_constraint("drift_metrics_pkey", "drift_metrics", type_="primary")
    op.create_primary_key(
        "drift_metrics_pkey",
        "drift_metrics",
        ["experiment_id", "deployment_id", "model_id", "value", "timestamp"],
    )
