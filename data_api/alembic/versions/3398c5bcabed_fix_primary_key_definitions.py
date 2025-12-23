"""fix primary key definitions

Revision ID: 3398c5bcabed
Revises: 07de3d447599
Create Date: 2025-12-23 16:01:32.262917

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3398c5bcabed"
down_revision: Union[str, None] = "07de3d447599"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add missing primary keys to tables that don't have them

    # data_hyperparameters - PK: run_id, data_id, technique_name, technique_parameter_name
    op.create_primary_key(
        "data_hyperparameters_pkey",
        "data_hyperparameters",
        ["run_id", "data_id", "technique_name", "technique_parameter_name"],
    )

    # build_and_integration_testing - PK: test_id, experiment_id, deployment_id, model_id
    op.create_primary_key(
        "build_and_integration_testing_pkey",
        "build_and_integration_testing",
        ["test_id", "experiment_id", "deployment_id", "model_id"],
    )

    # standards - PK: standard_id, experiment_id, deployment_id, model_id
    op.create_primary_key(
        "standards_pkey",
        "standards",
        ["standard_id", "experiment_id", "deployment_id", "model_id"],
    )

    # interfaces - PK: interface_id, experiment_id, deployment_id, model_id
    op.create_primary_key(
        "interfaces_pkey",
        "interfaces",
        ["interface_id", "experiment_id", "deployment_id", "model_id"],
    )

    # model_packaging - PK: packaging_id, experiment_id, deployment_id, model_id
    op.create_primary_key(
        "model_packaging_pkey",
        "model_packaging",
        ["packaging_id", "experiment_id", "deployment_id", "model_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove the primary keys added in upgrade
    op.drop_constraint("model_packaging_pkey", "model_packaging", type_="primary")
    op.drop_constraint("interfaces_pkey", "interfaces", type_="primary")
    op.drop_constraint("standards_pkey", "standards", type_="primary")
    op.drop_constraint(
        "build_and_integration_testing_pkey",
        "build_and_integration_testing",
        type_="primary",
    )
    op.drop_constraint(
        "data_hyperparameters_pkey", "data_hyperparameters", type_="primary"
    )
