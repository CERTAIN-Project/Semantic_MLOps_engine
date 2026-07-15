"""add run_id to ai_actors for run-scoped association

Revision ID: 20260703_add_runid_to_ai_actors
Revises: d2dcd3bc6f26
Create Date: 2026-07-03 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260703_add_runid_to_ai_actors"
down_revision = "d2dcd3bc6f26"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add nullable run_id column and foreign key to runs.run_id so future
    # inserts can use run_id. We keep experiment_id for backwards compatibility
    # (do not drop it in this migration).
    op.add_column("ai_actors", sa.Column("run_id", sa.String(), nullable=True))
    try:
        op.create_foreign_key(
            "ai_actors_run_id_fkey",
            "ai_actors",
            "runs",
            ["run_id"],
            ["run_id"],
        )
    except Exception:
        # If the runs table isn't present in the migration environment, skip FK.
        pass


def downgrade() -> None:
    op.drop_constraint("ai_actors_run_id_fkey", "ai_actors", type_="foreignkey")
    op.drop_column("ai_actors", "run_id")
