"""Add tokenizer_config and tokenization_stats tables

Revision ID: b9e1f2a3c4d5
Revises: 3398c5bcabed
Create Date: 2026-04-29

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision: str = "b9e1f2a3c4d5"
down_revision: Union[str, None] = "3398c5bcabed"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # tokenizer_config                                                     #
    # ------------------------------------------------------------------ #
    op.create_table(
        "tokenizer_config",
        sa.Column("tokenizer_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("tokenizer_type", sa.String(), nullable=False),
        sa.Column("model_name_or_path", sa.String(), nullable=True),
        sa.Column("vocab_size", sa.Integer(), nullable=True),
        sa.Column("max_length", sa.Integer(), nullable=True),
        sa.Column("padding", sa.String(), nullable=True),
        sa.Column("truncation", sa.Boolean(), nullable=True),
        sa.Column("stride", sa.Integer(), nullable=True),
        sa.Column("special_tokens", JSON, nullable=True),
        sa.Column(
            "creation_time", sa.Numeric(), nullable=False, server_default="1672531200"
        ),
        sa.PrimaryKeyConstraint("tokenizer_id", "run_id"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    )

    # ------------------------------------------------------------------ #
    # tokenization_stats                                                   #
    # ------------------------------------------------------------------ #
    op.create_table(
        "tokenization_stats",
        sa.Column("stats_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("split", sa.String(), nullable=False),
        sa.Column("total_sequences", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("avg_token_length", sa.Float(), nullable=True),
        sa.Column("max_token_length", sa.Integer(), nullable=True),
        sa.Column("min_token_length", sa.Integer(), nullable=True),
        sa.Column("truncation_rate", sa.Float(), nullable=True),
        sa.Column("padding_rate", sa.Float(), nullable=True),
        sa.Column("oov_rate", sa.Float(), nullable=True),
        sa.Column(
            "creation_time", sa.Numeric(), nullable=False, server_default="1672531200"
        ),
        sa.PrimaryKeyConstraint("stats_id", "run_id"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    )


def downgrade() -> None:
    op.drop_table("tokenization_stats")
    op.drop_table("tokenizer_config")
