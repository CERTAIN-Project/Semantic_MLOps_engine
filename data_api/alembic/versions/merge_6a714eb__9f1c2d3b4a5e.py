"""Merge heads and apply standards PK change

Revision ID: merge_6a714eb__9f1c2d3b4a5e
Revises: 6a714eb887eb, 9f1c2d3b4a5e
Create Date: 2026-08-14 12:45:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text


# revision identifiers, used by Alembic.
revision = 'merge_6a714eb__9f1c2d3b4a5e'
down_revision = ('6a714eb887eb', '9f1c2d3b4a5e')
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    # The merge revision intentionally no-ops besides ensuring a single
    # linear head. We also double-check that the standards PK change was
    # applied; if not, apply it here as a fallback.
    pk_query = text("""
    SELECT conname
    FROM pg_constraint
    WHERE conrelid = 'public.standards'::regclass AND contype = 'p';
    """)
    pk_name = conn.execute(pk_query).scalar()

    # If the current PK includes deployment_id (4 columns), we'll replace it
    # with the new PK (standard_id, experiment_id, model_id) and make
    # deployment_id nullable.
    if pk_name:
        # Inspect columns in the PK by querying pg_index/pg_attribute
        cols_q = text("""
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = 'public.standards'::regclass AND i.indisprimary;
        """)
        cols = [r[0] for r in conn.execute(cols_q).fetchall()]
        if 'deployment_id' in cols and len(cols) >= 4:
            # Drop the existing PK
            op.execute(sa.text(f'ALTER TABLE public.standards DROP CONSTRAINT IF EXISTS "{pk_name}"'))
            # Make deployment_id nullable
            op.alter_column('standards', 'deployment_id', existing_type=sa.String(), nullable=True)
            # Create new PK without deployment_id
            op.create_primary_key('standards_pkey', 'standards', ['standard_id', 'experiment_id', 'model_id'])


def downgrade():
    # Downgrade intentionally not implemented for merge migration
    pass
