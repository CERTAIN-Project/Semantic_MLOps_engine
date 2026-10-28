"""Make deployment_id nullable on standards and remove it from the PK

Revision ID: 9f1c2d3b4a5e
Revises: 38521158cbb7
Create Date: 2026-08-14 12:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text

# revision identifiers, used by Alembic.
revision = '9f1c2d3b4a5e'
down_revision = '38521158cbb7'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    # Find the current primary key constraint name for standards
    pk_query = text("""
    SELECT conname
    FROM pg_constraint
    WHERE conrelid = 'public.standards'::regclass AND contype = 'p';
    """)
    pk_name = conn.execute(pk_query).scalar()
    if pk_name:
        # Drop the existing PK constraint
        op.execute(sa.text(f'ALTER TABLE public.standards DROP CONSTRAINT IF EXISTS "{pk_name}"'))

    # Alter the deployment_id column to be nullable
    op.alter_column(
        'standards',
        'deployment_id',
        existing_type=sa.String(),
        nullable=True,
    )

    # Create a new PK that omits deployment_id
    # Keep standard_id, experiment_id and model_id as the composite PK
    op.create_primary_key('standards_pkey', 'standards', ['standard_id', 'experiment_id', 'model_id'])


def downgrade():
    conn = op.get_bind()

    # Drop the PK that omits deployment_id
    try:
        op.drop_constraint('standards_pkey', 'standards', type_='primary')
    except Exception:
        # If the constraint name differs, attempt to find and drop it
        pk_query = text("""
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'public.standards'::regclass AND contype = 'p';
        """)
        pk_name = conn.execute(pk_query).scalar()
        if pk_name:
            op.execute(sa.text(f'ALTER TABLE public.standards DROP CONSTRAINT IF EXISTS "{pk_name}"'))

    # Recreate the original PK including deployment_id
    op.create_primary_key('standards_pkey', 'standards', ['standard_id', 'experiment_id', 'deployment_id', 'model_id'])

    # Make deployment_id NOT NULL again
    op.alter_column(
        'standards',
        'deployment_id',
        existing_type=sa.String(),
        nullable=False,
    )
