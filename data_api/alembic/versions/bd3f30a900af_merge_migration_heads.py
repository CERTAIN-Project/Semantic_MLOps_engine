"""Merge migration heads

Revision ID: bd3f30a900af
Revises: 20260703_add_runid_to_ai_actors, 97ea440db74e
Create Date: 2026-07-03 16:38:03.726367

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bd3f30a900af'
down_revision: Union[str, None] = ('20260703_add_runid_to_ai_actors', '97ea440db74e')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
