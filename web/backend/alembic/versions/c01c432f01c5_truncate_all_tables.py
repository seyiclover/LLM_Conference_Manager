"""truncate all tables

Revision ID: c01c432f01c5
Revises: 0cd914a1724d
Create Date: 2024-07-09 17:33:06.699588

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c01c432f01c5'
down_revision: Union[str, None] = '0cd914a1724d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
