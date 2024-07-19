"""fileModel fix

Revision ID: e0333417a83b
Revises: 28ffa37319a4
Create Date: 2024-07-07 00:15:29.906791

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e0333417a83b'
down_revision: Union[str, None] = '28ffa37319a4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
