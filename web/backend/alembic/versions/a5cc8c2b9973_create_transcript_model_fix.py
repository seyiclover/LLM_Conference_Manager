"""create transcript model++++fix

Revision ID: a5cc8c2b9973
Revises: e63d2455b988
Create Date: 2024-07-06 23:41:28.332696

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a5cc8c2b9973'
down_revision: Union[str, None] = 'e63d2455b988'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
