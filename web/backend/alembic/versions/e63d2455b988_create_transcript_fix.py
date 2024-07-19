"""create transcript++++fix

Revision ID: e63d2455b988
Revises: 30922734fda8
Create Date: 2024-07-06 23:38:52.063945

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e63d2455b988'
down_revision: Union[str, None] = '30922734fda8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
