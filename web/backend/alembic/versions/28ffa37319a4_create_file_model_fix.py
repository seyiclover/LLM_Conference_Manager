"""create file  model++++fix

Revision ID: 28ffa37319a4
Revises: a5cc8c2b9973
Create Date: 2024-07-07 00:09:51.252891

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '28ffa37319a4'
down_revision: Union[str, None] = 'a5cc8c2b9973'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
