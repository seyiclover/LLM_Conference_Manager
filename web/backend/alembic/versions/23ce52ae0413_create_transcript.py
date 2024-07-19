"""create transcript

Revision ID: 23ce52ae0413
Revises: 4eab0b95407d
Create Date: 2024-07-06 23:18:49.290151

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '23ce52ae0413'
down_revision: Union[str, None] = '4eab0b95407d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
