"""create transcript++fix

Revision ID: 30922734fda8
Revises: 23ce52ae0413
Create Date: 2024-07-06 23:20:02.585885

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '30922734fda8'
down_revision: Union[str, None] = '23ce52ae0413'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
