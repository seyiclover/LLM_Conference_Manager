"""truncate all tables

Revision ID: 0cd914a1724d
Revises: d6048f8063e2
Create Date: 2024-07-09 17:27:56.757955

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0cd914a1724d'
down_revision: Union[str, None] = 'd6048f8063e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('SET FOREIGN_KEY_CHECKS = 0')
    
    op.execute('TRUNCATE TABLE users')
    op.execute('TRUNCATE TABLE files')
    op.execute('TRUNCATE TABLE transcripts')
    op.execute('TRUNCATE TABLE summaries')

    op.execute('SET FOREIGN_KEY_CHECKS = 1')
def downgrade() -> None:
    pass
