"""initial

Revision ID: 001
Revises: 
Create Date: 2024-03-13 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('building_details',
        sa.Column('building_id', sa.String(), nullable=False),
        sa.Column('full_research_question', sa.String(), nullable=True),
        sa.Column('full_building_type', sa.String(), nullable=True),
        sa.Column('full_donor_name', sa.String(), nullable=True),
        sa.Column('website', sa.String(), nullable=True),
        sa.Column('donation_amount', sa.Float(), nullable=True),
        sa.Column('creation_date', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('building_id')
    )


def downgrade() -> None:
    op.drop_table('building_details') 