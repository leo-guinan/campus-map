"""add regions

Revision ID: 002
Revises: 001
Create Date: 2024-03-13 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY

# revision identifiers
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Create regions table
    op.create_table('regions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('color', sa.String(), nullable=False),
        sa.Column('center', ARRAY(sa.Float()), nullable=False),
        sa.Column('is_hardcoded', sa.Boolean(), nullable=False, default=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create building_regions junction table
    op.create_table('building_regions',
        sa.Column('building_id', sa.String(), nullable=False),
        sa.Column('region_id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['building_id'], ['building_details.building_id']),
        sa.ForeignKeyConstraint(['region_id'], ['regions.id']),
        sa.PrimaryKeyConstraint('building_id', 'region_id')
    )

def downgrade() -> None:
    op.drop_table('building_regions')
    op.drop_table('regions') 