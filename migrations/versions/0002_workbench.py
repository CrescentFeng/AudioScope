"""Add analysis tasks and append-only human reviews without changing catalog data."""

import sqlalchemy as sa
from alembic import op

revision = "0002_workbench"
down_revision = "0001_audioscope_catalog"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "workbench_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("revision", sa.Integer, nullable=False),
        sa.Column("payload", sa.JSON, nullable=False),
    )
    op.create_table(
        "workbench_reviews",
        sa.Column("job_id", sa.String(36), primary_key=True),
        sa.Column("revision", sa.Integer, primary_key=True),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("payload", sa.JSON, nullable=False),
    )


def downgrade():
    op.drop_table("workbench_reviews")
    op.drop_table("workbench_jobs")
