"""Create the AudioScope catalog and release-readiness schema."""

from alembic import op

from audioscope.db import SCHEMA, Base

revision = "0001_audioscope_catalog"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
    op.execute(f"DROP SCHEMA IF EXISTS {SCHEMA}")
