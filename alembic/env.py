import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, text

from alembic import context

# Alembic Config object
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import project models so autogenerate can detect them
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import Base  # noqa: E402 — registers all ORM models

target_metadata = Base.metadata


def _db_url() -> str:
    """Build sync psycopg2 URL from env (Alembic uses sync driver)."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "regllm")
    user = os.getenv("POSTGRES_USER", "regllm")
    password = os.getenv("POSTGRES_PASSWORD", "changeme")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def run_migrations_offline() -> None:
    # Always prefer env vars; fall back to alembic.ini only if env vars are absent.
    url = _db_url() or config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _db_url() or config.get_main_option("sqlalchemy.url")
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = url

    connectable = engine_from_config(cfg, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        # Ensure pgvector extension exists before running migrations
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()

        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
