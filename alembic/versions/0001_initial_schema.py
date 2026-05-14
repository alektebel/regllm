"""Initial schema — all tables.

Revision ID: 0001
Revises:
Create Date: 2026-05-15
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # pgvector extension is enabled by env.py before migrations run
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("pregunta", sa.Text(), nullable=False),
        sa.Column("respuesta", sa.Text(), nullable=False),
        sa.Column("fuentes", sa.JSON(), nullable=True),
        sa.Column("score_confianza", sa.Float(), nullable=True),
        sa.Column("nivel_confianza", sa.String(length=20), nullable=True),
        sa.Column("advertencias", sa.JSON(), nullable=True),
        sa.Column("modelo", sa.String(length=200), nullable=True),
        sa.Column("latencia_ms", sa.Integer(), nullable=True),
        sa.Column("ip_cliente", sa.String(length=45), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "qa_interactions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("reference_answer", sa.Text(), nullable=True),
        sa.Column("model_answer", sa.Text(), nullable=False),
        sa.Column("question_embedding", sa.Text(), nullable=True),  # vector(384)
        sa.Column("answer_embedding", sa.Text(), nullable=True),    # vector(384)
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("source", sa.String(length=200), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        "ALTER TABLE qa_interactions "
        "ALTER COLUMN question_embedding TYPE vector(384) USING question_embedding::vector, "
        "ALTER COLUMN answer_embedding TYPE vector(384) USING answer_embedding::vector"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS qa_interactions_question_emb_idx "
        "ON qa_interactions USING hnsw (question_embedding vector_cosine_ops)"
    )

    op.create_table(
        "user_feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=True),
        sa.Column("question_text", sa.Text(), nullable=True),
        sa.Column("rating", sa.String(length=20), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["question_id"], ["qa_interactions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("backend", sa.String(length=20), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chunk_id", sa.String(length=500), nullable=False),
        sa.Column("texto", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("embedding", sa.Text(), nullable=True),  # vector(768)
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id"),
    )
    op.execute(
        "ALTER TABLE document_chunks "
        "ALTER COLUMN embedding TYPE vector(768) USING embedding::vector"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS document_chunks_emb_idx "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.create_index("ix_document_chunks_chunk_id", "document_chunks", ["chunk_id"])

    op.create_table(
        "citation_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chunk_id", sa.String(length=500), nullable=False),
        sa.Column("texto", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("embedding", sa.Text(), nullable=True),  # vector(384)
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id"),
    )
    op.execute(
        "ALTER TABLE citation_chunks "
        "ALTER COLUMN embedding TYPE vector(384) USING embedding::vector"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS citation_chunks_emb_idx "
        "ON citation_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.create_index("ix_citation_chunks_chunk_id", "citation_chunks", ["chunk_id"])


def downgrade() -> None:
    op.drop_table("citation_chunks")
    op.drop_table("document_chunks")
    op.drop_table("conversation_messages")
    op.drop_table("conversations")
    op.drop_table("users")
    op.drop_table("user_feedback")
    op.drop_table("qa_interactions")
    op.drop_table("query_logs")
