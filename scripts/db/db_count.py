#!/usr/bin/env python3
"""
Print row counts for the two main RegLLM PostgreSQL tables.

Usage:
    python scripts/db_count.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.db import get_engine
from sqlalchemy import text


async def count_tables() -> None:
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT 'query_logs'     AS table_name, COUNT(*) AS row_count FROM query_logs
            UNION ALL
            SELECT 'qa_interactions', COUNT(*) FROM qa_interactions
        """))
        rows = result.fetchall()

    print("\nPostgreSQL table counts:")
    print(f"  {'Table':<20} {'Rows':>8}")
    print(f"  {'-'*20} {'-'*8}")
    for table_name, row_count in rows:
        print(f"  {table_name:<20} {row_count:>8,}")
    print()

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(count_tables())
