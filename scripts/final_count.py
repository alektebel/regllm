import asyncio
import sys
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.db import get_session


async def count_all():
    async with get_session() as session:
        q_count = (
            await session.execute(text("SELECT COUNT(*) FROM qa_interactions"))
        ).scalar()
        l_count = (
            await session.execute(text("SELECT COUNT(*) FROM query_logs"))
        ).scalar()
        f_count = (
            await session.execute(text("SELECT COUNT(*) FROM user_feedback"))
        ).scalar()

        print(f"qa_interactions: {q_count}")
        print(f"query_logs: {l_count}")
        print(f"user_feedback: {f_count}")


if __name__ == "__main__":
    asyncio.run(count_all())
