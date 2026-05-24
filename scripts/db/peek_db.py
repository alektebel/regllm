import asyncio
import sys
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.db import get_session


async def peek():
    async with get_session() as session:
        result = await session.execute(
            text(
                "SELECT id, question, model_answer, created_at FROM qa_interactions ORDER BY created_at DESC LIMIT 3"
            )
        )
        rows = result.mappings().all()
        print("\n--- Extracción de la Base de Datos (Últimas 3 entradas) ---")
        for row in rows:
            print(f"\n[ID {row['id']}] | Fecha: {row['created_at']}")
            print(f"Pregunta: {row['question']}")
            # Limpiar HTML si existe en la respuesta guardada
            ans = (
                row["model_answer"]
                .replace('<div class="resp-card">', "")
                .replace("</div>", "\n")
            )
            if len(ans) > 400:
                ans = ans[:400] + "..."
            print(f"Respuesta: {ans.strip()}")
            print("-" * 60)


if __name__ == "__main__":
    asyncio.run(peek())
