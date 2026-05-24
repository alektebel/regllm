#!/usr/bin/env python3
"""
Enhanced QA generator with varied prompts simulating different user personas.

This script extends the base QA generator to create more diverse training data
by using different question styles, formality levels, and user personas.
"""

import argparse
import json
import logging
import random
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# Import the base QA generator components
sys.path.insert(0, str(Path(__file__).parent))
from generate_qa_from_docs import (
    RAGEnricher,
    TransformersBackend,
    OllamaBackend,
    load_document,
    detect_chapters,
    chunk_text,
    extract_pairs,
    LOADERS,
    SYSTEM_PROMPT,
    PROJECT_ROOT,
    # Display helpers
    print_doc_header,
    print_chapter_header,
    print_chunk_header,
    print_rag_info,
    print_pairs,
    print_summary,
    dim,
    bold,
    cyan,
    green,
    yellow,
    grey,
    blue,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User personas and question styles
# ---------------------------------------------------------------------------

USER_PERSONAS = [
    {
        "name": "Analista Junior",
        "style": "formal pero con dudas",
        "question_starters": [
            "Disculpa, ¿podrías explicarme",
            "Tengo una duda sobre",
            "¿Me podrías aclarar",
            "No entiendo bien",
            "¿Alguien me puede ayudar con",
        ],
        "formality": "medio",
    },
    {
        "name": "Risk Manager Senior",
        "style": "directo y técnico",
        "question_starters": [
            "¿Cuál es exactamente",
            "Necesito confirmar",
            "¿Qué establece",
            "¿Cuáles son los requisitos de",
            "Dame los detalles específicos sobre",
        ],
        "formality": "alto",
    },
    {
        "name": "Auditor Externo",
        "style": "muy formal y preciso",
        "question_starters": [
            "¿Podría indicarme la referencia normativa de",
            "¿Qué dice la regulación sobre",
            "Necesito la cita exacta de",
            "¿Cuál es el artículo que especifica",
            "¿Qué normativa aplica para",
        ],
        "formality": "muy alto",
    },
    {
        "name": "Estudiante MBA",
        "style": "curioso pero menos técnico",
        "question_starters": [
            "Hola, ¿me puedes explicar",
            "Estoy estudiando sobre",
            "¿Cómo funciona",
            "¿Qué significa",
            "No me queda claro el concepto de",
        ],
        "formality": "bajo",
    },
    {
        "name": "Director Financiero",
        "style": "enfocado en impacto de negocio",
        "question_starters": [
            "¿Cómo afecta esto a",
            "¿Qué implicaciones tiene",
            "¿Cuál es el impacto de",
            "¿Qué significa esto para",
            "¿Cómo debemos interpretar",
        ],
        "formality": "alto",
    },
    {
        "name": "Consultor Regulatorio",
        "style": "comparativo y analítico",
        "question_starters": [
            "¿Cuál es la diferencia entre",
            "Compara",
            "¿En qué se diferencia",
            "¿Qué cambió respecto a",
            "¿Cómo se relaciona",
        ],
        "formality": "medio",
    },
]

QUESTION_TYPES = [
    {
        "type": "definición",
        "templates": [
            "¿Qué es {concepto}?",
            "¿Puedes definir {concepto}?",
            "¿Cómo se define {concepto}?",
            "Explícame el concepto de {concepto}",
        ],
    },
    {
        "type": "procedimiento",
        "templates": [
            "¿Cómo se calcula {concepto}?",
            "¿Cuál es el proceso para {concepto}?",
            "¿Qué pasos hay que seguir para {concepto}?",
            "Explica el procedimiento de {concepto}",
        ],
    },
    {
        "type": "requisito",
        "templates": [
            "¿Qué requisitos establece {concepto}?",
            "¿Cuáles son los requerimientos de {concepto}?",
            "¿Qué exige {concepto}?",
            "¿Qué condiciones debe cumplir {concepto}?",
        ],
    },
    {
        "type": "comparación",
        "templates": [
            "¿Cuál es la diferencia entre {concepto1} y {concepto2}?",
            "Compara {concepto1} con {concepto2}",
            "¿En qué se diferencian {concepto1} y {concepto2}?",
            "¿Qué distingue {concepto1} de {concepto2}?",
        ],
    },
    {
        "type": "referencia",
        "templates": [
            "¿Qué dice {concepto} sobre esto?",
            "¿Qué establece {concepto}?",
            "¿Qué normativa hay en {concepto}?",
            "Dame la referencia de {concepto}",
        ],
    },
    {
        "type": "aplicación",
        "templates": [
            "¿Cuándo se aplica {concepto}?",
            "¿En qué casos se usa {concepto}?",
            "¿Qué entidades deben aplicar {concepto}?",
            "¿A quién afecta {concepto}?",
        ],
    },
]

# Conversational fillers and expressions
CONVERSATIONAL_FILLERS = [
    "Buenas tardes, ",
    "Hola, ",
    "Buenos días, ",
    "Disculpa, ",
    "Oye, ",
    "Perdona, ",
    "",  # No filler
    "",
    "",
]

FOLLOW_UP_EXPRESSIONS = [
    "Y además, ",
    "También quisiera saber ",
    "Otra duda: ",
    "Y ",
    "",
]


def generate_varied_prompt(
    chunk: str, n: int, rag_context: str, citation_paths: str, persona: dict
) -> str:
    """Generate a prompt that simulates a specific user persona."""

    rag_parts = []
    if rag_context:
        rag_parts.append("\nREFERENCIAS RELACIONADAS EN LA BASE DE CONOCIMIENTO:")
        rag_parts.append(rag_context)
    if citation_paths:
        rag_parts.append("\nRUTAS DE CITACIÓN DISPONIBLES:")
        rag_parts.append(citation_paths)
    if rag_parts:
        rag_parts.append(
            "\nUsa estas referencias para enriquecer las respuestas con citas regulatorias exactas."
        )
    rag_section = "\n".join(rag_parts)

    citation_instruction = (
        "citar exactamente las rutas de citación disponibles cuando sean relevantes"
        if citation_paths
        else "referenciar el texto regulatorio cuando sea posible"
    )

    # Adjust instructions based on persona formality
    if persona["formality"] == "muy alto":
        answer_style = (
            "extremadamente formal y precisa, con referencias normativas completas"
        )
    elif persona["formality"] == "alto":
        answer_style = "formal y técnica, con referencias claras"
    elif persona["formality"] == "medio":
        answer_style = "profesional pero accesible"
    else:
        answer_style = "clara y didáctica, explicando términos técnicos"

    prompt = textwrap.dedent(f"""\
        Eres un experto en regulación bancaria. Vas a generar pares pregunta-respuesta simulando a diferentes usuarios reales.
        
        CONTEXTO REGULATORIO:
        {chunk}
        {rag_section}
        
        PERFIL DE USUARIO A SIMULAR: {persona["name"]}
        Estilo: {persona["style"]}
        
        TAREA: Genera exactamente {n} pares pregunta-respuesta en ESPAÑOL basados en el fragmento anterior.
        
        INSTRUCCIONES PARA LAS PREGUNTAS:
        - Simula el estilo de pregunta de un {persona["name"]} real
        - Usa diferentes formas de preguntar (directa, con contexto, comparativa, etc.)
        - Varía la longitud y complejidad de las preguntas
        - Algunas preguntas pueden incluir errores menores de escritura o ser coloquiales
        - Usa expresiones naturales como las que usaría este perfil de usuario
        
        INSTRUCCIONES PARA LAS RESPUESTAS:
        - Las respuestas deben ser {answer_style}
        - Deben {citation_instruction}
        - Ser completas pero concisas
        - Adaptarse al nivel de la pregunta
        
        IMPORTANTE: Las preguntas deben sonar naturales y variadas, como si las hicieran personas reales diferentes.
        
        Devuelve SOLO un objeto JSON con esta estructura exacta (sin texto adicional antes ni después):
        {{
          "pairs": [
            {{"question": "...", "answer": "..."}},
            {{"question": "...", "answer": "..."}},
            {{"question": "...", "answer": "..."}}
          ]
        }}
    """)

    return prompt


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_qa_varied(
    docs_dir: Path,
    output_path: Path,
    backend: str,
    model: str,
    pairs_per_chunk: int,
    chunk_size: int,
    max_docs: int | None,
    use_4bit: bool,
    ollama_url: str,
    n_rag_results: int,
    single_file: Path | None = None,
    device: str = "auto",
    max_new_tokens: int = 512,
    vary_personas: bool = True,
):
    """Generate QA pairs with varied user personas."""

    # ── Collect documents ────────────────────────────────────────────────
    if single_file is not None:
        if not single_file.is_file():
            logger.error(f"--file no existe: {single_file}")
            sys.exit(1)
        if single_file.suffix.lower() not in LOADERS:
            logger.error(f"Formato no soportado: {single_file.suffix}")
            sys.exit(1)
        all_files = [single_file]
    else:
        all_files = [
            p
            for p in sorted(docs_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in LOADERS
        ]
        if not all_files:
            logger.error(f"No se encontraron documentos compatibles en {docs_dir}")
            sys.exit(1)
        if max_docs:
            all_files = all_files[:max_docs]
    logger.info(f"Documentos a procesar: {len(all_files)}")

    # ── RAG + RCT enricher ───────────────────────────────────────────────
    enricher = RAGEnricher(
        vector_db_dir=PROJECT_ROOT / "vector_db" / "chroma_db",
        citation_tree_dir=PROJECT_ROOT / "data" / "citation_trees",
    )

    # First-pass indexing if needed
    if enricher.rag_doc_count == 0:
        print(
            f"\n{yellow('  ChromaDB vacío — indexando documentos primero (pasada 1/2)...')}"
        )
        docs_to_index = []
        for fp in all_files:
            text = load_document(fp)
            if text:
                docs_to_index.append(
                    {
                        "texto": text,
                        "metadata": {
                            "documento": fp.stem,
                            "source": fp.name,
                            "documento_id": str(fp),
                            "tipo": fp.suffix.lstrip("."),
                        },
                    }
                )
        enricher.index_documents(docs_to_index)
        print(
            f"  {enricher.rag_doc_count} fragmentos indexados. Iniciando generación...\n"
        )

    # ── LLM backend ──────────────────────────────────────────────────────
    if backend == "transformers":
        llm = TransformersBackend(model, use_4bit=use_4bit, device=device)
    else:
        llm = OllamaBackend(model, base_url=ollama_url)

    # ── Generation loop ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for doc_idx, doc_path in enumerate(all_files, 1):
            text = load_document(doc_path)
            if not text:
                logger.warning(f"Sin texto: {doc_path.name}, omitiendo.")
                continue

            # ── Chapter detection ─────────────────────────────────────
            chapters = detect_chapters(text)
            if chapters:
                sections: list[tuple[str, list[str]]] = [
                    (title, chunk_text(body, chunk_size=chunk_size))
                    for title, body in chapters
                ]
                print_doc_header(
                    doc_path.name, doc_idx, len(all_files), n_chapters=len(chapters)
                )
            else:
                sections = [("", chunk_text(text, chunk_size=chunk_size))]
                print_doc_header(doc_path.name, doc_idx, len(all_files), n_chapters=0)

            print(dim(f"  {len(text.split())} palabras en total"))

            for ch_idx, (ch_title, chunks) in enumerate(sections):
                if ch_title:
                    print_chapter_header(ch_title, ch_idx + 1, len(sections))
                    print(dim(f"  {len(chunks)} sub-fragmento(s)"))

                for chunk_idx, chunk in enumerate(chunks):
                    # Select random persona for this chunk
                    persona = (
                        random.choice(USER_PERSONAS)
                        if vary_personas
                        else USER_PERSONAS[1]
                    )

                    print_chunk_header(chunk_idx, len(chunks))
                    print(
                        cyan(
                            f"  👤 Persona: {bold(persona['name'])} ({persona['style']})"
                        )
                    )

                    # ── RAG + RCT context ──────────────────────────────
                    search_query = (
                        f"{ch_title} {chunk[:400]}".strip() if ch_title else chunk[:500]
                    )
                    rag_context, citation_paths = enricher.get_context(
                        search_query, n_rag_results
                    )
                    print_rag_info(rag_context, citation_paths)

                    # ── Build persona-specific prompt ──────────────────
                    prompt = generate_varied_prompt(
                        chunk, pairs_per_chunk, rag_context, citation_paths, persona
                    )

                    # ── Generate (streams tokens live) ─────────────────
                    try:
                        raw = llm.generate(prompt, max_new_tokens=max_new_tokens)
                    except Exception as e:
                        logger.warning(f"Fallo generación fragmento {chunk_idx}: {e}")
                        continue

                    # ── Extract and display pairs ──────────────────────
                    pairs = extract_pairs(raw)
                    if not pairs:
                        print(dim("  (sin pares extraídos)"))
                        continue

                    print_pairs(pairs)

                    # ── Save to JSONL ──────────────────────────────────
                    for pair in pairs:
                        record = {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": pair["question"]},
                                {"role": "assistant", "content": pair["answer"]},
                            ],
                            "metadata": {
                                "source_file": doc_path.name,
                                "chapter": ch_title or None,
                                "chunk_index": chunk_idx,
                                "persona": persona["name"],
                                "formality": persona["formality"],
                                "rag_used": bool(rag_context),
                                "citations_found": bool(citation_paths),
                                "generated_at": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                                "generator": "varied_personas",
                            },
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        total_pairs += 1

                    print(dim(f"\n  Total acumulado: {total_pairs} pares guardados"))

    print_summary(total_pairs, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Genera pares QA con estilos variados simulando diferentes usuarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Procesar un único archivo",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directorio con documentos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "finetuning" / "generated_qa_varied.jsonl",
        help="Ruta del JSONL de salida",
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "ollama"],
        default="ollama",
        help="Backend LLM",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:14b-instruct-q4_K_M",
        help="ID de modelo",
    )
    parser.add_argument(
        "--pairs-per-chunk",
        type=int,
        default=3,
        help="Pares QA por fragmento",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="Tamaño de fragmento en palabras",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Máximo de documentos a procesar",
    )
    parser.add_argument(
        "--rag-results",
        type=int,
        default=3,
        help="Pasajes RAG a recuperar",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Dispositivo para el modelo",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=600,
        help="Tokens máximos a generar",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Desactivar cuantización 4-bit",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="URL del servidor Ollama",
    )
    parser.add_argument(
        "--no-vary-personas",
        action="store_true",
        help="No variar personas (usar solo estilo técnico)",
    )

    args = parser.parse_args()

    generate_qa_varied(
        docs_dir=args.docs_dir,
        output_path=args.output,
        backend=args.backend,
        model=args.model,
        pairs_per_chunk=args.pairs_per_chunk,
        chunk_size=args.chunk_size,
        max_docs=args.max_docs,
        use_4bit=not args.no_quantize,
        ollama_url=args.ollama_url,
        n_rag_results=args.rag_results,
        single_file=args.file,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        vary_personas=not args.no_vary_personas,
    )


if __name__ == "__main__":
    main()
