"""Build the Phase 1 domain corpus for continued pretraining.

Collects all domain-specific text (regulation, schemas, SAS code, docs,
KG ontology, structured data) into a single text file for causal LM
pretraining.

Augmentation strategies:
  1. Raw text files (regulation .md, docs, SAS code) — repeated per config weight
  2. Structured JSON → narrative Spanish text (EBA articles, mappings, KG graph)
  3. SQL DDL → descriptive Spanish text (table/column documentation)
  4. CSV sample data → natural language row descriptions
  5. AdaptLLM-style reading comprehension (domain text → Q&A pairs)
  6. PDF extraction (EBA GL/2017/16 raw text)
  7. Built-in glossaries (credit risk, KG patterns)

Documents are repeated according to config weights. Output is a plain
text file with <|endoftext|> separators.

Usage:
    python training/build_corpus.py [--config training/config.yaml]
    python training/build_corpus.py --stats   # just print token stats
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _collect_files(source_path: Path) -> list[Path]:
    """Collect all text files from a path (file or directory)."""
    if source_path.is_file():
        return [source_path]
    if source_path.is_dir():
        files = []
        for ext in ("*.md", "*.sql", "*.sas", "*.py", "*.txt", "*.json"):
            files.extend(sorted(source_path.rglob(ext)))
        return files
    return []


def _read_safe(path: Path) -> str:
    """Read a file, skipping binary/unreadable content."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return ""


# ── Credit risk glossary (always included) ─────────────────────────────────

CREDIT_RISK_GLOSSARY_ES = """
# Glosario de Riesgo de Crédito — Terminología IRB/IFRS9

## Probabilidad de Impago (PD)
La PD es la probabilidad estimada de que un deudor entre en situación de
impago dentro de un horizonte temporal (12 meses para Stage 1 IFRS9,
lifetime para Stage 2/3). Se estima mediante modelos estadísticos
calibrados con datos históricos. Regulada por CRR Art. 160-164.

## Severidad / Loss Given Default (LGD)
La LGD es la pérdida esperada como porcentaje de la exposición en caso
de impago. Depende del tipo de garantía, antigüedad de la deuda y ciclo
económico. CRR establece floors: 25% para exposiciones senior sin
garantía (corporativos), 10% con garantía inmobiliaria residencial.

## Exposición al Impago (EAD)
EAD = saldo dispuesto + CCF × saldo no dispuesto. Representa la
exposición total esperada en el momento del impago. CRR Art. 166.

## Factor de Conversión Crediticia (CCF)
Convierte compromisos fuera de balance en equivalentes de exposición.
Un CCF del 75% asume que el deudor dispondrá del 75% del límite no
utilizado antes del impago.

## Pérdida Crediticia Esperada (ECL)
ECL = PD × LGD × EAD. Bajo IFRS9:
- Stage 1: ECL 12 meses
- Stage 2: ECL lifetime
- Stage 3: ECL lifetime con PD = 100%

## Activos Ponderados por Riesgo (RWA)
RWA = EAD × función_de_ponderación(PD, LGD, M, ρ). Determina el
capital regulatorio mínimo (8% de RWA). CRR Art. 153-154.

## Stages IFRS9
- Stage 1: Sin incremento significativo de riesgo crediticio.
- Stage 2: Incremento significativo (SICR). Criterios: Δrating, DPDS>30.
- Stage 3: Impago/default. DPDS>90 o criterios cualitativos.

## Dotaciones y Provisiones
Las dotaciones son los importes reservados por el banco para cubrir
pérdidas esperadas. La Circular 4/2017 del BdE establece períodos
de dotación y mínimos según tipo de operación y garantía.

## Segmentación IRB
- RETAIL_HIP: hipotecas residenciales
- RETAIL_CONS: consumo
- CORP: corporativos
- PYME: pequeñas y medianas empresas
- SOB: soberanos/instituciones

## Cure Rate / CURE_FLAG
Indica si un ciclo de impago se ha curado (el deudor volvió a estar
al corriente de pago). Relevante para calibración de LGD workout.

## MOC (Margin of Conservatism)
Margen adicional sobre las estimaciones de PD/LGD para compensar
incertidumbre en los datos o modelos. Exigido por EBA GL/2017/16.

## DPDS (Días Pasados Desde impago)
Días transcurridos desde la fecha de entrada en impago. Criterio
cuantitativo principal para clasificación en Stage 3 (>90 días).
"""

# ── Cypher / KG patterns ──────────────────────────────────────────────────

KG_PATTERNS_DOC = """
# Patrones de Consulta del Grafo de Conocimiento Regulatorio

## Ontología del Grafo
Tipos de nodo: Regulation, Interpretation, Concept, Table, Column,
ValidationRule, Experience, Quirk, Insight.

Tipos de arista: GOVERNS, IMPLEMENTS, EXCEPTION_TO, CLARIFIED_BY,
RESOLVES, SUPERSEDES, RELATES_TO, SUBJECT_TO, DEFINITION_IN,
HAS_QUIRK, DISCOVERED_IN, VALIDATES, NEEDS_HUMAN_REVIEW.

## Ejemplos de Traversal

### Encontrar regulaciones que gobiernan una columna
MATCH (r:Entity {node_type: 'Regulation'})-[e:Edge {edge_type: 'GOVERNS'}]->(c:Entity {node_type: 'Column'})
WHERE c.label CONTAINS 'LGD_ESTIMADA'
RETURN r.label, e.edge_type, c.label

### Trazar cadena regulatoria completa
MATCH path = shortestPath(
  (reg:Entity {node_type: 'Regulation'})-[*..5]-(col:Entity {node_type: 'Column'})
)
WHERE reg.label CONTAINS 'art15' AND col.label CONTAINS 'LGD'
RETURN path

### Buscar excepciones a reglas de validación
MATCH (q:Entity {node_type: 'Quirk'})-[:Edge {edge_type: 'EXCEPTION_TO'}]->(vr:Entity {node_type: 'ValidationRule'})
RETURN q.label, vr.label

### Insights de alta prioridad (feedback del usuario)
MATCH (i:Entity {node_type: 'Insight'})
WHERE i.props CONTAINS '"priority": 1.0'
RETURN i.label, i.props
ORDER BY i.id DESC

## Herramientas del Agente para el Grafo
- query_regulation(concept, article, hops): Traversal multi-hop desde un concepto o artículo.
- find_governing_rules(column): Reglas y artículos que restringen una columna.
- trace_regulation_chain(article_id, target_column): Camino más corto regulación → columna.
- search_experience(query): Buscar insights y experiencias pasadas.
- save_insight(insight, related_fields): Persistir aprendizaje en el grafo.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Augmentation: Structured data → natural language
# ═══════════════════════════════════════════════════════════════════════════


def _augment_eba_articles_json(path: Path) -> list[str]:
    """Convert EBA GL/2017/16 articles JSON → narrative Spanish paragraphs."""
    if not path.exists():
        return []

    articles = json.loads(path.read_text(encoding="utf-8"))
    docs = []

    for art in articles:
        section = art.get("section", "").strip()
        subsection = art.get("subsection", "").strip()
        text = art.get("text", "").strip()
        par_num = art.get("paragraph", "")
        if not text:
            continue

        header = f"# EBA GL/2017/16 — Párrafo {par_num}"
        if section:
            header += f"\nSección: {section}"
        if subsection:
            header += f"\nSubsección: {subsection}"

        docs.append(f"{header}\n\n{text}")

    return docs


def _augment_mapping_json(path: Path) -> list[str]:
    """Convert audit mapping JSON → natural language field descriptions."""
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    mappings = data.get("mappings", data)
    if not isinstance(mappings, dict):
        return []

    docs = []
    for field_name, info in mappings.items():
        if not isinstance(info, dict):
            continue

        lines = [f"# Campo: {field_name}"]

        concept = info.get("regulation_concept", "")
        if concept:
            lines.append(f"Concepto regulatorio: {concept}")

        comp = info.get("computation_description", "")
        if comp:
            lines.append(f"\nCálculo: {comp}")

        reg_desc = info.get("regulation_description", "")
        if reg_desc:
            lines.append(f"\nRegulación aplicable: {reg_desc}")

        articles = info.get("articles", [])
        if articles:
            lines.append(f"\nArtículos relacionados: {', '.join(articles)}")

        lineage = info.get("sas_lineage", [])
        if lineage:
            lines.append(f"\nLinaje SAS (campos fuente): {', '.join(lineage)}")

        notes = info.get("notes", "")
        if notes:
            lines.append(f"\nNotas: {notes}")

        docs.append("\n".join(lines))

    return docs


def _augment_regulation_graph(path: Path) -> list[str]:
    """Convert knowledge graph JSON → natural language relationship descriptions."""
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    docs = []

    # Convert nodes to descriptions
    nodes_by_id = {}
    for node in data.get("nodes", []):
        nodes_by_id[node["id"]] = node
        ntype = node.get("type", "")
        label = node.get("label", "")
        body = node.get("body", "")
        if body:
            docs.append(
                f"# Nodo del grafo regulatorio [{ntype}]: {label}\n\n{body}"
            )

    # Convert edges to natural language
    edge_templates = {
        "REFERENCES": "hace referencia a",
        "CONTAINS": "contiene",
        "GOVERNS": "gobierna",
        "IMPLEMENTS": "implementa",
        "EXCEPTION_TO": "es una excepción a",
        "CLARIFIED_BY": "es clarificado por",
        "RELATES_TO": "se relaciona con",
        "SUBJECT_TO": "está sujeto a",
        "VALIDATES": "valida",
        "HAS_QUIRK": "tiene la particularidad",
    }

    for edge in data.get("edges", []):
        src = nodes_by_id.get(edge.get("source", ""), {})
        tgt = nodes_by_id.get(edge.get("target", ""), {})
        etype = edge.get("type", edge.get("label", ""))
        src_label = src.get("label", edge.get("source", ""))
        tgt_label = tgt.get("label", edge.get("target", ""))
        verb = edge_templates.get(etype, f"[{etype}]")

        docs.append(
            f"Relación regulatoria: «{src_label}» {verb} «{tgt_label}»."
        )

    return docs


def _augment_sql_schema(path: Path) -> list[str]:
    """Convert SQL DDL → descriptive Spanish text for each table/column."""
    if not path.exists():
        return []

    sql = path.read_text(encoding="utf-8")
    docs = []

    # Extract CREATE TABLE blocks
    table_pattern = re.compile(
        r'CREATE TABLE (\w+)\s*\((.*?)\);',
        re.DOTALL | re.IGNORECASE,
    )
    comment_pattern = re.compile(
        r"COMMENT ON COLUMN (\w+\.\w+) IS '(.*?)';",
        re.DOTALL,
    )

    # Collect comments
    comments: dict[str, str] = {}
    for m in comment_pattern.finditer(sql):
        comments[m.group(1).lower()] = m.group(2)

    for m in table_pattern.finditer(sql):
        table_name = m.group(1)
        body = m.group(2)

        lines = [f"# Tabla: {table_name}"]
        lines.append(f"\nEsta tabla forma parte del esquema IRB de riesgo de crédito.")
        lines.append(f"Columnas principales:\n")

        # Parse columns from body
        for col_line in body.split("\n"):
            col_line = col_line.strip().rstrip(",")
            if not col_line or col_line.startswith("FOREIGN") or col_line.startswith("--"):
                continue

            # Extract column name and inline comment
            parts = col_line.split()
            if not parts:
                continue
            col_name = parts[0]
            if col_name.upper() in ("FOREIGN", "CHECK", "CONSTRAINT", "PRIMARY", "UNIQUE"):
                continue

            inline_comment = ""
            if "--" in col_line:
                inline_comment = col_line.split("--", 1)[1].strip()

            full_key = f"{table_name}.{col_name}".lower()
            regulatory_comment = comments.get(full_key, "")

            desc = f"- **{col_name}**"
            if inline_comment:
                desc += f": {inline_comment}"
            if regulatory_comment:
                desc += f" Referencia regulatoria: {regulatory_comment}"

            lines.append(desc)

        docs.append("\n".join(lines))

    return docs


def _augment_csv_rows(path: Path, max_rows: int = 50) -> list[str]:
    """Convert CSV sample data rows → natural language descriptions."""
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(StringIO(text))
    rows = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        rows.append(row)

    if not rows:
        return []

    # Field name → Spanish description
    field_desc = {
        "CICLO_ID": "identificador de ciclo",
        "CONTRATO": "contrato",
        "SEGMENTO": "segmento",
        "RATING_GRADO": "grado de rating interno",
        "PD_ESTIMADA": "probabilidad de impago estimada",
        "LGD_ESTIMADA": "LGD estimada",
        "EAD": "exposición en caso de impago",
        "DPDS": "días en situación de impago",
        "STAGE_IFRS9": "estadio IFRS9",
        "ECL": "pérdida crediticia esperada",
        "PROVISION_PERIOD_MONTHS": "meses de período de dotación",
        "VENTANA_OBSERVACION_YEARS": "ventana de observación en años",
        "VENTANA_CALIBRACION_YEARS": "ventana de calibración en años",
        "COLATERAL_TIPO": "tipo de colateral",
        "LGD_FLOOR_APLICADO": "suelo LGD aplicado",
        "MoC": "margen de conservadurismo",
        "FECHA_INCUMPLIMIENTO": "fecha de incumplimiento",
        "CURE_FLAG": "indicador de curación",
        "RWA": "activos ponderados por riesgo",
    }

    docs = []
    for row in rows:
        parts = [f"Registro de ciclo de recuperación {row.get('CICLO_ID', '')}:"]
        for field, value in row.items():
            if not value or value == "":
                continue
            desc = field_desc.get(field, field)
            parts.append(f"  {desc} ({field}) = {value}")

        # Add interpretive text
        segment = row.get("SEGMENTO", "")
        dpds = row.get("DPDS", "0")
        stage = row.get("STAGE_IFRS9", "1")
        try:
            dpds_int = int(dpds)
            if dpds_int >= 360:
                parts.append("  → Fase de crisis (DPDS ≥ 360)")
            elif dpds_int >= 90:
                parts.append("  → Fase de contracción (90 ≤ DPDS < 360)")
            else:
                parts.append("  → Fase de expansión (DPDS < 90)")
        except ValueError:
            pass

        if stage == "3":
            parts.append("  → Exposición credit-impaired (Stage 3): PD efectiva = 1.0")
        elif stage == "2":
            parts.append("  → Incremento significativo de riesgo (Stage 2): ECL lifetime")

        docs.append("\n".join(parts))

    return docs


def _augment_reading_comprehension(text: str, source_name: str) -> list[str]:
    """AdaptLLM-style: convert domain text to reading comprehension Q&A.

    Generates deterministic Q&A pairs from the text structure
    (headers, key terms, regulatory references) without needing an LLM.
    """
    docs = []

    # Extract headers and their content
    sections = re.split(r'\n#{1,3}\s+', text)
    headers = re.findall(r'\n(#{1,3}\s+.+)', text)

    for i, header in enumerate(headers):
        header_text = header.strip("# \n")
        section_text = sections[i + 1] if i + 1 < len(sections) else ""
        section_text = section_text.strip()
        if len(section_text) < 50:
            continue

        # Truncate very long sections
        if len(section_text) > 800:
            section_text = section_text[:800] + "..."

        docs.append(
            f"Pregunta: ¿Qué establece la sección «{header_text}» en {source_name}?\n\n"
            f"Respuesta: {section_text}"
        )

    # Extract regulatory article references and generate Q&A
    art_refs = re.findall(
        r'(?:Art(?:ículo|\.)\s*(\d+(?:\.\d+)?(?:\s*(?:apartado|párrafo)\s*\d+)?))',
        text, re.IGNORECASE,
    )
    for ref in set(art_refs):
        # Find surrounding context
        pattern = re.compile(
            rf'[^.]*Art(?:ículo|\.)\s*{re.escape(ref)}[^.]*\.',
            re.IGNORECASE,
        )
        matches = pattern.findall(text)
        if matches:
            context = matches[0].strip()
            docs.append(
                f"Pregunta: ¿Qué dice el Artículo {ref} según {source_name}?\n\n"
                f"Respuesta: {context}"
            )

    # Extract field/variable references and generate Q&A
    field_refs = re.findall(r'`([A-Z_]{3,})`', text)
    for field in set(field_refs):
        pattern = re.compile(rf'[^.]*`{re.escape(field)}`[^.]*\.', re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            context = matches[0].strip()
            docs.append(
                f"Pregunta: ¿Qué es el campo {field} y cómo se utiliza?\n\n"
                f"Respuesta: {context}"
            )

    return docs


def _extract_pdf_text(path: Path) -> list[str]:
    """Extract text from a PDF file. Returns list of page texts."""
    if not path.exists():
        return []
    try:
        import pdfplumber
        docs = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    docs.append(f"# EBA GL/2017/16 — Página {page.page_number}\n\n{text}")
        return docs
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                docs.append(f"# EBA GL/2017/16 — Página {i + 1}\n\n{text}")
        return docs
    except ImportError:
        print(
            "WARNING: No PDF library available. Install pdfplumber or PyPDF2:\n"
            "  pip install pdfplumber",
            file=sys.stderr,
        )
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Main corpus builder
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 1 domain corpus")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--stats", action="store_true",
                        help="Print corpus statistics without writing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sources = cfg.get("domains", {}).get("sources", [])

    parts: list[str] = []

    # ── 1. Built-in glossaries ────────────────────────────────────────────
    for _ in range(5):
        parts.append(CREDIT_RISK_GLOSSARY_ES)
    for _ in range(3):
        parts.append(KG_PATTERNS_DOC)

    # ── 2. Raw text files from configured sources ─────────────────────────
    raw_count = 0
    for source in sources:
        source_path = PROJECT_ROOT / source["path"]
        repeat = source.get("repeat", 1)
        doc_type = source.get("type", "generic")

        files = _collect_files(source_path)
        for f in files:
            text = _read_safe(f)
            if not text.strip():
                continue
            header = f"\n\n# === {doc_type.upper()}: {f.name} ===\n\n"
            doc = header + text
            for _ in range(repeat):
                parts.append(doc)
                raw_count += 1

    print(f"[1/7] Raw text files: {raw_count} documents")

    # ── 3. Structured JSON → narrative text ───────────────────────────────
    # EBA articles
    eba_path = PROJECT_ROOT / "data" / "regulation" / "eba_gl_2017_16_articles.json"
    eba_docs = _augment_eba_articles_json(eba_path)
    for doc in eba_docs:
        for _ in range(2):  # repeat for emphasis
            parts.append(doc)
    print(f"[2/7] EBA articles (JSON→text): {len(eba_docs)} paragraphs")

    # Audit mapping
    mapping_path = PROJECT_ROOT / "data" / "audit" / "mapping.json"
    mapping_docs = _augment_mapping_json(mapping_path)
    for doc in mapping_docs:
        for _ in range(3):  # high weight — field-level knowledge
            parts.append(doc)
    print(f"[3/7] Audit field mappings: {len(mapping_docs)} fields")

    # Regulation graph
    graph_path = PROJECT_ROOT / "data" / "regulation" / "graph.json"
    graph_docs = _augment_regulation_graph(graph_path)
    for doc in graph_docs:
        parts.append(doc)
    print(f"[4/7] Regulation graph (nodes+edges): {len(graph_docs)} items")

    # ── 4. SQL schema → descriptive text ──────────────────────────────────
    schema_path = PROJECT_ROOT / "data" / "samples" / "irb_schema.sql"
    schema_docs = _augment_sql_schema(schema_path)
    for doc in schema_docs:
        for _ in range(3):  # high weight
            parts.append(doc)
    print(f"[5/7] SQL schema descriptions: {len(schema_docs)} tables")

    # ── 5. CSV sample data → natural language ─────────────────────────────
    csv_count = 0
    for csv_name in ("cycles_sample.csv", "cycles_v3.csv"):
        csv_path = PROJECT_ROOT / "data" / "samples" / csv_name
        csv_docs = _augment_csv_rows(csv_path, max_rows=50)
        for doc in csv_docs:
            parts.append(doc)
        csv_count += len(csv_docs)
    print(f"[6/7] CSV row descriptions: {csv_count} rows")

    # ── 6. PDF extraction ─────────────────────────────────────────────────
    pdf_path = PROJECT_ROOT / "raw" / "EBA-GL-2017-16-ES.pdf"
    pdf_docs = _extract_pdf_text(pdf_path)
    for doc in pdf_docs:
        for _ in range(2):  # repeat regulation text
            parts.append(doc)
    print(f"[7/7] PDF pages extracted: {len(pdf_docs)} pages")

    # ── 7. AdaptLLM-style reading comprehension ──────────────────────────
    rc_count = 0
    # Generate Q&A from regulation markdown files
    reg_dir = PROJECT_ROOT / "data" / "regulation"
    if reg_dir.is_dir():
        for f in sorted(reg_dir.glob("art_*.md")):
            text = _read_safe(f)
            if text:
                rc_docs = _augment_reading_comprehension(text, f.stem)
                parts.extend(rc_docs)
                rc_count += len(rc_docs)

    # Generate Q&A from the glossary itself
    rc_glossary = _augment_reading_comprehension(
        CREDIT_RISK_GLOSSARY_ES, "Glosario de Riesgo de Crédito"
    )
    parts.extend(rc_glossary)
    rc_count += len(rc_glossary)

    # Generate Q&A from doc markdown files
    docs_dir = PROJECT_ROOT / "data" / "docs"
    if docs_dir.is_dir():
        for f in sorted(docs_dir.glob("*.md")):
            text = _read_safe(f)
            if text:
                rc_docs = _augment_reading_comprehension(text, f.stem)
                parts.extend(rc_docs)
                rc_count += len(rc_docs)

    print(f"[+] Reading comprehension Q&A: {rc_count} pairs")

    # ── Shuffle and write ────────────────────────────────────────────────
    import random
    rng = random.Random(cfg.get("phase1", {}).get("seed", 42))
    rng.shuffle(parts)

    separator = "\n\n<|endoftext|>\n\n"
    corpus = separator.join(parts)

    total_chars = len(corpus)
    total_tokens_approx = total_chars // 4  # rough char/4 estimate

    print(f"\n{'='*60}")
    print(f"Total documents: {len(parts)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Approx tokens (chars/4): {total_tokens_approx:,}")
    print(f"{'='*60}")

    # Try accurate count with Qwen3 tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            cfg.get("base_model", "Qwen/Qwen3-4B"),
            trust_remote_code=True,
        )
        accurate_count = len(tok.encode(corpus))
        print(f"Accurate token count (Qwen3 tokenizer): {accurate_count:,}")
    except Exception as e:
        print(f"(Could not load tokenizer for accurate count: {e})")

    if args.stats:
        return

    out_path = PROJECT_ROOT / "training" / "data" / "phase1_corpus.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(corpus, encoding="utf-8")
    print(f"\nCorpus written to {out_path}")


if __name__ == "__main__":
    main()
