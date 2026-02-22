#!/usr/bin/env python3
"""
Generate Spanish QA training pairs from regulation documents using a local open-source LLM.

For each text chunk the script:
  1. Uses the RAG system (ChromaDB + BM25 hybrid search) to retrieve related
     passages already indexed in the vector DB.
  2. Looks up exact citation paths in the Regulation Citation Tree (RCT),
     e.g. "CRR > Article 92 > Paragraph 1".
  3. Injects both into the generation prompt so the LLM can write answers with
     precise regulatory references.
  4. Streams every generated token to the terminal in real-time.

Backends:
  transformers  — HuggingFace model running locally (GPU recommended)
  ollama        — Ollama server (no GPU required, https://ollama.ai)

Usage:
  # GPU — Qwen2.5-7B (default)
  python scripts/generate_qa_from_docs.py --docs-dir data/raw

  # Lighter model
  python scripts/generate_qa_from_docs.py --docs-dir data/raw --model Qwen/Qwen2.5-3B-Instruct

  # Ollama, no GPU  (run: ollama pull llama3.2)
  python scripts/generate_qa_from_docs.py --docs-dir data/raw --backend ollama --model llama3.2
"""

import argparse
import json
import logging
import re
import sys
import textwrap
import threading
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# Regex patterns for EU / Spanish regulation references
_REF_PATTERNS = [
    r'Art[ií]culo\s+\d+[a-z]?',
    r'Article\s+\d+[a-z]?',
    r'Secci[oó]n\s+[\d.]+',
    r'Ap[ae]rtado\s+\d+[a-z]?',
    r'P[aá]rrafo\s+\d+',
    r'Anexo\s+[IVX\d]+',
    r'EBA/(?:GL|RTS|ITS)/\d{4}/\d+',
    r'\bCRR\b',
    r'\bCRD\s*(?:IV|V)\b',
    r'\bIFRS\s*9\b',
    r'Basilea\s+(?:III|IV)',
    r'Basel\s+(?:III|IV)',
]
_REF_RE = re.compile("|".join(_REF_PATTERNS), re.IGNORECASE)


def extract_references(text: str) -> list[str]:
    """Extract regulation references from text (deduped, preserving order)."""
    seen = set()
    out = []
    for m in _REF_RE.finditer(text):
        ref = m.group(0).strip()
        if ref.lower() not in seen:
            seen.add(ref.lower())
            out.append(ref)
    return out


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def dim(t):    return _c("2",  t)
def bold(t):   return _c("1",  t)
def cyan(t):   return _c("36", t)
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def grey(t):   return _c("90", t)
def blue(t):   return _c("34", t)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Eres un experto en regulación bancaria y riesgo de crédito. "
    "Responde de forma precisa y detallada, citando siempre la fuente normativa cuando sea posible."
)

_BASE_GENERATION_PROMPT = textwrap.dedent("""\
    Eres un experto en regulación bancaria. Se te proporciona:

    FRAGMENTO DEL DOCUMENTO:
    {chunk}
    {rag_section}
    TAREA: Genera exactamente {n} pares pregunta-respuesta en ESPAÑOL basados en el fragmento anterior.
    - Las preguntas deben ser específicas y técnicas para profesionales del sector bancario.
    - Las respuestas deben ser completas y {citation_instruction}.

    Devuelve SOLO un objeto JSON con esta estructura exacta (sin texto adicional antes ni después):
    {{
      "pairs": [
        {{"question": "...", "answer": "..."}},
        {{"question": "...", "answer": "..."}},
        {{"question": "...", "answer": "..."}}
      ]
    }}
""")

def build_prompt(chunk: str, n: int, rag_context: str, citation_paths: str) -> str:
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
        if citation_paths else
        "referenciar el texto regulatorio cuando sea posible"
    )

    return _BASE_GENERATION_PROMPT.format(
        chunk=chunk,
        n=n,
        rag_section=rag_section,
        citation_instruction=citation_instruction,
    )


# ---------------------------------------------------------------------------
# RAG + Citation Tree enricher
# ---------------------------------------------------------------------------

class RAGEnricher:
    """
    Wraps RegulatoryRAGSystem (ChromaDB + BM25) and RegulationCitationTree.
    Provides per-chunk context enrichment for the generation prompt.
    """

    def __init__(self, vector_db_dir: Path, citation_tree_dir: Path):
        self.rag = None
        self.rct = None
        self._init_rag(vector_db_dir)
        self._init_rct(citation_tree_dir)

    # ------------------------------------------------------------------
    def _init_rag(self, db_dir: Path) -> None:
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from src.rag_system import RegulatoryRAGSystem  # noqa: PLC0415
            self.rag = RegulatoryRAGSystem(persist_directory=str(db_dir))
            n = self.rag.collection.count()
            logger.info(f"RAG inicializado: {n} fragmentos indexados en ChromaDB")
        except ImportError as e:
            logger.warning(f"RAG no disponible ({e}). Instala: sentence-transformers chromadb rank-bm25")
        except Exception as e:
            logger.warning(f"RAG no pudo cargarse: {e}")

    def _init_rct(self, tree_dir: Path) -> None:
        try:
            from src.citation_tree import RegulationCitationTree  # noqa: PLC0415
            self.rct = RegulationCitationTree(name="main")
            tree_files = sorted(tree_dir.glob("*.json")) if tree_dir.exists() else []
            total_nodes = 0
            for f in tree_files:
                try:
                    tmp = RegulationCitationTree()
                    tmp.load(str(f))
                    # Merge into main tree
                    for nid, node in tmp.nodes.items():
                        if nid not in self.rct.nodes:
                            self.rct.nodes[nid] = node
                    for rid in tmp.root_ids:
                        if rid not in self.rct.root_ids:
                            self.rct.root_ids.append(rid)
                    total_nodes += len(tmp.nodes)
                except Exception as e:
                    logger.warning(f"No se pudo cargar árbol de citación {f.name}: {e}")
            if total_nodes:
                self.rct._rebuild_indices()
                logger.info(f"RCT inicializado: {total_nodes} nodos de citación cargados")
            else:
                logger.info("RCT: no se encontraron archivos de citación (se usará árbol vacío)")
        except ImportError as e:
            logger.warning(f"Árbol de citación no disponible ({e})")

    # ------------------------------------------------------------------
    def index_documents(self, docs: list[dict]) -> None:
        """Index a list of {'texto': ..., 'metadata': {...}} dicts into ChromaDB."""
        if self.rag is None or not docs:
            return
        logger.info(f"Indexando {len(docs)} documentos en ChromaDB...")
        self.rag.procesar_documentos(docs)

    # ------------------------------------------------------------------
    def get_context(self, chunk_text: str, n_results: int = 3) -> tuple[str, str]:
        """
        Returns (rag_context_str, citation_paths_str) for a chunk of text.
        Falls back to ("", "") if RAG is not available or empty.
        """
        rag_context = ""
        citation_paths = ""

        if self.rag and self.rag.collection.count() > 0:
            try:
                results = self.rag.buscar_hibrida(chunk_text[:600], n_resultados=n_results)
                if results:
                    rag_context = self.rag.formatear_contexto(results)
                    citation_paths = self._resolve_citation_paths(results, chunk_text)
            except Exception as e:
                logger.debug(f"Error en búsqueda RAG: {e}")

        return rag_context, citation_paths

    # ------------------------------------------------------------------
    def _resolve_citation_paths(self, rag_results: list[dict], chunk_text: str) -> str:
        """
        Given RAG result chunks and the source chunk text, look up matching
        nodes in the RegulationCitationTree and return their full context paths.
        """
        if self.rct is None or not self.rct.nodes:
            return ""

        paths: set[str] = set()

        # 1. Extract regulation references from the chunk text + RAG passages
        combined_text = chunk_text[:1000] + " ".join(
            r.get("texto", "")[:200] for r in rag_results
        )
        refs = extract_references(combined_text)

        for ref in refs:
            # Keyword search in RCT
            nodes = self.rct.find_citations_by_keyword(ref, case_sensitive=False)
            for node in nodes:
                paths.add(node.context)

        # 2. Use RAG metadata to find context paths
        for result in rag_results:
            meta = result.get("metadata", {})
            for field_val in (
                meta.get("articulo", ""),
                meta.get("documento", ""),
                meta.get("source", ""),
            ):
                if not field_val:
                    continue
                nodes = self.rct.find_citations_by_keyword(field_val, case_sensitive=False)
                for node in nodes:
                    paths.add(node.context)
                # Also try context pattern search
                nodes_ctx = self.rct.find_citations_by_context(field_val)
                for node in nodes_ctx:
                    paths.add(node.context)

        return "\n".join(f"  - {p}" for p in sorted(paths)) if paths else ""

    # ------------------------------------------------------------------
    @property
    def rag_doc_count(self) -> int:
        if self.rag:
            try:
                return self.rag.collection.count()
            except Exception:
                return 0
        return 0


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> str:
    text = ""
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception as e:
        logger.debug(f"PyPDF2 falló para {path.name}: {e}, intentando pdfplumber")
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        except Exception as e2:
            logger.warning(f"No se pudo extraer texto de {path.name}: {e2}")
    return text.strip()


def _load_text(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
    return ""


def _load_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            parts = []
            for item in data:
                if isinstance(item, dict):
                    for key in ("text", "content", "body", "texto", "contenido"):
                        if key in item and isinstance(item[key], str):
                            parts.append(item[key])
                            break
                elif isinstance(item, str):
                    parts.append(item)
            return "\n\n".join(parts)
        if isinstance(data, dict):
            for key in ("text", "content", "body", "texto", "contenido"):
                if key in data and isinstance(data[key], str):
                    return data[key]
    except Exception as e:
        logger.warning(f"No se pudo parsear JSON {path.name}: {e}")
    return ""


def _load_jsonl(path: Path) -> str:
    parts = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                for key in ("text", "content", "body", "texto", "contenido"):
                    if key in item and isinstance(item[key], str):
                        parts.append(item[key])
                        break
    except Exception as e:
        logger.warning(f"No se pudo parsear JSONL {path.name}: {e}")
    return "\n\n".join(parts)


LOADERS = {
    ".pdf":  _load_pdf,
    ".txt":  _load_text,
    ".md":   _load_text,
    ".json": _load_json,
    ".jsonl": _load_jsonl,
}


def load_document(path: Path) -> str:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        return ""
    return loader(path)


# ---------------------------------------------------------------------------
# Chapter detection
# ---------------------------------------------------------------------------

# Matches chapter/section/title/part headings in EU and Spanish regulation docs.
# Handles: "CHAPTER I", "CAPÍTULO II", "SECTION 4", "TITLE III - General...",
#          "3. Scope", "IV. Background", "ANNEX I"
_HEADING_RE = re.compile(
    r'(?im)'
    r'^[ \t]*'
    r'(?:'
    # Keyword + Roman/Arabic: CHAPTER I, SECCIÓN 2, ANNEX III, …
    r'(?:CHAPTER|CAP[IÍ]TULO|T[IÍ]TULO?|TITLE|SECTION|SECCI[OÓ]N|PART|PARTE|ANNEX|ANEXO)'
    r'\s+(?:[IVXLCDM]+|\d+(?:\.\d+)?)'
    r'(?:[ \t]*[-–—:][ \t]*[^\n]{0,80})?'   # optional subtitle
    r'|'
    # Top-level numbered heading: "4. General provisions" or "4.1 Scope"
    r'\d{1,2}(?:\.\d{1,2}){0,1}\.?\s+[A-ZÁÉÍÓÚÑ][^\n]{5,70}'
    r'|'
    # Roman numeral heading: "IV. Purpose and background"
    r'[IVXLCDM]{2,6}\.[ \t]+[A-ZÁÉÍÓÚÑ][^\n]{5,70}'
    r')'
    r'[ \t]*$',
)

# Fallback: a line that is entirely upper-case (common in PDFs with lost formatting)
_ALLCAPS_RE = re.compile(r'(?m)^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s\-/\d]{7,69}$')


def detect_chapters(text: str) -> list[tuple[str, str]]:
    """
    Split a document into (chapter_title, chapter_text) pairs by detecting
    structural headings.

    Returns an empty list when no clear structure is found — the caller then
    falls back to plain word-based chunking.
    """
    matches = list(_HEADING_RE.finditer(text))

    # Fallback: all-caps standalone lines (common in extracted PDFs)
    if len(matches) < 2:
        matches = [
            m for m in _ALLCAPS_RE.finditer(text)
            if m.group().strip().isupper() and len(m.group().strip()) >= 8
        ]

    if len(matches) < 2:
        return []

    chapters: list[tuple[str, str]] = []

    # Preamble: text before the first heading
    preamble = text[: matches[0].start()].strip()
    if len(preamble.split()) >= 50:
        chapters.append(("Preámbulo", preamble))

    for i, match in enumerate(matches):
        title = match.group().strip()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()
        if len(content.split()) >= 10:          # skip near-empty sections
            chapters.append((title, content))

    return chapters


# ---------------------------------------------------------------------------
# Text chunking (used for sub-chunks within each chapter)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# LLM backends — both stream tokens to stdout in real-time
# ---------------------------------------------------------------------------

class TransformersBackend:
    """HuggingFace model with optional 4-bit quantization. Streams via TextIteratorStreamer."""

    def __init__(self, model_id: str, use_4bit: bool = True, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # CPU mode: quantization requires CUDA, so force it off
        on_cpu = device == "cpu"
        if on_cpu and use_4bit:
            logger.info("Modo CPU: desactivando cuantización 4-bit automáticamente.")
            use_4bit = False

        logger.info(f"Cargando modelo {model_id} (4-bit={use_4bit}, device={device}) ...")
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float32 if on_cpu else (torch.float16 if not use_4bit else None),
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        logger.info("Modelo cargado.")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        import torch
        from transformers import TextIteratorStreamer

        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        full_text = ""
        try:
            for token in streamer:
                print(grey(token), end="", flush=True)
                full_text += token
        except torch.cuda.OutOfMemoryError:
            print()
            torch.cuda.empty_cache()
            logger.warning(
                "OOM durante la generación. Fragmento omitido.\n"
                "  Prueba: --chunk-size 300 --max-new-tokens 256 o --device cpu"
            )
            return ""
        finally:
            thread.join()
            print()
        return full_text


class OllamaBackend:
    """Ollama server backend. Streams via the Ollama streaming API."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        import requests

        self.model = model
        self.base_url = base_url.rstrip("/")
        self._requests = requests

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            logger.info(f"Conectado a Ollama en {self.base_url}, modelo '{model}'")
        except Exception as e:
            logger.error(
                f"No se puede conectar con Ollama en {self.base_url}: {e}\n"
                "  Inícialo con: ollama serve\n"
                f"  Descarga el modelo con: ollama pull {model}"
            )
            sys.exit(1)

    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_new_tokens, "temperature": 0},
        }
        resp = self._requests.post(
            f"{self.base_url}/api/generate", json=payload, stream=True, timeout=180
        )
        resp.raise_for_status()

        full_text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("response", "")
            print(grey(token), end="", flush=True)
            full_text += token
            if data.get("done"):
                break
        print()
        return full_text


# ---------------------------------------------------------------------------
# QA pair extraction
# ---------------------------------------------------------------------------

def extract_pairs(raw_output: str) -> list[dict]:
    """Parse the JSON block from LLM output, tolerating minor formatting issues."""
    match = re.search(r'\{[\s\S]*"pairs"[\s\S]*\}', raw_output)
    if not match:
        return []
    json_str = match.group(0)
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)  # fix trailing commas
    try:
        data = json.loads(json_str)
        valid = []
        for p in data.get("pairs", []):
            if isinstance(p, dict) and p.get("question") and p.get("answer"):
                valid.append({"question": str(p["question"]), "answer": str(p["answer"])})
        return valid
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
_SEP = "─" * 60

def print_doc_header(name: str, idx: int, total: int, n_chapters: int = 0):
    print(f"\n{cyan(_SEP)}")
    chapter_info = f"  —  {n_chapters} capítulos detectados" if n_chapters else "  —  sin capítulos (chunking por palabras)"
    print(cyan(bold(f"  Documento [{idx}/{total}]: {name}")) + cyan(chapter_info))
    print(cyan(_SEP))

def print_chapter_header(title: str, idx: int, total: int):
    label = textwrap.shorten(title, width=55, placeholder="…")
    print(f"\n  {yellow(bold(f'[Capítulo {idx}/{total}]'))} {bold(label)}")
    print(dim("  " + "─" * 54))

def print_chunk_header(idx: int, total: int):
    print(f"\n  {dim(f'Fragmento {idx + 1}/{total}')}  {dim('(generando...)')}")
    print(dim("  " + "·" * 54))

def print_rag_info(rag_context: str, citation_paths: str):
    if not rag_context and not citation_paths:
        print(dim("  RAG: sin contexto adicional disponible"))
        return
    if rag_context:
        lines = [l for l in rag_context.splitlines() if l.strip()][:5]
        print(blue(bold("  RAG encontró:")))
        for line in lines:
            print(blue(f"    {line[:90]}"))
    if citation_paths:
        print(blue(bold("  Rutas RCT:")))
        for line in citation_paths.splitlines():
            if line.strip():
                print(blue(f"  {line}"))

def print_pairs(pairs: list[dict]):
    for i, pair in enumerate(pairs, 1):
        q = textwrap.fill(pair["question"], width=72, subsequent_indent="       ")
        a = textwrap.fill(pair["answer"],   width=72, subsequent_indent="       ")
        print(f"\n  {green(bold(f'[{i}] P:'))} {bold(q)}")
        print(f"  {green('    R:')} {a}")

def print_summary(total: int, path: Path):
    print(f"\n{cyan(_SEP)}")
    print(cyan(bold("  Generación completa")))
    print(f"  {bold(str(total))} pares guardados en {path}")
    print(cyan(_SEP) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_qa(
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
):
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
            p for p in sorted(docs_dir.rglob("*"))
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

    # First-pass indexing: if ChromaDB is empty, index all docs before generating
    if enricher.rag_doc_count == 0:
        print(f"\n{yellow('  ChromaDB vacío — indexando documentos primero (pasada 1/2)...')}")
        docs_to_index = []
        for fp in all_files:
            text = load_document(fp)
            if text:
                docs_to_index.append({
                    "texto": text,
                    "metadata": {
                        "documento": fp.stem,
                        "source": fp.name,
                        "documento_id": str(fp),
                        "tipo": fp.suffix.lstrip("."),
                    },
                })
        enricher.index_documents(docs_to_index)
        print(f"  {enricher.rag_doc_count} fragmentos indexados. Iniciando generación...\n")

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
                # Each chapter is (title, text); sub-chunk within each one
                sections: list[tuple[str, list[str]]] = [
                    (title, chunk_text(body, chunk_size=chunk_size))
                    for title, body in chapters
                ]
                print_doc_header(doc_path.name, doc_idx, len(all_files), n_chapters=len(chapters))
            else:
                # No structure detected — fall back to plain word chunking
                sections = [("", chunk_text(text, chunk_size=chunk_size))]
                print_doc_header(doc_path.name, doc_idx, len(all_files), n_chapters=0)

            print(dim(f"  {len(text.split())} palabras en total"))

            for ch_idx, (ch_title, chunks) in enumerate(sections):
                if ch_title:
                    print_chapter_header(ch_title, ch_idx + 1, len(sections))
                    print(dim(f"  {len(chunks)} sub-fragmento(s)"))

                for chunk_idx, chunk in enumerate(chunks):
                    print_chunk_header(chunk_idx, len(chunks))

                    # ── RAG + RCT context ──────────────────────────────
                    # Include chapter title in the search query for better recall
                    search_query = f"{ch_title} {chunk[:400]}".strip() if ch_title else chunk[:500]
                    rag_context, citation_paths = enricher.get_context(search_query, n_rag_results)
                    print_rag_info(rag_context, citation_paths)

                    # ── Build enriched prompt ──────────────────────────
                    prompt = build_prompt(chunk, pairs_per_chunk, rag_context, citation_paths)

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
                                {"role": "system",    "content": SYSTEM_PROMPT},
                                {"role": "user",      "content": pair["question"]},
                                {"role": "assistant", "content": pair["answer"]},
                            ],
                            "metadata": {
                                "source_file":     doc_path.name,
                                "chapter":         ch_title or None,
                                "chunk_index":     chunk_idx,
                                "rag_used":        bool(rag_context),
                                "citations_found": bool(citation_paths),
                                "generated_at":    datetime.now().isoformat(timespec="seconds"),
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
        description=(
            "Genera pares QA en español a partir de documentos regulatorios, "
            "enriquecidos con RAG (ChromaDB+BM25) y el árbol de citación regulatoria (RCT)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Ejemplos:
              # GPU — Qwen2.5-7B (por defecto)
              python scripts/generate_qa_from_docs.py --docs-dir data/raw

              # Modelo más ligero
              python scripts/generate_qa_from_docs.py --docs-dir data/raw \\
                --model Qwen/Qwen2.5-3B-Instruct

              # Ollama, sin GPU  (ejecutar antes: ollama pull llama3.2)
              python scripts/generate_qa_from_docs.py --docs-dir data/raw \\
                --backend ollama --model llama3.2
        """),
    )
    parser.add_argument(
        "--file", type=Path, default=None,
        metavar="PATH",
        help="Procesar un único archivo en lugar de un directorio.",
    )
    parser.add_argument(
        "--docs-dir", type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directorio con documentos (PDF, TXT, JSON, MD). Ignorado si se usa --file.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "finetuning" / "generated_qa.jsonl",
        help="Ruta del JSONL de salida.",
    )
    parser.add_argument(
        "--backend", choices=["transformers", "ollama"], default="transformers",
        help="Backend LLM: 'transformers' (GPU local) o 'ollama' (sin GPU).",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct",
        help="ID de modelo HuggingFace o nombre Ollama (ej: llama3.2, qwen2.5:7b).",
    )
    parser.add_argument(
        "--pairs-per-chunk", type=int, default=3,
        help="Pares QA a generar por fragmento (por defecto: 3).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=800,
        help="Tamaño de fragmento en palabras (por defecto: 800).",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Máximo de documentos a procesar (por defecto: todos).",
    )
    parser.add_argument(
        "--rag-results", type=int, default=3,
        help="Número de pasajes RAG a recuperar por fragmento (por defecto: 3).",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help=(
            "Dispositivo para el modelo: 'auto' (GPU si hay), 'cuda' (fuerza GPU), "
            "'cpu' (sin GPU, desactiva cuantización). Por defecto: auto."
        ),
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Tokens máximos a generar por fragmento (por defecto: 512). Reduce para ahorrar VRAM.",
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Desactivar cuantización 4-bit (solo backend transformers).",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="URL del servidor Ollama.",
    )

    args = parser.parse_args()

    if not args.docs_dir.exists():
        logger.error(f"--docs-dir no existe: {args.docs_dir}")
        sys.exit(1)

    generate_qa(
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
    )


if __name__ == "__main__":
    main()
