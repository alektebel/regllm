"""Procedural memory scenarios for sleep-cycle training (learn / organize / retrieve).

Analogous to ``bug_generator.py``: unlimited verifiable scenarios from
``data/experience/*.md`` seeds + synthetic graph mutations.

Usage:
    from training.memory_scenario_generator import MemoryScenarioGenerator

    gen = MemoryScenarioGenerator(seed=42)
    batch = gen.generate_batch(50, phase="retrieve")
    scenario = gen.generate_one("organize")
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIENCE_DIR = PROJECT_ROOT / "data" / "experience"

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)

MemoryPhase = Literal["learn", "organize", "consolidate", "retrieve"]

# ── Graph model (in-memory, no Kuzu) ─────────────────────────────────────

@dataclass
class MemoryNode:
    id: str
    label: str
    summary: str
    tags: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    priority: float = 0.5
    node_type: str = "insight"
    segment: str = ""
    is_historical: bool = False
    articles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "summary": self.summary,
            "tags": self.tags,
            "fields": self.fields,
            "priority": self.priority,
            "type": self.node_type,
            "segment": self.segment,
            "historical": self.is_historical,
        }

    def display_line(self) -> str:
        hist = " [histórico]" if self.is_historical else ""
        seg = f" ({self.segment})" if self.segment else ""
        tags = ", ".join(self.tags[:4]) if self.tags else "sin tags"
        return f"- **{self.id}**{seg}{hist}: {self.summary} [tags: {tags}]"


@dataclass
class MemoryScenario:
    """One verifiable memory task."""
    phase: MemoryPhase
    scenario_id: str
    graph_nodes: list[MemoryNode]
    user_prompt: str
    session_context: str | None = None
    question: str | None = None
    # Ground truth for reward / SFT golden
    should_save: bool | None = None
    expected_tool: str | None = None
    expected_ids: list[str] = field(default_factory=list)
    forbidden_ids: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    merge_pairs: list[tuple[str, str]] = field(default_factory=list)
    covers_ids: list[str] = field(default_factory=list)
    concept_label: str = ""
    concept_definition: str = ""
    should_create_concept: bool | None = None
    min_entropy_delta: float = 0.0
    golden_actions: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    subtype: str = ""

    def graph_text(self) -> str:
        if not self.graph_nodes:
            return "(grafo vacío)"
        return "\n".join(n.display_line() for n in self.graph_nodes)


# ── Experience corpus ────────────────────────────────────────────────────

def _parse_experience_md(path: Path) -> MemoryNode | None:
    text = path.read_text(encoding="utf-8")
    m = _FM_RE.match(text)
    if not m:
        return None
    meta: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k in ("tags", "fields", "articles"):
            meta[k] = [x.strip() for x in re.findall(r"[\w_]+", v)]
        elif k == "priority":
            try:
                meta[k] = float(v)
            except ValueError:
                meta[k] = 0.5
        elif k == "feedback":
            meta[k] = v.lower() in ("true", "yes", "1")
        else:
            meta[k] = v.strip('"')

    body = m.group(2).strip()
    title_m = re.search(r"^#\s+(.+)$", body, re.M)
    label = title_m.group(1) if title_m else path.stem
    summary = re.sub(r"^#.+$", "", body, flags=re.M).strip().split("\n")[0][:200]

    seg = ""
    for token in ("retail", "CORP", "hipoteca", "HIPOTECA", "soberano", "NINGUNA"):
        if token.lower() in (label + summary + str(meta.get("tags", ""))).lower():
            seg = token.upper() if token.isupper() or token == "retail" else token
            break

    return MemoryNode(
        id=str(meta.get("id", path.stem)),
        label=label,
        summary=summary or label,
        tags=list(meta.get("tags", [])),
        fields=list(meta.get("fields", [])),
        priority=float(meta.get("priority", 0.5)),
        node_type=str(meta.get("type", "insight")),
        segment=seg,
        articles=list(meta.get("articles", [])),
    )


class ExperienceCorpus:
    """Loads real experience markdown as mutation seeds."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or EXPERIENCE_DIR
        self.nodes: list[MemoryNode] = []
        for p in sorted(self.root.glob("*.md")):
            if p.name.upper() == "README.MD":
                continue
            node = _parse_experience_md(p)
            if node:
                self.nodes.append(node)

    def sample(self, n: int, rng: random.Random) -> list[MemoryNode]:
        if n >= len(self.nodes):
            return [copy.deepcopy(x) for x in self.nodes]
        return [copy.deepcopy(x) for x in rng.sample(self.nodes, n)]

    def by_tag(self, tag: str) -> list[MemoryNode]:
        t = tag.lower()
        return [n for n in self.nodes if t in [x.lower() for x in n.tags] or t in n.summary.lower()]


# ── Scenario generators ──────────────────────────────────────────────────

_NOISE_SESSIONS = [
    "Revisé el log de SAS del batch nocturno. Todos los jobs terminaron RC=0. Sin incidencias.",
    "El usuario preguntó la hora del cierre contable. Respondí 18:00 CET.",
    "Confirmé que el dataset WORK.CICLOS tiene 1.2M filas, igual que ayer.",
    "Leí el README del pipeline. No hay cambios respecto a la semana pasada.",
]

_PARAPHRASE_PREFIX = [
    "Hallazgo: ",
    "Nota de validación: ",
    "Insight detectado — ",
    "Registro de sesión: ",
]


class MemoryScenarioGenerator:
    """Procedural generator for memory sleep-cycle tasks."""

    def __init__(self, seed: int = 42, corpus: ExperienceCorpus | None = None) -> None:
        self.rng = random.Random(seed)
        self.corpus = corpus or ExperienceCorpus()
        if not self.corpus.nodes:
            raise FileNotFoundError(f"No experience seeds in {EXPERIENCE_DIR}")

    def generate_one(self, phase: MemoryPhase | None = None) -> MemoryScenario:
        phase = phase or self.rng.choice(["learn", "organize", "consolidate", "retrieve"])
        if phase == "learn":
            return self.rng.choice([self._gen_learn_save, self._gen_learn_skip])()
        if phase == "organize":
            return self.rng.choice([
                self._gen_organize_duplicate,
                self._gen_organize_untagged,
                self._gen_organize_conflict,
            ])()
        if phase == "consolidate":
            return self.rng.choice([
                self._gen_consolidate_lift,
                self._gen_consolidate_skip,
            ])()
        return self.rng.choice([
            self._gen_retrieve_segment,
            self._gen_retrieve_temporal,
            self._gen_retrieve_keyword,
            self._gen_retrieve_via_concept,
        ])()

    def generate_batch(
        self,
        n: int,
        phase: MemoryPhase | None = None,
        *,
        mix: dict[str, float] | None = None,
    ) -> list[MemoryScenario]:
        mix = mix or {"learn": 0.30, "organize": 0.20, "consolidate": 0.15, "retrieve": 0.35}
        phases: list[MemoryPhase] = []
        for p, w in mix.items():
            phases.extend([p] * max(1, int(n * w)))  # type: ignore[arg-type]
        while len(phases) < n:
            phases.append(self.rng.choice(["learn", "organize", "consolidate", "retrieve"]))
        self.rng.shuffle(phases)

        out: list[MemoryScenario] = []
        for i, ph in enumerate(phases[:n]):
            sc = self.generate_one(ph)
            sc.scenario_id = f"mem_{ph}_{i:04d}_{sc.subtype}"
            out.append(sc)
        return out

    # ── Learn ───────────────────────────────────────────────────────────

    def _gen_learn_save(self) -> MemoryScenario:
        seed = self.rng.choice(self.corpus.nodes)
        session = (
            f"Investigación de {', '.join(seed.fields[:2]) or 'pipeline LGD'}.\n"
            f"Tras trace_dependencies e inspect_lineage encontré: {seed.summary}\n"
            f"Archivo relacionado: data/experience/{seed.id}.md"
        )
        golden = (
            f"save_insight(id=\"{seed.id}\", label=\"{seed.label[:80]}\", "
            f"summary=\"{seed.summary[:120]}\", tags={seed.tags}, fields={seed.fields})"
        )
        return MemoryScenario(
            phase="learn",
            scenario_id="",
            graph_nodes=self.corpus.sample(3, self.rng),
            session_context=session,
            user_prompt=f"## Sesión del día\n\n{session}\n\n¿Qué merece persistirse en el grafo?",
            should_save=True,
            expected_tool="save_insight",
            must_contain=[seed.id.split("_")[0], seed.fields[0] if seed.fields else seed.tags[0] if seed.tags else "LGD"],
            golden_actions=[golden],
            difficulty="medium",
            subtype="learn_save",
        )

    def _gen_learn_skip(self) -> MemoryScenario:
        noise = self.rng.choice(_NOISE_SESSIONS)
        return MemoryScenario(
            phase="learn",
            scenario_id="",
            graph_nodes=self.corpus.sample(2, self.rng),
            session_context=noise,
            user_prompt=f"## Sesión del día\n\n{noise}\n\n¿Qué merece persistirse en el grafo?",
            should_save=False,
            expected_tool=None,
            must_not_contain=["save_insight", "save_quirk"],
            golden_actions=["# No guardar — sesión sin insight accionable"],
            difficulty="easy",
            subtype="learn_skip_noise",
        )

    # ── Organize ────────────────────────────────────────────────────────

    def _gen_organize_duplicate(self) -> MemoryScenario:
        base = self.rng.choice(self.corpus.nodes)
        dup = copy.deepcopy(base)
        dup.id = f"{base.id}_dup_{self.rng.randint(1, 99)}"
        dup.summary = self.rng.choice(_PARAPHRASE_PREFIX) + base.summary
        nodes = self.corpus.sample(2, self.rng) + [base, dup]
        self.rng.shuffle(nodes)
        return MemoryScenario(
            phase="organize",
            scenario_id="",
            graph_nodes=nodes,
            user_prompt=(
                f"## Estado del grafo\n\n{self._fmt_graph(nodes)}\n\n"
                "Organiza: deduplica, etiqueta y enlaza nodos relacionados."
            ),
            expected_tool="merge_nodes",
            merge_pairs=[(base.id, dup.id)],
            golden_actions=[f"merge_nodes(source=\"{dup.id}\", target=\"{base.id}\")"],
            difficulty="medium",
            subtype="organize_duplicate",
        )

    def _gen_organize_untagged(self) -> MemoryScenario:
        base = copy.deepcopy(self.rng.choice(self.corpus.nodes))
        base.tags = []
        nodes = self.corpus.sample(3, self.rng) + [base]
        self.rng.shuffle(nodes)
        return MemoryScenario(
            phase="organize",
            scenario_id="",
            graph_nodes=nodes,
            user_prompt=(
                f"## Estado del grafo\n\n{self._fmt_graph(nodes)}\n\n"
                f"El nodo `{base.id}` no tiene tags. Organiza el grafo."
            ),
            expected_tool="add_tags",
            expected_ids=[base.id],
            must_contain=base.fields[:1] or ["LGD", "PD"],
            golden_actions=[f"add_tags(node=\"{base.id}\", tags=[...])"],
            difficulty="easy",
            subtype="organize_untagged",
        )

    def _gen_organize_conflict(self) -> MemoryScenario:
        """Feedback node overrides an older generalization."""
        pd_nodes = [n for n in self.corpus.nodes if "pd" in n.summary.lower() or "PD" in n.fields]
        if len(pd_nodes) < 1:
            return self._gen_organize_duplicate()
        feedback = copy.deepcopy(pd_nodes[0])
        feedback.id = feedback.id + "_feedback"
        feedback.priority = 1.0
        feedback.node_type = "feedback"
        feedback.summary = "PD floor retail 0.03%; CORP/soberano 0.05% (corrección)"
        feedback.segment = "retail"
        old = copy.deepcopy(feedback)
        old.id = old.id.replace("_feedback", "_old")
        old.summary = "PD floor es 0.05% para todas las exposiciones"
        old.priority = 0.4
        old.is_historical = True
        nodes = [old, feedback] + self.corpus.sample(2, self.rng)
        return MemoryScenario(
            phase="organize",
            scenario_id="",
            graph_nodes=nodes,
            user_prompt=(
                f"## Estado del grafo\n\n{self._fmt_graph(nodes)}\n\n"
                "Detecta conflictos y prioriza el feedback vigente."
            ),
            expected_tool="flag_conflict",
            merge_pairs=[(old.id, feedback.id)],
            golden_actions=[
                f"flag_conflict(node_a=\"{old.id}\", node_b=\"{feedback.id}\")",
                f"merge_nodes(source=\"{old.id}\", target=\"{feedback.id}\")",
            ],
            difficulty="hard",
            subtype="organize_conflict",
        )

    # ── Consolidate (EverMemOS-style semantic lift) ───────────────────

    _CONSOLIDATE_THEMES = [
        {
            "tag": "LGD",
            "field": "LGD_ESTIMADA",
            "label": "LGD floor por segmento de calibración",
            "definition": (
                "El suelo de LGD varía por CALIBRATION_SEGMENT: retail, CORP/NINGUNA "
                "e hipoteca tienen valores regulatorios distintos."
            ),
            "variants": [
                ("lgd_seg_retail", "LGD floor retail", "LGD floor retail = 0.45%", "retail"),
                ("lgd_seg_corp", "LGD floor CORP", "LGD floor CORP/NINGUNA = 0.50%", "CORP"),
                ("lgd_seg_hip", "LGD floor hipoteca", "LGD floor HIPOTECA = 0.30 CRR Art.154", "HIPOTECA"),
            ],
        },
        {
            "tag": "PD",
            "field": "PD_ESTIMADA",
            "label": "PD floor por segmento",
            "definition": (
                "El suelo de PD depende del segmento: retail 0.03%, "
                "CORP/soberano 0.05%, hipoteca puede diferir."
            ),
            "variants": [
                ("pd_seg_retail", "PD floor retail", "PD floor retail = 0.03%", "retail"),
                ("pd_seg_corp", "PD floor CORP", "PD floor CORP/soberano = 0.05%", "CORP"),
                ("pd_seg_hip", "PD floor hipoteca", "PD floor hipotecas = 0.05%", "HIPOTECA"),
            ],
        },
    ]

    def _gen_consolidate_lift(self) -> MemoryScenario:
        theme = self.rng.choice(self._CONSOLIDATE_THEMES)
        variants = theme["variants"]
        k = min(len(variants), self.rng.randint(3, len(variants)))
        chosen = variants[:k]
        nodes: list[MemoryNode] = []
        covers: list[str] = []
        for nid, label, summary, seg in chosen:
            n = MemoryNode(
                id=nid,
                label=label,
                summary=summary,
                tags=[theme["tag"], seg.lower() if seg != "CORP" else "CORP"],
                fields=[theme["field"]],
                segment=seg,
            )
            nodes.append(n)
            covers.append(nid)
        distractors = self.corpus.sample(2, self.rng)
        nodes.extend(distractors)
        self.rng.shuffle(nodes)

        from src.knowledge.graph_entropy import compute_graph_entropy

        insight_nodes = [n for n in nodes if n.id in covers]
        h_before = compute_graph_entropy(insight_nodes).total
        h_after = compute_graph_entropy(
            insight_nodes, concept_covers={"concept:mem_theme": covers},
        ).total
        h_delta = round(h_before - h_after, 4)

        golden = (
            f'create_concept(label="{theme["label"]}", '
            f'definition="{theme["definition"]}", '
            f"covers={json.dumps(covers)}, "
            f'links_to=["{theme["field"]}"])'
        )
        return MemoryScenario(
            phase="consolidate",
            scenario_id="",
            graph_nodes=nodes,
            user_prompt=(
                f"## Estado del grafo\n\n{self._fmt_graph(nodes)}\n\n"
                "Consolida: detecta insights relacionados que comparten tema y "
                "crea un Concept que reduzca la entropía del grafo (abstracción semántica)."
            ),
            expected_tool="create_concept",
            covers_ids=covers,
            concept_label=theme["label"],
            concept_definition=theme["definition"],
            should_create_concept=True,
            min_entropy_delta=max(0.0, h_delta),
            must_contain=[theme["tag"], theme["field"]],
            golden_actions=[golden],
            difficulty="hard",
            subtype="consolidate_lift",
        )

    def _gen_consolidate_skip(self) -> MemoryScenario:
        """Single insight — creating a concept would be over-abstraction."""
        seed = copy.deepcopy(self.rng.choice(self.corpus.nodes))
        nodes = self.corpus.sample(2, self.rng) + [seed]
        self.rng.shuffle(nodes)
        return MemoryScenario(
            phase="consolidate",
            scenario_id="",
            graph_nodes=nodes,
            user_prompt=(
                f"## Estado del grafo\n\n{self._fmt_graph(nodes)}\n\n"
                "Consolida solo si hay compresión real (≥2 insights relacionados). "
                "Si no aplica, no crees conceptos."
            ),
            should_create_concept=False,
            must_not_contain=["create_concept"],
            golden_actions=["# No crear concepto — un solo insight aislado, sin compresión"],
            difficulty="easy",
            subtype="consolidate_skip",
        )

    # ── Retrieve ────────────────────────────────────────────────────────

    def _gen_retrieve_segment(self) -> MemoryScenario:
        """Fragile: same metric, different segment."""
        templates = [
            (
                [
                    MemoryNode("pd_corp", "PD floor CORP", "PD floor CORP/soberano = 0.05%", tags=["PD", "CORP"], fields=["PD_ESTIMADA"], segment="CORP"),
                    MemoryNode("pd_retail", "PD floor retail", "PD floor retail = 0.03% (feedback)", tags=["PD", "retail"], fields=["PD_ESTIMADA"], segment="retail", priority=1.0),
                    MemoryNode("pd_hip", "PD floor hipoteca", "PD floor hipotecas = 0.05%", tags=["PD", "hipoteca"], fields=["PD_ESTIMADA"], segment="HIPOTECA"),
                ],
                "¿Cuál es el PD floor para retail?",
                "pd_retail",
                ["pd_corp", "pd_hip"],
                ["0.03", "retail"],
            ),
            (
                [
                    MemoryNode("lgd_hip", "LGD hipoteca", "LGD floor HIPOTECA = 0.30 CRR Art.154", tags=["LGD", "hipoteca"], fields=["LGD_ESTIMADA"], segment="HIPOTECA"),
                    MemoryNode("lgd_corp", "LGD CORP", "LGD floor CORP/NINGUNA = 0.50", tags=["LGD", "CORP"], fields=["LGD_ESTIMADA"], segment="CORP"),
                ],
                "¿Qué suelo de LGD aplica a hipotecas?",
                "lgd_hip",
                ["lgd_corp"],
                ["0.30", "154"],
            ),
        ]
        nodes, question, correct, forbidden, must = self.rng.choice(templates)
        nodes = [copy.deepcopy(n) for n in nodes]
        return MemoryScenario(
            phase="retrieve",
            scenario_id="",
            graph_nodes=nodes,
            question=question,
            user_prompt=(
                f"## Experiencias en grafo\n{self._fmt_graph(nodes)}\n\n"
                f"## Consulta\n{question}\n\nRecupera EXACTAMENTE la experiencia correcta."
            ),
            expected_tool="search_experience",
            expected_ids=[correct],
            forbidden_ids=forbidden,
            must_contain=must,
            golden_actions=[f"search_experience(query=\"{question}\") → {correct}"],
            difficulty="hard",
            subtype="retrieve_segment",
        )

    def _gen_retrieve_temporal(self) -> MemoryScenario:
        nodes = [
            MemoryNode("lgd_old", "LGD CORP histórico", "LGD floor CORP era 0.45 pre-Circular 4/2022", tags=["LGD", "CORP"], fields=["LGD_ESTIMADA"], segment="CORP", is_historical=True, priority=0.3),
            MemoryNode("lgd_new", "LGD CORP vigente", "LGD floor CORP subió a 0.50 con Circular 4/2022", tags=["LGD", "CORP"], fields=["LGD_ESTIMADA"], segment="CORP", priority=0.8),
        ]
        question = "En datos de 2023, ¿qué LGD floor se aplica a CORP?"
        return MemoryScenario(
            phase="retrieve",
            scenario_id="",
            graph_nodes=nodes,
            question=question,
            user_prompt=(
                f"## Experiencias en grafo\n{self._fmt_graph(nodes)}\n\n"
                f"## Consulta\n{question}"
            ),
            expected_tool="search_experience",
            expected_ids=["lgd_new"],
            forbidden_ids=["lgd_old"],
            must_contain=["0.50", "2022"],
            golden_actions=["search_experience(query=\"LGD floor CORP 2023\") → lgd_new"],
            difficulty="hard",
            subtype="retrieve_temporal",
        )

    def _gen_retrieve_keyword(self) -> MemoryScenario:
        seed = self.rng.choice(self.corpus.nodes)
        distractors = [n for n in self.corpus.sample(4, self.rng) if n.id != seed.id][:2]
        nodes = [seed] + distractors
        self.rng.shuffle(nodes)
        q = f"¿Qué sabemos sobre {seed.fields[0] if seed.fields else seed.tags[0]} relacionado con {seed.label[:40]}?"
        return MemoryScenario(
            phase="retrieve",
            scenario_id="",
            graph_nodes=nodes,
            question=q,
            user_prompt=f"## Experiencias en grafo\n{self._fmt_graph(nodes)}\n\n## Consulta\n{q}",
            expected_tool="search_experience",
            expected_ids=[seed.id],
            forbidden_ids=[d.id for d in distractors],
            must_contain=[w for w in (seed.tags[:2] + seed.fields[:1]) if w],
            golden_actions=[f"search_experience(query=\"...\") → {seed.id}"],
            difficulty="medium",
            subtype="retrieve_keyword",
        )

    def _gen_retrieve_via_concept(self) -> MemoryScenario:
        """After consolidation: query resolves via concept, not raw insight list."""
        theme = self._CONSOLIDATE_THEMES[0]
        variants = theme["variants"][:3]
        concept = MemoryNode(
            id="concept:lgd_segmentation",
            label=theme["label"],
            summary=theme["definition"],
            node_type="concept",
            tags=["LGD"],
            fields=[theme["field"]],
        )
        insights = [
            MemoryNode(nid, label, summary, tags=["LGD"], fields=[theme["field"]], segment=seg)
            for nid, label, summary, seg in variants
        ]
        nodes = [concept] + insights
        question = "¿Cómo varía el LGD floor entre segmentos?"
        return MemoryScenario(
            phase="retrieve",
            scenario_id="",
            graph_nodes=nodes,
            question=question,
            user_prompt=(
                f"## Grafo (con concepto consolidado)\n{self._fmt_graph(nodes)}\n\n"
                f"## Consulta\n{question}\n\n"
                "Recupera vía el concepto o la experiencia más precisa."
            ),
            expected_tool="search_experience",
            expected_ids=["concept:lgd_segmentation"],
            must_contain=["segmento", "LGD"],
            golden_actions=[
                f'search_experience(query="LGD floor segmento") → concept:lgd_segmentation'
            ],
            difficulty="medium",
            subtype="retrieve_via_concept",
        )

    @staticmethod
    def _fmt_graph(nodes: list[MemoryNode]) -> str:
        return "\n".join(n.display_line() for n in nodes)


def make_toy_generator(seed: int = 0) -> MemoryScenarioGenerator:
    """Small corpus for tests."""
    return MemoryScenarioGenerator(seed=seed)


if __name__ == "__main__":
    gen = MemoryScenarioGenerator(seed=42)
    from collections import Counter
    batch = gen.generate_batch(200)
    print(Counter(s.phase for s in batch))
    print(Counter(s.subtype for s in batch))
    print(f"Corpus seeds: {len(gen.corpus.nodes)}")
