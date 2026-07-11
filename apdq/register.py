"""Atomic requirement register with regulation-hash pinning.

One row = one atomic, testable regulatory obligation. Each row cites its
regulation + paragraph, carries the SHA-256 of the paragraph text it was
extracted from, names the canonical concepts it constrains and the defect
classes that would violate it, and holds a sign-off state.

Gates (all machine-checkable):

* ``unsigned``   — rows not yet human-signed (certification requires 0);
* ``stale``      — rows whose pinned hash no longer matches the supplied
                   regulation text (amendment ⇒ automatic re-review);
* ``unbound``    — signed, applicable rows naming a concept the manifest
                   binds to no column (the data an auditor would ask for
                   does not exist — a finding by construction).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

from .manifest import Manifest

STATUSES = ("pending", "signed", "waived")


class RegisterError(ValueError):
    """Register fails validation."""


@dataclass(frozen=True)
class Requirement:
    id: str
    regulation: str
    section: str
    paragraph: str
    obligation: str                    # the atomic requirement, one sentence
    text_sha256: str                   # hash of the source paragraph text
    concepts: tuple[str, ...]
    defect_classes: tuple[int, ...]
    status: str                        # pending | signed | waived
    signed_by: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Register:
    regulation: str
    version: str
    requirements: tuple[Requirement, ...]
    source_path: Path | None = None


def _norm(text: str) -> str:
    """Whitespace/diacritics-stable normalization before hashing, so
    reflowing a PDF extract does not void every pin."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()


def load_register(path: Path | str, vocabulary: dict | None = None) -> Register:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("apdq_register") != 1:
        raise RegisterError(f"{path}: expected 'apdq_register: 1' header")
    reqs: list[Requirement] = []
    seen: set[str] = set()
    for raw in data.get("requirements", ()):
        rid = raw.get("id")
        if not rid:
            raise RegisterError(f"{path}: requirement without id")
        if rid in seen:
            raise RegisterError(f"{path}: duplicate requirement id {rid!r}")
        seen.add(rid)
        status = raw.get("status", "pending")
        if status not in STATUSES:
            raise RegisterError(f"{rid}: status must be one of {STATUSES}")
        if status in ("signed", "waived") and not raw.get("signed_by"):
            raise RegisterError(f"{rid}: {status} requires signed_by")
        concepts = tuple(raw.get("concepts", ()))
        if vocabulary is not None:
            unknown = [c for c in concepts if c not in vocabulary]
            if unknown:
                raise RegisterError(f"{rid}: unknown concepts {unknown}")
        classes = tuple(int(c) for c in raw.get("defect_classes", ()))
        bad = [c for c in classes if not 1 <= c <= 12]
        if bad:
            raise RegisterError(f"{rid}: defect classes out of range: {bad}")
        if not raw.get("obligation"):
            raise RegisterError(f"{rid}: missing obligation text")
        if not raw.get("text_sha256"):
            raise RegisterError(f"{rid}: missing text_sha256 pin")
        reqs.append(Requirement(
            id=rid,
            regulation=raw.get("regulation", data.get("regulation", "")),
            section=str(raw.get("section", "")),
            paragraph=str(raw.get("paragraph", "")),
            obligation=raw["obligation"],
            text_sha256=raw["text_sha256"],
            concepts=concepts,
            defect_classes=classes,
            status=status,
            signed_by=raw.get("signed_by", ""),
            notes=raw.get("notes", ""),
        ))
    return Register(
        regulation=data.get("regulation", ""),
        version=str(data.get("version", "")),
        requirements=tuple(reqs),
        source_path=path,
    )


# ── gates ────────────────────────────────────────────────────────────────────

def unsigned(register: Register) -> list[Requirement]:
    return [r for r in register.requirements if r.status == "pending"]


def stale(register: Register, paragraph_texts: dict[str, str]) -> list[Requirement]:
    """Requirements whose pinned hash mismatches the supplied regulation
    text. ``paragraph_texts`` maps paragraph id -> current text; rows whose
    paragraph is not supplied are not checked (partial verification)."""
    out = []
    for req in register.requirements:
        current = paragraph_texts.get(req.paragraph)
        if current is not None and text_hash(current) != req.text_sha256:
            out.append(req)
    return out


def unbound(register: Register, manifest: Manifest) -> list[tuple[Requirement, list[str]]]:
    """Signed, applicable requirements naming concepts the manifest does
    not bind. Each is a finding: the data an auditor would ask for is
    absent from the reporting schema."""
    bound = manifest.bound_concepts()
    out = []
    for req in register.requirements:
        if req.status != "signed":
            continue
        missing = [c for c in req.concepts if c not in bound]
        if missing:
            out.append((req, missing))
    return out


def summary(register: Register) -> dict:
    by_status: dict[str, int] = {}
    for req in register.requirements:
        by_status[req.status] = by_status.get(req.status, 0) + 1
    return {"total": len(register.requirements), **by_status}
