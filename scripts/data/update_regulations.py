"""
Regulatory Document Update Pipeline
=====================================

Downloads new/updated regulatory documents and re-indexes them into ChromaDB.
Designed to run as a weekly cron job.

TODO: Implement actual scraping per source.
  EBA:   https://www.eba.europa.eu/regulation-and-policy  (guidelines, RTS, ITS)
  BIS:   https://www.bis.org/bcbs/publications.htm        (Basel standards)
  EUR-Lex: https://eur-lex.europa.eu (CRR, CRD — search for 575/2013, 2013/36/EU)
  BOE:   https://www.boe.es (Circular 2/2016 Banco de España — Spanish impl.)

TODO: Use a proper scraper (see src/scraper/regulation_scraper.py) or
  the EBA's machine-readable API if available, rather than raw HTML scraping.
  The BIS publishes PDFs directly; use requests + PyPDF2 / pdfplumber.

TODO: Add change detection — hash each downloaded PDF; skip if unchanged.
  Store hashes in data/raw/checksums.json.

TODO: Set up as a cron or GitHub Actions schedule:
  # .github/workflows/update-regulations.yml
  on:
    schedule:
      - cron: '0 6 * * 1'   # every Monday at 06:00 UTC
  jobs:
    update:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: python scripts/update_regulations.py
        - run: python scripts/index_citations.py
        - uses: stefanzweifel/git-auto-commit-action@v5
          with:
            commit_message: "chore: update regulatory documents"

Usage:
    python scripts/update_regulations.py [--dry-run]
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data/raw"
CHECKSUMS_FILE = RAW_DIR / "checksums.json"

# TODO: populate with real URLs once scraper is implemented
SOURCES: list[dict] = [
    # {
    #     "name": "EBA Guidelines on credit risk",
    #     "url": "https://www.eba.europa.eu/...",
    #     "dest": "data/raw/eba/...",
    # },
]


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_checksums() -> dict:
    if CHECKSUMS_FILE.exists():
        return json.loads(CHECKSUMS_FILE.read_text())
    return {}


def _save_checksums(checksums: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHECKSUMS_FILE.write_text(json.dumps(checksums, indent=2))


def check_for_updates(dry_run: bool = False) -> list[str]:
    """Return list of files that were downloaded/updated."""
    import requests

    checksums = _load_checksums()
    updated = []

    if not SOURCES:
        logger.warning(
            "No sources configured. Edit SOURCES list in this file to add documents.\n"
            "See TODO comments above for EBA/BIS/EUR-Lex URLs."
        )
        return []

    for source in SOURCES:
        dest = PROJECT_ROOT / source["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Checking: {source['name']}")
        try:
            response = requests.get(source["url"], timeout=30, stream=True)
            response.raise_for_status()

            # Write to temp file, check hash
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)

            new_hash = _file_hash(tmp)
            old_hash = checksums.get(source["dest"], "")

            if new_hash == old_hash:
                logger.info(f"  No change: {dest.name}")
                tmp.unlink()
            else:
                if not dry_run:
                    tmp.rename(dest)
                    checksums[source["dest"]] = new_hash
                    updated.append(str(dest))
                    logger.info(f"  Updated: {dest.name}")
                else:
                    tmp.unlink()
                    logger.info(f"  [DRY RUN] Would update: {dest.name}")

        except Exception as e:
            logger.error(f"  Failed to download {source['name']}: {e}")

    if updated and not dry_run:
        _save_checksums(checksums)
        logger.info(f"\n{len(updated)} document(s) updated. Re-indexing...")
        _reindex()

    return updated


def _reindex() -> None:
    """Re-run the indexing pipeline after downloads."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/index_citations.py")],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        logger.error("Re-indexing failed")
    else:
        logger.info("Re-indexing complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update regulatory documents")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    updated = check_for_updates(dry_run=args.dry_run)
    logger.info(f"Done. {len(updated)} file(s) changed.")
