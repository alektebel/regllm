#!/usr/bin/env python3
"""
CLI entry point for GRPO training.

Usage:
    python scripts/run_grpo.py                    # default settings
    python scripts/run_grpo.py --epochs 3         # more epochs
    python scripts/run_grpo.py --group-size 8     # more completions per prompt
    python scripts/run_grpo.py --eval-only        # just evaluate current model
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rlhf.grpo_trainer import main

if __name__ == "__main__":
    main()
