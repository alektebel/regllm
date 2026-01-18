#!/usr/bin/env python3
"""Quick training wrapper for testing."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Now import and run
from training.train import main

if __name__ == "__main__":
    main()
