#!/usr/bin/env python
"""Fallback entry point for running qkit without pip install -e ."""

import sys
from pathlib import Path

root = str(Path(__file__).resolve().parent)
if root not in sys.path:
    sys.path.insert(0, root)

from qkit.cli import main

if __name__ == "__main__":
    main()
