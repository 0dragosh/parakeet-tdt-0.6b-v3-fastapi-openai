#!/usr/bin/env python3
"""Compatibility launcher for the structured src layout."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parakeet_server.server import app, main  # noqa: E402


if __name__ == "__main__":
    main()
