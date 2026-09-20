#!/usr/bin/env python
"""Phase 12 stub."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase_status import not_implemented

if __name__ == "__main__":
    not_implemented(12, "publication figures from completed experiments")
