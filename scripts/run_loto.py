#!/usr/bin/env python
"""Phase 8 stub."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase_status import not_implemented

if __name__ == "__main__":
    not_implemented(8, "leave-one-trip-out experiment")
