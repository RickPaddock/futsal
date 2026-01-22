# PROV: FUSBAL.PIPELINE.TESTS.CONFTEST.01
# REQ: SYS-ARCH-15
# WHY: Ensure tests can import the package from src/ without requiring installation.

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src))
