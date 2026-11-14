"""Test wrapper: ensure game package imports work cleanly, sys.path set up, etc.
Then import + run all new sub-tests from one location so unittest discover finds them.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for sub in (str(PROJECT_ROOT), str(PROJECT_ROOT / "game"), str(PROJECT_ROOT / "algorithms"),
            str(PROJECT_ROOT / "learning"), str(PROJECT_ROOT / "tools")):
    if sub not in sys.path:
        sys.path.insert(0, sub)
