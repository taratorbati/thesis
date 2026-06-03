# conftest.py  (repository root)
# ─────────────────────────────────────────────────────────────────────────────
# pytest discovers this file automatically and executes it before any test
# collection.  Inserting the repo root into sys.path here means that
#   from src.rl.gym_env import IrrigationEnv
# resolves on any machine, regardless of how pytest is invoked -- whether from
# the repo root, from tests/, or via an IDE test runner.
#
# NOT needed for training (train_*.py runs as a script; Python adds cwd to
# sys.path automatically).  Only needed for the local test runner.
# ─────────────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
