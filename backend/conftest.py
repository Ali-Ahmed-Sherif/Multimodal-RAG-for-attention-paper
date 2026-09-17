import sys
from pathlib import Path

# lets `pytest backend/tests` work from the repo root, not just from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent))
