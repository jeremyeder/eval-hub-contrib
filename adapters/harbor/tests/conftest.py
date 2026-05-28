"""Add adapter root to sys.path so `from main import ...` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
