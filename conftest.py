"""Makes `src` importable as `src.xxx` from anywhere in the repo when running pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
