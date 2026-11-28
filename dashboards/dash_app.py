"""dash_app — placeholder. Reads only from outputs/ and src/style.py."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.style import *  # noqa
