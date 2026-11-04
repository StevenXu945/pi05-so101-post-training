"""Lazy compatibility bridge to the independent value runtime in this checkout."""
from importlib import import_module
from pathlib import Path
import sys


def load_value_module(name):
    runtime = Path(__file__).resolve().parents[3] / "value_runtime"
    if not (runtime / "recap_value").is_dir():
        raise ImportError("RECAP value modules require value_runtime/ from the source checkout")
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    return import_module(f"recap_value.{name}")
