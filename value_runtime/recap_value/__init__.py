"""Independent RECAP value training and advantage labeling runtime.

Importing the package configures source paths only; heavy ML modules stay lazy.
"""
from pathlib import Path
import sys

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RUNTIME_ROOT.parent


def prepare_runtime():
    for path in (PROJECT_ROOT / "src", RUNTIME_ROOT, RUNTIME_ROOT / "vendor/gemma"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


prepare_runtime()
