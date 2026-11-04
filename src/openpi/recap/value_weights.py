"""Compatibility import; implementation lives in value_runtime/recap_value/weights.py."""
import sys
from openpi.recap._value_compat import load_value_module

# Alias the module itself so existing imports and monkeypatches share one state.
sys.modules[__name__] = load_value_module("weights")
