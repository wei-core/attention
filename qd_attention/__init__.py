"""Compatibility imports for the former qd_attention package name."""
import importlib
import sys
from attention import *  # noqa: F401,F403
from attention import __all__, __version__

for _module in ("core", "functional", "mla", "qwen", "sparse", "validation", "reference",
                "reference.gqa", "reference.mla", "reference.delta", "reference.sparse"):
    sys.modules[f"{__name__}.{_module}"] = importlib.import_module(f"attention.{_module}")

sys.modules[f"{__name__}.dense"] = importlib.import_module("attention.core")
