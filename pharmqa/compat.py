"""Environment workarounds that must run before importing peft."""
from __future__ import annotations

import importlib.metadata
import sys


def hide_incompatible_torchao(min_version: tuple[int, int] = (0, 16)) -> None:
    """Colab preinstalls torchao 0.10, and recent peft raises ImportError while building LoRA
    layers if an older torchao is installed. torchao is not used here, so make it invisible."""
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return
    parts = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
    if parts < min_version:
        sys.modules["torchao"] = None  # importlib.util.find_spec("torchao") now returns None
