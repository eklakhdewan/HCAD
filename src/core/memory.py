"""Host-RAM helpers (Kaggle's ~29 GB system RAM is the binding constraint)."""
from __future__ import annotations

import gc
import sys


def release_freed_memory() -> None:
    """Run the GC and hand freed heap pages back to the OS (glibc only).

    Without malloc_trim, CPython keeps freed arenas mapped, so RSS stays at
    the peak and the next stage (ollama spill, embeddings) gets OOM-killed.
    """
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:  # musl / no libc: nothing to trim
            pass
