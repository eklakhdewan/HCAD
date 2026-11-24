"""Judge-model registry: pick THE LLM judge by index + quantization level.

This module is the ONE place that knows which judge models exist. Everything
else (src/config.py -> JUDGE_OLLAMA_MODEL -> UnifiedLLMClient ->
BasePaperEvaluator) reads the resolved tag, so switching the judge never
requires editing source files:

    JUDGE_MODEL_INDEX=2 JUDGE_QUANT=qat python src/main.py base-paper ...

Exactly ONE judge scores a run. The registry is a menu of interchangeable
judges, not a panel: if the selected judge does not fit in memory (OOM),
pick another index and re-run. A list of indices ("0,3") is rejected.

Resolution (config.judge_ensemble_specs / config.JUDGE_OLLAMA_MODEL):
    JUDGE_OLLAMA_MODEL (exact tag) > JUDGE_MODEL_INDEX + JUDGE_QUANT > DEFAULT_JUDGE_TAG

The judge is STRICTLY 4-bit: the registry only lists 4-bit tags (q4_K_M,
plus Gemma 3's quantization-aware 'qat'), and an explicit JUDGE_OLLAMA_MODEL
tag must also be a 4-bit tag. Anything else (q8_0, fp16, an untagged
default) raises. The generator's precision is set separately by LLM_QUANT
in src/config.py.

Kept dependency-free on purpose: the Kaggle notebook kernel imports it
directly (different Python than the project env) to print the menu and
validate the selection before anything is pulled or run.

All tags below were verified against ollama.com/library/<model>/tags.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_QUANT = "q4_K_M"
DEFAULT_JUDGE_TAG = "gemma2:9b-instruct-q4_K_M"  # registry index 5, smallest (~5.8 GB at 4-bit)

# The ONLY judge precisions allowed. Strict: no 8-bit / fp16 judge.
ALLOWED_JUDGE_QUANTS = ("q4_K_M", "qat")
# Suffixes that identify a 4-bit Ollama tag given verbatim via JUDGE_OLLAMA_MODEL.
_FOUR_BIT_TAG = re.compile(r"[-:](q4_[0-9a-z_]+|qat)$", re.IGNORECASE)

# Model families that emit chain-of-thought unless told not to. Thinking
# tokens eat the judge's token budget and break JSON parsing, so the client
# sends think=False for these.
THINKING_FAMILIES = ("qwen3", "deepseek-r1")


@dataclass(frozen=True)
class JudgeModel:
    name: str                      # human-readable, shown in the notebook menu
    quants: Dict[str, str]         # quant level -> exact Ollama tag
    sizes_gb: Dict[str, float] = field(default_factory=dict)  # weights on disk/VRAM
    note: str = ""

    def tag(self, quant: str) -> str:
        if quant not in self.quants:
            raise ValueError(
                f"Quantization {quant!r} is not available for judge "
                f"{self.name!r}. Available: {', '.join(self.quants)}"
            )
        return self.quants[quant]


JUDGE_MODELS: List[JudgeModel] = [
    JudgeModel(  # 0
        name="Qwen3-14B",
        quants={
            "q4_K_M": "qwen3:14b-q4_K_M",
        },
        sizes_gb={"q4_K_M": 9.3},
        note="thinking disabled automatically",
    ),
    JudgeModel(  # 1
        name="DeepSeek-R1-Distill-Qwen-14B",
        quants={
            "q4_K_M": "deepseek-r1:14b-qwen-distill-q4_K_M",
        },
        sizes_gb={"q4_K_M": 9.0},
        note="reasoning model; thinking disabled automatically",
    ),
    JudgeModel(  # 2
        name="Gemma 3 12B IT",
        quants={
            "qat": "gemma3:12b-it-qat",
            "q4_K_M": "gemma3:12b-it-q4_K_M",
        },
        sizes_gb={"qat": 8.9, "q4_K_M": 8.2},
        note="'qat' = Google's quantization-aware 4-bit, best 4-bit quality",
    ),
    JudgeModel(  # 3
        name="Qwen2.5-14B-Instruct",
        quants={
            "q4_K_M": "qwen2.5:14b-instruct-q4_K_M",
        },
        sizes_gb={"q4_K_M": 9.0},
        note="previous project default judge",
    ),
    JudgeModel(  # 4
        name="Phi-4 14B",
        quants={
            "q4_K_M": "phi4:14b-q4_K_M",
        },
        sizes_gb={"q4_K_M": 9.1},
    ),
    JudgeModel(  # 5
        name="Gemma 2 9B IT",
        quants={
            "q4_K_M": "gemma2:9b-instruct-q4_K_M",
        },
        sizes_gb={"q4_K_M": 5.8},
        note="smallest / fastest option",
    ),
]


def get_judge_model(index: int) -> JudgeModel:
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"JUDGE_MODEL_INDEX must be an integer, got {index!r}")
    if not 0 <= index < len(JUDGE_MODELS):
        raise ValueError(
            f"JUDGE_MODEL_INDEX={index} is out of range. Valid indices:\n"
            + format_judge_menu()
        )
    return JUDGE_MODELS[index]


def check_judge_quant(quant: str) -> str:
    """Reject any judge precision that is not 4-bit."""
    quant = (quant or DEFAULT_QUANT).strip()
    if quant not in ALLOWED_JUDGE_QUANTS:
        raise ValueError(
            f"JUDGE_QUANT={quant!r} is not allowed: the judge is strictly 4-bit. "
            f"Choose one of {', '.join(ALLOWED_JUDGE_QUANTS)}."
        )
    return quant


def check_4bit_tag(tag: str) -> str:
    """Reject an explicit judge tag that is not a 4-bit Ollama tag."""
    if not _FOUR_BIT_TAG.search(tag or ""):
        raise ValueError(
            f"JUDGE_OLLAMA_MODEL={tag!r} is not a 4-bit tag: the judge is strictly "
            "4-bit. Use an explicit 4-bit tag (e.g. 'gemma2:9b-instruct-q4_K_M', "
            "'gemma3:12b-it-qat') or set JUDGE_MODEL_INDEX instead."
        )
    return tag


def resolve_judge_tag(index: int, quant: str = DEFAULT_QUANT) -> str:
    """Exact 4-bit Ollama tag for registry entry `index` at quantization `quant`."""
    return check_4bit_tag(get_judge_model(index).tag(check_judge_quant(quant)))


def _parse_index(raw: str) -> int:
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(
            f"JUDGE_MODEL_INDEX must be an integer, got {raw!r}. Options:\n"
            + format_judge_menu()
        ) from None


def parse_indices(raw) -> List[int]:
    """JUDGE_MODEL_INDEX value -> [index] (always exactly one judge).

    Accepts an int, "2" or [2]. Several indices are rejected: the menu is
    for swapping judges (e.g. after an OOM), not for a consensus panel.
    """
    if isinstance(raw, bool):
        raise ValueError(f"JUDGE_MODEL_INDEX must be an integer, got {raw!r}")
    if isinstance(raw, int):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [p for p in str(raw).replace(" ", "").split(",") if p != ""]
        items = [_parse_index(p) for p in items]
    if not items:
        raise ValueError("JUDGE_MODEL_INDEX is empty. Options:\n" + format_judge_menu())
    if len(items) > 1:
        raise ValueError(
            f"JUDGE_MODEL_INDEX selects ONE judge, got {items}. To switch judges "
            "(e.g. after an out-of-memory error), set a single other index:\n"
            + format_judge_menu()
        )
    get_judge_model(items[0])  # validates type and range, raises with the menu
    return items


def resolve_judge_tags(env: Optional[Dict[str, str]] = None) -> Optional[List[str]]:
    """[tag] of the selected judge, or None when no index is set."""
    env = os.environ if env is None else env
    raw_index = (env.get("JUDGE_MODEL_INDEX") or "").strip()
    if not raw_index:
        return None
    quant = (env.get("JUDGE_QUANT") or DEFAULT_QUANT).strip()
    return [resolve_judge_tag(i, quant) for i in parse_indices(raw_index)]


def resolve_judge_ollama_model(env: Optional[Dict[str, str]] = None) -> str:
    """Resolve the primary judge tag from the environment (see module doc)."""
    env = os.environ if env is None else env
    explicit = (env.get("JUDGE_OLLAMA_MODEL") or "").strip()
    if explicit:
        return check_4bit_tag(explicit)
    tags = resolve_judge_tags(env)
    if tags:
        return tags[0]
    return DEFAULT_JUDGE_TAG


def is_thinking_model(tag: str) -> bool:
    base = (tag or "").split("/")[-1].lower()
    return base.startswith(THINKING_FAMILIES)


def judge_slug(tag: str) -> str:
    """Filesystem-safe id for per-judge checkpoint/result file names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)


def format_judge_menu() -> str:
    lines = []
    for i, m in enumerate(JUDGE_MODELS):
        quants = ", ".join(
            f"{q}" + (f" ({m.sizes_gb[q]:g} GB)" if q in m.sizes_gb else "")
            for q in m.quants
        )
        note = f"  - {m.note}" if m.note else ""
        lines.append(f"  [{i}] {m.name}: {quants}{note}")
    return "\n".join(lines)
