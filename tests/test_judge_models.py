"""Judge registry: index/quant selection, ensemble wiring, preflight, resume."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.judge_models import (
    ALLOWED_JUDGE_QUANTS,
    DEFAULT_JUDGE_TAG,
    JUDGE_MODELS,
    is_thinking_model,
    judge_slug,
    parse_indices,
    resolve_judge_ollama_model,
    resolve_judge_tag,
    resolve_judge_tags,
)


def test_registry_is_strictly_4bit():
    for m in JUDGE_MODELS:
        assert "q4_K_M" in m.quants, m.name   # default precision
        assert set(m.quants) <= set(ALLOWED_JUDGE_QUANTS), m.name


def test_index_and_quant_resolve_to_exact_tags():
    assert resolve_judge_tag(0) == "qwen3:14b-q4_K_M"
    assert resolve_judge_tag(1) == "deepseek-r1:14b-qwen-distill-q4_K_M"
    assert resolve_judge_tag(2) == "gemma3:12b-it-q4_K_M"
    assert resolve_judge_tag(2, "qat") == "gemma3:12b-it-qat"
    assert resolve_judge_tag(3) == "qwen2.5:14b-instruct-q4_K_M"
    assert DEFAULT_JUDGE_TAG == "gemma2:9b-instruct-q4_K_M"


def test_precedence_explicit_tag_then_index_then_default():
    assert resolve_judge_ollama_model({}) == DEFAULT_JUDGE_TAG
    assert resolve_judge_ollama_model({"JUDGE_MODEL_INDEX": "3"}) == "qwen2.5:14b-instruct-q4_K_M"
    assert resolve_judge_ollama_model(
        {"JUDGE_MODEL_INDEX": "3", "JUDGE_OLLAMA_MODEL": "mistral-nemo:12b-instruct-2407-q4_K_M"}
    ) == "mistral-nemo:12b-instruct-2407-q4_K_M"


def test_exactly_one_judge_is_selected():
    assert parse_indices(2) == [2]
    assert parse_indices("2") == [2]
    assert parse_indices([4]) == [4]
    assert resolve_judge_tags({"JUDGE_MODEL_INDEX": "3"}) == ["qwen2.5:14b-instruct-q4_K_M"]
    assert resolve_judge_tags({}) is None
    for several in ("0, 3", [1, 4], "0,0"):
        with pytest.raises(ValueError, match="ONE judge"):
            parse_indices(several)


@pytest.mark.parametrize("env, match", [
    ({"JUDGE_MODEL_INDEX": "99"}, "out of range"),
    ({"JUDGE_MODEL_INDEX": "-1"}, "out of range"),
    ({"JUDGE_MODEL_INDEX": "two"}, "must be an integer"),
    ({"JUDGE_MODEL_INDEX": "0,3"}, "ONE judge"),
    ({"JUDGE_MODEL_INDEX": "0", "JUDGE_QUANT": "qat"}, "not available"),
    ({"JUDGE_MODEL_INDEX": "0", "JUDGE_QUANT": "q8_0"}, "strictly 4-bit"),
    ({"JUDGE_MODEL_INDEX": "5", "JUDGE_QUANT": "fp16"}, "strictly 4-bit"),
    ({"JUDGE_MODEL_INDEX": "5", "JUDGE_QUANT": "q3_K_M"}, "strictly 4-bit"),
    ({"JUDGE_OLLAMA_MODEL": "gemma2:9b-instruct-q8_0"}, "strictly 4-bit"),
    ({"JUDGE_OLLAMA_MODEL": "mistral-nemo:12b"}, "strictly 4-bit"),
])
def test_invalid_selection_fails_loudly(env, match):
    with pytest.raises(ValueError, match=match):
        resolve_judge_ollama_model(env)


def test_single_judge_follows_judge_index(monkeypatch):
    from src.config import judge_ensemble_specs

    for k in ("JUDGE_ENSEMBLE", "JUDGE_ENABLE_API_JUDGE", "JUDGE_MODEL_INDEX",
              "JUDGE_OLLAMA_MODEL", "JUDGE_QUANT"):
        monkeypatch.delenv(k, raising=False)
    assert judge_ensemble_specs() == [("ollama", DEFAULT_JUDGE_TAG)]

    monkeypatch.setenv("JUDGE_MODEL_INDEX", "2")
    assert judge_ensemble_specs() == [("ollama", "gemma3:12b-it-q4_K_M")]
    monkeypatch.setenv("JUDGE_QUANT", "qat")
    assert judge_ensemble_specs() == [("ollama", "gemma3:12b-it-qat")]
    monkeypatch.setenv("JUDGE_MODEL_INDEX", "0,5")
    with pytest.raises(ValueError, match="ONE judge"):
        judge_ensemble_specs()


@pytest.mark.parametrize("var, value", [
    ("JUDGE_ENSEMBLE", "ollama:phi4:14b-q8_0"),
    ("JUDGE_ENABLE_API_JUDGE", "1"),
])
def test_removed_ensemble_variables_fail_loudly(monkeypatch, var, value):
    from src.config import judge_ensemble_specs

    monkeypatch.delenv("JUDGE_MODEL_INDEX", raising=False)
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match="removed"):
        judge_ensemble_specs()


def test_thinking_detection_and_slug():
    assert is_thinking_model("qwen3:14b-q4_K_M")
    assert is_thinking_model("deepseek-r1:14b-qwen-distill-q4_K_M")
    assert not is_thinking_model("qwen2.5:14b-instruct-q4_K_M")
    assert not is_thinking_model("gemma3:12b-it-qat")
    assert judge_slug("ollama:gemma3:12b-it-qat") == "ollama_gemma3_12b-it-qat"


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_http_judge_payload_disables_thinking_and_sets_ctx():
    import src.config as cfg
    from src.core.llm_client import UnifiedLLMClient

    captured = {}

    def fake_post(url, json=None, timeout=None, **_):
        captured.clear()
        captured.update(json)
        return _Resp({"message": {"content": '{"ok": true}'}})

    client = UnifiedLLMClient(model="qwen3:14b-q4_K_M", provider="ollama",
                              strict=True, is_judge=True)
    with patch("src.core.llm_client.requests.post", fake_post):
        client.chat([{"role": "user", "content": "x"}], json_mode=True)
    assert captured["model"] == "qwen3:14b-q4_K_M"
    assert captured["think"] is False
    assert captured["format"] == "json"
    assert captured["options"]["num_ctx"] == cfg.JUDGE_NUM_CTX

    gen = UnifiedLLMClient(provider="ollama")  # generator: its own ctx, no think
    with patch("src.core.llm_client.requests.post", fake_post):
        gen.chat([{"role": "user", "content": "x"}])
    assert "think" not in captured
    assert captured["options"]["num_ctx"] == cfg.LLM_NUM_CTX


def test_check_ready_reports_missing_model():
    from src.core.llm_client import UnifiedLLMClient

    client = UnifiedLLMClient(model="gemma3:12b-it-q4_K_M", provider="ollama",
                              strict=True, is_judge=True)
    tags = _Resp({"models": [{"name": "llama3.1:8b"}]})
    with patch("src.core.llm_client.requests.get", return_value=tags):
        with pytest.raises(RuntimeError, match="ollama pull gemma3:12b-it-q4_K_M"):
            client.check_ready()


def test_ensemble_preflight_names_failing_judge():
    from src.evaluation.base_paper_evaluator import BasePaperEvaluator, MultiJudgeEvaluator

    good, bad = MagicMock(), MagicMock()
    bad.check_ready.side_effect = RuntimeError("not pulled")
    panel = MultiJudgeEvaluator([BasePaperEvaluator(judge_llm=good, name="ollama:a"),
                                 BasePaperEvaluator(judge_llm=bad, name="ollama:b")])
    with pytest.raises(RuntimeError, match="ollama:b: not pulled"):
        panel.check_ready()


def test_ensemble_counts_judge_failures():
    from src.evaluation.base_paper_evaluator import BasePaperEvaluator, MultiJudgeEvaluator

    judge = MagicMock()
    judge.chat.return_value = "not json at all"
    panel = MultiJudgeEvaluator([BasePaperEvaluator(judge_llm=judge, name="ollama:x")])
    panel.evaluate_scoring("q", "a", "ref", retries=0)
    panel.evaluate_selection("q", "a", "b", "ref", "A", "B", retries=0)
    assert panel.calls == {"ollama:x": 2} and panel.failures == {"ollama:x": 2}
    assert panel.failure_rate("ollama:x") == 1.0
    assert judge.chat.call_args.kwargs["json_mode"] is True
    assert panel.manifest()[0]["session_failures"] == 2


def test_checkpoint_resume_is_per_judge_and_generator(tmp_path):
    from src.config import LLM_MODEL
    from src.evaluation.run_base_paper_benchmark import (
        _completed_questions_from_checkpoint,
        _hcad_config_id,
        _records_from_checkpoint,
    )

    ckpt = tmp_path / "ckpt.jsonl"
    base = {"schema": 2, "question_pool": "test", "question_set_id": "s",
            "arms": {"hcad": "HCAD-RAG"}, "scores": {}, "selection": {},
            "hcad_config_id": _hcad_config_id(), "generator": LLM_MODEL}
    rows = [
        {**base, "question": "q1", "judges": ["ollama:qwen3:14b-q4_K_M"]},
        {**base, "question": "q2", "judges": ["ollama:gemma3:12b-it-q4_K_M"]},
        {**{k: v for k, v in base.items() if k != "hcad_config_id"},
         "question": "q3", "judges": ["ollama:qwen3:14b-q4_K_M"]},  # legacy record
        # same judge, answers from another generator precision -> never resumed
        {**base, "question": "q4", "judges": ["ollama:qwen3:14b-q4_K_M"],
         "generator": "llama3.1:8b-instruct-fp16"},
        # pre-generator-field record: written by the old 4-bit generator
        {**{k: v for k, v in base.items() if k != "generator"},
         "question": "q5", "judges": ["ollama:qwen3:14b-q4_K_M"]},
    ]
    ckpt.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    assert _completed_questions_from_checkpoint(
        ckpt, "test", "s", required_arms=["hcad"],
        required_judges=["ollama:qwen3:14b-q4_K_M"]) == (
            {"q1", "q5"} if LLM_MODEL == "llama3.1:8b-instruct-q4_K_M" else {"q1"})
    assert _completed_questions_from_checkpoint(
        ckpt, "test", "s", required_arms=["hcad"],
        required_judges=["ollama:phi4:14b-q4_K_M"]) == set()
    recs = _records_from_checkpoint(
        ckpt, {"q1", "q2", "q3"}, question_pool="test", question_set_id="s",
        required_judges=["ollama:gemma3:12b-it-q4_K_M"])
    assert [r["question"] for r in recs["per_question"]] == ["q2"]


def test_citation_tags_are_blinded_for_the_judge():
    from src.evaluation.run_base_paper_benchmark import _blind_citation_tags

    assert _blind_citation_tags("A [Evidence 1] and [Context 2, 3].") == \
        "A [Doc 1] and [Doc 2, 3]."
