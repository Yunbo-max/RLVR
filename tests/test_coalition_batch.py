"""Batched coalition teacher scoring must be numerically identical to the
sequential per-coalition scoring, and independent of the microbatch chunking.

These tests need torch (and transformers for a real small causal LM), so they
skip gracefully in the lightweight venv used for CPU-only tests.
"""
import importlib.util
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_learning_credit_five_round.py"
spec = importlib.util.spec_from_file_location("lc5_runner", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _fake_legacy():
    """Replace run_recursive_shapley_sdpo.prompt_ids/_logprobs_for_fixed_response."""
    class Legacy:
        @staticmethod
        def prompt_ids(tokenizer, row, fb):
            n = 0 if fb is None else len(fb)
            return list(range(100, 105 + n * 3))

        @staticmethod
        def _logprobs_for_fixed_response(model, tokenizer, row, fb, response_ids,
                                         *, return_distribution):
            ctx = Legacy.prompt_ids(tokenizer, row, fb)
            full = torch.tensor([ctx + response_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(input_ids=full, use_cache=False).logits[
                    0, len(ctx) - 1 : -1
                ].float()
            chosen = torch.as_tensor(response_ids, dtype=torch.long)
            selected = logits.gather(1, chosen.unsqueeze(1)).squeeze(1) - torch.logsumexp(
                logits, dim=-1
            )
            return selected.detach().cpu().numpy(), None

    return Legacy


def _teacher():
    cfg = transformers.GPT2Config(vocab_size=200, n_embd=32, n_layer=2, n_head=4)
    return transformers.GPT2LMHeadModel(cfg).eval()


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(m, "legacy", _fake_legacy())
    return m


def test_batch_matches_sequential_scoring(harness):
    teacher = _teacher()
    tokens = [10, 11, 12]
    feedback = [{"id": str(i)} for i in range(4)]
    masks = list(range(16))
    with torch.no_grad():
        batch = harness._coalition_score_batch(
            teacher, None, {}, tokens, feedback, masks, 8192, microbatch=4
        )
        seq = [
            harness._coalition_score(teacher, None, {}, tokens, feedback, mk, 8192)
            for mk in masks
        ]
    for a, b in zip(batch, seq):
        assert np.allclose(a, b, atol=1e-5)


def test_microbatch_invariance(harness):
    teacher = _teacher()
    tokens = [10, 11, 12]
    feedback = [{"id": str(i)} for i in range(4)]
    masks = list(range(16))
    with torch.no_grad():
        r1 = harness._coalition_score_batch(
            teacher, None, {}, tokens, feedback, masks, 8192, microbatch=1
        )
        r4 = harness._coalition_score_batch(
            teacher, None, {}, tokens, feedback, masks, 8192, microbatch=4
        )
        r16 = harness._coalition_score_batch(
            teacher, None, {}, tokens, feedback, masks, 8192, microbatch=16
        )
    for a, b in zip(r1, r4):
        assert np.allclose(a, b, atol=1e-5)
    for a, b in zip(r1, r16):
        assert np.allclose(a, b, atol=1e-5)


def test_microbatch_rejects_nonpositive(harness):
    teacher = _teacher()
    with pytest.raises(ValueError):
        harness._coalition_score_batch(
            teacher, None, {}, [10], [{"id": "0"}], [0], 8192, microbatch=0
        )
