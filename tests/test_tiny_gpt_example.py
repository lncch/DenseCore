"""The tiny GPT example trains and matches PyTorch over its first Adam steps."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("tiny_gpt", Path(__file__).resolve().parents[1] / "examples/tiny_gpt.py")
tiny_gpt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tiny_gpt)


def test_tiny_gpt_heldout_loss_falls_below_unigram_baseline():
    result = tiny_gpt.train(steps=150)
    assert result["heldout_loss_end"] < result["unigram_loss"] < result["heldout_loss_start"]
    assert len(result["sample"]) == 120


def test_tiny_gpt_first_adam_steps_match_torch():
    pytest.importorskip("torch")
    result = tiny_gpt.parity(steps=3)
    for mine, theirs in result["losses_densecore_vs_torch"]:
        assert abs(mine - theirs) <= 1e-5 * abs(theirs)
    assert result["max_weight_abs_diff"] < 2e-4
