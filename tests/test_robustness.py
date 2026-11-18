"""Tests for src/robustness.py."""


def test_block_bootstrap_ci_brackets_a_clear_difference():
    import numpy as np

    from src.robustness import block_bootstrap_diff

    rng = np.random.default_rng(1)
    a = rng.normal(10, 2, 120)
    b = a + 5 + rng.normal(0, 1, 120)
    r = block_bootstrap_diff(a, b, n_boot=500)
    assert r["ci_high"] < 0 and r["share_a_better"] == 1.0
