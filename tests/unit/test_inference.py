"""Dependence-robust inference: corrections and test size under common shocks."""

import math

import numpy as np
import pytest

from chronos.validation.inference import (
    benjamini_hochberg,
    holm,
    nonoverlap_origin_t_test,
    origin_block_bootstrap,
)


def test_holm_and_bh_known_values():
    p = [0.01, 0.04, 0.03, None, 0.20]
    h = holm(p)
    assert h[0] == pytest.approx(0.04) and h[2] == pytest.approx(0.09) and h[1] == pytest.approx(0.09)
    assert h[4] == pytest.approx(0.20) and h[3] is None
    b = benjamini_hochberg([0.01, 0.04, 0.03])
    assert b == pytest.approx([0.03, 0.04, 0.04])


def _panel_null(rng, rho=0.6, h=10, step=5, n_ent=40, n_orig=40):
    years = n_orig * step + h + 5
    common = rng.standard_t(4, size=years)
    idio = rng.standard_t(4, size=(n_ent, years))
    z = math.sqrt(rho) * common + math.sqrt(1 - rho) * idio
    return {k * step: list(z[:, k * step:k * step + h].sum(axis=1) / math.sqrt(h)) for k in range(n_orig)}


def test_nonoverlap_t_test_has_near_nominal_size_under_common_shocks():
    """Regression for the Phase 7 finding: the entity sign test rejected 52-78% of true nulls."""
    rng = np.random.default_rng(3)
    rej = sum(nonoverlap_origin_t_test(_panel_null(rng), 10, 5)["p_value"] < 0.05 for _ in range(300))
    assert rej / 300 < 0.11


def test_entity_sign_test_is_invalid_under_common_shocks():
    """Documents why the sign test was removed from verdicts: same null, very high false-positive rate."""
    from scipy import stats

    rng = np.random.default_rng(4)
    rej = 0
    for _ in range(200):
        d = _panel_null(rng)
        ent = np.array([d[o] for o in sorted(d)]).mean(axis=0)
        rej += stats.binomtest(int((ent < 0).sum()), len(ent), 0.5).pvalue < 0.05
    assert rej / 200 > 0.3


def test_tests_detect_a_real_improvement():
    rng = np.random.default_rng(5)
    d = _panel_null(rng, rho=0.3)
    shifted = {o: [x - 0.5 for x in v] for o, v in d.items()}
    assert nonoverlap_origin_t_test(shifted, 10, 5)["significant_improvement"]
    assert origin_block_bootstrap(shifted, 2, n_boot=500)["significant_improvement"]


def test_too_few_origins():
    assert nonoverlap_origin_t_test({1: [0.1], 2: [0.2]}, 10, 1)["p_value"] is None
    assert "note" in origin_block_bootstrap({1: [0.1]}, 1)
