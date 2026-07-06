"""Tests for onnesim.stats — exact CIs and McNemar, the statistical rigor the
reviewers asked for. Pure math, no network, so these run in CI."""
from onnesim import stats as ST


def test_clopper_pearson_known_bounds():
    # perfect and zero counts must hit the natural [0,1] rails on one side
    assert ST.clopper_pearson(10, 10)[1] == 1.0
    assert ST.clopper_pearson(0, 10)[0] == 0.0
    # 0/10 upper bound is the classic 0.308 (rule-of-three-ish), within tolerance
    lo, hi = ST.clopper_pearson(0, 10)
    assert 0.28 < hi < 0.34
    # a high proportion stays below 1 on the lower side and contains the point est
    lo, hi = ST.clopper_pearson(197, 200)
    assert lo < 197 / 200 < hi and lo > 0.94


def test_clopper_pearson_empty():
    assert ST.clopper_pearson(0, 0) == (0.0, 1.0)


def test_mcnemar_symmetric_is_insignificant():
    # equal discordant counts -> no evidence of difference
    assert ST.mcnemar_exact(5, 5)["p_value"] == 1.0
    assert ST.mcnemar_exact(0, 0)["p_value"] == 1.0


def test_mcnemar_lopsided_is_significant():
    # strongly lopsided discordant pairs -> tiny p (this is the classification case)
    p = ST.mcnemar_exact(2, 62)["p_value"]
    assert p < 1e-10
    # and it is symmetric in argument order
    assert ST.mcnemar_exact(62, 2)["p_value"] == p


def test_prop_ci_shape():
    d = ST.prop_ci(137, 200)
    assert d["k"] == 137 and d["n"] == 200 and d["ci_pct"] == 95
    assert d["lo"] < d["p"] < d["hi"]
