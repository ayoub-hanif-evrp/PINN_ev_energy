from evaluation.bootstrap import bootstrap_mean_ci, paired_difference_ci
import pandas as pd


def test_bootstrap_is_deterministic():
    x = pd.Series([0.2, 0.3, 0.1, 0.4], index=list("abcd"))
    a = bootstrap_mean_ci(x.to_numpy(), n_resamples=200, seed=1)
    b = bootstrap_mean_ci(x.to_numpy(), n_resamples=200, seed=1)
    assert a == b
    y = pd.Series([0.25, 0.35, 0.05, 0.5], index=list("abcd"))
    d = paired_difference_ci(x, y, n_resamples=200, seed=1)
    assert d["n_trips"] == 4
    assert d["ci_low"] <= d["mean"] <= d["ci_high"]
