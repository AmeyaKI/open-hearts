"""Bootstrap confidence intervals over per-deal observations.

One value per deal goes in (already averaged over the 4 seat rotations, see
`harness.rotated_match`). Resampling deals -- not individual games -- is the
whole point: the 4 rotations of one deal share that deal's luck, so they are
one independent observation, not four.
"""
import numpy as np


def bootstrap_ci(per_deal_values, n_boot: int = 10_000, rng=None):
    """Return (mean, lo95, hi95) by resampling deals with replacement.

    `rng` defaults to np.random.default_rng(0) so results are reproducible.
    """
    data = np.asarray(per_deal_values, dtype=float)
    assert data.ndim == 1 and data.size > 0, "expected a 1-D array of per-deal values"
    if rng is None:
        rng = np.random.default_rng(0)
    n = data.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = data[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(data.mean()), float(lo), float(hi)
