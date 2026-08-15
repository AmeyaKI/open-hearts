"""Phase 5 Task 3: profiler export/load round-trip, masked CE, determinism.

Torch is a TRAINING-ONLY dependency, so every torch-touching test here is
guarded by `pytest.importorskip("torch")`; the suite stays green in a
torch-free install. The first test asserts exactly that hygiene: importing
`openhearts.opponent` must not drag torch in. Mirrors
`tests/test_value_model.py`.
"""
import numpy as np
import pytest

from openhearts.engine import features
from openhearts.opponent import (PARAM_DIM, forward_numpy, load_npz,
                                 masked_probs_numpy)

H1, H2 = 32, 16


def _batch(n=256, seed=0, n_param_in=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, features.NF + n_param_in)) < 0.2).astype(np.float32)
    masks = np.zeros(n, dtype=np.int64)
    chosen = np.zeros(n, dtype=np.int64)
    for i in range(n):
        k = int(rng.integers(2, 10))
        cards_i = rng.choice(52, size=k, replace=False)
        m = 0
        for c in cards_i:
            m |= 1 << int(c)
        masks[i] = m
        chosen[i] = int(cards_i[0])
    return X, masks, chosen


def test_opponent_package_import_is_torch_free():
    import subprocess
    import sys
    code = (
        "import sys, openhearts.opponent;"
        "assert 'torch' not in sys.modules, "
        "'openhearts.opponent imported torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_export_load_forward_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    from openhearts.opponent import model as pmodel

    m = pmodel.make_model(7, conditioned=False, hidden1=H1, hidden2=H2)
    path = str(tmp_path / "w.npz")
    pmodel.export_npz(m, path)
    w, meta = load_npz(path)
    assert meta["features_v"] == features.FEATURES_V
    assert meta["profiler_v"] == 1
    assert meta["n_param_in"] == 0
    assert meta["layer_sizes"] == [features.NF, H1, H2, 52]

    X, masks, _c = _batch(512, seed=1)
    legal = pmodel.mask_bits(masks)
    with torch.no_grad():
        ref = torch.exp(pmodel.masked_log_probs(m(torch.from_numpy(X)),
                                                legal)).numpy().astype(float)
    got = masked_probs_numpy(forward_numpy(w, X), masks)
    # probabilities on [0,1]: an absolute gate is the meaningful one here,
    # and the reference is torch FLOAT32 (~6e-8 relative representation
    # error), so 1e-6 absolute is the honest agreement level.
    assert np.abs(got - ref).max() <= 1e-6
    assert np.all(got[ref == 0.0] == 0.0)


def test_conditioned_export_carries_param_width(tmp_path):
    pytest.importorskip("torch")
    from openhearts.opponent import model as pmodel
    m = pmodel.make_model(3, conditioned=True, hidden1=H1, hidden2=H2)
    path = str(tmp_path / "c.npz")
    pmodel.export_npz(m, path)
    _w, meta = load_npz(path)
    assert meta["n_param_in"] == PARAM_DIM
    assert meta["layer_sizes"][0] == features.NF + PARAM_DIM
    assert meta["variant"] == "conditioned"


def test_masked_ce_ignores_illegal_logits(tmp_path):
    """Changing an ILLEGAL card's logit must not change the loss at all."""
    torch = pytest.importorskip("torch")
    from openhearts.opponent import model as pmodel
    X, masks, chosen = _batch(64, seed=2)
    legal = pmodel.mask_bits(masks)
    rng = np.random.default_rng(0)
    logits = torch.from_numpy(rng.normal(0, 2, (64, 52)).astype(np.float32))
    y = torch.from_numpy(chosen)
    base = float(pmodel.masked_ce(logits, legal, y))
    bumped = logits.clone()
    bumped[~legal] += 100.0
    assert abs(float(pmodel.masked_ce(bumped, legal, y)) - base) < 1e-6


def test_training_is_deterministic():
    """Same seeds -> same weights, twice in a row (Phase-4 convention)."""
    torch = pytest.importorskip("torch")
    from openhearts.opponent import model as pmodel

    X, masks, chosen = _batch(512, seed=4)
    valX, valM, valC = _batch(128, seed=5)

    def batches(_ep):
        for i in range(0, X.shape[0], 128):
            yield (np.ascontiguousarray(X[i:i + 128]),
                   np.ascontiguousarray(masks[i:i + 128]),
                   np.ascontiguousarray(chosen[i:i + 128]))

    outs = []
    for _ in range(2):
        m = pmodel.make_model(99, hidden1=H1, hidden2=H2)
        pmodel.train(m, batches, valX, valM, valC, device="cpu", epochs=3,
                     patience=3, seed=99, log=lambda *a, **k: None)
        outs.append(pmodel.weights_dict(m))
    for k in outs[0]:
        assert np.array_equal(outs[0][k], outs[1][k]), f"{k} not deterministic"
    assert torch is not None


def test_make_model_seed_controls_init():
    pytest.importorskip("torch")
    from openhearts.opponent import model as pmodel
    a = pmodel.weights_dict(pmodel.make_model(1, hidden1=H1, hidden2=H2))
    b = pmodel.weights_dict(pmodel.make_model(1, hidden1=H1, hidden2=H2))
    c = pmodel.weights_dict(pmodel.make_model(2, hidden1=H1, hidden2=H2))
    assert np.array_equal(a["W1"], b["W1"])
    assert not np.array_equal(a["W1"], c["W1"])
