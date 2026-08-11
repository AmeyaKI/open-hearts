"""Phase 4 Task 4: numba value-inference kernel tests.

Torch-free throughout (no `importorskip("torch")` anywhere) -- the kernel and
this test both stay usable in a torch-free install. Agreement is pinned
against `npz_io.forward_numpy` (a torch-free float64 reference), not torch
itself.
"""
import numpy as np
import pytest

from openhearts.engine import features as feat_mod
from openhearts.value.infer import (load_weights, value_forward,
                                    value_forward_batch)
from openhearts.value.npz_io import forward_numpy

MODEL_PATH = "models/value_v1.npz"


def _rows_from_shard():
    """Real feature rows from the Task 2 self-play shards (preferred: less
    code and identical convention to `experiments/train_value.py`, which
    already reads these shards). Returns None if the shards aren't present."""
    import glob
    paths = sorted(glob.glob("results/value_data/value_test_*.npz"))
    rows = []
    n = 0
    for p in paths:
        d = np.load(p, allow_pickle=True)
        X = d["features"].astype(np.float64)
        rows.append(X)
        n += X.shape[0]
        if n >= 10000:
            break
    if rows:
        return np.concatenate(rows, axis=0)[:10000]
    return None


@pytest.fixture(scope="module")
def feature_rows():
    rows = _rows_from_shard()
    assert rows is not None, (
        "results/value_data/value_test_*.npz shards not found -- Task 2's "
        "data generator must have run first (see PHASE4_PLAN.md Task 2)")
    return rows


@pytest.fixture(scope="module")
def weights():
    return load_weights(MODEL_PATH)


def test_value_forward_batch_matches_forward_numpy(feature_rows, weights):
    W1, b1, W2, b2, W3, b3 = weights
    X = feature_rows
    assert X.shape[0] >= 10000
    out = np.zeros((X.shape[0], 4), dtype=np.float64)
    value_forward_batch(W1, b1, W2, b2, W3, b3, X, out)

    # load_weights returns W1 transposed (NF, h1) for the sparse first
    # layer; forward_numpy expects the stored (h1, NF) orientation.
    w_dict = {"W1": W1.T, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    ref = forward_numpy(w_dict, X)

    # Relative error, denominator floored at 1 point (house convention, see
    # experiments/train_value.py's npz round-trip check).
    rel_err = float(np.max(np.abs(out - ref) / np.maximum(np.abs(ref), 1.0)))
    assert rel_err <= 1e-6, f"max relative error {rel_err} > 1e-6"


def test_value_forward_single_row_matches_batch(feature_rows, weights):
    W1, b1, W2, b2, W3, b3 = weights
    X = feature_rows[:200]
    for i in range(X.shape[0]):
        got = value_forward(W1, b1, W2, b2, W3, b3, X[i])
        w_dict = {"W1": W1.T, "b1": b1,
                  "W2": W2, "b2": b2, "W3": W3, "b3": b3}
        ref = forward_numpy(w_dict, X[i:i + 1])[0]
        rel = np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1.0))
        assert rel <= 1e-6


def test_njit_matches_python_fallback(feature_rows, weights, monkeypatch):
    from openhearts.engine import kernel
    W1, b1, W2, b2, W3, b3 = weights
    X = feature_rows[:500]

    monkeypatch.delenv("OPENHEARTS_NO_JIT", raising=False)
    kernel.reset_jit_enabled()
    out_jit = np.zeros((X.shape[0], 4), dtype=np.float64)
    value_forward_batch(W1, b1, W2, b2, W3, b3, X, out_jit)
    single_jit = np.stack(
        [value_forward(W1, b1, W2, b2, W3, b3, X[i]) for i in range(20)])

    monkeypatch.setenv("OPENHEARTS_NO_JIT", "1")
    kernel.reset_jit_enabled()
    out_py = np.zeros((X.shape[0], 4), dtype=np.float64)
    value_forward_batch(W1, b1, W2, b2, W3, b3, X, out_py)
    single_py = np.stack(
        [value_forward(W1, b1, W2, b2, W3, b3, X[i]) for i in range(20)])

    kernel.reset_jit_enabled()

    assert np.array_equal(out_jit, out_py)
    assert np.array_equal(single_jit, single_py)


def test_load_weights_rejects_bad_features_v(tmp_path):
    from openhearts.engine.features import NF
    path = str(tmp_path / "bad_fv.npz")
    np.savez(path, W1=np.zeros((128, NF)), b1=np.zeros(128),
             W2=np.zeros((64, 128)), b2=np.zeros(64),
             W3=np.zeros((4, 64)), b3=np.zeros(4),
             features_v=np.int64(feat_mod.FEATURES_V + 1),
             nf=np.int64(NF),
             layer_sizes=np.array([NF, 128, 64, 4], dtype=np.int64),
             arch=np.array("mlp-relu"))
    with pytest.raises(ValueError):
        load_weights(path)


def test_load_weights_rejects_bad_shape(tmp_path):
    from openhearts.engine.features import NF
    path = str(tmp_path / "bad_shape.npz")
    # W1 declares a different input width than layer_sizes[0]/NF claims.
    np.savez(path, W1=np.zeros((128, NF - 1)), b1=np.zeros(128),
             W2=np.zeros((64, 128)), b2=np.zeros(64),
             W3=np.zeros((4, 64)), b3=np.zeros(4),
             features_v=np.int64(feat_mod.FEATURES_V),
             nf=np.int64(NF),
             layer_sizes=np.array([NF, 128, 64, 4], dtype=np.int64),
             arch=np.array("mlp-relu"))
    with pytest.raises(ValueError):
        load_weights(path)


def test_load_weights_rejects_wrong_output_dim(tmp_path):
    from openhearts.engine.features import NF
    path = str(tmp_path / "bad_out.npz")
    np.savez(path, W1=np.zeros((128, NF)), b1=np.zeros(128),
             W2=np.zeros((64, 128)), b2=np.zeros(64),
             W3=np.zeros((3, 64)), b3=np.zeros(3),
             features_v=np.int64(feat_mod.FEATURES_V),
             nf=np.int64(NF),
             layer_sizes=np.array([NF, 128, 64, 3], dtype=np.int64),
             arch=np.array("mlp-relu"))
    with pytest.raises((ValueError, AssertionError)):
        load_weights(path)
