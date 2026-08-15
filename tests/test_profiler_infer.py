"""Phase 5 Task 3: numba profiler-inference kernel tests.

Torch-free throughout (no `importorskip("torch")` anywhere) -- the kernel and
this test both stay usable in a torch-free install. Agreement is pinned
against `opponent/npz_io.forward_numpy` + `masked_probs_numpy` (torch-free
float64 references), not torch itself.

WEIGHTS ARE SYNTHESIZED HERE, NOT LOADED FROM `models/`. Phase 4's equivalent
test could point at a committed `models/value_v1.npz`; `models/profiler_v1.npz`
is written by the LEAD's full training run, so a fixture depending on it would
make this suite red for anyone who has not trained. The kernel contract under
test (sparse first layer, legal-only output rows, masked softmax) is
weight-independent, so a seeded random export exercises it exactly as well --
paired with REAL feature rows and REAL legal masks from the Task-2 shards,
which is where the sparsity and mask structure actually come from.
"""
import glob

import numpy as np
import pytest

from openhearts.engine import features as feat_mod
from openhearts.engine.features import NF
from openhearts.opponent.infer import (load_profiler, profiler_probs,
                                       profiler_probs_batch)
from openhearts.opponent.npz_io import (N_CARDS, PROFILER_V, forward_numpy,
                                        load_npz, masked_probs_numpy, save_npz)
from openhearts.opponent.params import PARAM_DIM

H1, H2 = 32, 16   # small: these tests check numerics, not capacity


def _rows_from_shard(n_want=10000):
    """Real (features, legal_mask, chosen) rows from the Task-2 shards."""
    paths = sorted(glob.glob("results/population_data/pop_test_*.npz"))
    Xs, Ls, Cs, n = [], [], [], 0
    for p in paths:
        d = np.load(p, allow_pickle=True)
        Xs.append(d["profiler_features"].astype(np.float64))
        Ls.append(d["legal_mask"].astype(np.int64))
        Cs.append(d["chosen_card"].astype(np.int64))
        n += Xs[-1].shape[0]
        if n >= n_want:
            break
    if not Xs:
        return None
    return (np.concatenate(Xs)[:n_want], np.concatenate(Ls)[:n_want],
            np.concatenate(Cs)[:n_want])


@pytest.fixture(scope="module")
def shard_rows():
    rows = _rows_from_shard()
    assert rows is not None, (
        "results/population_data/pop_test_*.npz shards not found -- Task 2's "
        "generator must have run first (see PHASE5_PLAN.md Task 2)")
    assert rows[0].shape[0] >= 10000
    return rows


def _make_npz(tmp_path, name, n_param_in=0, seed=0):
    rng = np.random.default_rng(seed)
    n_in = NF + n_param_in
    w = {"W1": rng.normal(0, 0.1, (H1, n_in)), "b1": rng.normal(0, 0.1, H1),
         "W2": rng.normal(0, 0.2, (H2, H1)), "b2": rng.normal(0, 0.1, H2),
         "W3": rng.normal(0, 0.3, (N_CARDS, H2)),
         "b3": rng.normal(0, 0.1, N_CARDS)}
    path = str(tmp_path / name)
    save_npz(path, w, [n_in, H1, H2, N_CARDS], n_param_in)
    return path


@pytest.fixture(scope="module")
def generic_weights(tmp_path_factory):
    path = _make_npz(tmp_path_factory.mktemp("w"), "generic.npz", 0, seed=11)
    return load_profiler(path)[0], path


# ------------------------------------------------------- masked-softmax core
def test_probs_sum_to_one_over_legal_and_zero_elsewhere(shard_rows,
                                                        generic_weights):
    (W1, b1, W2, b2, W3, b3), _ = generic_weights
    X, L, _C = shard_rows
    for i in range(300):
        p = profiler_probs(W1, b1, W2, b2, W3, b3, X[i], int(L[i]))
        assert p.shape == (N_CARDS,)
        legal = np.array([(int(L[i]) >> c) & 1 for c in range(N_CARDS)],
                         dtype=bool)
        # illegal cards are EXACTLY zero -- not merely tiny.
        assert np.all(p[~legal] == 0.0)
        assert np.all(p[legal] > 0.0)
        assert abs(float(p.sum()) - 1.0) <= 1e-12


def test_empty_mask_returns_zeros_not_nan(generic_weights):
    """Defensive: the kernel must not produce NaN on a degenerate mask.

    Real rows always have >= 2 legal cards, but the kernel is the play-path
    artifact and a -inf-based implementation would return NaN here.
    """
    (W1, b1, W2, b2, W3, b3), _ = generic_weights
    x = np.zeros(NF, dtype=np.float64)
    p = profiler_probs(W1, b1, W2, b2, W3, b3, x, 0)
    assert np.all(p == 0.0)


def test_single_legal_card_gets_probability_one(generic_weights):
    (W1, b1, W2, b2, W3, b3), _ = generic_weights
    x = np.zeros(NF, dtype=np.float64)
    x[7] = 1.0
    p = profiler_probs(W1, b1, W2, b2, W3, b3, x, 1 << 19)
    assert p[19] == 1.0
    assert p.sum() == 1.0


# ------------------------------------------------- agreement with reference
def _reference(path, X, L):
    w, _meta = load_npz(path)
    return masked_probs_numpy(forward_numpy(w, X), L)


def test_batch_matches_numpy_reference_on_real_rows(shard_rows,
                                                    generic_weights):
    (W1, b1, W2, b2, W3, b3), path = generic_weights
    X, L, _C = shard_rows
    assert X.shape[0] >= 10000
    got = np.zeros((X.shape[0], N_CARDS), dtype=np.float64)
    profiler_probs_batch(W1, b1, W2, b2, W3, b3, X, L, got)
    ref = _reference(path, X, L)
    # denominator floored at 1e-3: probabilities are legitimately tiny, and a
    # naive relative error on a 1e-9 probability measures float64 noise.
    rel = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-3)))
    assert rel <= 1e-6, f"max relative error {rel} > 1e-6"
    # the kernel's exact-zero guarantee, checked against the reference's mask
    assert np.all(got[ref == 0.0] == 0.0)


def test_single_row_matches_reference(shard_rows, generic_weights):
    (W1, b1, W2, b2, W3, b3), path = generic_weights
    X, L, _C = shard_rows
    ref = _reference(path, X[:200], L[:200])
    for i in range(200):
        got = profiler_probs(W1, b1, W2, b2, W3, b3, X[i], int(L[i]))
        rel = np.max(np.abs(got - ref[i]) / np.maximum(np.abs(ref[i]), 1e-3))
        assert rel <= 1e-6


def test_conditioned_input_width_matches_reference(shard_rows, tmp_path):
    """CONDITIONED weights (n_in = NF + PARAM_DIM) run through the same path."""
    path = _make_npz(tmp_path, "cond.npz", PARAM_DIM, seed=5)
    (W1, b1, W2, b2, W3, b3), meta = load_profiler(path)
    assert meta["n_param_in"] == PARAM_DIM
    assert W1.shape[0] == NF + PARAM_DIM
    X, L, _C = shard_rows
    rng = np.random.default_rng(3)
    XP = np.concatenate(
        [X[:500], rng.normal(0, 0.5, (500, PARAM_DIM))], axis=1)
    got = np.zeros((500, N_CARDS), dtype=np.float64)
    profiler_probs_batch(W1, b1, W2, b2, W3, b3, XP, L[:500], got)
    ref = _reference(path, XP, L[:500])
    rel = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-3)))
    assert rel <= 1e-6


# ------------------------------------------------------- njit vs pure Python
def test_njit_matches_python_fallback(shard_rows, generic_weights,
                                      monkeypatch):
    from openhearts.engine import kernel
    (W1, b1, W2, b2, W3, b3), _ = generic_weights
    X, L, _C = shard_rows
    X, L = X[:500], L[:500]

    monkeypatch.delenv("OPENHEARTS_NO_JIT", raising=False)
    kernel.reset_jit_enabled()
    out_jit = np.zeros((X.shape[0], N_CARDS), dtype=np.float64)
    profiler_probs_batch(W1, b1, W2, b2, W3, b3, X, L, out_jit)
    single_jit = np.stack([profiler_probs(W1, b1, W2, b2, W3, b3, X[i],
                                          int(L[i])) for i in range(20)])

    monkeypatch.setenv("OPENHEARTS_NO_JIT", "1")
    kernel.reset_jit_enabled()
    out_py = np.zeros((X.shape[0], N_CARDS), dtype=np.float64)
    profiler_probs_batch(W1, b1, W2, b2, W3, b3, X, L, out_py)
    single_py = np.stack([profiler_probs(W1, b1, W2, b2, W3, b3, X[i],
                                         int(L[i])) for i in range(20)])

    kernel.reset_jit_enabled()

    assert np.array_equal(out_jit, out_py)
    assert np.array_equal(single_jit, single_py)


# -------------------------------------------------------- metadata contracts
def _bad_npz(tmp_path, name, **over):
    n_in = over.pop("n_in", NF)
    fields = dict(
        W1=np.zeros((H1, n_in)), b1=np.zeros(H1),
        W2=np.zeros((H2, H1)), b2=np.zeros(H2),
        W3=np.zeros((N_CARDS, H2)), b3=np.zeros(N_CARDS),
        features_v=np.int64(feat_mod.FEATURES_V),
        profiler_v=np.int64(PROFILER_V), nf=np.int64(NF),
        n_param_in=np.int64(n_in - NF),
        layer_sizes=np.array([n_in, H1, H2, N_CARDS], dtype=np.int64),
        arch=np.array("profiler-mlp-relu"))
    fields.update(over)
    path = str(tmp_path / name)
    np.savez(path, **fields)
    return path


def test_rejects_bad_features_v(tmp_path):
    path = _bad_npz(tmp_path, "bad_fv.npz",
                    features_v=np.int64(feat_mod.FEATURES_V + 1))
    with pytest.raises(ValueError):
        load_profiler(path)


def test_rejects_bad_profiler_v(tmp_path):
    path = _bad_npz(tmp_path, "bad_pv.npz",
                    profiler_v=np.int64(PROFILER_V + 1))
    with pytest.raises(ValueError):
        load_profiler(path)


def test_rejects_missing_profiler_v(tmp_path):
    """A Phase-4 VALUE npz must not load as a profiler by accident."""
    path = str(tmp_path / "value_shaped.npz")
    np.savez(path, W1=np.zeros((H1, NF)), b1=np.zeros(H1),
             W2=np.zeros((H2, H1)), b2=np.zeros(H2),
             W3=np.zeros((4, H2)), b3=np.zeros(4),
             features_v=np.int64(feat_mod.FEATURES_V), nf=np.int64(NF),
             layer_sizes=np.array([NF, H1, H2, 4], dtype=np.int64),
             arch=np.array("mlp-relu"))
    with pytest.raises(ValueError):
        load_profiler(path)


def test_rejects_bad_nf(tmp_path):
    path = _bad_npz(tmp_path, "bad_nf.npz", nf=np.int64(NF + 1))
    with pytest.raises(ValueError):
        load_profiler(path)


def test_rejects_layer_shape_mismatch(tmp_path):
    path = _bad_npz(tmp_path, "bad_shape.npz", W1=np.zeros((H1, NF - 1)))
    with pytest.raises(ValueError):
        load_profiler(path)


def test_rejects_wrong_output_dim(tmp_path):
    path = _bad_npz(tmp_path, "bad_out.npz",
                    W3=np.zeros((51, H2)), b3=np.zeros(51),
                    layer_sizes=np.array([NF, H1, H2, 51], dtype=np.int64))
    with pytest.raises((ValueError, AssertionError)):
        load_profiler(path)


def test_save_npz_rejects_inconsistent_param_width(tmp_path):
    rng = np.random.default_rng(0)
    w = {"W1": rng.normal(0, .1, (H1, NF)), "b1": np.zeros(H1),
         "W2": np.zeros((H2, H1)), "b2": np.zeros(H2),
         "W3": np.zeros((N_CARDS, H2)), "b3": np.zeros(N_CARDS)}
    with pytest.raises(ValueError):
        save_npz(str(tmp_path / "x.npz"), w, [NF, H1, H2, N_CARDS], PARAM_DIM)


# ------------------------------------------------------ param vectorization
def test_param_vector_layout_and_ranges():
    from openhearts.opponent.params import param_table, param_vector
    from openhearts.players.personality import ANCHOR_IDS, make_population
    train, held = make_population(200, 50, 314159)
    V = np.stack([param_vector(p) for p in train + held])
    assert V.shape == (250, PARAM_DIM)
    # anchor one-hot is all-zero for personalities
    assert np.all(V[:, -len(ANCHOR_IDS):] == 0.0)
    # normalization keeps every axis on a comparable scale
    assert np.all(np.abs(V) <= 1.5), np.abs(V).max()
    # distinct personalities get distinct vectors
    assert len(set(map(tuple, V.tolist()))) == 250
    for name, pid in ANCHOR_IDS.items():
        v = param_vector(pid)
        assert v[:-len(ANCHOR_IDS)].sum() == 0.0
        assert v[-len(ANCHOR_IDS):].sum() == 1.0
    # the lookup table agrees with the per-id function
    lut = param_table(train[:5])
    for p in train[:5]:
        assert np.array_equal(lut[p], param_vector(p))
