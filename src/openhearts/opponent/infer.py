"""Torch-free numba inference kernel for the learned profiler (Phase 5 Task 3).

Loads a profiler `.npz` (via `npz_io.load_npz`, no duplicated parsing) and
runs the forward pass + MASKED SOFTMAX with plain loops, no `np.dot` inside
the `@njit` function -- same house convention as `engine/kernel.py`,
`engine/features.py` and `value/infer.py`: ONE Python-source function is
either compiled with `@njit` or used directly as the `OPENHEARTS_NO_JIT=1`
fallback, dispatched via `kernel.jit_enabled()`, so the two paths cannot
drift apart by construction.

Pinned against `npz_io.forward_numpy` + `npz_io.masked_probs_numpy` (the
torch-free float64 reference) to <=1e-6 relative. See
`tests/test_profiler_infer.py`.

TWO OPTIMIZATIONS, both documented because both are contracts:

1. SPARSE FIRST LAYER (Phase 4's trick, same feature family). `load_profiler`
   returns `W1` TRANSPOSED, `(n_in, h1)` rather than the stored `(h1, n_in)`,
   so the forward pass can walk input columns and skip exact zeros -- of the
   333 PROFILER_FEATURES_V=1 features only ~87 are nonzero, and the hidden
   blocks r=1..3 (156 dims) are STRUCTURALLY zero in every profiler row, so
   the skip is guaranteed, not merely typical. This transposed layout is a
   PRIVATE CONTRACT between `load_profiler` and `profiler_probs`: anyone
   rebuilding a dict for `forward_numpy` must transpose back (the tests do).

2. LEGAL-ONLY OUTPUT ROWS. Illegal logits are discarded by the mask anyway,
   so only the legal rows of `W3` are evaluated (typically 1-13 of 52). Each
   output is an independent dot product, so the legal entries are BIT-
   IDENTICAL to computing all 52 -- this is not an approximation. The
   reference in `npz_io` deliberately computes all 52 and masks afterwards,
   so it stays an independent check of both tricks.

The masked softmax never materializes `-inf`: it maxes/exponentiates over the
legal indices only and leaves the output array's other slots at their
initialized 0.0. Illegal cards therefore get EXACTLY 0.0, and an empty mask
returns all-zeros instead of NaN.

Hidden sizes are NEVER hardcoded: every loop bound comes from an array shape,
so the kernel serves any MLP-relu-relu-linear of this weight layout, GENERIC
(n_in = NF) and CONDITIONED (n_in = NF + PARAM_DIM) alike.
"""
import numpy as np

from ..engine.features import FEATURES_V, NF
from ..engine.kernel import HAVE_NUMBA, jit_enabled, njit
from .npz_io import N_CARDS, PROFILER_V, load_npz


def load_profiler(path):
    """Load a profiler `.npz` -> `((W1,b1,W2,b2,W3,b3), meta)`.

    `npz_io.load_npz` already asserts FEATURES_V / PROFILER_V / NF and
    per-layer shape consistency against `layer_sizes`; this adds the two
    edges of the whole-network contract the kernel relies on -- input width
    exactly `NF + n_param_in`, output width exactly 52 -- and hands back W1
    transposed for the sparse first layer (see module docstring).
    """
    w, meta = load_npz(path)
    sizes = meta["layer_sizes"]
    assert meta["features_v"] == FEATURES_V, (
        f"{path}: features_v={meta['features_v']} != {FEATURES_V}")
    assert meta["profiler_v"] == PROFILER_V, (
        f"{path}: profiler_v={meta['profiler_v']} != {PROFILER_V}")
    n_in = NF + int(meta["n_param_in"])
    assert sizes[0] == n_in, f"{path}: input dim {sizes[0]} != {n_in}"
    assert sizes[-1] == N_CARDS, f"{path}: output dim {sizes[-1]} != 52"
    W1 = np.ascontiguousarray(w["W1"].T, dtype=np.float64)
    b1 = np.ascontiguousarray(w["b1"], dtype=np.float64)
    W2 = np.ascontiguousarray(w["W2"], dtype=np.float64)
    b2 = np.ascontiguousarray(w["b2"], dtype=np.float64)
    W3 = np.ascontiguousarray(w["W3"], dtype=np.float64)
    b3 = np.ascontiguousarray(w["b3"], dtype=np.float64)
    assert W1.shape[0] == n_in, f"{path}: W1 input dim {W1.shape[0]} != {n_in}"
    assert W3.shape[0] == N_CARDS, f"{path}: W3 output dim {W3.shape[0]} != 52"
    return (W1, b1, W2, b2, W3, b3), meta


def _profiler_probs_py(W1, b1, W2, b2, W3, b3, features, legal_mask):
    """P(card | view) over the 52 cards: masked softmax, 0.0 off the mask.

    Single Python source shared by the `@njit` build and the NO_JIT fallback.
    """
    n_in = W1.shape[0]
    h1_size = W1.shape[1]
    h2_size = W2.shape[0]

    h1 = np.zeros(h1_size, dtype=np.float64)
    for i in range(h1_size):
        h1[i] = b1[i]
    for j in range(n_in):
        x = features[j]
        if x != 0.0:
            for i in range(h1_size):
                h1[i] += W1[j, i] * x
    for i in range(h1_size):
        if h1[i] < 0.0:
            h1[i] = 0.0

    h2 = np.zeros(h2_size, dtype=np.float64)
    for i in range(h2_size):
        s = b2[i]
        for j in range(h1_size):
            s += W2[i, j] * h1[j]
        if s < 0.0:
            s = 0.0
        h2[i] = s

    out = np.zeros(52, dtype=np.float64)
    idx = np.zeros(52, dtype=np.int64)
    n_legal = 0
    for c in range(52):
        if (legal_mask >> c) & 1:
            idx[n_legal] = c
            n_legal += 1
    if n_legal == 0:
        return out

    logits = np.zeros(n_legal, dtype=np.float64)
    best = -1.0e308
    for k in range(n_legal):
        c = idx[k]
        s = b3[c]
        for j in range(h2_size):
            s += W3[c, j] * h2[j]
        logits[k] = s
        if s > best:
            best = s
    total = 0.0
    for k in range(n_legal):
        e = np.exp(logits[k] - best)
        logits[k] = e
        total += e
    for k in range(n_legal):
        out[idx[k]] = logits[k] / total
    return out


_profiler_probs_njit = njit(cache=True)(_profiler_probs_py) if HAVE_NUMBA \
    else _profiler_probs_py


def profiler_probs(W1, b1, W2, b2, W3, b3, features, legal_mask):
    """Masked choice distribution for ONE position -> float64[52].

    `features` is float64[n_in] (333 PROFILER features, plus PARAM_DIM
    personality dims for the CONDITIONED variant); `legal_mask` the position's
    52-bit legal-move mask. Dispatches on `kernel.jit_enabled()` between the
    compiled kernel and the identical pure-Python source.
    """
    fn = _profiler_probs_njit if jit_enabled() else _profiler_probs_py
    return fn(W1, b1, W2, b2, W3, b3,
              np.asarray(features, dtype=np.float64), np.int64(legal_mask))


def _profiler_probs_batch_py(W1, b1, W2, b2, W3, b3, features, masks, out):
    for i in range(features.shape[0]):
        out[i, :] = _profiler_probs_py(W1, b1, W2, b2, W3, b3, features[i],
                                       masks[i])


def _profiler_probs_batch_njit_src(W1, b1, W2, b2, W3, b3, features, masks,
                                   out):
    for i in range(features.shape[0]):
        out[i, :] = _profiler_probs_njit(W1, b1, W2, b2, W3, b3, features[i],
                                         masks[i])


_profiler_probs_batch_njit = njit(cache=True)(_profiler_probs_batch_njit_src) \
    if HAVE_NUMBA else _profiler_probs_batch_njit_src


def profiler_probs_batch(W1, b1, W2, b2, W3, b3, features, masks, out):
    """N positions: `features` float64[N,n_in], `masks` int64[N], `out`
    preallocated float64[N,52], filled in place. Returns `out`."""
    fn = (_profiler_probs_batch_njit if jit_enabled()
          else _profiler_probs_batch_py)
    fn(W1, b1, W2, b2, W3, b3, np.asarray(features, dtype=np.float64),
       np.asarray(masks, dtype=np.int64), out)
    return out
