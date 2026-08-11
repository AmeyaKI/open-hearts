"""Torch-free numba inference kernel for the learned value net (Task 4).

Loads `models/value_*.npz` weights (via `npz_io.load_npz`, no duplicated
parsing logic) and runs the forward pass with plain loops, no `np.dot`/
`np.matmul` inside the `@njit` function -- same house convention as
`engine/kernel.py` / `engine/features.py`: a single Python-source function is
either compiled with `@njit` or used directly as the `OPENHEARTS_NO_JIT=1`
fallback, dispatched via `kernel.jit_enabled()`, so the two paths cannot drift
apart by construction.

This is NOT a bitwise port of anything (unlike `kernel.py`'s playout, which
pins itself against `HeuristicPlayer`) -- it is new numerics, pinned instead
against `npz_io.forward_numpy` (the torch-free float64 reference) to <=1e-6
relative agreement. See `tests/test_value_infer.py`.

Hidden sizes are NEVER hardcoded: loop bounds are derived from the array
shapes passed in (`W1.shape`, `W2.shape`, `W3.shape`), so the kernel works
for any MLP-relu-relu-linear architecture sharing this weight layout, not
just the shipped 333-128-64-4.
"""
import numpy as np

from ..engine.features import FEATURES_V, NF
from ..engine.kernel import HAVE_NUMBA, jit_enabled, njit
from .npz_io import load_npz


def load_weights(path):
    """Load an `.npz` weight file into a numba-friendly float64 tuple.

    Returns `(W1, b1, W2, b2, W3, b3)`, each `np.ascontiguousarray(float64)`.
    `npz_io.load_npz` already asserts FEATURES_V/NF and internal shape
    consistency (`layer_sizes[0] == NF`, per-layer `W`/`b` shapes agreeing
    with `layer_sizes`); this function additionally asserts the two edges of
    the whole-network contract this kernel relies on: the first layer's input
    width is exactly `NF` and the last layer's output width is exactly 4
    (four rotated seats).
    """
    w, meta = load_npz(path)
    sizes = meta["layer_sizes"]
    assert meta["features_v"] == FEATURES_V, (
        f"{path}: features_v={meta['features_v']} but this build is "
        f"{FEATURES_V}")
    assert sizes[0] == NF, f"{path}: input dim {sizes[0]} != NF={NF}"
    assert sizes[-1] == 4, f"{path}: output dim {sizes[-1]} != 4"
    # W1 is stored (h1, NF) but returned TRANSPOSED, (NF, h1): the forward
    # pass exploits feature sparsity (~87 of 333 nonzero) by walking input
    # columns and skipping zeros, which wants each input's weight row
    # contiguous. This tuple's layout is a private contract between
    # load_weights and value_forward/_batch -- anyone rebuilding the npz dict
    # for forward_numpy must transpose back (see tests/test_value_infer.py).
    W1 = np.ascontiguousarray(w["W1"].T, dtype=np.float64)
    b1 = np.ascontiguousarray(w["b1"], dtype=np.float64)
    W2 = np.ascontiguousarray(w["W2"], dtype=np.float64)
    b2 = np.ascontiguousarray(w["b2"], dtype=np.float64)
    W3 = np.ascontiguousarray(w["W3"], dtype=np.float64)
    b3 = np.ascontiguousarray(w["b3"], dtype=np.float64)
    assert W1.shape[0] == NF, f"{path}: W1 input dim {W1.shape[0]} != NF"
    assert W3.shape[0] == 4, f"{path}: W3 output dim {W3.shape[0]} != 4"
    return W1, b1, W2, b2, W3, b3


def _value_forward_py(W1, b1, W2, b2, W3, b3, features):
    """Plain-loop MLP forward pass: relu(relu(x@W1'+b1)@W2'+b2)@W3'+b3.

    No `np.dot`/`np.matmul` so this compiles as-is with `@njit` and is also
    the `OPENHEARTS_NO_JIT=1` fallback (identical source, per house
    convention). Loop bounds come from the passed-in array shapes -- hidden
    sizes 128/64 are never hardcoded.

    First layer is SPARSE: `W1` arrives transposed (NF, h1) from
    `load_weights`, and input columns with feature value exactly 0.0 are
    skipped (~87 of 333 features are nonzero in practice -- measured 4.5x on
    the layer that dominates cost). Skipping exact-zero terms changes only
    summation ORDER, so agreement with `forward_numpy` stays within the
    pinned <=1e-6 relative gate.
    """
    n_in = W1.shape[0]
    h1_size = W1.shape[1]
    h2_size = W2.shape[0]
    out_size = W3.shape[0]

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

    out = np.zeros(out_size, dtype=np.float64)
    for i in range(out_size):
        s = b3[i]
        for j in range(h2_size):
            s += W3[i, j] * h2[j]
        out[i] = s
    return out


_value_forward_njit = njit(cache=True)(_value_forward_py) if HAVE_NUMBA \
    else _value_forward_py


def value_forward(W1, b1, W2, b2, W3, b3, features):
    """Forward pass for one position -> float64[4] (remaining points per
    rotated seat). Dispatches on `kernel.jit_enabled()` between the compiled
    kernel and the identical pure-Python source, same pattern as
    `engine/features.py::featurize`."""
    fn = _value_forward_njit if jit_enabled() else _value_forward_py
    return fn(W1, b1, W2, b2, W3, b3, np.asarray(features, dtype=np.float64))


def _value_forward_batch_py(W1, b1, W2, b2, W3, b3, features, out):
    n = features.shape[0]
    for i in range(n):
        out[i, :] = _value_forward_py(W1, b1, W2, b2, W3, b3, features[i])


def _value_forward_batch_njit_src(W1, b1, W2, b2, W3, b3, features, out):
    n = features.shape[0]
    for i in range(n):
        out[i, :] = _value_forward_njit(W1, b1, W2, b2, W3, b3, features[i])


_value_forward_batch_njit = njit(cache=True)(_value_forward_batch_njit_src) \
    if HAVE_NUMBA else _value_forward_batch_njit_src


def value_forward_batch(W1, b1, W2, b2, W3, b3, features, out):
    """Forward pass for N positions: `features` float64[N,NF], `out`
    preallocated float64[N,4], filled in place. Returns `out`."""
    fn = _value_forward_batch_njit if jit_enabled() else _value_forward_batch_py
    fn(W1, b1, W2, b2, W3, b3, np.asarray(features, dtype=np.float64), out)
    return out
