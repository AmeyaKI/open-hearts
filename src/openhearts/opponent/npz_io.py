"""Torch-free reader/writer for exported profiler weights (Phase 5 Task 3).

Deliberately separate from `model.py` (which imports torch), exactly mirroring
the `openhearts.value` package split: the numba inference kernel
(`opponent/infer.py`) and `tests/test_profiler_infer.py` stay torch-free but
must still read the weight file and its metadata.

WHY NOT REUSE `value/npz_io.py`.  That reader hardcodes the network's input
width to `NF` (`payload["nf"] = NF`, and `load_npz` raises when
`layer_sizes[0] != NF`) and its output width is the value net's 4 rotated
seats.  The profiler's CONDITIONED variant takes `NF + n_param_in` inputs and
both variants emit 52 logits, so reusing that file would mean weakening
assertions in a Phase-4 module that is not ours to touch.  This is a parallel
reader with the same shape and the same conventions, plus two extra metadata
fields.

File format (float64 throughout -- the kernel is float64):
    W1 [h1, n_in]   b1 [h1]        n_in = NF + n_param_in
    W2 [h2, h1]     b2 [h2]
    W3 [52, h2]     b3 [52]
    features_v   int  (asserted against engine.features.FEATURES_V on load)
    profiler_v   int  (asserted against PROFILER_V on load)
    nf           int  (the FEATURE half of the input, asserted == NF)
    n_param_in   int  (0 for GENERIC, PARAM_DIM for CONDITIONED)
    layer_sizes  int64[4] = [n_in, h1, h2, 52]
    arch         str, e.g. "profiler-mlp-relu-333-256-128-52"

Forward pass: logits = W3 @ relu(W2 @ relu(W1 @ x + b1) + b2) + b3, then a
softmax restricted to the legal cards (see `masked_probs_numpy`).

PROFILER_V=1 pins the whole contract: PROFILER_FEATURES_V=1 inputs (the
FEATURES_V=1 333-dim layout with OFF_HANDS blocks r=1..3 structurally zero,
per `experiments/gen_population_data.py`), optionally followed by a
PARAM_DIM-vector of normalized personality parameters in the frozen draw
order (see `params.py`), and 52 card logits masked to the legal set.
"""
import numpy as np

from ..engine.features import FEATURES_V, NF

PROFILER_V = 1
N_CARDS = 52
ARCH = "profiler-mlp-relu"


def save_npz(path, weights, layer_sizes, n_param_in, extra=None):
    """Write a weight dict {W1,b1,W2,b2,W3,b3} (any float dtype) as float64."""
    sizes = [int(s) for s in layer_sizes]
    if sizes[0] != NF + int(n_param_in):
        raise ValueError(
            f"layer_sizes[0]={sizes[0]} != NF+n_param_in="
            f"{NF + int(n_param_in)}")
    if sizes[-1] != N_CARDS:
        raise ValueError(f"layer_sizes[-1]={sizes[-1]} != {N_CARDS}")
    payload = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
    payload["features_v"] = np.int64(FEATURES_V)
    payload["profiler_v"] = np.int64(PROFILER_V)
    payload["nf"] = np.int64(NF)
    payload["n_param_in"] = np.int64(n_param_in)
    payload["layer_sizes"] = np.asarray(sizes, dtype=np.int64)
    payload["arch"] = np.array(
        ARCH + "-" + "-".join(str(s) for s in sizes))
    if extra:
        payload["meta"] = np.array(dict(extra), dtype=object)
    np.savez(path, **payload)
    return path


def load_npz(path):
    """Load weights; assert FEATURES_V/PROFILER_V/NF/shapes. -> (dict, meta)."""
    d = np.load(path, allow_pickle=True)
    fv = int(d["features_v"])
    if fv != FEATURES_V:
        raise ValueError(
            f"{path}: FEATURES_V={fv} but this build is {FEATURES_V}")
    pv = int(d["profiler_v"]) if "profiler_v" in d.files else -1
    if pv != PROFILER_V:
        raise ValueError(
            f"{path}: PROFILER_V={pv} but this build is {PROFILER_V}")
    nf = int(d["nf"])
    if nf != NF:
        raise ValueError(f"{path}: NF={nf} but this build is {NF}")
    n_param_in = int(d["n_param_in"])
    sizes = [int(s) for s in d["layer_sizes"]]
    if sizes[0] != NF + n_param_in:
        raise ValueError(f"{path}: layer_sizes[0]={sizes[0]} != NF+"
                         f"n_param_in={NF + n_param_in}")
    if sizes[-1] != N_CARDS:
        raise ValueError(
            f"{path}: layer_sizes[-1]={sizes[-1]} != {N_CARDS}")
    w = {}
    for i, k in enumerate(("1", "2", "3")):
        W = np.ascontiguousarray(d["W" + k], dtype=np.float64)
        b = np.ascontiguousarray(d["b" + k], dtype=np.float64)
        if W.shape != (sizes[i + 1], sizes[i]) or b.shape != (sizes[i + 1],):
            raise ValueError(f"{path}: layer {k} shape {W.shape}/{b.shape} "
                             f"disagrees with layer_sizes {sizes}")
        w["W" + k], w["b" + k] = W, b
    meta = d["meta"].item() if "meta" in d.files else {}
    return w, dict(meta, layer_sizes=sizes, features_v=fv, profiler_v=pv,
                   nf=nf, n_param_in=n_param_in, arch=str(d["arch"]))


def forward_numpy(w, X):
    """Reference float64 forward pass on X [N, n_in] -> LOGITS [N, 52].

    Deliberately dense and unmasked: the numba kernel skips zero inputs and
    computes only the legal output rows, so the reference must NOT share
    either trick or it would stop being an independent check.
    """
    X = np.asarray(X, dtype=np.float64)
    h1 = np.maximum(X @ w["W1"].T + w["b1"], 0.0)
    h2 = np.maximum(h1 @ w["W2"].T + w["b2"], 0.0)
    return h2 @ w["W3"].T + w["b3"]


def masked_probs_numpy(logits, legal_masks):
    """Reference masked softmax: logits [N,52] + int64 bitmasks [N] -> [N,52].

    Illegal cards get EXACTLY 0.0 (they are never exponentiated), legal cards
    a softmax over the legal subset. Rows with an empty mask stay all-zero.
    """
    logits = np.asarray(logits, dtype=np.float64)
    masks = np.asarray(legal_masks, dtype=np.int64)
    out = np.zeros_like(logits)
    bits = ((masks[:, None] >> np.arange(N_CARDS, dtype=np.int64)[None, :])
            & 1).astype(bool)
    for i in range(logits.shape[0]):
        m = bits[i]
        if not m.any():
            continue
        z = logits[i][m]
        e = np.exp(z - z.max())
        out[i][m] = e / e.sum()
    return out
