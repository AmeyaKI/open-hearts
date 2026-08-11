"""Torch-free reader for exported value-net weights (`models/value_*.npz`).

Deliberately separate from `model.py` (which imports torch): Task 4's numba
inference kernel and `tests/test_value_infer.py` are specified to stay
torch-free, but they must still read the weight file and its FEATURES_V /
shape metadata. Putting the reader here means importing openhearts never
touches torch. `model.py` re-exports `load_npz` for convenience.

File format (float64 throughout -- the kernel is float64):
    W1 [128, NF]  b1 [128]
    W2 [64, 128]  b2 [64]
    W3 [4, 64]    b3 [4]
    features_v  int (asserted against engine.features.FEATURES_V on load)
    nf          int
    layer_sizes int64[4]  = [NF, 128, 64, 4]
    arch        str, e.g. "mlp-relu-333-128-64-4"
Forward pass: y = W3 @ relu(W2 @ relu(W1 @ x + b1) + b2) + b3.
"""
import numpy as np

from ..engine.features import FEATURES_V, NF

ARCH = "mlp-relu"


def save_npz(path, weights, layer_sizes, extra=None):
    """Write a weight dict {W1,b1,W2,b2,W3,b3} (any float dtype) as float64."""
    payload = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
    payload["features_v"] = np.int64(FEATURES_V)
    payload["nf"] = np.int64(NF)
    payload["layer_sizes"] = np.asarray(layer_sizes, dtype=np.int64)
    payload["arch"] = np.array(
        ARCH + "-" + "-".join(str(int(s)) for s in layer_sizes))
    if extra:
        payload["meta"] = np.array(dict(extra), dtype=object)
    np.savez(path, **payload)
    return path


def load_npz(path):
    """Load weights; assert FEATURES_V/NF/shape consistency. -> (dict, meta)."""
    d = np.load(path, allow_pickle=True)
    fv, nf = int(d["features_v"]), int(d["nf"])
    if fv != FEATURES_V:
        raise ValueError(
            f"{path}: FEATURES_V={fv} but this build is {FEATURES_V}")
    if nf != NF:
        raise ValueError(f"{path}: NF={nf} but this build is {NF}")
    sizes = [int(s) for s in d["layer_sizes"]]
    if sizes[0] != NF:
        raise ValueError(f"{path}: layer_sizes[0]={sizes[0]} != NF={NF}")
    w = {}
    for i, k in enumerate(("1", "2", "3")):
        W = np.ascontiguousarray(d["W" + k], dtype=np.float64)
        b = np.ascontiguousarray(d["b" + k], dtype=np.float64)
        if W.shape != (sizes[i + 1], sizes[i]) or b.shape != (sizes[i + 1],):
            raise ValueError(f"{path}: layer {k} shape {W.shape}/{b.shape} "
                             f"disagrees with layer_sizes {sizes}")
        w["W" + k], w["b" + k] = W, b
    meta = d["meta"].item() if "meta" in d.files else {}
    return w, dict(meta, layer_sizes=sizes, features_v=fv, nf=nf,
                   arch=str(d["arch"]))


def forward_numpy(w, X):
    """Reference float64 forward pass on X [N, NF] -> [N, 4]."""
    X = np.asarray(X, dtype=np.float64)
    h1 = np.maximum(X @ w["W1"].T + w["b1"], 0.0)
    h2 = np.maximum(h1 @ w["W2"].T + w["b2"], 0.0)
    return h2 @ w["W3"].T + w["b3"]
