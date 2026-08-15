"""Profiler definition + training loop (Phase 5 Task 3). TORCH ONLY.

The one module in `openhearts.opponent` that imports torch, and a
TRAINING-TIME module: nothing on the play path may import it (mirroring
`value/model.py`, Phase 4 Task 0). `opponent/__init__.py` deliberately does
not import it, so `import openhearts.opponent` works in a torch-free install.
Weights leave here as a flat float64 `.npz` (`export_npz`) and are read back
by the torch-free `npz_io.load_npz` / `infer.py`'s numba kernel.

Model v1 (per PHASE5_PLAN.md Task 3): MLP n_in -> 256 -> 128 -> 52, ReLU,
cross-entropy over LEGAL moves only, Adam, early stopping on val loss.

    GENERIC      n_in = NF                (333)  -- the deployment model
    CONDITIONED  n_in = NF + PARAM_DIM    (353)  -- the ceiling / headroom
                                                    measurement only

WIDTH RATIONALE (the plan says "start 256->128; you may adjust with
rationale"): 256->128 is kept. The task is harder than Phase 4's value net
(52-way classification vs 4 regression outputs) so the first hidden layer is
doubled from 128, but the inference budget is <=20us/call and the MAC count of
333->256->128->52 is ~3.1x Phase 4's measured 4.0us/call configuration, i.e.
~12-13us projected -- inside budget, with 512-wide clearly outside it. No
width change without a measurement.

MASKED CROSS-ENTROPY (`masked_ce`). Illegal logits are set to -inf BEFORE the
log-softmax, so illegal cards contribute exactly zero probability and no
gradient. This is the training-time counterpart of `infer.py`'s legal-only
softmax; the two agree because softmax-over-a-subset and
softmax-with-the-rest-at--inf are the same function. `-inf` is safe here (and
avoided in the kernel) because every training row has >=2 legal cards by
construction -- the generator only emits multi-legal decision events -- and
that invariant is ASSERTED per batch.

Determinism: `torch.manual_seed` fixes init; batch order is fixed by the
caller's shard/shuffle seeds. Documented residual nondeterminism (carried
verbatim from Phase 4): float reductions on MPS and multithreaded CPU BLAS
are not guaranteed bit-reproducible across runs or devices, so
"byte-identical model file" is a goal, not a guarantee, with
statistically-identical the documented fallback. Report, never hide.
"""
import numpy as np
import torch
import torch.nn as nn

from ..engine.features import NF
from .npz_io import N_CARDS, PROFILER_V, forward_numpy, load_npz, save_npz  # noqa: F401,E501
from .params import PARAM_DIM

HIDDEN1 = 256
HIDDEN2 = 128
NEG_INF = float("-inf")


def input_dim(conditioned):
    return NF + PARAM_DIM if conditioned else NF


class ProfilerMLP(nn.Module):
    """n_in -> h1 -> h2 -> 52 logits (masking happens in the loss / kernel)."""

    def __init__(self, conditioned=False, hidden1=HIDDEN1, hidden2=HIDDEN2):
        super().__init__()
        n_in = input_dim(conditioned)
        self.conditioned = bool(conditioned)
        self.n_param_in = PARAM_DIM if conditioned else 0
        self.fc1 = nn.Linear(n_in, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, N_CARDS)
        self.layer_sizes = [n_in, hidden1, hidden2, N_CARDS]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def make_model(seed=0, conditioned=False, hidden1=HIDDEN1, hidden2=HIDDEN2):
    """Deterministic init: same (seed, shape) -> same initial weights."""
    torch.manual_seed(seed)
    return ProfilerMLP(conditioned, hidden1, hidden2).float()


def weights_dict(model):
    sd = model.state_dict()
    return {
        "W1": sd["fc1.weight"].detach().cpu().numpy(),
        "b1": sd["fc1.bias"].detach().cpu().numpy(),
        "W2": sd["fc2.weight"].detach().cpu().numpy(),
        "b2": sd["fc2.bias"].detach().cpu().numpy(),
        "W3": sd["fc3.weight"].detach().cpu().numpy(),
        "b3": sd["fc3.bias"].detach().cpu().numpy(),
    }


def export_npz(model, path, extra=None):
    """Write trained weights as float64 `.npz` (+ FEATURES_V/PROFILER_V)."""
    meta = dict(extra or {})
    meta.setdefault("profiler_v", PROFILER_V)
    meta.setdefault("variant", "conditioned" if model.conditioned
                    else "generic")
    return save_npz(path, weights_dict(model), model.layer_sizes,
                    model.n_param_in, extra=meta)


# ------------------------------------------------------------------- masking
def mask_bits(masks_int64):
    """int64 bitmasks [N] -> bool torch tensor [N, 52] (numpy input)."""
    m = np.asarray(masks_int64, dtype=np.int64)
    bits = ((m[:, None] >> np.arange(N_CARDS, dtype=np.int64)[None, :]) & 1)
    return torch.from_numpy(bits.astype(np.bool_))


def masked_log_probs(logits, legal_bool):
    """log-softmax restricted to legal cards; illegal -> -inf."""
    return torch.log_softmax(logits.masked_fill(~legal_bool, NEG_INF), dim=1)


def masked_ce(logits, legal_bool, target):
    """Mean negative log-likelihood of the chosen card, legal cards only."""
    lp = masked_log_probs(logits, legal_bool)
    return -lp.gather(1, target.view(-1, 1)).mean()


# ------------------------------------------------------------------ devices
def available_devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def benchmark_device(device, n_steps=30, batch_size=1024, seed=0,
                     conditioned=False):
    """Rows/second of full training steps (fwd+bwd+opt) on one device.

    Same synthetic batch and same init on every device, so the only thing
    measured is the device. Includes a warmup (the first MPS step pays kernel
    compilation). Mirrors `value/model.py::benchmark_device`, with the
    profiler's masked-CE loss instead of MSE so the number reflects the loss
    actually trained.
    """
    import time
    dev = torch.device(device)
    model = make_model(seed, conditioned).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(batch_size, input_dim(conditioned), generator=g).to(dev)
    # a plausible legal set: ~8 of 52 cards legal, always at least 2.
    legal = (torch.rand(batch_size, N_CARDS, generator=g) < 0.15)
    legal[:, 0] = True
    legal[:, 1] = True
    y = torch.zeros(batch_size, dtype=torch.long)
    legal, y = legal.to(dev), y.to(dev)

    def steps(n):
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            masked_ce(model(X), legal, y).backward()
            opt.step()
        if device == "mps":
            torch.mps.synchronize()

    steps(5)  # warmup
    t0 = time.perf_counter()
    steps(n_steps)
    dt = time.perf_counter() - t0
    return {"device": device, "steps": n_steps, "batch_size": batch_size,
            "seconds": dt, "rows_per_s": n_steps * batch_size / dt}


def pick_device(n_steps=30, batch_size=1024, seed=0, conditioned=False):
    """Measure every available device; return (best_device, [results])."""
    results = [benchmark_device(d, n_steps, batch_size, seed, conditioned)
               for d in available_devices()]
    best = max(results, key=lambda r: r["rows_per_s"])["device"]
    return best, results


# ------------------------------------------------------------------ training
def evaluate(model, X, masks, y, device, batch_size=32768):
    """(mean NLL, top-1 accuracy) of the chosen card over legal moves."""
    model.eval()
    dev = torch.device(device)
    nll_sum, correct, n = 0.0, 0, X.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(
                X[i:i + batch_size], dtype=np.float32)).to(dev)
            lb = mask_bits(masks[i:i + batch_size]).to(dev)
            yb = torch.from_numpy(np.ascontiguousarray(
                y[i:i + batch_size], dtype=np.int64)).to(dev)
            lp = masked_log_probs(model(xb), lb)
            nll_sum += float(-lp.gather(1, yb.view(-1, 1)).sum())
            correct += int((lp.argmax(dim=1) == yb).sum())
    model.train()
    return nll_sum / max(n, 1), correct / max(n, 1)


def predict_log_probs(model, X, masks, device, batch_size=32768):
    """float64 masked log-probs on numpy X [N, n_in] -> [N, 52]."""
    model.eval()
    dev = torch.device(device)
    out = np.empty((X.shape[0], N_CARDS), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(
                X[i:i + batch_size], dtype=np.float32)).to(dev)
            lb = mask_bits(masks[i:i + batch_size]).to(dev)
            out[i:i + batch_size] = masked_log_probs(
                model(xb), lb).cpu().numpy()
    model.train()
    return out.astype(np.float64)


def train(model, epoch_batches, val_X, val_masks, val_y, *, device="cpu",
          epochs=20, lr=1e-3, patience=3, seed=0, log=print):
    """Train with early stopping on val NLL.

    `epoch_batches(epoch)` yields (X float32 [B, n_in], masks int64 [B],
    y int64 [B]) numpy batches for that epoch -- the caller owns shard
    streaming and shuffling (and its seeds), so this stays memory-agnostic.

    Returns a history dict; the model is left holding the BEST val weights.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"epoch": [], "train_nll": [], "val_nll": [], "val_top1": [],
            "n_rows": []}
    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    for ep in range(epochs):
        tot, n = 0.0, 0
        for Xb, Mb, Yb in epoch_batches(ep):
            xb = torch.from_numpy(Xb).to(dev)
            lb = mask_bits(Mb).to(dev)
            yb = torch.from_numpy(Yb).to(dev)
            # generator invariant: multi-legal rows only, chosen card legal.
            assert bool(lb.sum(dim=1).min() >= 2), \
                "a training row has fewer than 2 legal cards"
            assert bool(lb.gather(1, yb.view(-1, 1)).all()), \
                "a chosen card is outside its legal mask"
            opt.zero_grad(set_to_none=True)
            loss = masked_ce(model(xb), lb, yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * xb.shape[0]
            n += xb.shape[0]
        train_nll = tot / max(n, 1)
        val_nll, val_top1 = evaluate(model, val_X, val_masks, val_y, device)
        hist["epoch"].append(ep)
        hist["train_nll"].append(train_nll)
        hist["val_nll"].append(val_nll)
        hist["val_top1"].append(val_top1)
        hist["n_rows"].append(n)
        log(f"epoch {ep}: train_nll={train_nll:.4f} val_nll={val_nll:.4f} "
            f"val_top1={val_top1:.4f} rows={n}")
        if val_nll < best_val - 1e-6:
            best_val, best_epoch, bad = val_nll, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop at epoch {ep} (no val improvement for "
                    f"{patience} epochs; best epoch {best_epoch})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch
    hist["best_val_nll"] = best_val
    return hist
