"""Value-net definition + training loop (Phase 4 Task 3). TORCH ONLY.

This is the one module in `openhearts` that imports torch, and it is a
TRAINING-TIME module: nothing on the play path may import it (see
PHASE4_PLAN.md Task 0). `openhearts.value/__init__.py` deliberately does not
import it, so `import openhearts` works in a torch-free install. Weights leave
here as a flat float64 `.npz` (`export_npz`) and are read back by the
torch-free `npz_io.load_npz` / Task 4's numba kernel.

Model v1 (per plan): MLP NF -> 128 -> 64 -> 4, ReLU, MSE, Adam, early stopping
on val MSE. The 4 outputs are remaining points for the 4 ROTATED seats, so
output 0 is always the evaluated seat.

Determinism: `torch.manual_seed` fixes init; batch order is fixed by the
caller's shard/shuffle seeds. Documented residual nondeterminism: float
reductions on MPS (and multithreaded CPU BLAS) are not guaranteed
bit-reproducible across runs or devices, so "byte-identical model file" is a
goal, not a guarantee -- the plan's documented fallback is
statistically-identical. Report, never hide.
"""
import numpy as np
import torch
import torch.nn as nn

from ..engine.features import NF
from .npz_io import forward_numpy, load_npz, save_npz  # noqa: F401 re-export

HIDDEN1 = 128
HIDDEN2 = 64
N_OUT = 4
LAYER_SIZES = [NF, HIDDEN1, HIDDEN2, N_OUT]


class ValueMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NF, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, N_OUT)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def make_model(seed=0):
    """Deterministic init: same seed -> same initial weights."""
    torch.manual_seed(seed)
    return ValueMLP().float()


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
    """Write the trained weights as float64 `.npz` (+ FEATURES_V / shapes)."""
    return save_npz(path, weights_dict(model), LAYER_SIZES, extra=extra)


# ------------------------------------------------------------------ devices
def available_devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def benchmark_device(device, n_steps=30, batch_size=1024, seed=0):
    """Rows/second of full training steps (fwd+bwd+opt) on one device.

    Same synthetic batch every step and the same init on every device, so the
    only difference measured is the device. Includes a short warmup (first MPS
    step pays kernel compilation).
    """
    import time
    dev = torch.device(device)
    model = make_model(seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(batch_size, NF, generator=g).to(dev)
    Y = (torch.rand(batch_size, N_OUT, generator=g) * 26.0).to(dev)
    lossf = nn.MSELoss()

    def steps(n):
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            lossf(model(X), Y).backward()
            opt.step()
        if device == "mps":
            torch.mps.synchronize()

    steps(5)  # warmup
    t0 = time.perf_counter()
    steps(n_steps)
    dt = time.perf_counter() - t0
    return {"device": device, "steps": n_steps, "batch_size": batch_size,
            "seconds": dt, "rows_per_s": n_steps * batch_size / dt}


def pick_device(n_steps=30, batch_size=1024, seed=0):
    """Measure every available device; return (best_device, [results])."""
    results = [benchmark_device(d, n_steps, batch_size, seed)
               for d in available_devices()]
    best = max(results, key=lambda r: r["rows_per_s"])["device"]
    return best, results


# ------------------------------------------------------------------ training
def evaluate(model, X, Y, device, batch_size=65536):
    """Mean-squared error over all 4 outputs, and over output 0 alone."""
    model.eval()
    dev = torch.device(device)
    se_all = se0 = 0.0
    n = X.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(
                X[i:i + batch_size], dtype=np.float32)).to(dev)
            yb = torch.from_numpy(np.ascontiguousarray(
                Y[i:i + batch_size], dtype=np.float32)).to(dev)
            p = model(xb)
            se_all += float(((p - yb) ** 2).sum())
            se0 += float(((p[:, 0] - yb[:, 0]) ** 2).sum())
    model.train()
    return se_all / (n * N_OUT), se0 / n


def predict(model, X, device, batch_size=65536):
    """float32 forward pass on numpy X [N, NF] -> numpy [N, 4]."""
    model.eval()
    dev = torch.device(device)
    out = np.empty((X.shape[0], N_OUT), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(
                X[i:i + batch_size], dtype=np.float32)).to(dev)
            out[i:i + batch_size] = model(xb).cpu().numpy()
    model.train()
    return out


def train(model, epoch_batches, val_X, val_Y, *, device="cpu", epochs=20,
          lr=1e-3, patience=3, seed=0, log=print):
    """Train with early stopping on val MSE.

    `epoch_batches(epoch)` yields (X float32 [B, NF], Y float32 [B, 4]) numpy
    batches for that epoch -- the caller owns shard streaming and shuffling
    (and its seeds), so this function stays memory-agnostic.

    Returns a history dict; the model is left holding the BEST val weights.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    hist = {"epoch": [], "train_mse": [], "val_mse": [], "val_mse_seat0": [],
            "n_rows": []}
    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    for ep in range(epochs):
        se, n = 0.0, 0
        for Xb, Yb in epoch_batches(ep):
            xb = torch.from_numpy(Xb).to(dev)
            yb = torch.from_numpy(Yb).to(dev)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            se += float(loss.detach()) * xb.shape[0]
            n += xb.shape[0]
        train_mse = se / max(n, 1)
        val_mse, val_mse0 = evaluate(model, val_X, val_Y, device)
        hist["epoch"].append(ep)
        hist["train_mse"].append(train_mse)
        hist["val_mse"].append(val_mse)
        hist["val_mse_seat0"].append(val_mse0)
        hist["n_rows"].append(n)
        log(f"epoch {ep}: train_mse={train_mse:.4f} val_mse={val_mse:.4f} "
            f"val_mse_seat0={val_mse0:.4f} rows={n}")
        if val_mse < best_val - 1e-6:
            best_val, best_epoch, bad = val_mse, ep, 0
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
    hist["best_val_mse"] = best_val
    return hist
