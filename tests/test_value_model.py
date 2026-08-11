"""Phase 4 Task 3: value-net export/load round-trip and import hygiene.

Torch is a TRAINING-ONLY dependency, so every torch-touching test here is
guarded by `pytest.importorskip("torch")`; the suite stays green in a
torch-free install. The first test asserts exactly that hygiene: importing
`openhearts.value` must not drag torch in.
"""
import numpy as np
import pytest

from openhearts.engine import features
from openhearts.value import forward_numpy, load_npz


def test_value_package_import_is_torch_free():
    import subprocess
    import sys
    code = (
        "import sys, openhearts.value;"
        "assert 'torch' not in sys.modules, 'openhearts.value imported torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_export_load_forward_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    from openhearts.value import model as vmodel

    m = vmodel.make_model(7)
    # A random-init net outputs ~0.03, where torch's own float32 representation
    # error (~6e-8 relative) dominates any relative comparison. The gate is
    # about agreement on the REMAINING-POINTS scale (0..26), so put the outputs
    # there BEFORE exporting: nudge the output bias to a typical target.
    with torch.no_grad():
        m.fc3.bias.fill_(6.5)

    path = str(tmp_path / "w.npz")
    vmodel.export_npz(m, path)
    w, meta = load_npz(path)
    assert meta["features_v"] == features.FEATURES_V
    assert meta["layer_sizes"] == [features.NF, 128, 64, 4]

    rng = np.random.default_rng(0)
    X = (rng.random((1500, features.NF)) < 0.2).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(X)).numpy().astype(np.float64)
    got = forward_numpy(w, X)
    # The plan's stated gate is RELATIVE agreement <= 1e-6 (PHASE4_PLAN.md
    # Task 4). The absolute check is a loose sanity companion, not the gate:
    # at a 6.5-point output scale, torch's float32 accumulation alone costs
    # ~3e-6 absolute (~5e-7 relative), so no float32 reference can meet 1e-6
    # ABSOLUTE here -- that number would be measuring float32, not the export.
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= 1e-6
    assert np.abs(got - ref).max() < 1e-4


def test_load_npz_rejects_wrong_features_v(tmp_path):
    path = str(tmp_path / "bad.npz")
    nf = features.NF
    np.savez(path, W1=np.zeros((128, nf)), b1=np.zeros(128),
             W2=np.zeros((64, 128)), b2=np.zeros(64),
             W3=np.zeros((4, 64)), b3=np.zeros(4),
             features_v=np.int64(features.FEATURES_V + 1),
             nf=np.int64(nf),
             layer_sizes=np.array([nf, 128, 64, 4], dtype=np.int64),
             arch=np.array("mlp-relu"))
    with pytest.raises(ValueError):
        load_npz(path)
