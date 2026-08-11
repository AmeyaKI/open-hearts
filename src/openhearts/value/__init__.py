"""Learned value function (Phase 4).

Import-safe WITHOUT torch: this package only pulls in the torch-free
`npz_io` reader. `model.py` (torch, training only) is NOT imported here --
import it explicitly, and only from training code.
"""
from .npz_io import ARCH, forward_numpy, load_npz, save_npz

__all__ = ["ARCH", "forward_numpy", "load_npz", "save_npz"]
