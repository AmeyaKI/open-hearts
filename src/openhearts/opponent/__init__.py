"""Learned opponent model -- the Phase-5 "profiler": P(card | view).

Import-safe WITHOUT torch, exactly like `openhearts.value`: this package pulls
in only the torch-free `npz_io` reader and the `params` vectorization.
`model.py` (torch, TRAINING ONLY) is NOT imported here -- import it
explicitly, and only from training code.
"""
from .npz_io import (ARCH, N_CARDS, PROFILER_V, forward_numpy, load_npz,
                     masked_probs_numpy, save_npz)
from .params import PARAM_DIM, param_table, param_vector

__all__ = ["ARCH", "N_CARDS", "PROFILER_V", "PARAM_DIM", "forward_numpy",
           "load_npz", "masked_probs_numpy", "save_npz", "param_table",
           "param_vector"]
