"""Backend selector for the C++ port (Track C, CPP_PORT_PLAN.md).

This is the third-backend socket. It mirrors `kernel.jit_enabled()` exactly:

* ``OPENHEARTS_BACKEND`` unset  -> "default": the existing numba/pure-Python
  behaviour, byte for byte. The compiled extension is NEVER imported.
* ``OPENHEARTS_BACKEND=cpp``    -> the C++ extension serves every function
  that has been ported. If the extension is missing this is a LOUD
  ImportError, not a silent fallback: a benchmark that quietly measured the
  Python path would be worse than one that crashed.
* anything else                 -> ValueError (typos must not select Python).

The pure-Python implementations stay the definition of correct; the gate test
(`tests/test_cpp_gate.py`) compares the extension against them directly,
independent of this switch, so the gate can run while the default backend is
active.
"""
import importlib
import os

_BACKEND = None   # cached like kernel._ENABLED; reset_backend() re-reads
_EXT = None       # the imported extension module, once loaded

_VALID = ("default", "cpp")

BUILD_HINT = ("C++ extension not built. Run `scripts/cpp_build.sh` "
              "(needs cmake + the venv's pybind11).")


def backend() -> str:
    """The selected backend name, looked up once and cached."""
    global _BACKEND
    if _BACKEND is None:
        raw = os.environ.get("OPENHEARTS_BACKEND", "").strip().lower()
        chosen = "default" if raw == "" else raw
        if chosen not in _VALID:
            raise ValueError(
                f"OPENHEARTS_BACKEND={raw!r} is not one of {_VALID}")
        _BACKEND = chosen
    return _BACKEND


def cpp_enabled() -> bool:
    """True when ported functions should be served by the C++ extension."""
    return backend() == "cpp"


def reset_backend() -> None:
    """Forget the cached choice (tests toggle the env var and call this)."""
    global _BACKEND
    _BACKEND = None


def cpp_module():
    """Import and return `openhearts._hearts_cpp`, building nothing.

    Deliberately independent of `cpp_enabled()`: the gate test needs the
    extension while the *engine* keeps running the default backend.
    """
    global _EXT
    if _EXT is None:
        try:
            _EXT = importlib.import_module("openhearts._hearts_cpp")
        except ImportError as e:
            raise ImportError(BUILD_HINT) from e
    return _EXT


# --------------------------------------------------------------------------
# dispatchers: one per ported function. Each has the pure-Python reference
# inline so the default path never touches the extension.
# --------------------------------------------------------------------------
def popcount(mask: int) -> int:
    """Number of cards in a 52-bit hand mask."""
    if cpp_enabled():
        return cpp_module().popcount52(mask)
    return int(mask).bit_count()
