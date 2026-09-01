"""Bitwise gate for the C++ backend (Track C).

Every function ported to C++ is listed in GATES with its pure-Python
reference and is checked for *identical* output over the corpus: the hands
and masks from `test_kernel_equivalence.generate_states` (random-length game
prefixes, mid-trick states included), the random 52-bit masks used by the
legal-moves port test, and the edge cases. Bitwise-identical is REQUIRED
before any timing number is quoted (CPP_PORT_PLAN.md §7.5, amendment 3).

Behaviour when the extension is not built:
  * OPENHEARTS_BACKEND unset -> the gate SKIPS (a fresh checkout without cmake
    must stay green);
  * OPENHEARTS_BACKEND=cpp   -> the gate FAILS loudly (you asked for C++ and
    are not getting it).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from openhearts.engine import backend, cards
from test_kernel_equivalence import generate_states

FULL = (1 << 52) - 1


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def ext():
    try:
        return backend.cpp_module()
    except ImportError:
        if os.environ.get("OPENHEARTS_BACKEND", "").lower() == "cpp":
            raise
        pytest.skip(backend.BUILD_HINT)


@pytest.fixture(scope="module")
def mask_corpus():
    """52-bit masks: real hands/legal masks from game states + random + edges."""
    masks = {0, FULL, 1, 1 << 51, 1 << cards.QUEEN_SPADES,
             cards.HEARTS_MASK, cards.POINTS_MASK, *cards.SUIT_MASK}
    for state in generate_states(600, seed=7):
        masks.update(state.hands)
        seat = state.to_play
        masks.add(state.view_for(seat).legal_moves)
        played = 0
        for _s, c in state.history:
            played |= 1 << c
        masks.add(played)
        masks.add(FULL & ~played)
    rng = np.random.default_rng(3)
    masks.update(int(x) for x in rng.integers(1, 1 << 52, size=3000))
    return sorted(masks)


# ---------------------------------------------------------------- the gate
# (name, python_reference, cpp_call). Add one row per ported function; the
# corpus and the assertion below are shared.
GATES = [
    ("popcount52", lambda m: m.bit_count(), lambda e, m: e.popcount52(m)),
    ("popcount52_loop", lambda m: m.bit_count(),
     lambda e, m: e.popcount52_loop(m)),
]


@pytest.mark.parametrize("name,ref,call", GATES, ids=[g[0] for g in GATES])
def test_bitwise_identical_over_corpus(ext, mask_corpus, name, ref, call):
    assert len(mask_corpus) > 3000
    for m in mask_corpus:
        expected = ref(m)
        got = call(ext, m)
        assert type(got) is int, (name, m, type(got))
        assert got == expected, f"{name} diverged on mask {m:#x}"


def test_boundary_rejects_bad_masks(ext):
    with pytest.raises(ValueError):          # std::invalid_argument -> ValueError
        ext.popcount52(1 << 52)
    with pytest.raises(TypeError):           # negative can't become uint64_t
        ext.popcount52(-1)
    with pytest.raises(TypeError):           # 2**64 overflows uint64_t
        ext.popcount52(1 << 64)


# ---------------------------------------------------------------- the switch
def _fresh(code, **env_overrides):
    env = dict(os.environ)
    env.pop("OPENHEARTS_BACKEND", None)
    env.update(env_overrides)
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True)
    return out


PROBE = ("import sys; from openhearts.engine import backend;"
         "n = backend.popcount((1 << 52) - 1);"
         "print(backend.backend(), backend.cpp_enabled(), n,"
         "      'openhearts._hearts_cpp' in sys.modules)")


def test_unset_env_var_means_default_backend_and_no_extension_import():
    out = _fresh(PROBE)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["default", "False", "52", "False"]


def test_backend_cpp_serves_from_the_extension(ext):
    out = _fresh(PROBE, OPENHEARTS_BACKEND="cpp")
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["cpp", "True", "52", "True"]


def test_backend_typo_is_loud():
    out = _fresh(PROBE, OPENHEARTS_BACKEND="c++")
    assert out.returncode != 0
    assert "ValueError" in out.stderr


def test_reset_rereads_environment(monkeypatch):
    backend.reset_backend()
    monkeypatch.delenv("OPENHEARTS_BACKEND", raising=False)
    assert backend.backend() == "default"
    monkeypatch.setenv("OPENHEARTS_BACKEND", "cpp")
    assert backend.backend() == "default"      # still cached
    backend.reset_backend()
    assert backend.backend() == "cpp"
    monkeypatch.delenv("OPENHEARTS_BACKEND", raising=False)
    backend.reset_backend()
    assert backend.backend() == "default"
