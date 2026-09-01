// =============================================================================
// cpp/tests/test_bits.cpp — C++ unit tests for bits.hpp (a plain executable)
// =============================================================================
// This is the OTHER kind of target CMake builds: an executable with a main().
// It links the same header-only core the Python module links, so what passes
// here is exactly the code Python calls. The pytest gate then checks the
// Python-visible answers against the pure-Python reference; this file checks
// C++-internal invariants that Python never sees (loop == hardware popcount).
#include <cstdint>
#include <random>    // std::mt19937_64 — the Mersenne Twister, same family numpy uses

#include <openhearts/bits.hpp>

#include "check.hpp"

// `using` inside a function (or a .cpp) is fine; it is only in headers that
// it leaks. Pulling two names is clearer than `using namespace openhearts;`.
using openhearts::kFullDeck;
using openhearts::popcount52;
using openhearts::popcount52_loop;

int main() {
    // Edge cases first: empty, full, one bit at each end.
    CHECK_EQ(popcount52(0), 0);
    CHECK_EQ(popcount52(kFullDeck), 52);
    CHECK_EQ(popcount52(std::uint64_t{1}), 1);           // 2♣
    CHECK_EQ(popcount52(std::uint64_t{1} << 51), 1);     // A♥
    CHECK_EQ(popcount52_loop(kFullDeck), 52);

    // Random 52-bit masks: hardware popcount must equal the Kernighan loop.
    // A FIXED seed makes the test deterministic — the same discipline the
    // playout port will need when we replicate numpy's stream exactly.
    std::mt19937_64 rng(2026);
    for (int i = 0; i < 100000; ++i) {
        const std::uint64_t mask = rng() & kFullDeck;
        CHECK_EQ(popcount52(mask), popcount52_loop(mask));
    }

    return oh_test::finish();
}
