// =============================================================================
// openhearts/bits.hpp — raw 52-bit mask helpers (the worked example)
// =============================================================================
//
// HEADER FILE 101. A header is text that `#include` pastes into every .cpp that
// names it. Consequences:
//   * `#pragma once` stops the paste happening twice in one .cpp (which would
//     redefine everything and fail to compile).
//   * Anything DEFINED (with a body) in a header must be `inline`, `constexpr`
//     (which implies inline), or a template. Otherwise two .cpp files that both
//     include it each get a copy of the function, and the LINKER refuses:
//     "duplicate symbol". This is the One Definition Rule, the #1 early trap.
//   * Headers should include only what they need (<bit>, <cstdint>), not the
//     kitchen sink — every include is paid for by every file that includes you.
//
// Python analogy: a header is a module's *signature*; the .cpp is its body.
// Python has one file for both because it resolves names at runtime. C++
// resolves them at compile time (declarations) and link time (definitions).
#pragma once

#include <bit>       // std::popcount, std::countr_zero — C++20, hardware ops
#include <cstdint>   // std::uint64_t — a *fixed-width* 64-bit unsigned integer

// A namespace is a Python module without the file boundary. Everything the
// engine defines lives in openhearts:: so it cannot collide with the standard
// library or pybind11. Never `using namespace std;` in a header — that leaks
// into every file that includes you.
namespace openhearts {

// ---- Constants ---------------------------------------------------------------
// `inline constexpr` (C++17): one shared compile-time constant, no storage
// duplicated per translation unit, usable in static_assert and array sizes.
//
// TRAP: `(1 << 52) - 1` is UNDEFINED BEHAVIOUR in C++. The literal `1` is a
// plain `int` (32 bits), and shifting an int past its width is UB — the
// compiler may compute anything, including 0, and it will not warn you in
// every case. Python promotes to bignum silently; C++ does not promote at all.
// The fix is to make the ONE 64-bit *before* shifting: `std::uint64_t{1}`
// (or `1ULL`). This exact mistake is the first thing to check whenever a
// bitboard port disagrees with Python.
inline constexpr int kNumCards = 52;
inline constexpr std::uint64_t kFullDeck = (std::uint64_t{1} << kNumCards) - 1;

// ---- popcount (the end-to-end example) -------------------------------------
// [[nodiscard]]: the compiler warns if a caller ignores the return value —
// calling a pure function for nothing is always a bug.
// constexpr: may run at compile time (see the static_asserts below) AND at
// run time; it also implies `inline`, so it is header-safe.
// noexcept: promises not to throw; lets the optimizer skip unwinding tables.
// Pass-by-value on purpose: a uint64_t is one register. Passing `const&` here
// would hand the callee a pointer to look through — slower, not safer.
//
// CONTRACT: `mask` must be a 52-bit hand (mask <= kFullDeck). The core does
// not re-check that (hot path); the Python boundary in module.cpp does.
[[nodiscard]] constexpr int popcount52(std::uint64_t mask) noexcept {
    // std::popcount compiles to a single CNT instruction on Apple Silicon.
    return std::popcount(mask);
}

// The same function the way kernel.py::_popcount writes it (Kernighan's
// trick: `m & (m - 1)` clears the lowest set bit). Kept as the reference so
// the C++ unit test can pin the hardware version against the textbook one.
// Note `mask` is a COPY (by value), so mutating it inside is harmless to the
// caller — in Python you would be rebinding a local name; here you are
// overwriting a local object. Same effect, different mechanism.
[[nodiscard]] constexpr int popcount52_loop(std::uint64_t mask) noexcept {
    int n = 0;
    while (mask != 0) {
        mask &= mask - 1;
        ++n;
    }
    return n;
}

// ---- Compile-time tests ------------------------------------------------------
// static_assert runs INSIDE the compiler: if any of these were false the build
// would fail before a single test executable existed. They cost nothing at
// run time. This only works because the functions are constexpr — one of the
// real payoffs of that keyword.
static_assert(popcount52(0) == 0);
static_assert(popcount52(kFullDeck) == 52);
static_assert(popcount52(std::uint64_t{1} << 36) == 1);          // Q♠ alone
static_assert(popcount52_loop(kFullDeck) == popcount52(kFullDeck));

}  // namespace openhearts
