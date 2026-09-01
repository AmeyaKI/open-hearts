// =============================================================================
// cpp/tests/check.hpp — a 30-line test "framework" so we depend on nothing
// =============================================================================
// Real projects use GoogleTest or Catch2/doctest. Those are worth learning
// later; for now the interesting part is *how* a test framework works, and
// that fits in a page. Usage in a test file:
//
//     #include "check.hpp"
//     int main() {
//         CHECK(popcount52(0) == 0);
//         CHECK_EQ(popcount52(kFullDeck), 52);
//         return oh_test::finish();
//     }
//
// The executable's exit code (0 = pass) is what ctest looks at.
#pragma once

#include <cstdio>   // std::printf, std::fprintf — C's I/O, fine for tests

namespace oh_test {

// `inline` on a VARIABLE (C++17) is what makes a header-defined global legal:
// every .cpp that includes this header shares ONE counter instead of each
// getting its own (which would violate the One Definition Rule).
inline int checks = 0;
inline int failures = 0;

inline int finish() {
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}

}  // namespace oh_test

// MACROS. `#define` is textual substitution before compilation — no types, no
// scope. They are the wrong tool for almost everything EXCEPT this: a test
// macro needs __FILE__ and __LINE__ of the *call site* and the *source text*
// of the expression (`#expr` turns the argument into a string literal). No
// function can do that.
//
// The `do { ... } while (0)` wrapper is the standard trick that makes a
// multi-statement macro behave like ONE statement, so `if (x) CHECK(y);
// else ...` parses the way it looks. Every macro pro writes it this way.
#define CHECK(expr)                                                          \
    do {                                                                     \
        ++oh_test::checks;                                                   \
        if (!(expr)) {                                                       \
            ++oh_test::failures;                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,     \
                         #expr);                                             \
        }                                                                    \
    } while (0)

// Prints both values on failure. `(long long)` casts so one %lld format works
// for every integer width we use; a real framework would use templates.
#define CHECK_EQ(a, b)                                                       \
    do {                                                                     \
        ++oh_test::checks;                                                   \
        auto oh_a_ = (a);                                                    \
        auto oh_b_ = (b);                                                    \
        if (!(oh_a_ == oh_b_)) {                                             \
            ++oh_test::failures;                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s == %s  (got %lld vs %lld)\n", \
                         __FILE__, __LINE__, #a, #b,                         \
                         (long long)oh_a_, (long long)oh_b_);                \
        }                                                                    \
    } while (0)
