// =============================================================================
// cpp/src/module.cpp — the pybind11 glue: this file IS `openhearts._hearts_cpp`
// =============================================================================
//
// HOW AN EXTENSION MODULE WORKS. `import openhearts._hearts_cpp` makes Python
// dlopen() the shared library `_hearts_cpp.cpython-312-darwin.so`, look up one
// specially named C function (PyInit__hearts_cpp), and call it. That function
// builds a module object and registers our functions/classes on it. The
// PYBIND11_MODULE macro below writes PyInit__hearts_cpp for us; the body is
// where we say what the module contains.
//
// THE RULE OF THIS FILE: the engine core (include/openhearts/*.hpp) knows
// nothing about Python. This file is the ONLY place that includes pybind11.
// Keeping the boundary in one file is what lets the same core be unit-tested
// from a plain C++ executable (tests/test_bits.cpp) and, later, be compiled
// by nvcc for CUDA without dragging the interpreter along.
//
// WHERE COPIES HAPPEN. Every call across this boundary converts arguments:
// a Python int becomes a C++ uint64_t (cheap, one register), but a Python
// list would become a std::vector (a full copy, O(n), every call). This is
// the "why is my C++ slower than numba?!" lesson from CPP_PORT_PLAN.md §4,
// and the reason the playout kernel will later take numpy arrays by
// reference (py::array_t) rather than lists. For popcount there is nothing
// to copy, which is exactly why it is the first example.
// =============================================================================

#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>   // std::invalid_argument

#include <openhearts/bits.hpp>

// A namespace alias: `py::` is the universal convention in pybind11 code.
namespace py = pybind11;

// An ANONYMOUS namespace = "private to this .cpp file" (C++'s `static` for
// functions, but the modern spelling). Nothing here is visible to the linker
// from other translation units, so it can never collide with anything.
namespace {

// The boundary validates; the core assumes. Python callers can hand us any
// int at all, so we check the 52-bit contract HERE, once, and throw. pybind11
// translates a thrown std::invalid_argument into a Python ValueError
// automatically (std::out_of_range -> IndexError, etc.). Negative ints and
// ints >= 2**64 never even reach us: pybind11 refuses to convert them to
// uint64_t and raises TypeError on the Python side.
std::uint64_t require_mask52(std::uint64_t mask) {
    if (mask > openhearts::kFullDeck) {
        throw std::invalid_argument(
            "mask has bits above card 51 set: not a 52-bit hand");
    }
    return mask;
}

}  // namespace

// The first argument MUST match the target name in CMakeLists.txt
// (pybind11_add_module(_hearts_cpp ...)); `m` is the module being built.
PYBIND11_MODULE(_hearts_cpp, m) {
    m.doc() = "open-hearts C++ backend (Track C). Selected at run time by "
              "OPENHEARTS_BACKEND=cpp; never imported otherwise.";

    // Bump this whenever the exported surface changes incompatibly, so the
    // Python side can detect a stale build (.so older than the sources).
    m.attr("BACKEND_VERSION") = 1;

    // m.def(name, callable, arg names..., docstring).
    // The callable is a LAMBDA: `[](std::uint64_t mask) { ... }` is an
    // anonymous function, like Python's `lambda mask: ...` but with typed
    // parameters. The `[]` is the capture list (what outer variables it can
    // see) — empty here. pybind11 reads the parameter types off the lambda
    // to generate the int -> uint64_t conversion.
    // We wrap instead of passing &openhearts::popcount52 directly so the
    // range check runs on every Python call and never on C++-internal calls.
    m.def(
        "popcount52",
        [](std::uint64_t mask) {
            return openhearts::popcount52(require_mask52(mask));
        },
        py::arg("mask"),
        "Number of set bits in a 52-bit hand mask. Reference: int.bit_count().");

    m.def(
        "popcount52_loop",
        [](std::uint64_t mask) {
            return openhearts::popcount52_loop(require_mask52(mask));
        },
        py::arg("mask"),
        "Kernighan-loop popcount (the kernel.py::_popcount algorithm).");
}
