# =============================================================================
# cpp/CMakeLists.txt — the build recipe for the open-hearts C++ backend
# =============================================================================
#
# WHAT CMAKE IS (in one breath): CMake does NOT compile anything. It reads this
# file and *generates* a build system (Ninja files here) that knows how to turn
# our .cpp files into (a) a Python extension module and (b) test executables.
# Two separate steps, always:
#     cmake -S cpp -B cpp/build      # "configure": read this file, emit Makefiles
#     cmake --build cpp/build        # "build": run the Makefiles -> call clang++
# Python analogy: pyproject.toml describes the package; pip does the work.
# CMake describes the *targets*; the generated Ninja file does the work.
# (Generators are interchangeable: -G Ninja, -G "Unix Makefiles", -G Xcode all
# consume this same file. We use Ninja — see scripts/cpp_build.sh for why.)
#
# The unit of thought in CMake is the TARGET: a library, an executable, or a
# module. You attach properties to targets (include dirs, flags, dependencies)
# and CMake propagates them along the dependency edges. Almost every CMake
# mistake is "I set a global variable when I should have set a target property".
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# LANGUAGES CXX: only C++ — no C compiler probe, faster configure, fewer surprises.
project(openhearts_cpp LANGUAGES CXX)

# ---- Language standard -------------------------------------------------------
# C++20 gives us <bit> (std::popcount, std::countr_zero), std::span, defaulted
# operator==, and designated initializers. STANDARD_REQUIRED means "error if the
# compiler can't do it" instead of silently downgrading (a classic trap: your
# code fails to compile with cryptic errors because CMake quietly fell back to
# C++14). EXTENSIONS OFF asks for -std=c++20, not -std=gnu++20: portable code.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimized build. Debug (-O0 -g) is what you want inside a
# debugger; Release (-O3 -DNDEBUG) is what any timing must use. NOTE: -DNDEBUG
# deletes `assert()` — never put side effects inside an assert.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ---- Python + pybind11 -------------------------------------------------------
# Development.Module = "I am building an extension module", which on macOS
# means we do NOT link against libpython (the interpreter that imports us
# already has the symbols). Linking libpython is the classic macOS mistake that
# produces a module that crashes on import inside a venv.
# The build script passes -DPython_EXECUTABLE=<repo>/.venv/bin/python so the
# module is built for *that* interpreter (ABI tag cpython-312-darwin).
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11 was pip-installed into the venv; it ships its own CMake config
# files. Ask the interpreter where they are instead of hard-coding the path.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11 ${pybind11_VERSION} from ${pybind11_DIR}")

# ---- The core library --------------------------------------------------------
# For now the engine core is HEADER-ONLY (everything lives in include/ as
# constexpr/inline functions), so the "library" produces no .a/.so — it is an
# INTERFACE target: a bundle of usage requirements (include path, flags) that
# anything linking to it inherits. When we add .cpp files with real function
# bodies (the playout loop), this becomes `add_library(openhearts_core STATIC
# src/kernel.cpp ...)` and nothing downstream changes. That is the point of
# targets: consumers say "link openhearts_core" and never learn the details.
add_library(openhearts_core INTERFACE)
target_include_directories(openhearts_core INTERFACE
  "${PROJECT_SOURCE_DIR}/include")
target_compile_features(openhearts_core INTERFACE cxx_std_20)

# Warnings are the cheapest bug-finder you will ever own. -Wall -Wextra is the
# baseline every serious C++ shop uses; -Wpedantic rejects compiler-specific
# extensions. We do NOT add -Werror yet (it turns every new warning into a
# build break, which is great in CI and annoying while learning).
#
# -ffp-contract=off: forbid fusing a*b+c into one FMA instruction. Irrelevant
# for integer code, but the playout loop (C2 target 3) carries float weights,
# and FMA changes rounding — the bitwise gate would fail for no visible reason.
# Setting it on day one means we never have to hunt for it later.
target_compile_options(openhearts_core INTERFACE
  -Wall -Wextra -Wpedantic -ffp-contract=off)

# ---- The Python extension module --------------------------------------------
# pybind11_add_module = add_library(... MODULE ...) plus all the fiddly bits
# (correct suffix `.cpython-312-darwin.so`, -fvisibility=hidden, Python include
# dirs, link flags). The module name here MUST equal the name inside
# PYBIND11_MODULE(...) in module.cpp, or the import fails with
# "dynamic module does not define module export function".
pybind11_add_module(_hearts_cpp src/module.cpp)
target_link_libraries(_hearts_cpp PRIVATE openhearts_core)

# Drop the finished .so straight into the Python package so
# `import openhearts._hearts_cpp` works with no install step. (*.so is
# gitignored: it is a build product, like __pycache__.)
set_target_properties(_hearts_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../src/openhearts")

# ---- C++ unit tests ----------------------------------------------------------
# A second kind of target: a plain EXECUTABLE with its own main(). It links the
# same core library, so the code under test is byte-for-byte what the Python
# module sees. `ctest --test-dir cpp/build` runs every add_test() below.
enable_testing()

add_executable(test_bits tests/test_bits.cpp)
target_link_libraries(test_bits PRIVATE openhearts_core)
add_test(NAME test_bits COMMAND test_bits)
