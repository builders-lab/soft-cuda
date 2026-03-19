cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 23)

# ── GTest ──────────────────────────────────────────────────────────────────────
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ── Project source files (main.cpp chod do — uska apna main() hai) ────────────
file(GLOB_RECURSE TEST_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.h"
)
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*main\\.cpp$")

# ── Test executable ────────────────────────────────────────────────────────────
add_executable(tensor_tests
  soft-cuda/tensor/test_tensor.cpp
  ${TEST_SOURCES}        # project ka actual code seedha compile ho raha hai
)

target_include_directories(tensor_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/core/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend_cpu/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src"
)

target_link_libraries(tensor_tests PRIVATE
  gtest_main
)

# ── CTest registration ─────────────────────────────────────────────────────────
enable_testing()
include(GoogleTest)
gtest_discover_tests(tensor_tests)
