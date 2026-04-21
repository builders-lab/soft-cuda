cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 23)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_compile_definitions(tensor_tests PRIVATE HAS_CUDA=1)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tensor_tests
  soft-cuda/tensor/test_tensor.cpp
  test_mul.cpp
  test_ops.cpp
  test_graph.cpp
  test_backprop.cpp
  test_backend.cpp
)

target_include_directories(tensor_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/core/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/core/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend_cpu/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src"
)

target_link_libraries(tensor_tests PRIVATE
  gtest_main
  soft_lib
)

enable_testing()
include(GoogleTest)
gtest_discover_tests(tensor_tests)
