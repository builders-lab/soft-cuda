cmake_minimum_required(VERSION 3.16)

# C++ benchmarks against soft_lib
add_executable(bench_softcuda bench_softcuda.cpp)
set_target_properties(bench_softcuda PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
target_include_directories(bench_softcuda
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src/core/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src"
)
target_link_libraries(bench_softcuda PRIVATE soft_cuda_python)
target_include_directories(bench_softcuda PRIVATE
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)
target_link_directories(bench_softcuda PRIVATE
    "/opt/cuda/targets/x86_64-linux/lib"
)
target_link_libraries(bench_softcuda PRIVATE cublas)
target_link_libraries(bench_softcuda PRIVATE cublas cudart)
add_executable(bench_deep_mlp bench_deep_mlp.cpp)
set_target_properties(bench_deep_mlp PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
target_include_directories(bench_deep_mlp
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src/core/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src"
)
target_link_libraries(bench_deep_mlp PRIVATE soft_cuda_python)
