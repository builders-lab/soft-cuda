cmake_minimum_required(VERSION 3.16)
project(soft-cuda LANGUAGES CXX CUDA)

# Core library sources — explicitly exclude the Python bridge subdir
file(GLOB_RECURSE LIB_SOURCE src/*.cu src/*.cpp src/*.hpp src/*.h)
list(FILTER LIB_SOURCE EXCLUDE REGEX ".*/python/.*")
list(FILTER LIB_SOURCE EXCLUDE REGEX ".*/profiler\.cu$")

set(CMAKE_CUDA_ARCHITECTURES native)

add_library(soft_lib SHARED ${LIB_SOURCE})
target_link_libraries(soft_lib PRIVATE cublas)

set_target_properties(soft_lib PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)
target_include_directories(soft_lib
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_cpu/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
  PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

target_compile_options(soft_lib PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror -pedantic-errors -ggdb>
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0>
  $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O2 -DNDEBUG>
  $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O2 -DNDEBUG>
)

set_target_properties(soft_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
#
# target_compile_options(soft_lib PUBLIC
#   $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=address>
# )
target_compile_options(soft_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g --expt-relaxed-constexpr>)
#
# target_link_options(soft_lib PUBLIC
#   $<$<CONFIG:Debug>:-fsanitize=address>
# )



# Python C bridge — flat-C shared library for use with ctypes / cffi

add_library(soft_cuda_python SHARED src/python/sc_bridge.cpp)

set_target_properties(soft_cuda_python PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

target_include_directories(soft_cuda_python
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_cpu/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_executable(soft main.cpp)
target_link_libraries(soft_cuda_python PRIVATE soft_lib)
target_link_libraries(soft PRIVATE soft_cuda_python)
# Ensure all sc_* symbols are exported on Linux/macOS
if(NOT WIN32)
  target_compile_options(soft_cuda_python PRIVATE -fvisibility=default)
endif()

enable_testing()
add_subdirectory(tests)

# HARDWARE PROFILER

add_executable(soft_profiler src/init/config/profiler.cu)
set_target_properties(soft_profiler PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
target_include_directories(soft_profiler
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(soft_profiler PRIVATE soft_lib)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
  add_subdirectory(benchmarks)
endif()


