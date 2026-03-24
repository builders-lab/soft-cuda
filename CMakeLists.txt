cmake_minimum_required(VERSION 3.16)
project(soft-cuda LANGUAGES CXX CUDA)

file(GLOB_RECURSE LIB_SOURCE src/*.cu src/*.cpp src/*.hpp src/*.h)

set(CMAKE_CUDA_ARCHITECTURES native)

add_library(soft_lib SHARED ${LIB_SOURCE})

target_include_directories(soft_lib
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_cpu/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_compile_options(soft_lib PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror -pedantic-errors -ggdb -fsanitize=address>
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -G>
  $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O2 -DNDEBUG>
  $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O2 -DNDEBUG>
)

set_target_properties(soft_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(soft_lib PUBLIC
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=address>
)
target_link_options(soft_lib PUBLIC
  $<$<CONFIG:Debug>:-fsanitize=address>
)
add_executable(soft main.cpp)

target_link_libraries(soft PRIVATE soft_lib)

enable_testing()
add_subdirectory(tests)
