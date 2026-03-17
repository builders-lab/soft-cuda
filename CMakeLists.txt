cmake_minimum_required(VERSION 3.16)
project(soft-cuda)

file(GLOB_RECURSE LIB_SOURCE src/*.cpp src/*.hpp src/*.h)

set(CMAKE_CXX_STANDARD 23)

add_library(soft_lib STATIC ${LIB_SOURCE})

target_include_directories(soft_lib
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_cpu/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_compile_options(soft_lib PRIVATE
  $<$<CONFIG:Debug>:-O0 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror -pedantic-errors -ggdb -fsanitize=address>
  $<$<CONFIG:Release>:-O2 -DNDEBUG>
)

target_compile_options(soft_lib PUBLIC
  $<$<CONFIG:Debug>:-fsanitize=address>
)
target_link_options(soft_lib PUBLIC
  $<$<CONFIG:Debug>:-fsanitize=address>
)
add_executable(soft main.cpp)

target_link_libraries(soft PRIVATE soft_lib)

enable_testing()
add_subdirectory(tests)
