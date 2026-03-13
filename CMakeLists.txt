cmake_minimum_required(VERSION 3.16)
project(soft-cuda)

file(GLOB_RECURSE SOURCE_FILE src/*.cpp src/*.hpp src/*.h)

set(CMAKE_CXX_STANDARD 23)

add_executable(soft ${SOURCE_FILE})

target_include_directories(soft
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_cpu/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
