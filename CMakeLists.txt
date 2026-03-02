cmake_minimum_required(VERSION 3.16)

project(soft-cuda)

file(GLOB_RECURSE SOURCE_FILE src/*.cpp src/*.hpp src/*.h)

set(CMAKE_CXX_STANDARD 23)

add_executable(soft ${SOURCE_FILE})

target_include_directories(soft PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
