cmake_minimum_required(VERSION 3.25)
project(crystal_quantize_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include paths
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/_deps/json-src/include
    ${CMAKE_BINARY_DIR}/_deps/llama.cpp-src
)

# Link library
add_executable(crystal-quantize main.cpp)
target_link_libraries(crystal-quantize crystal)