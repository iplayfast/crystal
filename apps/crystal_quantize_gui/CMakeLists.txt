cmake_minimum_required(VERSION 3.16)
project(crystal_quantize_gui LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

add_executable(crystal_quantize_gui
    main.cpp
)

target_link_libraries(crystal_quantize_gui
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        crystal
)

target_include_directories(crystal_quantize_gui
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
