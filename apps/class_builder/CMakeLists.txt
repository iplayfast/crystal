cmake_minimum_required(VERSION 3.16)
project(class_builder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

add_executable(class_builder
    main.cpp
)

target_link_libraries(class_builder
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        crystal
)

target_include_directories(class_builder
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
