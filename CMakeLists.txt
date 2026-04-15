cmake_minimum_required(VERSION 3.25)
project(crystal
    VERSION 0.1.0
    DESCRIPTION "Modern C++20 computational intelligence library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(CRYSTAL_BUILD_TESTS "Build tests" ON)
option(CRYSTAL_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CRYSTAL_BUILD_APPS "Build CLI applications" OFF)
option(CRYSTAL_ENABLE_SANITIZERS "Enable ASan/UBSan" OFF)

# Compiler warnings
include(cmake/CompilerWarnings.cmake)

# Find dependencies (optional - will fall back to FetchContent if not found)
find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, using FetchContent")
    include(FetchContent)
    FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
    FetchContent_MakeAvailable(json)
endif()

# llama.cpp for GGUF I/O and model loading
include(FetchContent)
FetchContent_Declare(llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG b8784
)
FetchContent_MakeAvailable(llama_cpp)

if(CRYSTAL_BUILD_TESTS)
    find_package(Catch2 3 CONFIG QUIET)
    if(NOT Catch2_FOUND)
        message(STATUS "Catch2 not found, using FetchContent")
    endif()
endif()

if(CRYSTAL_BUILD_BENCHMARKS)
    find_package(benchmark CONFIG REQUIRED)
endif()

if(CRYSTAL_BUILD_APPS)
    find_package(Qt6 COMPONENTS Widgets REQUIRED)
endif()

# Main library
add_subdirectory(src)

# Tests
if(CRYSTAL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# CLI Apps
if(CRYSTAL_BUILD_APPS)
    add_subdirectory(apps/crystal_quantize)
else()
    # Still build the app for convenience - just don't expose option
    add_subdirectory(apps/crystal_quantize)
endif()

# GUI Apps
add_subdirectory(apps/crystal_quantize_gui)
add_subdirectory(apps/fuzzy_builder)
add_subdirectory(apps/class_builder)

# Install (skip for now - llama.cpp dependency isn't exportable)
# include(GNUInstallDirs)
# install(DIRECTORY include/crystal
#     DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )
# 
# install(TARGETS crystal
#     EXPORT CrystalTargets
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )
# 
# install(EXPORT CrystalTargets
#     FILE CrystalTargets.cmake
#     NAMESPACE Crystal::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Crystal
# )
# 
# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#     cmake/CrystalConfig.cmake.in
#     "${CMAKE_CURRENT_BINARY_DIR}/CrystalConfig.cmake"
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Crystal
# )
# 
# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/CrystalConfigVersion.cmake"
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY SameMajorVersion
# )
# 
# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/CrystalConfig.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/CrystalConfigVersion.cmake"
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Crystal
# )