add_library(crystal STATIC
    fuzzy/fuzzy_set.cpp
    nn/backprop.cpp
    nn/blob_network.cpp
    nn/quantization.cpp
    io/gguf.cpp
    math/huge_int.cpp
    quantize/model_reader.cpp
    quantize/ternary_converter.cpp
    quantize/pipeline.cpp
    quantize/ensemble.cpp
    quantize/gguf_writer.cpp
    quantize/calibration.cpp
)

target_include_directories(crystal
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/_deps/json-src/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(crystal
    PUBLIC
        llama
)

target_compile_features(crystal PUBLIC cxx_std_20)
