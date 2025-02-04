project(onnx_runtime_sample)

add_executable(${PROJECT_NAME} main.cc)

target_link_libraries(${PROJECT_NAME} onnxruntime)
