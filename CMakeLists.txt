cmake_minimum_required (VERSION 3.18)

project(xinyiok)

include_directories(thirdparty/onnxruntime-linux-x64-1.15.1/include)
link_directories(thirdparty/onnxruntime-linux-x64-1.15.1/lib)

add_subdirectory(sample)