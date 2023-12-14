cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(detectron2_ops)

find_package(Torch REQUIRED)

add_library(detectron2_ops SHARED detectron2_ops.cpp)

target_compile_features(detectron2_ops PRIVATE cxx_std_14)

target_link_libraries(detectron2_ops PUBLIC
    c10
    torch
    torch_cpu)