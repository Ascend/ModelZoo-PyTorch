cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(mmdet_ops)

find_package(Torch REQUIRED)

add_library(mmdet_ops SHARED mmdet_ops.cpp)

target_compile_features(mmdet_ops PRIVATE cxx_std_14)

target_link_libraries(mmdet_ops PUBLIC
    c10
    torch
    torch_cpu)