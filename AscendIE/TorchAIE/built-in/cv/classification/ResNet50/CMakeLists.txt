project(SAMPLE)
CMAKE_MINIMUM_REQUIRED(VERSION 3.13)

add_compile_options(-fPIE -fstack-protector-all -fPIC -Wall -Wfatal-errors -O2)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(TORCH_AIE "/usr/lib/python3.9.0/lib/python3.9/site-packages/torch_aie" CACHE STRING "")
set(TORCH "/usr/lib/python3.9.0/lib/python3.9/site-packages/torch" CACHE STRING "")

add_executable(sample ${CMAKE_CURRENT_LIST_DIR}/resnet50.cpp)
set_property(TARGET sample PROPERTY CXX_STANDARD 14)

target_include_directories(sample PUBLIC
    ${TORCH_AIE}/include
    ${TORCH}/include
    ${TORCH}/include/torch/csrc/api/include
)
target_link_directories(sample PUBLIC
    ${TORCH_AIE}/lib
    ${TORCH}/lib)

target_link_libraries(sample PUBLIC
    c10
    torch
    torch_cpu
    torch_aie
)