CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
project(SAMPLE)
add_compile_options(-fPIE -fstack-protector-all -fPIC -Wall -Wfatal-errors)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(TORCH_AIE_PATH "path/to/torch_aie" CACHE STRING "")
set(TORCH_ROOT "path/to/libtorch/install/" CACHE STRING "")

add_executable(sample yolov5.cpp)
set_property(TARGET sample PROPERTY CXX_STANDARD 14)
include_directories(sample ${TORCH_AIE_PATH}/include)
target_link_directories(sample PUBLIC ${TORCH_AIE_PATH}/lib)
target_link_libraries(sample "${TORCH_LIBRARIES}" torch_aie)