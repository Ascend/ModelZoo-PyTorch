cmake_minimum_required(VERSION 3.10.2)
project(hourglass)
set(TARGET hourglass)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)
set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*cpp )

include_directories(${MX_SDK_HOME}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/opencv4)
include_directories(${MX_SDK_HOME}/lib)

link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
        /usr/local/Ascend/driver/lib64/
)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_executable(${TARGET} ${SRC_FILES})
target_link_libraries(${TARGET}
        glog
        mxbase
        cpprest
        opencv_world
        )
install(TARGETS ${TARGET} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/)