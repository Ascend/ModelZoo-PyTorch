cmake_minimum_required(VERSION 3.5.2)
project(deepfm)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)


set(TARGET_MAIN deepfm)

set(ACL_LIB_PATH $ENV{ASCEND_HOME}/ascend-toolkit/latest/acllib)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories($ENV{MX_SDK_HOME}/include)
include_directories($ENV{MX_SDK_HOME}/opensource/include)
include_directories($ENV{MX_SDK_HOME}/opensource/include/opencv4)
include_directories($ENV{MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories($ENV{MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories($ENV{MX_SDK_HOME}/opensource/lib/glib-2.0/include)

link_directories($ENV{MX_SDK_HOME}/lib)
link_directories($ENV{MX_SDK_HOME}/opensource/lib/)


add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")
add_compile_options("-Dgoogle=mindxsdk_private")

add_definitions(-DENABLE_DVPP_INTERFACE)

include_directories(${ACL_LIB_PATH}/include)
link_directories(${ACL_LIB_PATH}/lib64/)



add_executable(${TARGET_MAIN} src/main.cpp src/DeepFM.cpp)
target_link_libraries(${TARGET_MAIN} ${TARGET_LIBRARY} glog  cpprest mxbase libascendcl.so)
install(TARGETS ${TARGET_MAIN} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/)
