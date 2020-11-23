cmake_minimum_required(VERSION 3.17)
project(opencl_demo)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)

add_executable(opencl_demo main.cpp)

target_include_directories (opencl_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(opencl_demo OpenCL::OpenCL)