cmake_minimum_required(VERSION 3.12)
project(cppsolver)

set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 REQUIRED)

add_library(cppsolver MODULE py_solver_bindings.cpp cpp_solver_stub.cpp)

target_link_libraries(cppsolver PRIVATE pybind11::module)

# On Windows, set the output name to cppsolver.pyd
if (WIN32)
    set_target_properties(cppsolver PROPERTIES SUFFIX ".pyd")
endif() 