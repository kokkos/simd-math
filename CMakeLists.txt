
cmake_minimum_required(VERSION 3.17)

project(simd-math)

find_package(Kokkos REQUIRED)

# Set the local module path so custom cmake scripts can be located automatically
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules/")
set(SIMD_MATH_LIB simd_math CACHE INTERNAL "" FORCE )

add_subdirectory(src)

include(CTest)
add_subdirectory(test)
