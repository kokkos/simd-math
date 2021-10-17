add_library(${SIMD_MATH_LIB} INTERFACE)
target_include_directories(${SIMD_MATH_LIB} INTERFACE .)
target_link_libraries(${SIMD_MATH_LIB} INTERFACE Kokkos::kokkos)
target_compile_features(${SIMD_MATH_LIB} INTERFACE cxx_std_14)