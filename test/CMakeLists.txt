set(PROJECT_GTEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest/include)

# Add dependency googletest CMake - gtest is exported via 'cxx_library'.
# Override googletest options.
set(BUILD_GMOCK OFF CACHE BOOL "Builds the googlemock subproject" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Enable installation of googletest. (Projects embedding googletest may want to turn this OFF.)" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest)

include(GoogleTest)

# Hide various options from UI-based property editors
mark_as_advanced(
  BUILD_GTEST INSTALL_GTEST BUILD_SHARED_LIBS
  gtest_build_samples gtest_build_tests
  gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
  #BUILD_GMOCK
  #gmock_build_tests
)

set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)

list(APPEND
  unit_test_names
  TestMath
  TestSimdMask
  TestSimdStorage
)

foreach(TEST_NAME ${unit_test_names})
  add_executable(
    ${TEST_NAME}
    "${TEST_NAME}.cpp"
    main.cpp
  )

  target_link_libraries(${TEST_NAME} PUBLIC gtest ${SIMD_MATH_LIB})
  # add_dependencies(unit_tests ${unit_test_name})

  target_include_directories(${TEST_NAME} PUBLIC ${PROJECT_GTEST_INCLUDE_DIR})

  gtest_add_tests(
      TARGET                 ${TEST_NAME}
      WORKING_DIRECTORY      ${CMAKE_CURRENT_BINARY_DIR}
      TEST_PREFIX            simd_math:
  )
endforeach()
