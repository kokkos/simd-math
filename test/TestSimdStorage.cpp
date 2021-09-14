/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <simd.hpp>

namespace Test {

template <typename ScalarType>
using simd_t = simd::simd<ScalarType, simd::simd_abi::native>;

template <typename ScalarType>
using storage_t = typename simd_t<ScalarType>::storage_type;

template<typename ScalarType>
void compare(const std::string &test_name, int i, ScalarType value, ScalarType expected);

template<>
void compare(const std::string &test_name, int i, double value, double expected){
    EXPECT_DOUBLE_EQ(value, expected) << "Failure during " + test_name + " with i = " + std::to_string(i);;
}

template<>
void compare(const std::string &test_name, int i, float value, float expected){
    EXPECT_FLOAT_EQ(value, expected) << "Failure during " + test_name + " with i = " + std::to_string(i);;
}

template <typename ScalarType>
void check_storage_const_content(const storage_t<ScalarType> storage, ScalarType value) {

    // using const data
    auto storage_by_simd_data = storage.data();
    for(int i = 0; i < storage.size(); ++i) {
        compare("do_simd_storage_basic_api_test", i, storage_by_simd_data[i], value);
    }

    // using operator
    for(int i = 0; i < storage.size(); ++i) {
        compare("do_simd_storage_basic_api_test", i, storage[i], value);
    }

}

template <typename ScalarType>
void check_storage_content(storage_t<ScalarType> storage, ScalarType value) {

    // using const data
    auto storage_by_simd_data = storage.data();
    for(int i = 0; i < storage.size(); ++i) {
        compare("do_simd_storage_basic_api_test", i, storage_by_simd_data[i], value);
    }

    // using operator
    for(int i = 0; i < storage.size(); ++i) {
        compare("do_simd_storage_basic_api_test", i, storage[i], value);
    }

}
template <typename ScalarType>
void do_simd_storage_basic_api_test() {

  // constructs storage using all constructors
  const storage_t<ScalarType> storage_by_simd(simd::simd<ScalarType, simd::simd_abi::native>{4.0});
  const storage_t<ScalarType> storage_by_value(6.0);
  const storage_t<ScalarType> storage_by_copy(storage_by_simd);
  storage_t<ScalarType> storage_by_affectation;
  storage_by_affectation = simd::simd<ScalarType, simd::simd_abi::native>{6.0};

  // check size aren't null
  EXPECT_GT(storage_by_simd.size(), 0);
  EXPECT_GT(storage_by_value.size(), 0);
  EXPECT_GT(storage_by_copy.size(), 0);
  EXPECT_GT(storage_by_affectation.size(), 0);

  // verify content using all methods
  check_storage_const_content<ScalarType>(storage_by_simd, 4.0);
  check_storage_const_content<ScalarType>(storage_by_value, 6.0);
  check_storage_const_content<ScalarType>(storage_by_copy, 4.0);
  check_storage_const_content<ScalarType>(storage_by_affectation, 6.0);

  check_storage_content<ScalarType>(storage_by_simd, 4.0);
  check_storage_content<ScalarType>(storage_by_value, 6.0);
  check_storage_content<ScalarType>(storage_by_copy, 4.0);
  check_storage_content<ScalarType>(storage_by_affectation, 6.0);
}

TEST(simd_storage, simd_storage_basic_api) {
  do_simd_storage_basic_api_test<float>();
  do_simd_storage_basic_api_test<double>();
}

}  // namespace Test
