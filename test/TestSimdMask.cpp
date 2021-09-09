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

constexpr int extentArray[14] = {1,      2,      8,       9,      100,
                                 101,    1000,   1001,    10000,  10001,
                                 100000, 100001, 1000000, 1000001};

template <typename T>
bool is_true(const T &mask) {
  return simd::all_of(mask);
}

template <typename T>
bool is_false(const T &mask) {
  return !simd::all_of(mask) && !simd::any_of(mask);
}

template <typename ScalarType>
void do_simd_mask_basic_api_test() {
  using simd_t = simd::simd<ScalarType, simd::simd_abi::native>;
  using mask_t = typename simd_t::mask_type;

  const mask_t a(true);
  const mask_t b(false);

  EXPECT_TRUE(is_true(a));
  EXPECT_TRUE(is_false(b));

  const auto a_or_b = a || b;
  EXPECT_TRUE(is_true(a_or_b));

  const auto a_and_b = a && b;
  EXPECT_TRUE(is_false(a_and_b));

  const auto not_a = !a;
  const auto not_b = !b;

  EXPECT_TRUE(is_false(not_a));
  EXPECT_TRUE(is_true(not_b));

  EXPECT_TRUE(simd::all_of(a));
  EXPECT_FALSE(simd::all_of(b));

  EXPECT_TRUE(simd::any_of(a));
  EXPECT_FALSE(simd::any_of(b));
}

TEST(simd_mask, simd_mask_basic_api) {
  do_simd_mask_basic_api_test<float>();
  do_simd_mask_basic_api_test<double>();
}

template <class ScalarType>
bool test_is_mask_smaller(
    Kokkos::View<simd::simd<ScalarType, simd::simd_abi::native> *> data) {
  using simd_t = simd::simd<ScalarType, simd::simd_abi::native>;
  using mask_t = typename simd_t::mask_type;

  Kokkos::View<simd_t *> results("Test results", data.extent(0));

  simd_t min = 0.0;
  Kokkos::parallel_for(
      "is_mask_smaller", data.extent(0), KOKKOS_LAMBDA(const int i) {
        const auto a_i        = data(i);
        mask_t is_data_bigger = min < a_i;
        results(i)            = all_of(is_data_bigger);
      });

  Kokkos::View<ScalarType *> results_scalar(
      (ScalarType *)(results.data()), results.extent(0) * simd_t ::size());

  int result = 0;
  Kokkos::parallel_reduce(
      "Reduce results", results_scalar.extent(0),
      KOKKOS_LAMBDA(const int i, int &r) { r += results_scalar(i); }, result);

  return result == results_scalar.extent(0);
}

template <typename ScalarType>
void do_test_simd_mask_smaller(const int viewExtent) {
  using simd_t = simd::simd<ScalarType, simd::simd_abi::native>;

  const int data_size      = viewExtent * simd_t ::size();
  const int simd_data_size = viewExtent;

  Kokkos::View<ScalarType *> a("Test data", data_size);

  auto h_a = Kokkos::create_mirror_view(a);
  for (int i = 0; i < data_size; ++i) {
    h_a(i) = i + 1;
  }

  Kokkos::deep_copy(a, h_a);

  Kokkos::View<simd_t *> a_v((simd_t *)(a.data()), simd_data_size);
  EXPECT_TRUE(test_is_mask_smaller<ScalarType>(a_v));
}

TEST(simd_mask, simd_mask_smaller) {
  for (auto extent : extentArray) {
    do_test_simd_mask_smaller<float>(extent);
    do_test_simd_mask_smaller<double>(extent);
  }
}

}  // namespace Test
