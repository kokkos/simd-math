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

using simd_t = simd::simd<double, simd::simd_abi::native>;
using mask_t = simd_t::mask_type;

TEST(simd_mask, simd_mask_basic_api) {
  const mask_t a(true);
  const mask_t b(false);

  EXPECT_TRUE(a.get());
  EXPECT_FALSE(b.get());

  const auto a_or_b = a || b;
  EXPECT_TRUE(a_or_b.get());

  const auto a_and_b = a && b;
  EXPECT_FALSE(a_and_b.get());

  const auto not_a = !a;
  const auto not_b = !b;

  EXPECT_FALSE(not_a.get());
  EXPECT_TRUE(not_b.get());

  EXPECT_TRUE(simd::all_of(a));
  EXPECT_FALSE(simd::all_of(b));

  EXPECT_TRUE(simd::any_of(a));
  EXPECT_FALSE(simd::any_of(b));
}

bool test_is_data_bigger(Kokkos::View<simd_t *> data) {
  using simd_bool = simd::simd<bool, simd::simd_abi::native>;
  Kokkos::View<simd_bool *> results("Test results", data.extent(0));

  simd_t min = 0.0;
  Kokkos::parallel_for(
      "is_data_bigger", data.extent(0), KOKKOS_LAMBDA(const int i) {
        const auto a_i   = data(i);
        mask_t is_data_bigger = min < a_i;
        results(i)       = all_of(is_data_bigger);
      });

  Kokkos::View<bool *> results_scalar((bool *)(results.data()),
                                      results.extent(0) * simd_bool::size());

  int result = 0;
  Kokkos::parallel_reduce(
      "Reduce results", results_scalar.extent(0),
      KOKKOS_LAMBDA(const int i, int &r) { r += results_scalar(i); }, result);

  return result == results_scalar.extent(0);
}

TEST(simd_mask, simd_mask_less) {
  constexpr int data_size      = 10 * simd_t::size();
  constexpr int simd_data_size = data_size / simd_t::size();

  Kokkos::View<double *> a("Test data", data_size);

  auto h_a = Kokkos::create_mirror_view(a);
  for (int i = 0; i < data_size; ++i) {
    h_a(i) = i + 1;
  }

  Kokkos::deep_copy(a, h_a);

  Kokkos::View<simd_t *> a_v((simd_t *)(a.data()), simd_data_size);
  EXPECT_TRUE(test_is_data_bigger(a_v));
}

}  // namespace Test