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
#ifdef __CUDACC__

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <simd.hpp>
#include "TestHelpers.hpp"

namespace Test {

template <class StorageType, typename ScalarType>
static Kokkos::View<StorageType *> create_data_with_unique_value(
    const std::string &name, int size, ScalarType value) {
  Kokkos::View<StorageType *> data(name, size);
  Kokkos::View<ScalarType *> data_scalar((ScalarType *)(data.data()),
                                         size * StorageType::size());
  Kokkos::deep_copy(data_scalar, value);

  return data;
}

template <class StorageType>
Kokkos::View<StorageType *> create_data_positive(std::string name, int size) {
  using ScalarType = typename StorageType::value_type;
  Kokkos::View<StorageType *> data(name, size);
  Kokkos::View<ScalarType *> data_scalar(
      reinterpret_cast<ScalarType *>(data.data()), StorageType::size() * size);

  Kokkos::parallel_for(
      data_scalar.extent(0), KOKKOS_LAMBDA(const int i) {
        data_scalar(i) = static_cast<ScalarType>(i);
      });

  return data;
}

template <class StorageType>
Kokkos::View<StorageType *> create_data_negative(std::string name, int size) {
  using ScalarType = typename StorageType::value_type;
  Kokkos::View<StorageType *> data(name, size);
  Kokkos::View<ScalarType *> data_scalar(
      reinterpret_cast<ScalarType *>(data.data()), StorageType::size() * size);

  Kokkos::parallel_for(
      data_scalar.extent(0), KOKKOS_LAMBDA(const int i) {
        data_scalar(i) = -static_cast<ScalarType>(i);
      });

  return data;
}

template <class StorageType>
Kokkos::View<StorageType *> create_simd_data_sqrt(std::string name, int size) {
  using ScalarType = typename StorageType::value_type;
  Kokkos::View<StorageType *> data(name, size);
  Kokkos::View<ScalarType *> data_scalar(
      reinterpret_cast<ScalarType *>(data.data()), StorageType::size() * size);

  Kokkos::parallel_for(
      data_scalar.extent(0), KOKKOS_LAMBDA(const int i) {
        auto value     = i >= sqrtMaxIndex ? i - sqrtMaxIndex : i;
        data_scalar(i) = static_cast<ScalarType>(value * value);
      });

  return data;
}

template <class StorageType>
Kokkos::View<StorageType *> create_simd_data_cbrt(std::string name, int size) {
  using ScalarType = typename StorageType::value_type;

  Kokkos::View<StorageType *> data(name, size);
  Kokkos::View<ScalarType *> data_scalar(
      reinterpret_cast<ScalarType *>(data.data()), StorageType::size() * size);

  Kokkos::parallel_for(
      data_scalar.extent(0), KOKKOS_LAMBDA(const int i) {
        auto value     = i >= cbrtMaxIndex ? i - cbrtMaxIndex : i;
        data_scalar(i) = static_cast<ScalarType>(value * value * value);
      });

  return data;
}

template <typename ScalarType>
using simd_warp_t = simd::simd<ScalarType, simd::simd_abi::cuda_warp<32>>;

template <typename ScalarType>
using simd_storage_t = typename simd_warp_t<ScalarType>::storage_type;

template <class StorageType>
void test_warp_add(Kokkos::View<StorageType *> data,
                   Kokkos::View<StorageType *> results) {
  Kokkos::parallel_for(
      "Combine", Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i     = team.league_rank();
        results(i) = simd_warp_t<typename StorageType::value_type>(data(i)) +
                     simd_warp_t<typename StorageType::value_type>(0.0);
      });
  Kokkos::fence();
}

template <typename ScalarType, typename StorageType>
void test_view_result(const std::string &test_name,
                      Kokkos::View<StorageType *> results,
                      Kokkos::View<ScalarType *, Kokkos::HostSpace> expected) {
  Kokkos::View<ScalarType *> results_scalar(
      (ScalarType *)results.data(), results.extent(0) * StorageType::size());
  auto results_h = Kokkos::create_mirror_view(results_scalar);
  Kokkos::deep_copy(results_h, results_scalar);

  for (size_t i = 0; i < results_h.extent(0); ++i) {
    compare(test_name, i, results_h(i), expected(i));
  }
}

template <class StorageType>
void test_abs(Kokkos::View<StorageType *> data) {
  Kokkos::parallel_for(
      "simd::abs", Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i = team.league_rank();
        data(i) =
            simd::abs(simd_warp_t<typename StorageType::value_type>(data(i)));
      });

  Kokkos::fence();
}

template <class StorageType>
void test_sqrt(Kokkos::View<StorageType *> data) {
  Kokkos::parallel_for(
      "simd::sqrt",
      Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i = team.league_rank();
        data(i) =
            simd::sqrt(simd_warp_t<typename StorageType::value_type>(data(i)));
      });

  Kokkos::fence();
}

template <typename ScalarType>
void do_test_abs(int viewExtent) {
  using storage_t        = simd_storage_t<ScalarType>;
  const int simdSize     = storage_t::size();
  const int expectedSize = viewExtent * storage_t::size();

  auto data = create_data_with_unique_value<storage_t>("Test View", viewExtent,
                                                       ScalarType{-4});

  test_abs(data);
  Kokkos::View<ScalarType *, Kokkos::HostSpace> expectedData("expectedData",
                                                             expectedSize);
  Kokkos::deep_copy(expectedData, static_cast<ScalarType>(4.0));
  test_view_result("test_abs_1", data, expectedData);

  data = create_data_positive<storage_t>("Test View 2", viewExtent);

  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      expectedData(i * simdSize + j) =
          static_cast<ScalarType>(i * simdSize + j);
    }
  }

  test_abs(data);
  test_view_result("test_abs_2", data, expectedData);

  data = create_data_negative<storage_t>("Test View 3", viewExtent);

  test_abs(data);
  test_view_result("test_abs_3", data, expectedData);
}

template <typename ScalarType>
void do_test_sqrt(int viewExtent) {
  using storage_t        = simd_storage_t<ScalarType>;
  const int simdSize     = storage_t::size();
  const int expectedSize = viewExtent * storage_t::size();

  auto data = create_data_with_unique_value<storage_t>("Test View", viewExtent,
                                                       ScalarType{16});

  test_sqrt(data);

  Kokkos::View<ScalarType *, Kokkos::HostSpace> expectedData("expectedData",
                                                             expectedSize);
  Kokkos::deep_copy(expectedData, static_cast<ScalarType>(4.0));

  test_view_result("test_sqrt_1", data, expectedData);

  data = create_simd_data_sqrt<storage_t>("Test View 2", viewExtent);

  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      auto index = i * simdSize + j;
      auto value = index >= sqrtMaxIndex ? index - sqrtMaxIndex : index;
      expectedData(index) = static_cast<ScalarType>(value);
    }
  }

  test_sqrt(data);
  test_view_result("test_sqrt_2", data, expectedData);
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct TestCudaWarp : testing::Test {};

TYPED_TEST_SUITE_P(TestCudaWarp);

TYPED_TEST_P(TestCudaWarp, test_abs) {
  constexpr auto extentSize = std::tuple_element<0, TypeParam>::type::value;
  using ScalarType          = typename std::tuple_element<1, TypeParam>::type;

  do_test_abs<ScalarType>(extentSize);
}

TYPED_TEST_P(TestCudaWarp, test_sqrt) {
  constexpr auto extentSize = std::tuple_element<0, TypeParam>::type::value;
  using ScalarType          = typename std::tuple_element<1, TypeParam>::type;

  do_test_sqrt<ScalarType>(extentSize);
}

// TEST(simd_cuda_warp, test_cbrt) {
//   int N_in = 64;
//   double a = 2.0;
//   int N    = N_in / simd_warp_t<double>::size();

//   auto data = create_simd_data_cbrt<simd_storage_t>("whatever", N);

//   Kokkos::View<simd_storage_t *> results("R", N);

//   test_warp_add(data, results);
//   verify(results, N_in);
// }

// TEST(simd_cuda_warp, test_exp) {
//   int N_in = 64;
//   double a = 2.0;
//   int N    = N_in / simd_warp_t<double>::size();

//   auto data = create_simd_data_cbrt<simd_storage_t>("whatever", N);

//   Kokkos::View<simd_storage_t *> results("R", N);

//   test_warp_add(data, results);
//   verify(results, N_in);
// }

using TestTypes =
    testing::Types<std::tuple<std::integral_constant<int, 1>, float>,
                   std::tuple<std::integral_constant<int, 2>, float>,
                   std::tuple<std::integral_constant<int, 8>, float>,
                   std::tuple<std::integral_constant<int, 9>, float>,
                   std::tuple<std::integral_constant<int, 100>, float>,
                   std::tuple<std::integral_constant<int, 101>, float>,
                   std::tuple<std::integral_constant<int, 1000>, float>,
                   std::tuple<std::integral_constant<int, 1001>, float>,

                   std::tuple<std::integral_constant<int, 1>, double>,
                   std::tuple<std::integral_constant<int, 2>, double>,
                   std::tuple<std::integral_constant<int, 8>, double>,
                   std::tuple<std::integral_constant<int, 9>, double>,
                   std::tuple<std::integral_constant<int, 100>, double>,
                   std::tuple<std::integral_constant<int, 101>, double>,
                   std::tuple<std::integral_constant<int, 1000>, double>,
                   std::tuple<std::integral_constant<int, 1001>, double>>;

REGISTER_TYPED_TEST_SUITE_P(TestCudaWarp, test_abs, test_sqrt);

INSTANTIATE_TYPED_TEST_SUITE_P(test_simd_cuda_warp_set, TestCudaWarp,
                               TestTypes);

}  // namespace Test

#endif