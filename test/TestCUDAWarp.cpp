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
        ScalarType value = i % sqrtMaxIndex;
        data_scalar(i)   = static_cast<ScalarType>(value * value);
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
        ScalarType value = i % cbrtMaxIndex;
        data_scalar(i)   = static_cast<ScalarType>(value * value * value);
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

template <class StorageType>
void test_cbrt(Kokkos::View<StorageType *> data) {
  Kokkos::parallel_for(
      "simd::cbrt",
      Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i = team.league_rank();
        data(i) =
            simd::cbrt(simd_warp_t<typename StorageType::value_type>(data(i)));
      });

  Kokkos::fence();
}

template <class StorageType>
void test_exp(Kokkos::View<StorageType *> data) {
  Kokkos::parallel_for(
      "simd::exp", Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i = team.league_rank();
        data(i) =
            simd::exp(simd_warp_t<typename StorageType::value_type>(data(i)));
      });

  Kokkos::fence();
}

template <class StorageType>
void test_fma(Kokkos::View<StorageType *> data,
              simd_warp_t<typename StorageType::value_type> val_1,
              simd_warp_t<typename StorageType::value_type> val_2) {
  Kokkos::parallel_for(
      "simd::fma", Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i = team.league_rank();
        data(i) =
            simd::fma(simd_warp_t<typename StorageType::value_type>(data(i)),
                      val_1, val_2);
      });

  Kokkos::fence();
}

template <class StorageType>
void test_max(Kokkos::View<StorageType *> data,
              simd_warp_t<typename StorageType::value_type> val) {
  Kokkos::parallel_for(
      "simd::max", Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i  = team.league_rank();
        data(i) = simd::max(
            simd_warp_t<typename StorageType::value_type>(data(i)), val);
      });

  Kokkos::fence();
}

template <class StorageType>
void test_min(Kokkos::View<StorageType *> data,
              simd_warp_t<typename StorageType::value_type> val) {
  Kokkos::parallel_for(
      "simd::min", Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i  = team.league_rank();
        data(i) = simd::min(
            simd_warp_t<typename StorageType::value_type>(data(i)), val);
      });

  Kokkos::fence();
}

template <class StorageType>
void test_add(Kokkos::View<StorageType *> data,
              simd_warp_t<typename StorageType::value_type> val) {
  Kokkos::parallel_for(
      "simd::add", Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i  = team.league_rank();
        data(i) = simd_warp_t<typename StorageType::value_type>(data(i)) + val;
      });

  Kokkos::fence();
}

template <class StorageType>
void test_subtract(Kokkos::View<StorageType *> data,
                   simd_warp_t<typename StorageType::value_type> val) {
  Kokkos::parallel_for(
      "simd::subtract",
      Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i  = team.league_rank();
        data(i) = simd_warp_t<typename StorageType::value_type>(data(i)) - val;
      });

  Kokkos::fence();
}

template <class StorageType>
void test_multiply(Kokkos::View<StorageType *> data,
                   simd_warp_t<typename StorageType::value_type> val) {
  Kokkos::parallel_for(
      "simd::multiply",
      Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i  = team.league_rank();
        data(i) = simd_warp_t<typename StorageType::value_type>(data(i)) * val;
      });

  Kokkos::fence();
}

template <class StorageType>
void test_divide(Kokkos::View<StorageType *> data,
                 simd_warp_t<typename StorageType::value_type> val) {
  Kokkos::parallel_for(
      "simd::divide",
      Kokkos::TeamPolicy<>(data.extent(0), 1, StorageType::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        auto i  = team.league_rank();
        data(i) = simd_warp_t<typename StorageType::value_type>(data(i)) / val;
      });

  Kokkos::fence();
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

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
      const auto index    = i * simdSize + j;
      expectedData(index) = index % sqrtMaxIndex;
    }
  }

  test_sqrt(data);
  test_view_result("test_sqrt_2", data, expectedData);
}

template <typename ScalarType>
void do_test_cbrt(int viewExtent) {
  using storage_t        = simd_storage_t<ScalarType>;
  const int simdSize     = storage_t::size();
  const int expectedSize = viewExtent * storage_t::size();

  auto data = create_data_with_unique_value<storage_t>("Test View", viewExtent,
                                                       ScalarType{27});

  test_cbrt(data);

  Kokkos::View<ScalarType *, Kokkos::HostSpace> expectedData("expectedData",
                                                             expectedSize);
  Kokkos::deep_copy(expectedData, static_cast<ScalarType>(3.0));

  test_view_result("test_cbrt_1", data, expectedData);

  data = create_simd_data_cbrt<storage_t>("Test View 2", viewExtent);

  Kokkos::View<ScalarType *> results_scalar((ScalarType *)data.data(),
                                            data.extent(0) * storage_t::size());
  auto results_h = Kokkos::create_mirror_view(results_scalar);
  Kokkos::deep_copy(results_h, results_scalar);

  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      auto index          = i * simdSize + j;
      expectedData(index) = static_cast<ScalarType>(index % cbrtMaxIndex);
    }
  }

  test_cbrt(data);
  test_view_result("test_cbrt_2", data, expectedData);
}

template <typename ScalarType>
void do_test_exp(int viewExtent) {
  using storage_t        = simd_storage_t<ScalarType>;
  const int simdSize     = storage_t::size();
  const int expectedSize = viewExtent * storage_t::size();

  auto data = create_data_with_unique_value<storage_t>("Test View", viewExtent,
                                                       ScalarType{1});

  test_exp(data);

  Kokkos::View<ScalarType *, Kokkos::HostSpace> expectedData("expectedData",
                                                             expectedSize);
  Kokkos::deep_copy(expectedData, static_cast<ScalarType>(std::exp(1.0)));

  test_view_result("test_exp_1", data, expectedData);

  data = create_data_positive<storage_t>("Test View 2", viewExtent);

  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      const auto index    = i * simdSize + j;
      expectedData(index) = static_cast<ScalarType>(std::exp(index));
    }
  }

  test_exp(data);
  test_view_result("test_exp_2", data, expectedData);
}

template <typename ScalarType>
void do_test_fma(int viewExtent) {
  using storage_t        = simd_storage_t<ScalarType>;
  const int simdSize     = storage_t::size();
  const int expectedSize = viewExtent * storage_t::size();

  auto data = create_data_with_unique_value<storage_t>("Test View", viewExtent,
                                                       ScalarType{-4});

  test_fma(data, simd_warp_t<ScalarType>{2.0}, simd_warp_t<ScalarType>{5.0});

  Kokkos::View<ScalarType *, Kokkos::HostSpace> expectedData("expectedData",
                                                             expectedSize);
  Kokkos::deep_copy(expectedData, static_cast<ScalarType>(-3.0));

  test_view_result("test_fma_1", data, expectedData);

  data = create_data_positive<storage_t>("Test View 2", viewExtent);

  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      const auto index    = i * simdSize + j;
      expectedData(index) = static_cast<ScalarType>(7.0 + 4 * (index));
    }
  }

  test_fma(data, simd_warp_t<ScalarType>{4.0}, simd_warp_t<ScalarType>{7.0});
  test_view_result("test_fma_2", data, expectedData);

  data = create_data_negative<storage_t>("Test View 2", viewExtent);
  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      const auto index    = i * simdSize + j;
      expectedData(index) = static_cast<ScalarType>(7.0 - 4 * (index));
    }
  }

  test_fma(data, simd_warp_t<ScalarType>{4.0}, simd_warp_t<ScalarType>{7.0});
  test_view_result("test_fma_3", data, expectedData);
}

template <typename ScalarType>
void do_test_max(int viewExtent) {
  using storage_t        = simd_storage_t<ScalarType>;
  const int simdSize     = storage_t::size();
  const int expectedSize = viewExtent * storage_t::size();

  auto data = create_data_with_unique_value<storage_t>("Test View", viewExtent,
                                                       ScalarType{1.0});

  test_max(data, simd_warp_t<ScalarType>{10.0});
  Kokkos::View<ScalarType *, Kokkos::HostSpace> expectedData("expectedData",
                                                             expectedSize);
  Kokkos::deep_copy(expectedData, static_cast<ScalarType>(10.0));

  test_view_result("test_max_1", data, expectedData);

  data = create_simd_data_sqrt<storage_t>("Test View 2", viewExtent);

  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      const auto index       = i * simdSize + j;
      const ScalarType value = index % sqrtMaxIndex;
      expectedData(index)    = std::max(ScalarType{10.0}, value * value);
    }
  }

  test_max(data, simd_warp_t<ScalarType>{10.0});
  test_view_result("test_max_2", data, expectedData);
}

template <typename ScalarType>
void do_test_min(int viewExtent) {
  using storage_t        = simd_storage_t<ScalarType>;
  const int simdSize     = storage_t::size();
  const int expectedSize = viewExtent * storage_t::size();

  auto data = create_data_with_unique_value<storage_t>("Test View", viewExtent,
                                                       ScalarType{4.0});

  test_min(data, simd_warp_t<ScalarType>{1.0});
  Kokkos::View<ScalarType *, Kokkos::HostSpace> expectedData("expectedData",
                                                             expectedSize);
  Kokkos::deep_copy(expectedData, static_cast<ScalarType>(1.0));

  test_view_result("test_min_1", data, expectedData);

  data = create_simd_data_sqrt<storage_t>("Test View 2", viewExtent);

  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      const auto index       = i * simdSize + j;
      const ScalarType value = index % sqrtMaxIndex;
      expectedData(index)    = std::min(ScalarType{10.0}, value * value);
    }
  }

  test_min(data, simd_warp_t<ScalarType>{10.0});
  test_view_result("test_min_2", data, expectedData);
}

template <typename ScalarType>
void do_test_add(int viewExtent) {
  using storage_t        = simd_storage_t<ScalarType>;
  const int simdSize     = storage_t::size();
  const int expectedSize = viewExtent * storage_t::size();

  auto data = create_data_with_unique_value<storage_t>("Test View", viewExtent,
                                                       ScalarType{4.0});
  test_add(data, simd_warp_t<ScalarType>{10.0});
  Kokkos::View<ScalarType *, Kokkos::HostSpace> expectedData("expectedData",
                                                             expectedSize);
  Kokkos::deep_copy(expectedData, static_cast<ScalarType>(14.0));

  test_view_result("test_add_1", data, expectedData);

  data = create_data_positive<storage_t>("Test View 2", viewExtent);

  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      const auto index    = i * simdSize + j;
      expectedData(index) = ScalarType{4.0} + static_cast<ScalarType>(index);
    }
  }

  test_add(data, simd_warp_t<ScalarType>{4.0});
  test_view_result("test_add_2", data, expectedData);
}

template <typename ScalarType>
void do_test_subtract(int viewExtent) {
  using storage_t        = simd_storage_t<ScalarType>;
  const int simdSize     = storage_t::size();
  const int expectedSize = viewExtent * storage_t::size();

  auto data = create_data_with_unique_value<storage_t>("Test View", viewExtent,
                                                       ScalarType{4.0});
  test_subtract(data, simd_warp_t<ScalarType>{2.0});
  Kokkos::View<ScalarType *, Kokkos::HostSpace> expectedData("expectedData",
                                                             expectedSize);
  Kokkos::deep_copy(expectedData, static_cast<ScalarType>(2.0));

  test_view_result("test_subtract_1", data, expectedData);

  data = create_data_positive<storage_t>("Test View 2", viewExtent);

  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      const auto index    = i * simdSize + j;
      expectedData(index) = static_cast<ScalarType>(index) - ScalarType{4.0};
    }
  }

  test_subtract(data, simd_warp_t<ScalarType>{4.0});
  test_view_result("test_subtract_2", data, expectedData);
}

template <typename ScalarType>
void do_test_multiply(int viewExtent) {
  using storage_t        = simd_storage_t<ScalarType>;
  const int simdSize     = storage_t::size();
  const int expectedSize = viewExtent * storage_t::size();

  auto data = create_data_with_unique_value<storage_t>("Test View", viewExtent,
                                                       ScalarType{4.0});
  test_multiply(data, simd_warp_t<ScalarType>{2.0});
  Kokkos::View<ScalarType *, Kokkos::HostSpace> expectedData("expectedData",
                                                             expectedSize);
  Kokkos::deep_copy(expectedData, static_cast<ScalarType>(8.0));

  test_view_result("test_multiply_1", data, expectedData);

  data = create_data_positive<storage_t>("Test View 2", viewExtent);

  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      const auto index    = i * simdSize + j;
      expectedData(index) = static_cast<ScalarType>(index) * ScalarType{4.0};
    }
  }

  test_multiply(data, simd_warp_t<ScalarType>{4.0});
  test_view_result("test_multiply_2", data, expectedData);
}

template <typename ScalarType>
void do_test_divide(int viewExtent) {
  using storage_t        = simd_storage_t<ScalarType>;
  const int simdSize     = storage_t::size();
  const int expectedSize = viewExtent * storage_t::size();

  auto data = create_data_with_unique_value<storage_t>("Test View", viewExtent,
                                                       ScalarType{4.0});
  test_divide(data, simd_warp_t<ScalarType>{2.0});
  Kokkos::View<ScalarType *, Kokkos::HostSpace> expectedData("expectedData",
                                                             expectedSize);
  Kokkos::deep_copy(expectedData, static_cast<ScalarType>(2.0));

  test_view_result("test_divide_1", data, expectedData);

  data = create_data_positive<storage_t>("Test View 2", viewExtent);

  for (int i = 0; i < viewExtent; ++i) {
    for (int j = 0; j < simdSize; ++j) {
      const auto index    = i * simdSize + j;
      expectedData(index) = static_cast<ScalarType>(index) / ScalarType{4.0};
    }
  }

  test_divide(data, simd_warp_t<ScalarType>{4.0});
  test_view_result("test_divide_2", data, expectedData);
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

TYPED_TEST_P(TestCudaWarp, test_cbrt) {
  constexpr auto extentSize = std::tuple_element<0, TypeParam>::type::value;
  using ScalarType          = typename std::tuple_element<1, TypeParam>::type;

  do_test_cbrt<ScalarType>(extentSize);
}

TYPED_TEST_P(TestCudaWarp, test_exp) {
  constexpr auto extentSize = std::tuple_element<0, TypeParam>::type::value;
  using ScalarType          = typename std::tuple_element<1, TypeParam>::type;

  do_test_exp<ScalarType>(extentSize);
}

TYPED_TEST_P(TestCudaWarp, test_fma) {
  constexpr auto extentSize = std::tuple_element<0, TypeParam>::type::value;
  using ScalarType          = typename std::tuple_element<1, TypeParam>::type;

  do_test_fma<ScalarType>(extentSize);
}

TYPED_TEST_P(TestCudaWarp, test_max) {
  constexpr auto extentSize = std::tuple_element<0, TypeParam>::type::value;
  using ScalarType          = typename std::tuple_element<1, TypeParam>::type;

  do_test_max<ScalarType>(extentSize);
}

TYPED_TEST_P(TestCudaWarp, test_min) {
  constexpr auto extentSize = std::tuple_element<0, TypeParam>::type::value;
  using ScalarType          = typename std::tuple_element<1, TypeParam>::type;

  do_test_min<ScalarType>(extentSize);
}

TYPED_TEST_P(TestCudaWarp, test_add) {
  constexpr auto extentSize = std::tuple_element<0, TypeParam>::type::value;
  using ScalarType          = typename std::tuple_element<1, TypeParam>::type;

  do_test_add<ScalarType>(extentSize);
}

TYPED_TEST_P(TestCudaWarp, test_subtract) {
  constexpr auto extentSize = std::tuple_element<0, TypeParam>::type::value;
  using ScalarType          = typename std::tuple_element<1, TypeParam>::type;

  do_test_subtract<ScalarType>(extentSize);
}

TYPED_TEST_P(TestCudaWarp, test_multiply) {
  constexpr auto extentSize = std::tuple_element<0, TypeParam>::type::value;
  using ScalarType          = typename std::tuple_element<1, TypeParam>::type;

  do_test_multiply<ScalarType>(extentSize);
}
TYPED_TEST_P(TestCudaWarp, test_divide) {
  constexpr auto extentSize = std::tuple_element<0, TypeParam>::type::value;
  using ScalarType          = typename std::tuple_element<1, TypeParam>::type;

  do_test_divide<ScalarType>(extentSize);
}

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

REGISTER_TYPED_TEST_SUITE_P(TestCudaWarp, test_abs, test_sqrt, test_cbrt,
                            test_exp, test_fma, test_max, test_min, test_add,
                            test_subtract, test_multiply, test_divide);

INSTANTIATE_TYPED_TEST_SUITE_P(test_simd_cuda_warp_set, TestCudaWarp,
                               TestTypes);

}  // namespace Test

#endif