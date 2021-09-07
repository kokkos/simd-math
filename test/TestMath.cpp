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

// Methods to create data
Kokkos::View<simd_t *> createDataWithUniqValue(std::string name, int size, double value) {
    Kokkos::View<simd_t *> data("Test View", 8);
    Kokkos::deep_copy(data, simd_t(value));
    return data;
}

Kokkos::View<simd_t *> createDataZeroToSizePositive(std::string name, int size) {
    Kokkos::View<simd_t *> data(name, size);
    auto data_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), data); // this lives on host
    for(int i = 0; i < data_h.extent(0); ++i)
    {
      data_h(i) = simd_t{static_cast<double>(i)};
    }
    Kokkos::deep_copy(data, data_h);

    return data;
}

Kokkos::View<simd_t *> createDataSQRT(std::string name, int size) {
    Kokkos::View<simd_t *> data(name, size);
    auto data_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), data); // this lives on host
    for(int i = 0; i < data_h.extent(0); ++i)
    {
      data_h(i) = simd_t{static_cast<double>(i)} * simd_t{static_cast<double>(i)};
    }
    Kokkos::deep_copy(data, data_h);

    return data;
}

Kokkos::View<simd_t *> createDataCBRT(std::string name, int size) {
    Kokkos::View<simd_t *> data(name, size);
    auto data_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), data); // this lives on host
    for(int i = 0; i < data_h.extent(0); ++i)
    {
      data_h(i) = simd_t{static_cast<double>(i)} * simd_t{static_cast<double>(i)} * simd_t{static_cast<double>(i)};
    }
    Kokkos::deep_copy(data, data_h);

    return data;
}

Kokkos::View<simd_t *> createDataZeroToSizeNegative(std::string name, int size) {
    Kokkos::View<simd_t *> data(name, size);
    auto data_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), data); // this lives on host
    for(int i = 0; i < data_h.extent(0); ++i)
    {
      data_h(i) = -simd_t{static_cast<double>(i)};
    }
    Kokkos::deep_copy(data, data_h);

    return data;
}

// Methods for testing
void test_abs(Kokkos::View<simd_t *> data) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::abs(data(i)); });
}

void test_sqrt(Kokkos::View<simd_t *> data) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::sqrt(data(i)); });
}

void test_cbrt(Kokkos::View<simd_t *> data) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::cbrt(data(i)); });
}

void test_exp(Kokkos::View<simd_t *> data) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::exp(data(i)); });
}

void test_fma(Kokkos::View<simd_t *> data, simd_t val_1, simd_t val_2) {
  Kokkos::parallel_for(
      data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::fma(data(i), val_1, val_2);
      });
}

void test_max(Kokkos::View<simd_t *> data, simd_t val) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::max(data(i), val); });
}

void test_min(Kokkos::View<simd_t *> data, simd_t val) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::min(data(i), val); });
}

void test_copysign(Kokkos::View<simd_t *> data, simd_t sng) {
  Kokkos::parallel_for(
      data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::copysign(data(i), sng);
      });
}

void test_multiplysign(Kokkos::View<simd_t *> data, simd_t sng) {
  Kokkos::parallel_for(
      data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::multiplysign(data(i), sng);
      });
}

void test_view_result(const std::string &test_name, Kokkos::View<simd_t *> data,
                      double expected) {
  auto data_h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(data_h, data);


  Kokkos::View<double *, Kokkos::HostSpace> scalar_view(
      reinterpret_cast<double *>(data_h.data()),
      data_h.extent(0) * simd_t::size());

  for (int i = 0; i < scalar_view.extent(0); ++i) {
    EXPECT_DOUBLE_EQ(scalar_view(i), expected)
        << "Failure during " + test_name + " with i = " + std::to_string(i);
  }
}

void test_view_result(const std::string &test_name, Kokkos::View<simd_t *> data,
                      double *expected) {
  auto data_h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(data_h, data);

  Kokkos::View<double *, Kokkos::HostSpace> scalar_view(
      reinterpret_cast<double *>(data_h.data()),
      data_h.extent(0) * simd_t::size());

  for (int i = 0; i < scalar_view.extent(0); ++i) {
    EXPECT_DOUBLE_EQ(scalar_view(i), expected[i])
        << "Failure during " + test_name + " with i = " + std::to_string(i);
  }
}

TEST(simd_math, test_abs) {
  Kokkos::View<simd_t *> data = createDataWithUniqValue("Test View", 8, -4);

  test_abs(data);
  test_view_result("test_abs_1", data, 4.0);

  data = createDataZeroToSizePositive("Test View 2", 8);
  const int expectedSize = 16;
  double expectedData[expectedSize]= {0.0,0.0,1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0,5.0,5.0,6.0,6.0,7.0,7.0};

  test_abs(data);
  test_view_result("test_abs_2", data, expectedData);

  data = createDataZeroToSizeNegative("Test View 3", 8);

  test_abs(data);
  test_view_result("test_abs_3", data, expectedData);
}

TEST(simd_math, test_sqrt) {
  Kokkos::View<simd_t *> data = createDataWithUniqValue("Test View", 8, 16.0);

  test_sqrt(data);
  test_view_result("test_sqrt_1", data, 4.0);

  data = createDataSQRT("Test View 2", 8);
  const int expectedSize = 16;
  double expectedData[expectedSize]= {0.0,0.0,1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0,5.0,5.0,6.0,6.0,7.0,7.0};

  test_sqrt(data);
  test_view_result("test_sqrt_2", data, expectedData);
}

TEST(simd_math, test_cbrt) {
  Kokkos::View<simd_t *> data = createDataWithUniqValue("Test View", 8, 27.0);

  test_cbrt(data);
  test_view_result("test_cbrt_1", data, 3.0);

  data = createDataCBRT("Test View 2", 8);
  const int expectedSize = 16;
  double expectedData[expectedSize]= {0.0,0.0,1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0,5.0,5.0,6.0,6.0,7.0,7.0};

  test_cbrt(data);
  test_view_result("test_cbrt_2", data, expectedData);
}

TEST(simd_math, test_exp) {
  Kokkos::View<simd_t *> data = createDataWithUniqValue("Test View", 8, 1.0);

  test_exp(data);
  test_view_result("test_exp_1", data, std::exp(1.0));

  data = createDataZeroToSizePositive("Test View 2", 8);
  const int expectedSimdEntries = 8;
  const int expectedSize = 16;
  double expectedData[expectedSize];
  for(int i = 0; i < expectedSimdEntries; ++i) {
      double currentExp = std::exp(static_cast<double>(i));
      expectedData[2*i] = currentExp;
      expectedData[2*i + 1] = currentExp;
  }

  test_exp(data);
  test_view_result("test_exp_2", data, expectedData);
}

TEST(simd_math, test_fma) {
  Kokkos::View<simd_t *> data = createDataWithUniqValue("Test View", 8, -4.0);

  test_fma(data, simd_t{2.0}, simd_t{5.0});
  test_view_result("test_fma_1", data, -3.0);

  data = createDataZeroToSizePositive("Test View 2", 8);
  const int expectedSize = 16; // 8 * 2
  test_fma(data, simd_t{4.0}, simd_t{7.0});
  double expectedData[expectedSize]= {7.0,7.0,11.0,11.0,15.0,15.0,19.0,19.0,23.0,23.0,27.0,27.0,31.0,31.0,35.0,35.0};
  test_view_result("test_fma_2", data, expectedData);

  data = createDataZeroToSizeNegative("Test View 2", 8);
  test_fma(data, simd_t{4.0}, simd_t{7.0});
  double expectedData2[expectedSize]= {7.0,7.0,3.0,3.0,-1.0,-1.0,-5.0,-5.0,-9.0,-9.0,-13.0,-13.0,-17.0,-17.0,-21.0,-21.0};
  test_view_result("test_fma_3", data, expectedData2);

}

TEST(simd_math, test_max) {
  Kokkos::View<simd_t *> data = createDataWithUniqValue("Test View", 8, 4.0);

  test_max(data, simd_t{10.0});
  test_view_result("test_max_1", data, 10.0);

  data = createDataSQRT("Test View 2", 8);
  const int expectedSize = 16;
  double expectedData[expectedSize]= {10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,16.0,16.0,25.0,25.0,36.0,36.0,49.0,49.0};

  test_max(data, simd_t{10.0});
  test_view_result("test_max_2", data, expectedData);
}

TEST(simd_math, test_min) {
  Kokkos::View<simd_t *> data = createDataWithUniqValue("Test View", 8, 4.0);

  test_min(data, simd_t{1.0});
  test_view_result("test_min", data, 1.0);

  data = createDataSQRT("Test View 2", 8);
  const int expectedSize = 16;
  double expectedData[expectedSize]= {0.0,0.0,1.0,1.0,4.0,4.0,9.0,9.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0};

  test_min(data, simd_t{10.0});
  test_view_result("test_min_2", data, expectedData);
}

TEST(simd_math, test_copysign) {
    Kokkos::View<simd_t *> data = createDataZeroToSizePositive("Test View", 8);

    const int expectedSize = 16; // 8 * 2
    test_copysign(data, simd_t{2.0});
    double expectedData[expectedSize]= {0.0,0.0,1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0,5.0,5.0,6.0,6.0,7.0,7.0};
    test_view_result("test_copysign_1", data, expectedData);

    test_copysign(data, simd_t{-2.0});
    for(int i = 0; i < expectedSize; ++i) {
       expectedData[i] = -expectedData[i];
    }
    test_view_result("test_copysign_2", data, expectedData);

    test_copysign(data, simd_t{4.0});
    for(int i = 0; i < expectedSize; ++i) {
       expectedData[i] = -expectedData[i];
    }
    test_view_result("test_copysign_3", data, expectedData);
}

TEST(simd_math, test_multiplysign) {
    Kokkos::View<simd_t *> data("Test View", 8);
    Kokkos::deep_copy(data, simd_t(4.0));

    test_multiplysign(data, simd_t{2.0});
    test_view_result("test_multiplysign_1", data, 4.0);

    test_multiplysign(data, simd_t{-2.0});
    test_view_result("test_multiplysign_2", data, -4.0);

    test_multiplysign(data, simd_t{2.0});
    test_view_result("test_multiplysign_3", data, -4.0);

    data = createDataZeroToSizePositive("Test View", 8);
    const int expectedSize = 16; // 8 * 2
    test_multiplysign(data, simd_t{2.0});
    double expectedData[expectedSize]= {0.0,0.0,1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0,5.0,5.0,6.0,6.0,7.0,7.0};
    test_view_result("test_multiplysign_4", data, expectedData);

    test_multiplysign(data, simd_t{-2.0});
    for(int i = 0; i < expectedSize; ++i) {
       expectedData[i] = -expectedData[i];
    }
    test_view_result("test_multiplysign_5", data, expectedData);

    test_multiplysign(data, simd_t{-4.0});
    for(int i = 0; i < expectedSize; ++i) {
       expectedData[i] = -expectedData[i];
    }
    test_view_result("test_multiplysign_6", data, expectedData);

    test_multiplysign(data, simd_t{4.0});
    test_view_result("test_multiplysign_7", data, expectedData);


}

}  // namespace Test
