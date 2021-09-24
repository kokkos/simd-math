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

#pragma once

#include <simd.hpp>
#include <Kokkos_Core.hpp>

namespace Test {

#ifdef SIMD_FORCE_SCALAR
#define SIMD_USING_SCALAR_ABI
#endif

#ifdef __CUDACC__
#define SIMD_USING_SCALAR_ABI
#endif

#ifdef __HIPCC__
#define SIMD_USING_SCALAR_ABI
#endif

template <typename ScalarType>
using simd_t = simd::simd<ScalarType, simd::simd_abi::native>;

template <typename ScalarType>
using storage_t = typename simd_t<ScalarType>::storage_type;

constexpr int extentArray[14] = {1,      2,      8,       9,      100,
                                 101,    1000,   1001,    10000,  10001,
                                 100000, 100001, 1000000, 1000001};
constexpr int sqrtMaxIndex    = 10000;
constexpr int cbrtMaxIndex    = 5000;

enum SIMD_CONSTRUCTOR {
  SIMD_CONSTRUCTOR_SCALAR           = 0,
  SIMD_CONSTRUCTOR_MUTLI_SCALAR     = 1,
  SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG   = 2,
  SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE = 3,
  SIMD_CONSTRUCTOR_STORAGE          = 4,
#ifndef SIMD_USING_SCALAR_ABI
  SIMD_CONSTRUCTOR_SPECIALIZED = 5,
#endif
};

// Method to create simd
template <typename ScalarType>
simd_t<ScalarType> create_simd_data(SIMD_CONSTRUCTOR constructor,
                                    ScalarType i) {
  return create_simd_data<ScalarType, simd_t<ScalarType>::size()>(constructor,
                                                                  i);
}

template <typename ScalarType, std::size_t vectorLength>
simd_t<ScalarType> create_simd_data(SIMD_CONSTRUCTOR constructor, ScalarType i);

#if defined(SIMD_USING_SCALAR_ABI)
template <>
simd_t<float> create_simd_data<float, 1>(SIMD_CONSTRUCTOR constructor,
                                         float i) {
  constexpr auto num_scalars = 1;
  float array[num_scalars]   = {i};

  switch (constructor) {
    case SIMD_CONSTRUCTOR_MUTLI_SCALAR: return simd_t<float>{array[0]};
    case SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG:
      return simd_t<float>(array, simd::element_aligned_tag());
    case SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE: return simd_t<float>(array, 1);
    case SIMD_CONSTRUCTOR_STORAGE: return simd_t<float>(storage_t<float>(i));
    case SIMD_CONSTRUCTOR_SCALAR:
    default: return simd_t<float>{i};
  }
}

template <>
simd_t<double> create_simd_data<double, 1>(SIMD_CONSTRUCTOR constructor,
                                           double i) {
  constexpr auto num_scalars = 1;
  double array[num_scalars]  = {i};

  switch (constructor) {
    case SIMD_CONSTRUCTOR_MUTLI_SCALAR: return simd_t<double>{array[0]};
    case SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG:
      return simd_t<double>(array, simd::element_aligned_tag());
    case SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE: return simd_t<double>(array, 1);
    case SIMD_CONSTRUCTOR_STORAGE: return simd_t<double>(storage_t<double>(i));
    case SIMD_CONSTRUCTOR_SCALAR:
    default: return simd_t<double>{i};
  }
}

#else

#if defined(__VSX__) || defined(__SSE2__) || defined(__SSE__) || \
    (defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE_BITS) &&  \
     !defined(__ARM_FEATURE_SVE))

template <>
simd_t<float> create_simd_data<float, 4>(SIMD_CONSTRUCTOR constructor,
                                         float i) {
  constexpr auto num_scalars = 4;
  float array[num_scalars]   = {i, i + 1.0f, i + 2.0f, i + 3.0f};

  switch (constructor) {
    case SIMD_CONSTRUCTOR_MUTLI_SCALAR:
      return simd_t<float>{array[0], array[1], array[2], array[3]};
    case SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG:
      return simd_t<float>(array, simd::element_aligned_tag());
    case SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE: return simd_t<float>(array, 1);
    case SIMD_CONSTRUCTOR_STORAGE: return simd_t<float>(storage_t<float>(i));
    case SIMD_CONSTRUCTOR_SPECIALIZED:
      return simd_t<float>({array[0], array[1], array[2], array[3]});
    case SIMD_CONSTRUCTOR_SCALAR:
    default: return simd_t<float>{i};
  }
}

template <>
simd_t<double> create_simd_data<double, 2>(SIMD_CONSTRUCTOR constructor,
                                           double i) {
  constexpr auto num_scalars = 2;
  double array[num_scalars]  = {i, i + 1.0};

  switch (constructor) {
    case SIMD_CONSTRUCTOR_MUTLI_SCALAR:
      return simd_t<double>{array[0], array[1]};
    case SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG:
      return simd_t<double>(array, simd::element_aligned_tag());
    case SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE: return simd_t<double>(array, 1);
    case SIMD_CONSTRUCTOR_STORAGE: return simd_t<double>(storage_t<double>(i));
    case SIMD_CONSTRUCTOR_SPECIALIZED:
      return simd_t<double>({array[0], array[1]});
    case SIMD_CONSTRUCTOR_SCALAR:
    default: return simd_t<double>{i};
  }
}
#elif !defined(SIMD_FORCE_SCALAR) && !defined(__CUDACC__) &&            \
    !defined(__HIPCC__) && !defined(__AVX512F__) && !defined(avx512) && \
    !defined(__AVX__) && !defined(__SSE2__) &&                          \
    (!(defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE_BITS) &&       \
       !defined(__ARM_FEATURE_SVE))) &&                                 \
    !defined(__VSX__) && !defined(SIMD_ENABLE_VECTOR_SIZE)
// from simd.hpp: native = pack<8>
template <>
simd_t<float> create_simd_data<float, 8>(SIMD_CONSTRUCTOR constructor,
                                         float i) {
  constexpr auto num_scalars = 8;
  float array[num_scalars];
  for (int j = 0; j < 8; ++j) {
    array[j] = j + i;
  }

  switch (constructor) {
    case SIMD_CONSTRUCTOR_SPECIALIZED:
    case SIMD_CONSTRUCTOR_MUTLI_SCALAR:
    case SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE:
    case SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG:
      return simd_t<float>(array, simd::element_aligned_tag());
    case SIMD_CONSTRUCTOR_STORAGE: return simd_t<float>(storage_t<float>(i));
    case SIMD_CONSTRUCTOR_SCALAR:
    default: return simd_t<float>{i};
  }
}

template <>
simd_t<double> create_simd_data<double, 8>(SIMD_CONSTRUCTOR constructor,
                                           double i) {
  constexpr auto num_scalars = 8;
  double array[num_scalars];
  for (int j = 0; j < 8; ++j) {
    array[j] = j + i;
  }
  switch (constructor) {
    case SIMD_CONSTRUCTOR_SPECIALIZED:
    case SIMD_CONSTRUCTOR_MUTLI_SCALAR:
    case SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE:
    case SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG:
      return simd_t<double>(array, simd::element_aligned_tag());
    case SIMD_CONSTRUCTOR_STORAGE: return simd_t<double>(storage_t<double>(i));
    case SIMD_CONSTRUCTOR_SCALAR:
    default: return simd_t<double>{i};
  }
}

#elif defined(__AVX__)

template <>
simd_t<float> create_simd_data<float, 8>(SIMD_CONSTRUCTOR constructor,
                                         float i) {
  constexpr auto num_scalars = 8;
  float array[num_scalars]   = {i,        i + 1.0f, i + 2.0f, i + 3.0f,
                              i + 4.0f, i + 5.0f, i + 6.0f, i + 7.0f};

  switch (constructor) {
    case SIMD_CONSTRUCTOR_MUTLI_SCALAR:
      return simd_t<float>{array[0], array[1], array[2], array[3],
                           array[4], array[5], array[6], array[7]};
    case SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG:
      return simd_t<float>(array, simd::element_aligned_tag());
    case SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE: return simd_t<float>(array, 1);
    case SIMD_CONSTRUCTOR_STORAGE: return simd_t<float>(storage_t<float>(i));
    case SIMD_CONSTRUCTOR_SPECIALIZED:
      return simd_t<float>({array[0], array[1], array[2], array[3], array[4],
                            array[5], array[6], array[7]});
    case SIMD_CONSTRUCTOR_SCALAR:
    default: return simd_t<float>{i};
  }
}

template <>
simd_t<double> create_simd_data<double, 4>(SIMD_CONSTRUCTOR constructor,
                                           double i) {
  constexpr auto num_scalars = 4;
  double array[num_scalars]  = {i, i + 1.0, i + 2.0, i + 3.0};

  switch (constructor) {
    case SIMD_CONSTRUCTOR_MUTLI_SCALAR:
      return simd_t<double>{array[0], array[1], array[2], array[3]};
    case SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG:
      return simd_t<double>(array, simd::element_aligned_tag());
    case SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE: return simd_t<double>(array, 1);
    case SIMD_CONSTRUCTOR_STORAGE: return simd_t<double>(storage_t<double>(i));
    case SIMD_CONSTRUCTOR_SPECIALIZED:
      return simd_t<double>({array[0], array[1], array[2], array[3]});
    case SIMD_CONSTRUCTOR_SCALAR:
    default: return simd_t<double>{i};
  }
}

#elif defined(__AVX512F__)

template <>
simd_t<float> create_simd_data<float, 16>(SIMD_CONSTRUCTOR constructor,
                                          float i) {
  constexpr auto num_scalars = 16;
  float array[num_scalars]   = {i,         i + 1.0f,  i + 2.0f,  i + 3.0f,
                              i + 4.0f,  i + 5.0f,  i + 6.0f,  i + 7.0f,
                              i + 8.0f,  i + 9.0f,  i + 10.0f, i + 11.0f,
                              i + 12.0f, i + 13.0f, i + 14.0f, i + 15.0f};

  switch (constructor) {
    case SIMD_CONSTRUCTOR_MUTLI_SCALAR:
      return simd_t<float>{
          array[0],  array[1],  array[2],  array[3],  array[4],  array[5],
          array[6],  array[7],  array[8],  array[9],  array[10], array[11],
          array[12], array[13], array[14], array[15], array[16]};
    case SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG:
      return simd_t<float>(array, simd::element_aligned_tag());
    case SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE: return simd_t<float>(array, 1);
    case SIMD_CONSTRUCTOR_STORAGE: return simd_t<float>(storage_t<float>(i));
    case SIMD_CONSTRUCTOR_SPECIALIZED:
      return simd_t<float>({array[0], array[1], array[2], array[3], array[4],
                            array[5], array[6], array[7], array[8], array[9],
                            array[10], array[11], array[12], array[13],
                            array[14], array[15], array[16]});
    case SIMD_CONSTRUCTOR_SCALAR:
    default: return simd_t<float>{i};
  }
}

template <>
simd_t<double> create_simd_data<double, 8>(SIMD_CONSTRUCTOR constructor,
                                           double i) {
  constexpr auto num_scalars = 8;
  double array[num_scalars]  = {i,       i + 1.0, i + 2.0, i + 3.0,
                               i + 4.0, i + 5.0, i + 6.0, i + 7.0};

  switch (constructor) {
    case SIMD_CONSTRUCTOR_MUTLI_SCALAR:
      return simd_t<double>{array[0], array[1], array[2], array[3],
                            array[4], array[5], array[6], array[7]};
    case SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG:
      return simd_t<double>(array, simd::element_aligned_tag());
    case SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE: return simd_t<double>(array, 1);
    case SIMD_CONSTRUCTOR_STORAGE: return simd_t<double>(storage_t<double>(i));
    case SIMD_CONSTRUCTOR_SPECIALIZED:
      return simd_t<double>({array[0], array[1], array[2], array[3], array[4],
                             array[5], array[6], array[7]});
    case SIMD_CONSTRUCTOR_SCALAR:
    default: return simd_t<double>{i};
  }
}

#endif
#endif

// Methods to create data
template <class StorageType>
Kokkos::View<StorageType *> create_data_with_uniq_value(
    const std::string &name, int size, typename StorageType::value_type value) {
  Kokkos::View<StorageType *> data(name, size);
  Kokkos::deep_copy(data, StorageType(value));

  return data;
}

template <class ValueType>
Kokkos::View<simd_t<ValueType> *> create_data_zero_to_size_positive(
    std::string name, int size,
    SIMD_CONSTRUCTOR constructor = SIMD_CONSTRUCTOR_SCALAR) {
  Kokkos::View<simd_t<ValueType> *> data(name, size);
  auto data_h = Kokkos::create_mirror_view(Kokkos::HostSpace(),
                                           data);  // this lives on host
  for (size_t i = 0; i < data_h.extent(0); ++i) {
    data_h(i) = create_simd_data<ValueType>(constructor, i);
  }
  Kokkos::deep_copy(data, data_h);

  return data;
}

template <class ValueType>
Kokkos::View<simd_t<ValueType> *> create_data_zero_to_size_negative(
    std::string name, int size,
    SIMD_CONSTRUCTOR constructor = SIMD_CONSTRUCTOR_SCALAR) {
  Kokkos::View<simd_t<ValueType> *> data(name, size);
  auto data_h = Kokkos::create_mirror_view(Kokkos::HostSpace(),
                                           data);  // this lives on host
  for (size_t i = 0; i < data_h.extent(0); ++i) {
    data_h(i) = -create_simd_data<ValueType>(constructor, i);
  }
  Kokkos::deep_copy(data, data_h);

  return data;
}

template <class ValueType>
Kokkos::View<simd_t<ValueType> *> create_data_sqrt(
    std::string name, int size,
    SIMD_CONSTRUCTOR constructor = SIMD_CONSTRUCTOR_SCALAR) {
  Kokkos::View<simd_t<ValueType> *> data(name, size);
  auto data_h = Kokkos::create_mirror_view(Kokkos::HostSpace(),
                                           data);  // this lives on host
  int j = 0;  // We don't want a square higher than 10k *10k. After this  we
              // restart the square from 0.
  for (size_t i = 0; i < data_h.extent(0); ++i) {
    data_h(i) = create_simd_data<ValueType>(constructor, j) *
                create_simd_data<ValueType>(constructor, j);
    j >= sqrtMaxIndex ? j = 0 : ++j;
  }
  Kokkos::deep_copy(data, data_h);

  return data;
}

template <class ValueType>
Kokkos::View<simd_t<ValueType> *> create_data_cbrt(
    std::string name, int size,
    SIMD_CONSTRUCTOR constructor = SIMD_CONSTRUCTOR_SCALAR) {
  Kokkos::View<simd_t<ValueType> *> data(name, size);
  auto data_h = Kokkos::create_mirror_view(Kokkos::HostSpace(),
                                           data);  // this lives on host
  int j = 0;  // We don't want a cbrt higher than 5k * 5k * 5k. After this  we
              // restart the square from 0.
  for (size_t i = 0; i < data_h.extent(0); ++i) {
    data_h(i) = create_simd_data<ValueType>(constructor, j) *
                create_simd_data<ValueType>(constructor, j) *
                create_simd_data<ValueType>(constructor, j);
    j >= cbrtMaxIndex ? j = 0 : ++j;
  }
  Kokkos::deep_copy(data, data_h);

  return data;
}

template <typename ScalarType>
void compare(const std::string &test_name, int i, ScalarType value,
             ScalarType expected);

template <>
void compare(const std::string &test_name, int i, double value,
             double expected) {
  EXPECT_DOUBLE_EQ(value, expected)
      << "Failure during " + test_name + "<double> with i = " + std::to_string(i);
  ;
}

template <>
void compare(const std::string &test_name, int i, float value, float expected) {
  EXPECT_FLOAT_EQ(value, expected)
      << "Failure during " + test_name + "<float> with i = " + std::to_string(i);
  ;
}

}  // namespace Test
