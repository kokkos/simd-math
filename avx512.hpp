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

#include "simd_common.hpp"

#ifdef __AVX512F__

#include <immintrin.h>

namespace SIMD_NAMESPACE {

namespace simd_abi {

class avx512 {};

}

template <>
class simd_mask<float, simd_abi::avx512> {
  __mmask16 m_value;
 public:
  using value_type = bool;
  using simd_type = simd<float, simd_abi::avx512>;
  using abi_type = simd_abi::avx512;
  SIMD_ALWAYS_INLINE inline simd_mask() = default;
  SIMD_ALWAYS_INLINE inline simd_mask(bool value)
    :m_value(-std::int16_t(value))
  {}
  SIMD_ALWAYS_INLINE inline static constexpr int size() { return 16; }
  SIMD_ALWAYS_INLINE inline constexpr simd_mask(__mmask16 const& value_in)
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE inline constexpr __mmask16 get() const { return m_value; }
  SIMD_ALWAYS_INLINE inline simd_mask operator||(simd_mask const& other) const {
    return simd_mask(_kor_mask16(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE inline simd_mask operator&&(simd_mask const& other) const {
    return simd_mask(_kand_mask16(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE inline simd_mask operator!() const {
    return simd_mask(_knot_mask16(m_value));
  }
};

SIMD_ALWAYS_INLINE inline bool all_of(simd_mask<float, simd_abi::avx512> const& a) {
  return _ktestc_mask16_u8(a.get(),
      simd_mask<float, simd_abi::avx512>(true).get());
}

SIMD_ALWAYS_INLINE inline bool any_of(simd_mask<float, simd_abi::avx512> const& a) {
  return !_ktestc_mask16_u8(
      simd_mask<float, simd_abi::avx512>(false).get(), a.get());
}

template <>
class simd<float, simd_abi::avx512> {
  __m512 m_value;
 public:
  SIMD_ALWAYS_INLINE simd() = default;
  using value_type = float;
  using abi_type = simd_abi::avx512;
  using mask_type = simd_mask<float, abi_type>;
  using storage_type = simd_storage<float, abi_type>;
  SIMD_ALWAYS_INLINE inline static constexpr int size() { return 16; }
  SIMD_ALWAYS_INLINE inline simd(float value)
    :m_value(_mm512_set1_ps(value))
  {}
  SIMD_ALWAYS_INLINE inline simd(
      float a, float b, float c, float d,
      float e, float f, float g, float h,
      float i, float j, float k, float l,
      float m, float n, float o, float p)
    :m_value(_mm512_setr_ps(
          a, b, c, d, e, f, g, h,
          i, j, k, l, m, n, o, p))
  {}
  SIMD_ALWAYS_INLINE inline
  simd(storage_type const& value) {
    copy_from(value.data(), element_aligned_tag());
  }
  SIMD_ALWAYS_INLINE inline
  simd& operator=(storage_type const& value) {
    copy_from(value.data(), element_aligned_tag());
    return *this;
  }
  template <class Flags>
  SIMD_ALWAYS_INLINE inline simd(float const* ptr, Flags /*flags*/)
    :m_value(_mm512_loadu_ps(ptr))
  {}
  SIMD_ALWAYS_INLINE inline simd(float const* ptr, int stride)
    :simd(ptr[0],        ptr[stride],   ptr[2*stride], ptr[3*stride],
          ptr[4*stride], ptr[5*stride], ptr[6*stride], ptr[7*stride],
          ptr[8*stride], ptr[9*stride], ptr[10*stride], ptr[11*stride],
          ptr[12*stride], ptr[13*stride], ptr[14*stride], ptr[15*stride])
  {}
  SIMD_ALWAYS_INLINE inline constexpr simd(__m512 const& value_in)
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE inline simd operator*(simd const& other) const {
    return simd(_mm512_mul_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE inline simd operator/(simd const& other) const {
    return simd(_mm512_div_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE inline simd operator+(simd const& other) const {
    return simd(_mm512_add_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE inline simd operator-(simd const& other) const {
    return simd(_mm512_sub_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd operator-() const {
    return simd(_mm512_sub_ps(_mm512_set1_ps(0.0), m_value));
  }
  SIMD_ALWAYS_INLINE inline void copy_from(float const* ptr, element_aligned_tag) {
    m_value = _mm512_loadu_ps(ptr);
  }
  SIMD_ALWAYS_INLINE inline void copy_to(float* ptr, element_aligned_tag) const {
    _mm512_storeu_ps(ptr, m_value);
  }
  SIMD_ALWAYS_INLINE inline constexpr __m512 get() const { return m_value; }
  SIMD_ALWAYS_INLINE inline simd_mask<float, simd_abi::avx512> operator<(simd const& other) const {
    return simd_mask<float, simd_abi::avx512>(_mm512_cmp_ps_mask(m_value, other.m_value, _CMP_LT_OS));
  }
  SIMD_ALWAYS_INLINE inline simd_mask<float, simd_abi::avx512> operator==(simd const& other) const {
    return simd_mask<float, simd_abi::avx512>(_mm512_cmp_ps_mask(m_value, other.m_value, _CMP_EQ_OS));
  }
};

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> multiplysign(simd<float, simd_abi::avx512> const& a, simd<float, simd_abi::avx512> const& b) {
  static simd<float, simd_abi::avx512> sign_mask(-0.f);
  return simd<float, simd_abi::avx512>(_mm512_xor_ps(a.get(), _mm512_and_ps(sign_mask.get(), b.get())));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> copysign(simd<float, simd_abi::avx512> const& a, simd<float, simd_abi::avx512> const& b) {
  static simd<float, simd_abi::avx512> sign_mask(-0.f);
  return simd<float, simd_abi::avx512>(_mm512_xor_ps(_mm512_andnot_ps(sign_mask.get(), a.get()) , _mm512_and_ps(sign_mask.get(), b.get())));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> abs(simd<float, simd_abi::avx512> const& a) {
  __m512 const rhs = a.get();
  return reinterpret_cast<__m512>(_mm512_and_epi32(reinterpret_cast<__m512i>(rhs), _mm512_set1_epi32(0x7fffffff)));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> sqrt(simd<float, simd_abi::avx512> const& a) {
  return simd<float, simd_abi::avx512>(_mm512_sqrt_ps(a.get()));
}

#ifdef __INTEL_COMPILER
SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> cbrt(simd<float, simd_abi::avx512> const& a) {
  return simd<float, simd_abi::avx512>(_mm512_cbrt_ps(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> exp(simd<float, simd_abi::avx512> const& a) {
  return simd<float, simd_abi::avx512>(_mm512_exp_ps(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> log(simd<float, simd_abi::avx512> const& a) {
  return simd<float, simd_abi::avx512>(_mm512_log_ps(a.get()));
}
#endif

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> fma(
    simd<float, simd_abi::avx512> const& a,
    simd<float, simd_abi::avx512> const& b,
    simd<float, simd_abi::avx512> const& c) {
  return simd<float, simd_abi::avx512>(_mm512_fmadd_ps(a.get(), b.get(), c.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> max(
    simd<float, simd_abi::avx512> const& a, simd<float, simd_abi::avx512> const& b) {
  return simd<float, simd_abi::avx512>(_mm512_max_ps(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> min(
    simd<float, simd_abi::avx512> const& a, simd<float, simd_abi::avx512> const& b) {
  return simd<float, simd_abi::avx512>(_mm512_min_ps(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> choose(
    simd_mask<float, simd_abi::avx512> const& a, simd<float, simd_abi::avx512> const& b, simd<float, simd_abi::avx512> const& c) {
  return simd<float, simd_abi::avx512>(_mm512_mask_blend_ps(a.get(), c.get(), b.get()));
}

template <>
class simd_mask<double, simd_abi::avx512> {
  __mmask8 m_value;
 public:
  using value_type = bool;
  SIMD_ALWAYS_INLINE inline simd_mask() = default;
  SIMD_ALWAYS_INLINE inline simd_mask(bool value)
    :m_value(-std::int16_t(value))
  {}
  SIMD_ALWAYS_INLINE inline static constexpr int size() { return 8; }
  SIMD_ALWAYS_INLINE inline constexpr simd_mask(__mmask8 const& value_in)
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE inline constexpr __mmask8 get() const { return m_value; }
  SIMD_ALWAYS_INLINE simd_mask operator||(simd_mask const& other) const {
    return simd_mask(_kor_mask8(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd_mask operator&&(simd_mask const& other) const {
    return simd_mask(_kand_mask8(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd_mask operator!() const {
    return simd_mask(_knot_mask8(m_value));
  }
};

SIMD_ALWAYS_INLINE inline bool all_of(simd_mask<double, simd_abi::avx512> const& a) {
  return _ktestc_mask8_u8(a.get(),
      simd_mask<double, simd_abi::avx512>(true).get());
}

SIMD_ALWAYS_INLINE inline bool any_of(simd_mask<double, simd_abi::avx512> const& a) {
  return !_ktestc_mask8_u8(
      simd_mask<double, simd_abi::avx512>(false).get(), a.get());
}

template <>
class simd<double, simd_abi::avx512> {
  __m512d m_value;
 public:
  using value_type = double;
  using abi_type = simd_abi::avx512;
  using mask_type = simd_mask<double, abi_type>;
  using storage_type = simd_storage<double, abi_type>;
  SIMD_ALWAYS_INLINE inline simd() = default;
  SIMD_ALWAYS_INLINE inline simd(simd const&) = default;
  SIMD_ALWAYS_INLINE inline simd(simd&&) = default;
  SIMD_ALWAYS_INLINE inline simd& operator=(simd const&) = default;
  SIMD_ALWAYS_INLINE inline simd& operator=(simd&&) = default;
  SIMD_ALWAYS_INLINE inline static constexpr int size() { return 8; }
  SIMD_ALWAYS_INLINE inline simd(double value)
    :m_value(_mm512_set1_pd(value))
  {}
  SIMD_ALWAYS_INLINE inline simd(
      double a, double b, double c, double d,
      double e, double f, double g, double h)
    :m_value(_mm512_setr_pd(a, b, c, d, e, f, g, h))
  {}
  SIMD_ALWAYS_INLINE inline
  simd(storage_type const& value) {
    copy_from(value.data(), element_aligned_tag());
  }
#ifdef STK_VOLATILE_SIMD
  SIMD_ALWAYS_INLINE inline
  simd(simd const volatile& value)
    :m_value(value.m_value)
  {}
#endif
  SIMD_ALWAYS_INLINE inline
  simd& operator=(storage_type const& value) {
    copy_from(value.data(), element_aligned_tag());
    return *this;
  }
  template <class Flags>
  SIMD_ALWAYS_INLINE inline simd(double const* ptr, Flags /*flags*/)
    :m_value(_mm512_loadu_pd(ptr))
  {}
  SIMD_ALWAYS_INLINE inline simd(double const* ptr, int stride)
    :simd(ptr[0],        ptr[stride],   ptr[2*stride], ptr[3*stride],
          ptr[4*stride], ptr[5*stride], ptr[6*stride], ptr[7*stride])
  {}
  SIMD_ALWAYS_INLINE inline constexpr simd(__m512d const& value_in)
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE inline simd operator*(simd const& other) const {
    return simd(_mm512_mul_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE inline simd operator/(simd const& other) const {
    return simd(_mm512_div_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE inline simd operator+(simd const& other) const {
    return simd(_mm512_add_pd(m_value, other.m_value));
  }
#ifdef STK_VOLATILE_SIMD
  SIMD_ALWAYS_INLINE inline void plus_equals(simd const volatile& other) volatile {
    m_value = _mm512_add_pd(m_value, other.m_value);
  }
#endif
  SIMD_ALWAYS_INLINE inline simd operator-(simd const& other) const {
    return simd(_mm512_sub_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd operator-() const {
    return simd(_mm512_sub_pd(_mm512_set1_pd(0.0), m_value));
  }
  SIMD_ALWAYS_INLINE inline void copy_from(double const* ptr, element_aligned_tag) {
    m_value = _mm512_loadu_pd(ptr);
  }
  SIMD_ALWAYS_INLINE inline void copy_to(double* ptr, element_aligned_tag) const {
    _mm512_storeu_pd(ptr, m_value);
  }
  SIMD_ALWAYS_INLINE inline constexpr __m512d get() const { return m_value; }
  SIMD_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator<(simd const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm512_cmp_pd_mask(m_value, other.m_value, _CMP_LT_OS));
  }
  SIMD_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator==(simd const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm512_cmp_pd_mask(m_value, other.m_value, _CMP_EQ_OS));
  }
};

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> multiplysign(simd<double, simd_abi::avx512> const& a, simd<double, simd_abi::avx512> const& b) {
  static simd<double, simd_abi::avx512> sign_mask(-0.0);
  return simd<double, simd_abi::avx512>(_mm512_xor_pd(a.get(), _mm512_and_pd(sign_mask.get(), b.get())));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> copysign(simd<double, simd_abi::avx512> const& a, simd<double, simd_abi::avx512> const& b) {
  static simd<double, simd_abi::avx512> sign_mask(-0.0);
  return simd<double, simd_abi::avx512>(_mm512_xor_pd(_mm512_andnot_pd(sign_mask.get(), a.get()) , _mm512_and_pd(sign_mask.get(), b.get())));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> abs(simd<double, simd_abi::avx512> const& a) {
  __m512d const rhs = a.get();
  return reinterpret_cast<__m512d>(_mm512_and_epi64(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF),
        reinterpret_cast<__m512i>(rhs)));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> sqrt(simd<double, simd_abi::avx512> const& a) {
  return simd<double, simd_abi::avx512>(_mm512_sqrt_pd(a.get()));
}

#ifdef __INTEL_COMPILER
SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> cbrt(simd<double, simd_abi::avx512> const& a) {
  return simd<double, simd_abi::avx512>(_mm512_cbrt_pd(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> exp(simd<double, simd_abi::avx512> const& a) {
  return simd<double, simd_abi::avx512>(_mm512_exp_pd(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> log(simd<double, simd_abi::avx512> const& a) {
  return simd<double, simd_abi::avx512>(_mm512_log_pd(a.get()));
}
#endif

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> fma(
    simd<double, simd_abi::avx512> const& a,
    simd<double, simd_abi::avx512> const& b,
    simd<double, simd_abi::avx512> const& c) {
  return simd<double, simd_abi::avx512>(_mm512_fmadd_pd(a.get(), b.get(), c.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> max(
    simd<double, simd_abi::avx512> const& a, simd<double, simd_abi::avx512> const& b) {
  return simd<double, simd_abi::avx512>(_mm512_max_pd(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> min(
    simd<double, simd_abi::avx512> const& a, simd<double, simd_abi::avx512> const& b) {
  return simd<double, simd_abi::avx512>(_mm512_min_pd(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> choose(
    simd_mask<double, simd_abi::avx512> const& a, simd<double, simd_abi::avx512> const& b, simd<double, simd_abi::avx512> const& c) {
  return simd<double, simd_abi::avx512>(_mm512_mask_blend_pd(a.get(), c.get(), b.get()));
}

  // SIMD MASK FOR SIMD INT
  // Essentially this is the same as the mask for simd<float> but I tried
  // Deriving from that and that didn't work for me.
  // The only difference is the line 'using simd_type='
template <>
class simd_mask<int, simd_abi::avx512>
{
  __mmask16 m_value;
 public:
  using value_type = bool;
  using simd_type = simd<int, simd_abi::avx512>;
  using abi_type = simd_abi::avx512;
  SIMD_ALWAYS_INLINE inline simd_mask() = default;
  SIMD_ALWAYS_INLINE inline simd_mask(bool value)
    :m_value(-std::int16_t(value))
  {}
  SIMD_ALWAYS_INLINE inline static constexpr int size() { return 16; }
  SIMD_ALWAYS_INLINE inline constexpr simd_mask(__mmask16 const& value_in)
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE inline constexpr __mmask16 get() const { return m_value; }
  SIMD_ALWAYS_INLINE inline simd_mask operator||(simd_mask const& other) const {
    return simd_mask(_kor_mask16(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE inline simd_mask operator&&(simd_mask const& other) const {
    return simd_mask(_kand_mask16(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE inline simd_mask operator!() const {
    return simd_mask(_knot_mask16(m_value));
  }
};

  // An integer SIMD class for AVX512
template <>
class simd<int, simd_abi::avx512> {
  __m512i m_value;
 public:
  SIMD_ALWAYS_INLINE simd() = default;
  using value_type = float;
  using abi_type = simd_abi::avx512;
  using mask_type = simd_mask<float, abi_type>;
  using storage_type = simd_storage<int, abi_type>;
  SIMD_ALWAYS_INLINE inline static constexpr int size() { return 16; }
  SIMD_ALWAYS_INLINE inline simd(int value)
    :m_value(_mm512_set1_epi32(value))
  {}
  SIMD_ALWAYS_INLINE inline simd(
      int a, int b, int c, int d,
      int e, int f, int g, int h,
      int i, int j, int k, int l,
      int m, int n, int o, int p)
    :m_value(_mm512_setr_epi32(
          a, b, c, d, e, f, g, h,
          i, j, k, l, m, n, o, p))
  {}

  /*
   * CAVEAT: Loading 8 element here
   *  sets every second element as needed
   *  for permutes. Is this a general pattern?
   *  This is different from AVX & AVX2 where
   *  there is not a DP shuffle actually,
   *  We can use the 256 bit permute from AVX512
   *  buf I believe that expects zeros at the high
   *  end only 
   */
  SIMD_ALWAYS_INLINE inline simd(
      int a, int b, int c, int d,
      int e, int f, int g, int h)
    :m_value(_mm512_setr_epi32(
			       a, 0, b, 0, c, 0, d, 0,
			       e, 0, f, 0, g, 0, h, 0))  {}
  SIMD_ALWAYS_INLINE inline
  simd(storage_type const& value) {
    copy_from(value.data(), element_aligned_tag());
  }
  SIMD_ALWAYS_INLINE inline
  simd& operator=(storage_type const& value) {
    copy_from(value.data(), element_aligned_tag());
    return *this;
  }
  template <class Flags>
  SIMD_ALWAYS_INLINE inline simd(int const* ptr, Flags /*flags*/)
    :m_value(_mm512_load_epi32(static_cast<void const*>(ptr)))
  {}
  SIMD_ALWAYS_INLINE inline simd(int const* ptr, int stride)
    :simd(ptr[0],        ptr[stride],   ptr[2*stride], ptr[3*stride],
          ptr[4*stride], ptr[5*stride], ptr[6*stride], ptr[7*stride],
          ptr[8*stride], ptr[9*stride], ptr[10*stride], ptr[11*stride],
          ptr[12*stride], ptr[13*stride], ptr[14*stride], ptr[15*stride])
  {}
  SIMD_ALWAYS_INLINE inline constexpr simd(__m512i const& value_in)
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE inline simd operator*(simd const& other) const {
    return simd(_mm512_mul_epi32(m_value, other.m_value));
  }

#if 0
  // This needs SVML extension
  SIMD_ALWAYS_INLINE inline simd operator/(simd const& other) const {
    return simd(_mm512_div_epi32(m_value, other.m_value));
  }
#endif

  SIMD_ALWAYS_INLINE inline simd operator+(simd const& other) const {
    return simd(_mm512_add_epi32(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE inline simd operator-(simd const& other) const {
    return simd(_mm512_sub_epi32(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd operator-() const {
    return simd(_mm512_sub_epi32(_mm512_set1_epi32(0), m_value));
  }
  SIMD_ALWAYS_INLINE inline void copy_from(int const* ptr, element_aligned_tag) {
    m_value = _mm512_load_epi32(static_cast<void const*>(ptr));
  }
  SIMD_ALWAYS_INLINE inline void copy_to(int* ptr, element_aligned_tag) const {
    _mm512_store_epi32(ptr, m_value);
  }
  SIMD_ALWAYS_INLINE inline constexpr __m512i get() const { return m_value; }
  SIMD_ALWAYS_INLINE inline simd_mask<int, simd_abi::avx512> operator<(simd const& other) const {
    return simd_mask<int, simd_abi::avx512>(_mm512_cmp_epi32_mask(m_value, other.m_value, _CMP_LT_OS));
  }
  SIMD_ALWAYS_INLINE inline simd_mask<int, simd_abi::avx512> operator==(simd const& other) const {
    return simd_mask<int, simd_abi::avx512>(_mm512_cmp_epi32_mask(m_value, other.m_value, _CMP_EQ_OS));
  }
};

// Specialized permute
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE
inline simd<float, simd_abi::avx512> permute(simd<int, simd_abi::avx512> const& control,
		simd<float, simd_abi::avx512> const& a) {
   	simd<float,simd_abi::avx512> result(_mm512_permutexvar_ps(control.get(),a.get())  );
  return result;
}

// Specialized permute
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE
inline simd<double, simd_abi::avx512> permute(simd<int, simd_abi::avx512> const& control,
		simd<double, simd_abi::avx512> const& a) {
   	simd<double,simd_abi::avx512> result(_mm512_permutexvar_pd(control.get(),a.get())  );
  return result;
}



}

#endif



