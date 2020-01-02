#pragma once

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __SSE3__
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __VSX__
#include <altivec.h>
// undefine the really dangerous macros from this file
#undef vector
#undef pixel
#undef bool
#endif

#include <cmath>

//#define SIMD_ALWAYS_INLINE [[gnu::always_inline]]
// DI-QUESTION: always-inline gives several errors for gcc 7.2.0 linux build?
#define SIMD_ALWAYS_INLINE

#ifdef __CUDACC__
#define SIMD_HOST_DEVICE __host__ __device__
#else
#define SIMD_HOST_DEVICE
#endif

#if defined(_OPENMP)
#define SIMD_PRAGMA _Pragma("omp simd")
#elif defined(__clang__)
#define SIMD_PRAGMA _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define SIMD_PRAGMA _Pragma("GCC ivdep")
#endif

#ifndef SIMD_NAMESPACE
#define SIMD_NAMESPACE simd
#endif

namespace SIMD_NAMESPACE {

template <class T, class Abi>
class simd;

template <class T, class Abi>
class simd_mask;

class element_aligned_tag {};

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi>& operator+=(simd<T, Abi>& a, simd<T, Abi> const& b) noexcept {
  a = a + b;
  return a;
}

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi>& operator-=(simd<T, Abi>& a, simd<T, Abi> const& b) noexcept {
  a = a - b;
  return a;
}

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi>& operator*=(simd<T, Abi>& a, simd<T, Abi> const& b) noexcept {
  a = a * b;
  return a;
}

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi>& operator/=(simd<T, Abi>& a, simd<T, Abi> const& b) noexcept {
  a = a / b;
  return a;
}

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi> operator+(T const& a, simd<T, Abi> const& b) noexcept {
  return simd<T, Abi>(a) + b;
}

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi> operator+(simd<T, Abi> const& a, T const& b) noexcept {
  return a + simd<T, Abi>(b);
}

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi> operator-(T const& a, simd<T, Abi> const& b) noexcept {
  return simd<T, Abi>(a) - b;
}

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi> operator-(simd<T, Abi> const& a, T const& b) noexcept {
  return a - simd<T, Abi>(b);
}

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi> operator*(T const& a, simd<T, Abi> const& b) noexcept {
  return simd<T, Abi>(a) * b;
}

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi> operator*(simd<T, Abi> const& a, T const& b) noexcept {
  return a * simd<T, Abi>(b);
}

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi> operator/(T const& a, simd<T, Abi> const& b) noexcept {
  return simd<T, Abi>(a) / b;
}

template <class T, class Abi>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd<T, Abi> operator/(simd<T, Abi> const& a, T const& b) noexcept {
  return a / simd<T, Abi>(b);
}

namespace simd_abi {

class scalar {};

}

template <class T>
class simd_mask<T, simd_abi::scalar> {
  bool m_value;
 public:
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE static constexpr int size() noexcept { return 1; }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd_mask(bool value) noexcept
    :m_value(value)
  {}
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE constexpr bool get() const noexcept { return m_value; }
};

template <class T>
class simd<T, simd_abi::scalar> {
  T m_value;
 public:
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd() noexcept = default;
  using value_type = T;
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE static constexpr int size() noexcept { return 1; }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd(T value) noexcept
    :m_value(value)
  {}
  template <class Flags>
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd(T const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd operator*(simd const& other) const noexcept {
    return simd(m_value * other.m_value);
  }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd operator/(simd const& other) const noexcept {
    return simd(m_value / other.m_value);
  }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd operator+(simd const& other) const noexcept {
    return simd(m_value + other.m_value);
  }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd operator-(simd const& other) const noexcept {
    return simd(m_value - other.m_value);
  }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE void copy_from(T const* ptr, element_aligned_tag) noexcept {
    m_value = *ptr;
  }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE void copy_to(T* ptr, element_aligned_tag) const noexcept {
    *ptr = m_value;
  }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE constexpr T get() const noexcept { return m_value; }
  SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE simd_mask<T, simd_abi::scalar> operator<(simd const& other) const noexcept {
    return simd_mask<T, simd_abi::scalar>(m_value < other.m_value);
  }
};

template <class T>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd<T, simd_abi::scalar> sqrt(simd<T, simd_abi::scalar> const& a) noexcept {
  return simd<T, simd_abi::scalar>(std::sqrt(a.get()));
}

template <class T>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd<T, simd_abi::scalar> max(
    simd<T, simd_abi::scalar> const& a, simd<T, simd_abi::scalar> const& b) noexcept {
  using std::max;
  return simd<T, simd_abi::scalar>(max(a.get(), b.get()));
}

template <class T>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd<T, simd_abi::scalar> min(
    simd<T, simd_abi::scalar> const& a, simd<T, simd_abi::scalar> const& b) noexcept {
  using std::min;
  return simd<T, simd_abi::scalar>(min(a.get(), b.get()));
}

template <class T>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd<T, simd_abi::scalar> choose(
    simd_mask<T, simd_abi::scalar> const& a, simd<T, simd_abi::scalar> const& b, simd<T, simd_abi::scalar> const& c) noexcept {
  return simd<T, simd_abi::scalar>(a.get() ? b.get() : c.get());
}

#ifdef SIMD_PRAGMA

namespace simd_abi {

template <std::size_t NBytes>
class directive;

}

template <std::size_t NBytes>
class simd_mask<float, simd_abi::directive<NBytes>> {
  static_assert(NBytes % sizeof(float) == 0, "bytes not a multiple of sizeof(float)");
  int m_value[NBytes / sizeof(float)];
 public:
  SIMD_ALWAYS_INLINE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return NBytes / sizeof(float); }
  SIMD_ALWAYS_INLINE simd_mask(bool value) noexcept
  {
    SIMD_PRAGMA for (int i = 0; i < size(); ++i) m_value[i] = value;
  }
  SIMD_ALWAYS_INLINE constexpr bool operator[](int i) const noexcept { return m_value[i]; }
  SIMD_ALWAYS_INLINE int& operator[](int i) noexcept { return m_value[i]; }
};

template <std::size_t NBytes>
class simd_mask<double, simd_abi::directive<NBytes>> {
  static_assert(NBytes % sizeof(double) == 0, "bytes not a multiple of sizeof(double)");
  long long m_value[NBytes / sizeof(double)];
 public:
  SIMD_ALWAYS_INLINE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return NBytes / sizeof(double); }
  SIMD_ALWAYS_INLINE simd_mask(bool value) noexcept
  {
    SIMD_PRAGMA for (int i = 0; i < size(); ++i) m_value[i] = value;
  }
  SIMD_ALWAYS_INLINE constexpr bool operator[](int i) const noexcept { return m_value[i]; }
  SIMD_ALWAYS_INLINE long long& operator[](int i) noexcept { return m_value[i]; }
};

template <class T, std::size_t NBytes>
class simd<T, simd_abi::directive<NBytes>> {
  static_assert(NBytes % sizeof(T) == 0, "bytes not a multiple of sizeof(T)");
  T m_value[NBytes / sizeof(T)];
 public:
  SIMD_ALWAYS_INLINE simd() noexcept = default;
  using value_type = T;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return NBytes / sizeof(T); }
  SIMD_ALWAYS_INLINE simd(T value) noexcept
  {
    SIMD_PRAGMA for (int i = 0; i < size(); ++i) m_value[i] = value;
  }
  template <class Flags>
  SIMD_ALWAYS_INLINE simd(T const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE simd operator*(simd const& other) const noexcept {
    simd result;
    SIMD_PRAGMA for (int i = 0; i < size(); ++i) result[i] = m_value[i] * other.m_value[i];
    return result;
  }
  SIMD_ALWAYS_INLINE simd operator/(simd const& other) const noexcept {
    simd result;
    SIMD_PRAGMA for (int i = 0; i < size(); ++i) result[i] = m_value[i] / other.m_value[i];
    return result;
  }
  SIMD_ALWAYS_INLINE simd operator+(simd const& other) const noexcept {
    simd result;
    SIMD_PRAGMA for (int i = 0; i < size(); ++i) result[i] = m_value[i] + other.m_value[i];
    return result;
  }
  SIMD_ALWAYS_INLINE simd operator-(simd const& other) const noexcept {
    simd result;
    SIMD_PRAGMA for (int i = 0; i < size(); ++i) result[i] = m_value[i] - other.m_value[i];
    return result;
  }
  SIMD_ALWAYS_INLINE void copy_from(T const* ptr, element_aligned_tag) noexcept {
    SIMD_PRAGMA for (int i = 0; i < size(); ++i) m_value[i] = ptr[i];
  }
  SIMD_ALWAYS_INLINE void copy_to(T* ptr, element_aligned_tag) const noexcept {
    SIMD_PRAGMA for (int i = 0; i < size(); ++i) ptr[i] = m_value[i];
  }
  SIMD_ALWAYS_INLINE constexpr T operator[](int i) const noexcept { return m_value[i]; }
  SIMD_ALWAYS_INLINE T& operator[](int i) noexcept { return m_value[i]; }
  SIMD_ALWAYS_INLINE simd_mask<T, simd_abi::directive<NBytes>> operator<(simd const& other) const noexcept {
    simd_mask<T, simd_abi::directive<NBytes>> result;
    SIMD_PRAGMA for (int i = 0; i < size(); ++i) result[i] = m_value[i] < other.m_value[i];
    return result;
  }
};

template <class T, std::size_t NBytes>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd<T, simd_abi::directive<NBytes>> sqrt(simd<T, simd_abi::directive<NBytes>> const& a) noexcept {
  simd<T, simd_abi::directive<NBytes>> result;
  using std::sqrt;
  SIMD_PRAGMA for (int i = 0; i < a.size(); ++i) result[i] = sqrt(a[i]);
  return result;
}

template <class T, std::size_t NBytes>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd<T, simd_abi::directive<NBytes>> max(
    simd<T, simd_abi::directive<NBytes>> const& a, simd<T, simd_abi::directive<NBytes>> const& b) noexcept {
  simd<T, simd_abi::directive<NBytes>> result;
  using std::max;
  SIMD_PRAGMA for (int i = 0; i < a.size(); ++i) result[i] = max(a[i], b[i]);
  return result;
}

template <class T, std::size_t NBytes>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd<T, simd_abi::directive<NBytes>> min(
    simd<T, simd_abi::directive<NBytes>> const& a, simd<T, simd_abi::directive<NBytes>> const& b) noexcept {
  simd<T, simd_abi::directive<NBytes>> result;
  using std::min;
  SIMD_PRAGMA for (int i = 0; i < a.size(); ++i) result[i] = min(a[i], b[i]);
  return result;
}

template <class T, std::size_t NBytes>
SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd<T, simd_abi::directive<NBytes>> choose(
    simd_mask<T, simd_abi::directive<NBytes>> const& a, simd<T, simd_abi::directive<NBytes>> const& b, simd<T, simd_abi::directive<NBytes>> const& c) noexcept {
  simd<T, simd_abi::directive<NBytes>> result;
  SIMD_PRAGMA for (int i = 0; i < a.size(); ++i) result[i] = a[i] ? b[i] : c[i];
  return result;
}

#endif

#ifdef __SSE__

namespace simd_abi {

class sse {};

}

template <>
class simd_mask<float, simd_abi::sse> {
  __m128 m_value;
 public:
  SIMD_ALWAYS_INLINE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 4; }
  SIMD_ALWAYS_INLINE simd_mask(bool value_in) noexcept
    :m_value(value_in ? _mm_cmpeq_ps(_mm_setzero_ps(),_mm_setzero_ps()) : _mm_setzero_ps())
  {}
  SIMD_ALWAYS_INLINE constexpr simd_mask(__m128 const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE constexpr __m128 get() const noexcept { return m_value; }
};

template <>
class simd<float, simd_abi::sse> {
  __m128 m_value;
 public:
  SIMD_ALWAYS_INLINE simd() noexcept = default;
  using value_type = float;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 4; }
  SIMD_ALWAYS_INLINE simd(float value) noexcept
    :m_value(_mm_set1_ps(value))
  {}
  template <class Flags>
  SIMD_ALWAYS_INLINE simd(float const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE constexpr simd(__m128 const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE simd operator*(simd const& other) const noexcept {
    return simd(_mm_mul_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator/(simd const& other) const noexcept {
    return simd(_mm_div_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator+(simd const& other) const noexcept {
    return simd(_mm_add_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator-(simd const& other) const noexcept {
    return simd(_mm_sub_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE void copy_from(float const* ptr, element_aligned_tag) noexcept {
    m_value = _mm_loadu_ps(ptr);
  }
  SIMD_ALWAYS_INLINE void copy_to(float* ptr, element_aligned_tag) const noexcept {
    _mm_storeu_ps(ptr, m_value);
  }
  SIMD_ALWAYS_INLINE constexpr __m128 get() const noexcept { return m_value; }
  SIMD_ALWAYS_INLINE simd_mask<float, simd_abi::sse> operator<(simd const& other) const noexcept {
    return simd_mask<float, simd_abi::sse>(_mm_cmplt_ps(m_value, other.m_value));
  }
};

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sse> sqrt(simd<float, simd_abi::sse> const& a) noexcept {
  return simd<float, simd_abi::sse>(_mm_sqrt_ps(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sse> max(
    simd<float, simd_abi::sse> const& a, simd<float, simd_abi::sse> const& b) noexcept {
  return simd<float, simd_abi::sse>(_mm_max_ps(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sse> min(
    simd<float, simd_abi::sse> const& a, simd<float, simd_abi::sse> const& b) noexcept {
  return simd<float, simd_abi::sse>(_mm_min_ps(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sse> choose(
    simd_mask<float, simd_abi::sse> const& a, simd<float, simd_abi::sse> const& b, simd<float, simd_abi::sse> const& c) noexcept {
  return simd<float, simd_abi::sse>(_mm_add_ps(_mm_and_ps(a.get(), b.get()), _mm_andnot_ps(a.get(), c.get())));
}

#endif

#ifdef __SSE2__

template <>
class simd_mask<double, simd_abi::sse> {
  __m128d m_value;
 public:
  SIMD_ALWAYS_INLINE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 4; }
  SIMD_ALWAYS_INLINE simd_mask(bool value_in) noexcept
    :m_value(value_in ? _mm_cmpeq_pd(_mm_setzero_pd(),_mm_setzero_pd()) : _mm_setzero_pd())
  {}
  SIMD_ALWAYS_INLINE constexpr simd_mask(__m128d const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE constexpr __m128d get() const noexcept { return m_value; }
};

template <>
class simd<double, simd_abi::sse> {
  __m128d m_value;
 public:
  SIMD_ALWAYS_INLINE simd() noexcept = default;
  using value_type = double;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 2; }
  SIMD_ALWAYS_INLINE simd(double value) noexcept
    :m_value(_mm_set1_pd(value))
  {}
  template <class Flags>
  SIMD_ALWAYS_INLINE simd(double const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE constexpr simd(__m128d const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE simd operator*(simd const& other) const noexcept {
    return simd(_mm_mul_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator/(simd const& other) const noexcept {
    return simd(_mm_div_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator+(simd const& other) const noexcept {
    return simd(_mm_add_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator-(simd const& other) const noexcept {
    return simd(_mm_sub_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE void copy_from(double const* ptr, element_aligned_tag) noexcept {
    m_value = _mm_loadu_pd(ptr);
  }
  SIMD_ALWAYS_INLINE void copy_to(double* ptr, element_aligned_tag) const noexcept {
    _mm_storeu_pd(ptr, m_value);
  }
  SIMD_ALWAYS_INLINE constexpr __m128d get() const noexcept { return m_value; }
  SIMD_ALWAYS_INLINE simd_mask<double, simd_abi::sse> operator<(simd const& other) const noexcept {
    return simd_mask<double, simd_abi::sse>(_mm_cmplt_pd(m_value, other.m_value));
  }
};

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sse> sqrt(simd<double, simd_abi::sse> const& a) noexcept {
  return simd<double, simd_abi::sse>(_mm_sqrt_pd(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sse> max(
    simd<double, simd_abi::sse> const& a, simd<double, simd_abi::sse> const& b) noexcept {
  return simd<double, simd_abi::sse>(_mm_max_pd(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sse> min(
    simd<double, simd_abi::sse> const& a, simd<double, simd_abi::sse> const& b) noexcept {
  return simd<double, simd_abi::sse>(_mm_min_pd(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sse> choose(
    simd_mask<double, simd_abi::sse> const& a, simd<double, simd_abi::sse> const& b, simd<double, simd_abi::sse> const& c) noexcept {
  return simd<double, simd_abi::sse>(_mm_add_pd(_mm_and_pd(a.get(), b.get()), _mm_andnot_pd(a.get(), c.get())));
}

#endif

#ifdef __AVX__

namespace simd_abi {

class avx {};

}

template <>
class simd_mask<float, simd_abi::avx> {
  __m256 m_value;
 public:
  SIMD_ALWAYS_INLINE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 8; }
  SIMD_ALWAYS_INLINE simd_mask(bool value_in) noexcept
    :m_value(value_in ? _mm256_cmp_ps(_mm256_setzero_ps(),_mm256_setzero_ps(),_CMP_EQ_OQ) : _mm256_setzero_ps())
  {}
  SIMD_ALWAYS_INLINE constexpr simd_mask(__m256 const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE constexpr __m256 get() const noexcept { return m_value; }
};

template <>
class simd<float, simd_abi::avx> {
  __m256 m_value;
 public:
  SIMD_ALWAYS_INLINE simd() noexcept = default;
  using value_type = float;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 8; }
  SIMD_ALWAYS_INLINE simd(float value) noexcept
    :m_value(_mm256_set1_ps(value))
  {}
  template <class Flags>
  SIMD_ALWAYS_INLINE simd(float const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE constexpr simd(__m256 const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE simd operator*(simd const& other) const noexcept {
    return simd(_mm256_mul_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator/(simd const& other) const noexcept {
    return simd(_mm256_div_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator+(simd const& other) const noexcept {
    return simd(_mm256_add_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator-(simd const& other) const noexcept {
    return simd(_mm256_sub_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE void copy_from(float const* ptr, element_aligned_tag) noexcept {
    m_value = _mm256_loadu_ps(ptr);
  }
  SIMD_ALWAYS_INLINE void copy_to(float* ptr, element_aligned_tag) const noexcept {
    _mm256_storeu_ps(ptr, m_value);
  }
  SIMD_ALWAYS_INLINE constexpr __m256 get() const noexcept { return m_value; }
  SIMD_ALWAYS_INLINE simd_mask<float, simd_abi::avx> operator<(simd const& other) const noexcept {
    return simd_mask<float, simd_abi::avx>(_mm256_cmp_ps(m_value, other.m_value, _CMP_LT_OS));
  }
};

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx> sqrt(simd<float, simd_abi::avx> const& a) noexcept {
  return simd<float, simd_abi::avx>(_mm256_sqrt_ps(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx> max(
    simd<float, simd_abi::avx> const& a, simd<float, simd_abi::avx> const& b) noexcept {
  return simd<float, simd_abi::avx>(_mm256_max_ps(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx> min(
    simd<float, simd_abi::avx> const& a, simd<float, simd_abi::avx> const& b) noexcept {
  return simd<float, simd_abi::avx>(_mm256_min_ps(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx> choose(
    simd_mask<float, simd_abi::avx> const& a, simd<float, simd_abi::avx> const& b, simd<float, simd_abi::avx> const& c) noexcept {
  return simd<float, simd_abi::avx>(_mm256_blendv_ps(c.get(), b.get(), a.get()));
}

template <>
class simd_mask<double, simd_abi::avx> {
  __m256d m_value;
 public:
  SIMD_ALWAYS_INLINE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 4; }
  SIMD_ALWAYS_INLINE simd_mask(bool value_in) noexcept
    :m_value(value_in ? _mm256_cmp_pd(_mm256_setzero_pd(),_mm256_setzero_pd(),_CMP_EQ_OQ) : _mm256_setzero_pd())
  {}
  SIMD_ALWAYS_INLINE constexpr simd_mask(__m256d const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE constexpr __m256d get() const noexcept { return m_value; }
};

template <>
class simd<double, simd_abi::avx> {
  __m256d m_value;
 public:
  SIMD_ALWAYS_INLINE simd() noexcept = default;
  using value_type = double;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 4; }
  SIMD_ALWAYS_INLINE simd(double value) noexcept
    :m_value(_mm256_set1_pd(value))
  {}
  template <class Flags>
  SIMD_ALWAYS_INLINE simd(double const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE constexpr simd(__m256d const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE simd operator*(simd const& other) const noexcept {
    return simd(_mm256_mul_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator/(simd const& other) const noexcept {
    return simd(_mm256_div_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator+(simd const& other) const noexcept {
    return simd(_mm256_add_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator-(simd const& other) const noexcept {
    return simd(_mm256_sub_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE void copy_from(double const* ptr, element_aligned_tag) noexcept {
    m_value = _mm256_loadu_pd(ptr);
  }
  SIMD_ALWAYS_INLINE void copy_to(double* ptr, element_aligned_tag) const noexcept {
    _mm256_storeu_pd(ptr, m_value);
  }
  SIMD_ALWAYS_INLINE constexpr __m256d get() const noexcept { return m_value; }
  SIMD_ALWAYS_INLINE simd_mask<double, simd_abi::avx> operator<(simd const& other) const noexcept {
    return simd_mask<double, simd_abi::avx>(_mm256_cmp_pd(m_value, other.m_value, _CMP_LT_OS));
  }
};

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx> sqrt(simd<double, simd_abi::avx> const& a) noexcept {
  return simd<double, simd_abi::avx>(_mm256_sqrt_pd(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx> max(
    simd<double, simd_abi::avx> const& a, simd<double, simd_abi::avx> const& b) noexcept {
  return simd<double, simd_abi::avx>(_mm256_max_pd(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx> min(
    simd<double, simd_abi::avx> const& a, simd<double, simd_abi::avx> const& b) noexcept {
  return simd<double, simd_abi::avx>(_mm256_min_pd(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx> choose(
    simd_mask<double, simd_abi::avx> const& a, simd<double, simd_abi::avx> const& b, simd<double, simd_abi::avx> const& c) noexcept {
  return simd<double, simd_abi::avx>(_mm256_blendv_pd(c.get(), b.get(), a.get()));
}

#endif

#ifdef __AVX512F__

namespace simd_abi {

class avx512 {};

}

template <>
class simd_mask<float, simd_abi::avx512> {
  __mmask16 m_value;
 public:
  SIMD_ALWAYS_INLINE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 16; }
  SIMD_ALWAYS_INLINE constexpr simd_mask(__mmask16 const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE constexpr __mmask16 get() const noexcept { return m_value; }
};

template <>
class simd<float, simd_abi::avx512> {
  __m512 m_value;
 public:
  SIMD_ALWAYS_INLINE simd() noexcept = default;
  using value_type = float;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 16; }
  SIMD_ALWAYS_INLINE simd(float value) noexcept
    :m_value(_mm512_set1_ps(value))
  {}
  template <class Flags>
  SIMD_ALWAYS_INLINE simd(float const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE constexpr simd(__m512 const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE simd operator*(simd const& other) const noexcept {
    return simd(_mm512_mul_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator/(simd const& other) const noexcept {
    return simd(_mm512_div_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator+(simd const& other) const noexcept {
    return simd(_mm512_add_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator-(simd const& other) const noexcept {
    return simd(_mm512_sub_ps(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE void copy_from(float const* ptr, element_aligned_tag) noexcept {
    m_value = _mm512_loadu_ps(ptr);
  }
  SIMD_ALWAYS_INLINE void copy_to(float* ptr, element_aligned_tag) const noexcept {
    _mm512_storeu_ps(ptr, m_value);
  }
  SIMD_ALWAYS_INLINE constexpr __m512 get() const noexcept { return m_value; }
  SIMD_ALWAYS_INLINE simd_mask<float, simd_abi::avx512> operator<(simd const& other) const noexcept {
    return simd_mask<float, simd_abi::avx512>(_mm512_cmp_ps_mask(m_value, other.m_value, _CMP_LT_OS));
  }
};

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> sqrt(simd<float, simd_abi::avx512> const& a) noexcept {
  return simd<float, simd_abi::avx512>(_mm512_sqrt_ps(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> max(
    simd<float, simd_abi::avx512> const& a, simd<float, simd_abi::avx512> const& b) noexcept {
  return simd<float, simd_abi::avx512>(_mm512_max_ps(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> min(
    simd<float, simd_abi::avx512> const& a, simd<float, simd_abi::avx512> const& b) noexcept {
  return simd<float, simd_abi::avx512>(_mm512_min_ps(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::avx512> choose(
    simd_mask<float, simd_abi::avx512> const& a, simd<float, simd_abi::avx512> const& b, simd<float, simd_abi::avx512> const& c) noexcept {
  return simd<float, simd_abi::avx512>(_mm512_mask_blend_ps(a.get(), c.get(), b.get()));
}

template <>
class simd_mask<double, simd_abi::avx512> {
  __mmask8 m_value;
 public:
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 8; }
  SIMD_ALWAYS_INLINE constexpr simd_mask(__mmask8 const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE constexpr __mmask8 get() const noexcept { return m_value; }
};

template <>
class simd<double, simd_abi::avx512> {
  __m512d m_value;
 public:
  SIMD_ALWAYS_INLINE simd() noexcept = default;
  using value_type = double;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 8; }
  SIMD_ALWAYS_INLINE simd(double value) noexcept
    :m_value(_mm512_set1_pd(value))
  {}
  template <class Flags>
  SIMD_ALWAYS_INLINE simd(double const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE constexpr simd(__m512d const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE simd operator*(simd const& other) const noexcept {
    return simd(_mm512_mul_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator/(simd const& other) const noexcept {
    return simd(_mm512_div_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator+(simd const& other) const noexcept {
    return simd(_mm512_add_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator-(simd const& other) const noexcept {
    return simd(_mm512_sub_pd(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE void copy_from(double const* ptr, element_aligned_tag) noexcept {
    m_value = _mm512_loadu_pd(ptr);
  }
  SIMD_ALWAYS_INLINE void copy_to(double* ptr, element_aligned_tag) const noexcept {
    _mm512_storeu_pd(ptr, m_value);
  }
  SIMD_ALWAYS_INLINE constexpr __m512d get() const noexcept { return m_value; }
  SIMD_ALWAYS_INLINE simd_mask<double, simd_abi::avx512> operator<(simd const& other) const noexcept {
    return simd_mask<double, simd_abi::avx512>(_mm512_cmp_pd_mask(m_value, other.m_value, _CMP_LT_OS));
  }
};

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> sqrt(simd<double, simd_abi::avx512> const& a) noexcept {
  return simd<double, simd_abi::avx512>(_mm512_sqrt_pd(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> max(
    simd<double, simd_abi::avx512> const& a, simd<double, simd_abi::avx512> const& b) noexcept {
  return simd<double, simd_abi::avx512>(_mm512_max_pd(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> min(
    simd<double, simd_abi::avx512> const& a, simd<double, simd_abi::avx512> const& b) noexcept {
  return simd<double, simd_abi::avx512>(_mm512_min_pd(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::avx512> choose(
    simd_mask<double, simd_abi::avx512> const& a, simd<double, simd_abi::avx512> const& b, simd<double, simd_abi::avx512> const& c) noexcept {
  return simd<double, simd_abi::avx512>(_mm512_mask_blend_pd(a.get(), c.get(), b.get()));
}

#endif

#ifdef __ARM_NEON

namespace simd_abi {

class neon {};

}

template <>
class simd_mask<float, simd_abi::neon> {
  uint32x4_t m_value;
 public:
  SIMD_ALWAYS_INLINE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 4; }
  SIMD_ALWAYS_INLINE constexpr simd_mask(uint32x4_t const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE constexpr uint32x4_t get() const noexcept { return m_value; }
};

template <>
class simd<float, simd_abi::neon> {
  float32x4_t m_value;
 public:
  SIMD_ALWAYS_INLINE simd() noexcept = default;
  using value_type = float;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 4; }
  SIMD_ALWAYS_INLINE simd(float value) noexcept
    :m_value(vdupq_n_f32(value))
  {}
  template <class Flags>
  SIMD_ALWAYS_INLINE simd(float const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE constexpr simd(float32x4_t const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE simd operator*(simd const& other) const noexcept {
    return simd(vmulq_f32(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator/(simd const& other) const noexcept {
    return simd(vdivq_f32(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator+(simd const& other) const noexcept {
    return simd(vaddq_f32(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator-(simd const& other) const noexcept {
    return simd(vsubq_f32(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE void copy_from(float const* ptr, element_aligned_tag) noexcept {
    m_value = vld1q_f32(ptr);
  }
  SIMD_ALWAYS_INLINE void copy_to(float* ptr, element_aligned_tag) const noexcept {
    vst1q_f32(ptr, m_value);
  }
  SIMD_ALWAYS_INLINE constexpr float32x4_t get() const noexcept { return m_value; }
  SIMD_ALWAYS_INLINE simd_mask<float, simd_abi::neon> operator<(simd const& other) const noexcept {
    return simd_mask<float, simd_abi::neon>(vcltq_f32(m_value, other.m_value));
  }
};

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::neon> sqrt(simd<float, simd_abi::neon> const& a) noexcept {
  return simd<float, simd_abi::neon>(vsqrtq_f32(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::neon> max(
    simd<float, simd_abi::neon> const& a, simd<float, simd_abi::neon> const& b) noexcept {
  return simd<float, simd_abi::neon>(vmaxq_f32(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::neon> min(
    simd<float, simd_abi::neon> const& a, simd<float, simd_abi::neon> const& b) noexcept {
  return simd<float, simd_abi::neon>(vminq_f32(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::neon> choose(
    simd_mask<float, simd_abi::neon> const& a, simd<float, simd_abi::neon> const& b, simd<float, simd_abi::neon> const& c) noexcept {
  return simd<float, simd_abi::neon>(
    vreinterpretq_f32_u32(
      vbslq_u32(
        a.get(),
        vreinterpretq_u32_f32(b.get()),
        vreinterpretq_u32_f32(c.get()))));
}

template <>
class simd_mask<double, simd_abi::neon> {
  uint64x2_t m_value;
 public:
  SIMD_ALWAYS_INLINE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 4; }
  SIMD_ALWAYS_INLINE constexpr simd_mask(uint64x2_t const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE constexpr uint64x2_t get() const noexcept { return m_value; }
};

template <>
class simd<double, simd_abi::neon> {
  float64x2_t m_value;
 public:
  SIMD_ALWAYS_INLINE simd() noexcept = default;
  using value_type = double;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 2; }
  SIMD_ALWAYS_INLINE simd(double value) noexcept
    :m_value(vdupq_n_f64(value))
  {}
  template <class Flags>
  SIMD_ALWAYS_INLINE simd(double const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE constexpr simd(float64x2_t const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE simd operator*(simd const& other) const noexcept {
    return simd(vmulq_f64(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator/(simd const& other) const noexcept {
    return simd(vdivq_f64(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator+(simd const& other) const noexcept {
    return simd(vaddq_f64(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator-(simd const& other) const noexcept {
    return simd(vsubq_f64(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE void copy_from(double const* ptr, element_aligned_tag) noexcept {
    m_value = vld1q_f64(ptr);
  }
  SIMD_ALWAYS_INLINE void copy_to(double* ptr, element_aligned_tag) const noexcept {
    vst1q_f64(ptr, m_value);
  }
  SIMD_ALWAYS_INLINE constexpr float64x2_t get() const noexcept { return m_value; }
  SIMD_ALWAYS_INLINE simd_mask<double, simd_abi::neon> operator<(simd const& other) const noexcept {
    return simd_mask<double, simd_abi::neon>(vcltq_f64(m_value, other.m_value));
  }
};

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::neon> sqrt(simd<double, simd_abi::neon> const& a) noexcept {
  return simd<double, simd_abi::neon>(vsqrtq_f64(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::neon> max(
    simd<double, simd_abi::neon> const& a, simd<double, simd_abi::neon> const& b) noexcept {
  return simd<double, simd_abi::neon>(vmaxq_f64(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::neon> min(
    simd<double, simd_abi::neon> const& a, simd<double, simd_abi::neon> const& b) noexcept {
  return simd<double, simd_abi::neon>(vminq_f64(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::neon> choose(
    simd_mask<double, simd_abi::neon> const& a, simd<double, simd_abi::neon> const& b, simd<double, simd_abi::neon> const& c) noexcept {
  return simd<double, simd_abi::neon>(
    vreinterpretq_f64_u64(
      vbslq_u64(
        a.get(),
        vreinterpretq_u64_f64(b.get()),
        vreinterpretq_u64_f64(c.get()))));
}

#endif

#ifdef __VSX__

namespace simd_abi {

class vsx {};

}

template <>
class simd_mask<float, simd_abi::vsx> {
  __vector __bool int m_value;
 public:
  SIMD_ALWAYS_INLINE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 4; }
  SIMD_ALWAYS_INLINE constexpr simd_mask(__vector __bool int const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE constexpr __vector __bool int get() const noexcept { return m_value; }
};

template <>
class simd<float, simd_abi::vsx> {
  __vector float m_value;
 public:
  SIMD_ALWAYS_INLINE simd() noexcept = default;
  using value_type = float;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 4; }
  SIMD_ALWAYS_INLINE simd(float value) noexcept
    :m_value(vec_splats(value))
  {}
  template <class Flags>
  SIMD_ALWAYS_INLINE simd(float const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE constexpr simd(__vector float const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE simd operator*(simd const& other) const noexcept {
    return simd(vec_mul(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator/(simd const& other) const noexcept {
    return simd(vec_div(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator+(simd const& other) const noexcept {
    return simd(vec_add(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator-(simd const& other) const noexcept {
    return simd(vec_sub(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE void copy_from(float const* ptr, element_aligned_tag) noexcept {
    m_value = vec_vsx_ld(0, ptr);
  }
  SIMD_ALWAYS_INLINE void copy_to(float* ptr, element_aligned_tag) const noexcept {
    vec_vsx_st(m_value, 0, ptr);
  }
  SIMD_ALWAYS_INLINE constexpr __vector float get() const noexcept { return m_value; }
  SIMD_ALWAYS_INLINE simd_mask<float, simd_abi::vsx> operator<(simd const& other) const noexcept {
    return simd_mask<float, simd_abi::vsx>(vec_cmplt(m_value, other.m_value));
  }
};

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::vsx> sqrt(simd<float, simd_abi::vsx> const& a) noexcept {
  return simd<float, simd_abi::vsx>(vec_sqrt(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::vsx> max(
    simd<float, simd_abi::vsx> const& a, simd<float, simd_abi::vsx> const& b) noexcept {
  return simd<float, simd_abi::vsx>(vec_max(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::vsx> min(
    simd<float, simd_abi::vsx> const& a, simd<float, simd_abi::vsx> const& b) noexcept {
  return simd<float, simd_abi::vsx>(vec_min(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::vsx> choose(
    simd_mask<float, simd_abi::vsx> const& a, simd<float, simd_abi::vsx> const& b, simd<float, simd_abi::vsx> const& c) noexcept {
  return simd<float, simd_abi::vsx>(vec_sel(c.get(), b.get(), a.get()));
}

template <>
class simd_mask<double, simd_abi::vsx> {
  __vector __bool long long m_value;
 public:
  SIMD_ALWAYS_INLINE simd_mask() noexcept = default;
  using value_type = bool;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 2; }
  SIMD_ALWAYS_INLINE constexpr simd_mask(__vector __bool long long const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE constexpr __vector __bool long long get() const noexcept { return m_value; }
};

template <>
class simd<double, simd_abi::vsx> {
  __vector double m_value;
 public:
  SIMD_ALWAYS_INLINE simd() noexcept = default;
  using value_type = double;
  SIMD_ALWAYS_INLINE static constexpr int size() noexcept { return 2; }
  SIMD_ALWAYS_INLINE simd(double value) noexcept
    :m_value(vec_splats(value))
  {}
  template <class Flags>
  SIMD_ALWAYS_INLINE simd(double const* ptr, Flags flags) noexcept {
    copy_from(ptr, flags);
  }
  SIMD_ALWAYS_INLINE constexpr simd(__vector double const& value_in) noexcept
    :m_value(value_in)
  {}
  SIMD_ALWAYS_INLINE simd operator*(simd const& other) const noexcept {
    return simd(vec_mul(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator/(simd const& other) const noexcept {
    return simd(vec_div(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator+(simd const& other) const noexcept {
    return simd(vec_add(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE simd operator-(simd const& other) const noexcept {
    return simd(vec_sub(m_value, other.m_value));
  }
  SIMD_ALWAYS_INLINE void copy_from(double const* ptr, element_aligned_tag) noexcept {
    m_value = vec_vsx_ld(0, ptr);
  }
  SIMD_ALWAYS_INLINE void copy_to(double* ptr, element_aligned_tag) const noexcept {
    vec_vsx_st(m_value, 0, ptr);
  }
  SIMD_ALWAYS_INLINE constexpr __vector double get() const noexcept { return m_value; }
  SIMD_ALWAYS_INLINE simd_mask<double, simd_abi::vsx> operator<(simd const& other) const noexcept {
    return simd_mask<double, simd_abi::vsx>(vec_cmplt(m_value, other.m_value));
  }
};

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::vsx> sqrt(simd<double, simd_abi::vsx> const& a) noexcept {
  return simd<double, simd_abi::vsx>(vec_sqrt(a.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::vsx> max(
    simd<double, simd_abi::vsx> const& a, simd<double, simd_abi::vsx> const& b) noexcept {
  return simd<double, simd_abi::vsx>(vec_max(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::vsx> min(
    simd<double, simd_abi::vsx> const& a, simd<double, simd_abi::vsx> const& b) noexcept {
  return simd<double, simd_abi::vsx>(vec_min(a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::vsx> choose(
    simd_mask<double, simd_abi::vsx> const& a, simd<double, simd_abi::vsx> const& b, simd<double, simd_abi::vsx> const& c) noexcept {
  return simd<double, simd_abi::vsx>(vec_sel(c.get(), b.get(), a.get()));
}

#endif

namespace simd_abi {

#if defined(__AVX512F__)
using native = avx512;
#elif defined(__AVX__)
using native = avx;
#elif defined(__SSE2__)
using native = sse;
#elif defined(__ARM_NEON)
using native = neon;
#elif defined(__VSX__)
using native = vsx;
#elif defined(HPC_PRAGMA_SIMD)
using native = directive<256 / 8>;
#else
using native = scalar;
#endif

}

}
