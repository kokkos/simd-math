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

#include <iostream>
#include <iomanip>

#include "simd.hpp"

#define ASSERT_EQ(a, b) \
  if ((a) != (b)) { \
    std::abort(); \
  }

template <class Abi, class T, class BinaryOp>
void test_binary_op(
    T const* a,
    T const* b,
    BinaryOp const& binary_op) {
  simd::simd<T, Abi> native_a(a, simd::element_aligned_tag());
  simd::simd<T, Abi> native_b(b, simd::element_aligned_tag());
  simd::simd<T, Abi> native_answer = binary_op(native_a, native_b);
  constexpr int size = simd::simd<T, Abi>::size();
  using pack_abi = simd::simd_abi::pack<size>;
  simd::simd<T, pack_abi> pack_a(a, simd::element_aligned_tag());
  simd::simd<T, pack_abi> pack_b(b, simd::element_aligned_tag());
  simd::simd<T, pack_abi> pack_answer = binary_op(pack_a, pack_b);
  simd::simd_storage<T, Abi> stored_native_answer(native_answer);
  simd::simd_storage<T, pack_abi> stored_pack_answer(pack_answer);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(stored_native_answer[i], stored_pack_answer[i]);
  }
}

struct plus {
  template <class T>
  T operator()(T const& a, T const& b) const {
    return a + b;
  }
};

struct minus {
  template <class T>
  T operator()(T const& a, T const& b) const {
    return a - b;
  }
};

struct multiplies {
  template <class T>
  T operator()(T const& a, T const& b) const {
    return a * b;
  }
};

struct divides {
  template <class T>
  T operator()(T const& a, T const& b) const {
    return a / b;
  }
};

int main() {
  double const a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  double const b[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
  test_binary_op<simd::simd_abi::native>(a, b, plus());
  test_binary_op<simd::simd_abi::native>(a, b, minus());
  test_binary_op<simd::simd_abi::native>(a, b, multiplies());
  test_binary_op<simd::simd_abi::native>(a, b, divides());
}
