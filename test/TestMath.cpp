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

namespace Test{

using simd_t = simd::simd<double, simd::simd_abi::native>;
using mask_t = simd_t::mask_type;

void test_abs(Kokkos::View<simd_t*> data){
    Kokkos::parallel_for(data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::abs(data(i));
    });
}

void test_sqrt(Kokkos::View<simd_t*> data){
    Kokkos::parallel_for(data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::sqrt(data(i));
    });
}

void test_cbrt(Kokkos::View<simd_t*> data){
    Kokkos::parallel_for(data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::cbrt(data(i));
    });
}

void test_exp(Kokkos::View<simd_t*> data){
    Kokkos::parallel_for(data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::exp(data(i));
    });
}

void test_fma(Kokkos::View<simd_t*> data, simd_t val_1, simd_t val_2){
    Kokkos::parallel_for(data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::fma(data(i), val_1, val_2);
    });
}

void test_max(Kokkos::View<simd_t*> data, simd_t val){
    Kokkos::parallel_for(data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::max(data(i), val);
    });
}

void test_min(Kokkos::View<simd_t*> data, simd_t val){
    Kokkos::parallel_for(data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::min(data(i), val);
    });
}



TEST(simd_math, test_abs) {
    Kokkos::View<simd_t*> data("Test View", 8);
    Kokkos::deep_copy(data, simd_t(-4.0));

    test_abs(data);

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);

    for(int i = 0; i < data.extent(0); ++i) {
        EXPECT_EQ(data_h(i).get(), 4.0);
    }
}

TEST(simd_math, test_sqrt){
    Kokkos::View<simd_t*> data("Test View", 8);
    Kokkos::deep_copy(data, simd_t(16.0));

    test_sqrt(data);

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);

    for(int i = 0; i < data.extent(0); ++i) {
        EXPECT_EQ(data_h(i).get(), 4.0);
    }
}

TEST(simd_math, test_cbrt){
    Kokkos::View<simd_t*> data("Test View", 8);
    Kokkos::deep_copy(data, simd_t(27.0));

    test_cbrt(data);

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);

    for(int i = 0; i < data.extent(0); ++i) {
        EXPECT_EQ(data_h(i).get(), 3.0);
    }
}

TEST(simd_math, test_exp){
    Kokkos::View<simd_t*> data("Test View", 8);
    Kokkos::deep_copy(data, simd_t(-4.0));

    test_exp(data);

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);

    for(int i = 0; i < data.extent(0); ++i) {
        EXPECT_EQ(data_h(i).get(), 4.0);
    }
}

TEST(simd_math, test_fma){
    Kokkos::View<simd_t*> data("Test View", 8);
    Kokkos::deep_copy(data, simd_t(-4.0));

    test_fma(data, simd_t{2.0}, simd_t{5.0});

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);

    for(int i = 0; i < data.extent(0); ++i) {
        EXPECT_EQ(data_h(i).get(), -3.0);
    }
}

TEST(simd_math, test_max){
    Kokkos::View<simd_t*> data("Test View", 8);
    Kokkos::deep_copy(data, simd_t(-4.0));

    test_max(data, simd_t{10.0});

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);

    for(int i = 0; i < data.extent(0); ++i) {
        EXPECT_EQ(data_h(i).get(), 10.0);
    }
}

TEST(simd_math, test_min){
    Kokkos::View<simd_t*> data("Test View", 8);
    Kokkos::deep_copy(data, simd_t(4.0));

    test_min(data, simd_t{1.0});

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);

    for(int i = 0; i < data.extent(0); ++i) {
        EXPECT_EQ(data_h(i).get(), 1.0);
    }
}

}