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
#include "TestHelpers.hpp"
#include <simd.hpp>

namespace Test {

template<class ScalarType>
void test_abs(Kokkos::View<simd_t<ScalarType> *> data) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::abs(data(i)); });
}

template<class ScalarType>
void test_sqrt(Kokkos::View<simd_t<ScalarType> *> data) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::sqrt(data(i)); });
}

template<class ScalarType>
void test_cbrt(Kokkos::View<simd_t<ScalarType> *> data) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::cbrt(data(i)); });
}

template<class ScalarType>
void test_exp(Kokkos::View<simd_t<ScalarType> *> data) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::exp(data(i)); });
}

template<class ScalarType>
void test_fma(Kokkos::View<simd_t<ScalarType> *> data, simd_t<ScalarType> val_1, simd_t<ScalarType> val_2) {
  Kokkos::parallel_for(
      data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::fma(data(i), val_1, val_2);
      });
}

template<typename ScalarType>
void test_max(Kokkos::View<simd_t<ScalarType> *> data, simd_t<ScalarType> val) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::max(data(i), val); });
}

template<class ScalarType>
void test_min(Kokkos::View<simd_t<ScalarType> *> data, simd_t<ScalarType> val) {
  Kokkos::parallel_for(
      data.extent(0),
      KOKKOS_LAMBDA(const int i) { data(i) = simd::min(data(i), val); });
}

template<class ScalarType>
void test_copysign(Kokkos::View<simd_t<ScalarType> *> data, simd_t<ScalarType> sng) {
  Kokkos::parallel_for(
      data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::copysign(data(i), sng);
      });
}

template<class ScalarType>
void test_multiplysign(Kokkos::View<simd_t<ScalarType> *> data, simd_t<ScalarType> sng) {
  Kokkos::parallel_for(
      data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = simd::multiplysign(data(i), sng);
      });
}

template<class ScalarType>
void test_op_add(Kokkos::View<simd_t<ScalarType> *> data, simd_t<ScalarType> val) {
  Kokkos::parallel_for(
      data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = data(i) + val;
      });
}

template<class ScalarType>
void test_op_sub(Kokkos::View<simd_t<ScalarType> *> data, simd_t<ScalarType> val) {
  Kokkos::parallel_for(
      data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = data(i) - val;
      });
}

template<class ScalarType>
void test_op_mul(Kokkos::View<simd_t<ScalarType> *> data, simd_t<ScalarType> val) {
  Kokkos::parallel_for(
      data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = data(i) * val;
      });
}

template<class ScalarType>
void test_op_div(Kokkos::View<simd_t<ScalarType> *> data, simd_t<ScalarType> val) {
  Kokkos::parallel_for(
      data.extent(0), KOKKOS_LAMBDA(const int i) {
        data(i) = data(i) / val;
      });
}

template<typename ScalarType>
void test_view_result(const std::string &test_name, Kokkos::View<simd_t<ScalarType> *> data,
                      ScalarType expected) {
  auto data_h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(data_h, data);


  Kokkos::View<ScalarType *, Kokkos::HostSpace> scalar_view(
      reinterpret_cast<ScalarType *>(data_h.data()),
      data_h.extent(0) * simd_t<ScalarType>::size());

  for (size_t i = 0; i < scalar_view.extent(0); ++i) {
    compare(test_name, i, scalar_view(i), static_cast<ScalarType>(expected));
  }
}

template<typename ScalarType>
void test_view_result(const std::string &test_name, Kokkos::View<simd_t<ScalarType> *> data,
                      ScalarType *expected) {
  auto data_h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(data_h, data);

  Kokkos::View<ScalarType *, Kokkos::HostSpace> scalar_view(
      reinterpret_cast<ScalarType *>(data_h.data()),
      data_h.extent(0) * simd_t<ScalarType>::size());

  for (size_t i = 0; i < scalar_view.extent(0); ++i) {
    compare(test_name, i, scalar_view(i), static_cast<ScalarType>(expected[i]));
  }
}

template<class ScalarType>
void do_test_constructor(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();
    auto data = create_data_zero_to_size_positive<ScalarType>("Test View", viewExtent, constructor);

    ScalarType *expectedData = new ScalarType[expectedSize];
    for(int i = 0; i < viewExtent; ++i) {

        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(i);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(i + j);
            }
        }
    }
    test_view_result("test_test_constructor_1", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_abs(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, -4);

    test_abs(data);
    test_view_result("test_abs_1", data, static_cast<ScalarType>(4.0));

    data = create_data_zero_to_size_positive<ScalarType>("Test View 2", viewExtent, constructor);


    ScalarType *expectedData = new ScalarType[expectedSize];
    for(int i = 0; i < viewExtent; ++i) {

        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(i);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(i + j);
            }
        }
    }

    test_abs(data);
    test_view_result("test_abs_2", data, expectedData);

    data = create_data_zero_to_size_negative<ScalarType>("Test View 3", viewExtent, constructor);

    test_abs(data);
    test_view_result("test_abs_3", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_sqrt(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, 16.0);

    test_sqrt(data);
    test_view_result("test_sqrt_1", data, static_cast<ScalarType>(4.0));

    data = create_data_sqrt<ScalarType>("Test View 2", viewExtent, constructor);
    ScalarType *expectedData = new ScalarType[expectedSize];
    int k = 0;
    for(int i = 0; i < viewExtent; ++i) {

        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(k);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(k + j);
            }
        }
        k >= sqrtMaxIndex ? k = 0 : ++k;
    }

    test_sqrt(data);
    test_view_result("test_sqrt_2", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_cbrt(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, 27.0);

    test_cbrt(data);
    test_view_result("test_cbrt_1", data, static_cast<ScalarType>(3.0));

    data = create_data_cbrt<ScalarType>("Test View 2", viewExtent, constructor);
    ScalarType *expectedData = new ScalarType[expectedSize];
    int k = 0;
    for(int i = 0; i < viewExtent; ++i) {

        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(k);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(k + j);
            }
        }
        k >= cbrtMaxIndex ? k = 0 : ++k;
    }

    test_cbrt(data);
    test_view_result("test_cbrt_2", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_exp(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, 1.0);

    test_exp(data);
    test_view_result("test_exp_1", data, static_cast<ScalarType>(std::exp(1.0)));

    data = create_data_zero_to_size_positive<ScalarType>("Test View 2", viewExtent, constructor);
    ScalarType *expectedData = new ScalarType[expectedSize];
    for(int i = 0; i < viewExtent; ++i) {
        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(std::exp(i));
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(std::exp(i + j));
            }
        }
    }

    test_exp(data);
    test_view_result("test_exp_2", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_fma(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, -4.0);

    test_fma(data, simd_t<ScalarType>{2.0}, simd_t<ScalarType>{5.0});
    test_view_result("test_fma_1", data, static_cast<ScalarType>(-3.0));

    data = create_data_zero_to_size_positive<ScalarType>("Test View 2", viewExtent, constructor);
    ScalarType *expectedData = new ScalarType[expectedSize];
    for(int i = 0; i < viewExtent; ++i) {
        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(7.0 + 4*i);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(7.0 + 4*(i + j));
            }
        }
    }

    test_fma(data, simd_t<ScalarType>{4.0}, simd_t<ScalarType>{7.0});
    test_view_result("test_fma_2", data, expectedData);

    data = create_data_zero_to_size_negative<ScalarType>("Test View 2", viewExtent, constructor);
    for(int i = 0; i < viewExtent; ++i) {

        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(7.0 - 4*i);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(7.0 - 4*(i + j));
            }
        }
    }

    test_fma(data, simd_t<ScalarType>{4.0}, simd_t<ScalarType>{7.0});
    test_view_result("test_fma_3", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_max(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, 1.0);

    test_max(data, simd_t<ScalarType>{10.0});
    test_view_result("test_max_1", data, static_cast<ScalarType>(10.0));

    data = create_data_sqrt<ScalarType>("Test View 2", viewExtent, constructor);
    ScalarType *expectedData = new ScalarType[expectedSize];
    int k = 0;
    for(int i = 0; i < viewExtent; ++i) {

        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                if( k  < 4) {
                    expectedData[i * simdSize + j] = static_cast<ScalarType>(10.0);
                } else {
                    expectedData[i * simdSize + j] = static_cast<ScalarType>(k * k);
                }
            } else {
                if( (k + j)  < 4) {
                    expectedData[i * simdSize + j] = static_cast<ScalarType>(10.0);
                } else {
                    expectedData[i * simdSize + j] = static_cast<ScalarType>((k + j) * (k + j));
                }
            }
        }
        k >= sqrtMaxIndex ? k = 0 : ++k;
    }

    test_max(data, simd_t<ScalarType>{10.0});
    test_view_result("test_max_2", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_min(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, 4.0);

    test_min(data, simd_t<ScalarType>{1.0});
    test_view_result("test_min_1", data, static_cast<ScalarType>(1.0));

    data = create_data_sqrt<ScalarType>("Test View 2", viewExtent, constructor);
    ScalarType *expectedData = new ScalarType[expectedSize];
    int k = 0;
    for(int i = 0; i < viewExtent; ++i) {

        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                if( k  < 4) {
                    expectedData[i * simdSize + j] = static_cast<ScalarType>(k * k);
                } else {
                    expectedData[i * simdSize + j] = static_cast<ScalarType>(10.0);
                }
            } else {
                if( (k + j)  < 4) {
                    expectedData[i * simdSize + j] = static_cast<ScalarType>((k + j) * (k + j));
                } else {
                    expectedData[i * simdSize + j] = static_cast<ScalarType>(10.0);
                }
            }
        }

        k >= sqrtMaxIndex ? k = 0 : ++k;
    }

    test_min(data, simd_t<ScalarType>{10.0});
    test_view_result("test_min_2", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_op_add(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, 4.0);
    test_op_add(data, simd_t<ScalarType>{10.0});
    test_view_result("test_op_add_1", data, static_cast<ScalarType>(14.0));

    data = create_data_zero_to_size_positive<ScalarType>("Test View 2", viewExtent, constructor);
    ScalarType *expectedData = new ScalarType[expectedSize];
    for(int i = 0; i < viewExtent; ++i) {
        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(4.0 + i);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(4.0 + i + j);
            }
        }
    }

    test_op_add(data, simd_t<ScalarType>{4.0});
    test_view_result("test_op_add_2", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_op_sub(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, 4.0);

    test_op_sub(data, simd_t<ScalarType>{10.0});
    test_view_result("test_op_sub_1", data, static_cast<ScalarType>(-6.0));

    data = create_data_zero_to_size_positive<ScalarType>("Test View 2", viewExtent, constructor);
    ScalarType *expectedData = new ScalarType[expectedSize];
    for(int i = 0; i < viewExtent; ++i) {
        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(-4.0 + i);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(-4.0 + i + j);
            }
        }
    }

    test_op_sub(data, simd_t<ScalarType>{4.0});
    test_view_result("test_op_sub_1", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_op_mul(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, 4.0);

    test_op_mul(data, simd_t<ScalarType>{10.0});
    test_view_result("test_op_mul_1", data, static_cast<ScalarType>(40.0));

    data = create_data_zero_to_size_positive<ScalarType>("Test View 2", viewExtent, constructor);
    ScalarType *expectedData = new ScalarType[expectedSize];
    for(int i = 0; i < viewExtent; ++i) {
        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(4.0 * i);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(4.0 * (i + j));
            }
        }
    }

    test_op_mul(data, simd_t<ScalarType>{4.0});
    test_view_result("test_op_mul_2", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_op_div(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, 4.0);

    test_op_div(data, simd_t<ScalarType>{10.0});
    test_view_result("test_op_div_1", data, static_cast<ScalarType>(0.4));

    data = create_data_zero_to_size_positive<ScalarType>("Test View 2", viewExtent, constructor);
    ScalarType *expectedData = new ScalarType[expectedSize];
    for(int i = 0; i < viewExtent; ++i) {
        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(i / 4.0);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>((i + j) / 4.0);
            }
        }
    }

    test_op_div(data, simd_t<ScalarType>{4.0});
    test_view_result("test_op_div_2", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_copysign(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_zero_to_size_positive<ScalarType>("Test View", viewExtent, constructor);
    test_copysign(data, simd_t<ScalarType>{2.0});
    ScalarType *expectedData = new ScalarType[expectedSize];
    for(int i = 0; i < viewExtent; ++i) {

        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(i);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(i + j);
            }
        }
    }
    test_view_result("test_copysign_1", data, expectedData);

    test_copysign(data, simd_t<ScalarType>{-2.0});
    for(int i = 0; i < expectedSize; ++i) {
        expectedData[i] = -expectedData[i];
    }
    test_view_result("test_copysign_2", data, expectedData);

    test_copysign(data, simd_t<ScalarType>{4.0});
    for(int i = 0; i < expectedSize; ++i) {
        expectedData[i] = -expectedData[i];
    }
    test_view_result("test_copysign_3", data, expectedData);
    delete[] expectedData;
}

template<typename ScalarType>
void do_test_multiplysign(SIMD_CONSTRUCTOR constructor, int viewExtent) {
    const int simdSize = simd_t<ScalarType>::size();
    const int expectedSize = viewExtent *  simd_t<ScalarType>::size();

    auto data = create_data_with_uniq_value<simd_t<ScalarType>>("Test View", viewExtent, 4.0);

    test_multiplysign(data, simd_t<ScalarType>{2.0});
    test_view_result("test_multiplysign_1", data, static_cast<ScalarType>(4.0));

    test_multiplysign(data, simd_t<ScalarType>{-2.0});
    test_view_result("test_multiplysign_2", data, static_cast<ScalarType>(-4.0));

    test_multiplysign(data, simd_t<ScalarType>{2.0});
    test_view_result("test_multiplysign_3", data, static_cast<ScalarType>(-4.0));

    data = create_data_zero_to_size_positive<ScalarType>("Test View", viewExtent, constructor);
    test_multiplysign(data, simd_t<ScalarType>{2.0});
    ScalarType *expectedData = new ScalarType[expectedSize];
    for(int i = 0; i < viewExtent; ++i) {

        for(int j = 0; j < simdSize; ++j) {
            if(constructor == SIMD_CONSTRUCTOR_SCALAR || constructor == SIMD_CONSTRUCTOR_STORAGE) {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(i);
            } else {
                expectedData[i * simdSize + j] = static_cast<ScalarType>(i + j);
            }
        }
    }
    test_view_result("test_multiplysign_4", data, expectedData);

    test_multiplysign(data, simd_t<ScalarType>{-2.0});
    for(int i = 0; i < expectedSize; ++i) {
        expectedData[i] = -expectedData[i];
    }
    test_view_result("test_multiplysign_5", data, expectedData);

    test_multiplysign(data, simd_t<ScalarType>{-4.0});
    for(int i = 0; i < expectedSize; ++i) {
        expectedData[i] = -expectedData[i];
    }
    test_view_result("test_multiplysign_6", data, expectedData);

    test_multiplysign(data, simd_t<ScalarType>{4.0});
    test_view_result("test_multiplysign_7", data, expectedData);
    delete[] expectedData;
}

TEST(simd_math, test_constructor_1) {

    for (auto extent: extentArray) {
        do_test_constructor<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_constructor<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_constructor<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_constructor<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_constructor<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_constructor<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_constructor<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_constructor<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_constructor<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_constructor<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_constructor<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_constructor<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
 }

TEST(simd_math, test_abs) {
    for (auto extent: extentArray) {
        do_test_abs<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_abs<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_abs<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_abs<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_abs<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_abs<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_abs<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_abs<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_abs<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_abs<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_abs<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_abs<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
 }

TEST(simd_math, test_sqrt) {
    for (auto extent: extentArray) {
        do_test_sqrt<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_sqrt<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_sqrt<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_sqrt<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_sqrt<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_sqrt<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_sqrt<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_sqrt<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_sqrt<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_sqrt<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_sqrt<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_sqrt<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

TEST(simd_math, test_cbrt) {
    for (auto extent: extentArray) {
        do_test_cbrt<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_cbrt<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_cbrt<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_cbrt<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_cbrt<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_cbrt<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_cbrt<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_cbrt<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_cbrt<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_cbrt<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_cbrt<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_cbrt<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

TEST(simd_math, test_exp) {
    for (auto extent: extentArray) {
        do_test_exp<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_exp<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_exp<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_exp<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_exp<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_exp<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_exp<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_exp<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_exp<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_exp<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_exp<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_exp<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

TEST(simd_math, test_fma) {
    for (auto extent: extentArray) {
        do_test_fma<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_fma<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_fma<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_fma<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_fma<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_fma<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_fma<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_fma<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_fma<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_fma<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_fma<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_fma<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

TEST(simd_math, test_max) {
    for (auto extent: extentArray) {
        do_test_max<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_max<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_max<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_max<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_max<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_max<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_max<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_max<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_max<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_max<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_max<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_max<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

TEST(simd_math, test_min) {
    for (auto extent: extentArray) {
        do_test_min<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_min<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_min<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_min<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_min<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_min<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_min<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_min<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_min<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_min<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_min<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_min<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

TEST(simd_math, test_op_add) {
    for (auto extent: extentArray) {
        do_test_op_add<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_op_add<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_op_add<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_op_add<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_op_add<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_op_add<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_op_add<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_op_add<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_op_add<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_op_add<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_op_add<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_op_add<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

TEST(simd_math, test_op_sub) {
    for (auto extent: extentArray) {
        do_test_op_sub<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_op_sub<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_op_sub<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_op_sub<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_op_sub<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_op_sub<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_op_sub<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_op_sub<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_op_sub<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_op_sub<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_op_sub<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_op_sub<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

TEST(simd_math, test_op_mul) {
    for (auto extent: extentArray) {
        do_test_op_mul<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_op_mul<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_op_mul<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_op_mul<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_op_mul<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_op_mul<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_op_mul<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_op_mul<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_op_mul<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_op_mul<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_op_mul<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_op_mul<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

TEST(simd_math, test_op_div) {
    for (auto extent: extentArray) {
        do_test_op_div<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_op_div<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_op_div<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_op_div<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_op_div<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_op_div<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_op_div<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_op_div<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_op_div<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_op_div<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_op_div<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_op_div<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

TEST(simd_math, test_copysign) {
    for (auto extent: extentArray) {
        do_test_copysign<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_copysign<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_copysign<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_copysign<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_copysign<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_copysign<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_copysign<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_copysign<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_copysign<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_copysign<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_copysign<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_copysign<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

TEST(simd_math, test_multiplysign) {
    for (auto extent: extentArray) {
        do_test_multiplysign<double>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_multiplysign<float>(SIMD_CONSTRUCTOR_SCALAR, extent);
        do_test_multiplysign<double>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_multiplysign<float>(SIMD_CONSTRUCTOR_MUTLI_SCALAR, extent);
        do_test_multiplysign<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_multiplysign<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_FLAG, extent);
        do_test_multiplysign<double>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_multiplysign<float>(SIMD_CONSTRUCTOR_MUTLI_PTR_STRIDE, extent);
        do_test_multiplysign<double>(SIMD_CONSTRUCTOR_STORAGE, extent);
        do_test_multiplysign<float>(SIMD_CONSTRUCTOR_STORAGE, extent);
#ifndef SIMD_USING_SCALAR_ABI
        do_test_multiplysign<double>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
        do_test_multiplysign<float>(SIMD_CONSTRUCTOR_SPECIALIZED, extent);
#endif
    }
}

}  // namespace Test
