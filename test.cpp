#include <iostream>
#include <iomanip>

#include "simd.hpp"

int main() {
  simd::simd_storage<double, simd::simd_abi::native> aa;
  simd::simd_storage<double, simd::simd_abi::native> ab;
  simd::simd_storage<double, simd::simd_abi::native> ac;
  simd::simd_storage<double, simd::simd_abi::native> ad;
  for (int i = 0; i < simd::simd<double, simd::simd_abi::native>::size(); ++i) {
    aa[i] = 1.0 * (i + 1);
    ab[i] = 0.5 * (i + 1);
    ac[i] = 0.1 * (i + 1);
    ad[i] = 0.0;
  }
  simd::simd<double, simd::simd_abi::native> sa;
  simd::simd<double, simd::simd_abi::native> sb;
  simd::simd<double, simd::simd_abi::native> sc;
  simd::simd<double, simd::simd_abi::native> sd;
  sa = aa;
  sb = ab;
  sc = ac;
  simd::simd_mask<double, simd::simd_abi::native> ma(false);
  sd = simd::choose(ma, cbrt(sa), fma(sa, sa, sc));
  ad = sd;
  std::cout << std::setprecision(6);
  for (int i = 0; i < simd::simd<double, simd::simd_abi::native>::size(); ++i) {
    std::cout << ad[i] << '\n';
  }
}
