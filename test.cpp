#include <iostream>
#include <iomanip>

#include "simd.hpp"

int main() {
  double aa[16];
  double ab[16];
  double ac[16];
  double ad[16];
  for (int i = 0; i < 16; ++i) {
    aa[i] = 1.0 * (i + 1);
    ab[i] = 0.5 * (i + 1);
    ac[i] = 0.1 * (i + 1);
    ad[i] = 0.0;
  }
  simd::simd<double, simd::simd_abi::native> sa;
  simd::simd<double, simd::simd_abi::native> sb;
  simd::simd<double, simd::simd_abi::native> sc;
  simd::simd<double, simd::simd_abi::native> sd;
  sa.copy_from(aa, simd::element_aligned_tag());
  sb.copy_from(ab, simd::element_aligned_tag());
  sc.copy_from(ac, simd::element_aligned_tag());
  simd::simd_mask<double, simd::simd_abi::native> ma(true);
  sd = simd::choose(ma, cbrt(sa), sb - sc);
  sd.copy_to(ad, simd::element_aligned_tag());
  std::cout << std::setprecision(6);
  for (int i = 0; i < 16; ++i) {
    std::cout << ad[i] << '\n';
  }
}
