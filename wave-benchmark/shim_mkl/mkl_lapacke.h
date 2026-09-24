// LAPACKE_dgels replacement: least squares min ||b - A x||_2 for a full-rank tall system,
// row-major, trans = 'N', nrhs = 1, which is the only way XIndex and FINEdex call it.
// On return b[0..n-1] holds x. Returns i > 0 when column i (1-based) is linearly dependent,
// matching LAPACK's "the i-th diagonal element of R is zero", which XIndex uses to drop
// features. The common case, one feature plus a bias column, is solved as a centered
// regression in long double, which is exact up to rounding and stable for 64-bit keys.
#pragma once
#include <cmath>
#include <vector>
#ifndef LAPACK_ROW_MAJOR
#define LAPACK_ROW_MAJOR 101
#endif
inline int LAPACKE_dgels(int layout, char trans, int m, int n, int nrhs, double *a, int lda,
                         double *b, int ldb) {
  if (layout != LAPACK_ROW_MAJOR || (trans != 'N' && trans != 'n') || nrhs != 1 || m < 1 ||
      n < 1)
    return -1;
  (void)ldb;
  auto A = [&](int i, int j) -> long double { return a[(size_t)i * lda + j]; };
  // One feature and a constant column (in either order).
  if (n == 2) {
    int fc = -1, bc = -1;
    for (int j = 0; j < 2; j++) {
      bool constant = true;
      for (int i = 1; i < m && constant; i++) constant = (A(i, j) == A(0, j));
      if (constant && A(0, j) != 0) bc = j; else fc = j;
    }
    if (bc >= 0 && fc >= 0) {
      long double mx = 0, my = 0;
      for (int i = 0; i < m; i++) { mx += A(i, fc); my += b[i]; }
      mx /= m; my /= m;
      long double sxx = 0, sxy = 0;
      for (int i = 0; i < m; i++) {
        long double dx = A(i, fc) - mx;
        sxx += dx * dx;
        sxy += dx * (b[i] - my);
      }
      if (sxx == 0) return fc + 1;
      long double slope = sxy / sxx, icpt = (my - slope * mx) / A(0, bc);
      b[fc] = (double)slope;
      b[bc] = (double)icpt;
      return 0;
    }
  }
  // General case: Householder QR in long double.
  std::vector<long double> R((size_t)m * n), y(b, b + m), cn(n, 0);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      R[(size_t)i * n + j] = A(i, j);
      cn[j] += A(i, j) * A(i, j);
    }
  for (int j = 0; j < n; j++) cn[j] = std::sqrt(cn[j]);
  for (int k = 0; k < n && k < m; k++) {
    long double norm = 0;
    for (int i = k; i < m; i++) norm += R[(size_t)i * n + k] * R[(size_t)i * n + k];
    norm = std::sqrt(norm);
    if (norm <= 1e-12L * cn[k]) return k + 1;  // column k depends on the earlier ones
    long double alpha = R[(size_t)k * n + k] > 0 ? -norm : norm;
    std::vector<long double> v(m, 0);
    v[k] = R[(size_t)k * n + k] - alpha;
    for (int i = k + 1; i < m; i++) v[i] = R[(size_t)i * n + k];
    long double vnorm = 0;
    for (int i = k; i < m; i++) vnorm += v[i] * v[i];
    if (vnorm == 0) continue;
    for (int j = k; j < n; j++) {
      long double s = 0;
      for (int i = k; i < m; i++) s += v[i] * R[(size_t)i * n + j];
      s = 2 * s / vnorm;
      for (int i = k; i < m; i++) R[(size_t)i * n + j] -= s * v[i];
    }
    long double s = 0;
    for (int i = k; i < m; i++) s += v[i] * y[i];
    s = 2 * s / vnorm;
    for (int i = k; i < m; i++) y[i] -= s * v[i];
  }
  if (m < n) return n;  // underdetermined: XIndex never asks for this
  for (int k = n - 1; k >= 0; k--) {
    long double d = R[(size_t)k * n + k];
    if (std::fabs(d) <= 1e-12L * cn[k]) return k + 1;
    long double s = y[k];
    for (int j = k + 1; j < n; j++) s -= R[(size_t)k * n + j] * y[j];
    y[k] = s / d;
  }
  for (int k = 0; k < n; k++) b[k] = (double)y[k];
  return 0;
}
