#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

const char* dgemv_desc = "OpenMP dgemv.";

enum DgemvMode {
  DGEMV_OUTER,
  DGEMV_INNER,
  DGEMV_NESTED
};

void my_dgemv(int n, double* A, double* x, double* y, DgemvMode mode) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    switch (mode) {
      case DGEMV_OUTER:
        for (int j = 0; j < n; j++) {
          sum += A[i * n + j] * x[j];
        }
        break;
      case DGEMV_INNER:
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < n; j++) {
          sum += A[i * n + j] * x[j];
        }
        break;
      case DGEMV_NESTED:
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < n; j++) {
          sum += A[i * n + j] * x[j];
        }
        break;
    }
    #pragma omp atomic
    y[i] += sum;
  }
}

extern "C" {
  void benchmark(int n, double* A, double* x, double* y, int mode) {
    DgemvMode dgemvMode;
    switch (mode) {
      case 0:
        dgemvMode = DGEMV_OUTER;
        break;
      case 1:
        dgemvMode = DGEMV_INNER;
        break;
      case 2:
        dgemvMode = DGEMV_NESTED;
        break;
      default:
        fprintf(stderr, "Invalid mode\n");
        return;
    }

    my_dgemv(n, A, x, y, dgemvMode);
  }
}
