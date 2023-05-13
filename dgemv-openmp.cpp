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