#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

const char* dgemv_desc = "OpenMP dgemv.";

void my_dgemv_outer(int n, double* A, double* x, double* y) {
   #pragma omp parallel for
   for (int i = 0; i < n; i++) {
      double sum = 0.0;
      for (int j = 0; j < n; j++) {
         sum += A[i*n+j] * x[j];
      }
      #pragma omp atomic
      y[i] += sum;
   }
}

void my_dgemv_inner(int n, double* A, double* x, double* y) {
   for (int i = 0; i < n; i++) {
      double sum = 0.0;
      #pragma omp parallel for reduction(+:sum)
      for (int j = 0; j < n; j++) {
         sum += A[i*n+j] * x[j];
      }
      y[i] += sum;
   }
}

void my_dgemv_nested(int n, double* A, double* x, double* y) {
   #pragma omp parallel for
   for (int i = 0; i < n; i++) {
      double sum = 0.0;
      #pragma omp simd reduction(+:sum)
      for (int j = 0; j < n; j++) {
         sum += A[i*n+j] * x[j];
      }
      #pragma omp atomic
      y[i] += sum;
   }
}
