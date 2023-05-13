#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

const char* dgemv_desc = "OpenMP dgemv.";

void my_dgemv(int n, double* A, double* x, double* y) {
   switch (mode) {
      case 0: // my_dgemv_outer
         #pragma omp parallel for
         for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
               sum += A[i * n + j] * x[j];
            }
            #pragma omp atomic
            y[i] += sum;
         }
         break;
      case 1: // my_dgemv_inner
         for (int i = 0; i < n; i++) {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (int j = 0; j < n; j++) {
               sum += A[i * n + j] * x[j];
            }
            y[i] += sum;
         }
         break;
      case 2: // my_dgemv_nested
         #pragma omp parallel for
         for (int i = 0; i < n; i++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < n; j++) {
               sum += A[i * n + j] * x[j];
            }
            #pragma omp atomic
            y[i] += sum;
         }
         break;
      default:
         printf("Invalid mode!\n");
   }
}
