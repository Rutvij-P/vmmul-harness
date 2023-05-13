#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

const char* dgemv_desc = "OpenMP dgemv.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */

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

int main() {
   int problem_sizes[] = {100, 1000, 10000};  // Example problem sizes
   int num_sizes = sizeof(problem_sizes) / sizeof(problem_sizes[0]);

   for (int k = 0; k < num_sizes; k++) {
      int n = problem_sizes[k];

      double* A = (double*)malloc(n * n * sizeof(double));
      double* x = (double*)malloc(n * sizeof(double));
      double* y = (double*)malloc(n * sizeof(double));

      // Initialize A, x, and y with appropriate values

      double start_time, end_time, elapsed_time;

      // Measure the runtime for my_dgemv_outer
      start_time = omp_get_wtime();
      my_dgemv_outer(n, A, x, y);
      end_time = omp_get_wtime();
      elapsed_time = end_time - start_time;
      printf("Runtime (Outer): %f seconds\n", elapsed_time);

      // Measure the runtime for my_dgemv_inner
      start_time = omp_get_wtime();
      my_dgemv_inner(n, A, x, y);
      end_time = omp_get_wtime();
      elapsed_time = end_time - start_time;
      printf("Runtime (Inner): %f seconds\n", elapsed_time);

      // Measure the runtime for my_dgemv_nested
      start_time = omp_get_wtime();
      my_dgemv_nested(n, A, x, y);
      end_time = omp_get_wtime();
      elapsed_time = end_time - start_time;
      printf("Runtime (Nested): %f seconds\n", elapsed_time);

      // Free allocated memory
      free(A);
      free(x);
      free(y);
   }

   return 0;
}

