#include <iostream>
using namespace std;

const char* dgemv_desc = "Basic implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
    // implementation of basic matrix multiply
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i*n+j] * x[j];
        }
        y[i] += sum;
    }
}

int main() {
    int n = 3;
    double A[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double x[3] = {1, 2, 3};
    double y[3] = {0, 0, 0};

    my_dgemv(n, A, x, y);

    // print the result vector
    for (int i = 0; i < n; i++) {
        cout << y[i] << " ";
    }
    cout << endl;

    return 0;
}
