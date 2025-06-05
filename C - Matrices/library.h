#ifndef MATRICES_LIBRARY_H
#define MATRICES_LIBRARY_H

// Matrix Structure
typedef struct
{
    int rows;
    int cols;
    double** entries;
} Matrix;

// Code Operation
Matrix* newMatrix(int rows, int cols, double init);
void delMatrix(Matrix* matrix);
Matrix* copyMatrix(Matrix* matrix);

// Basic Operations
Matrix* matrixAddition(Matrix *A, Matrix *B);
Matrix* matrixMultiplication(Matrix* A, Matrix* B);
double matrixTrace(Matrix* matrix);

// Advanced Operations
Matrix* matrixTranspose(Matrix* matrix);
Matrix* matrixInverse(Matrix* matrix); // Using Gaussian Elimination
// ~ Eigenvalues


// Decompositions
// ~ LU Decomposition
// ~ QR Decomposition

// Assignment
void randomizeEntries(Matrix* matrix, const double min, const double max);

// Display
void printMatrixGrid(Matrix* matrix);
void printMatrixList(Matrix* matrix);


#endif //MATRICES_LIBRARY_H