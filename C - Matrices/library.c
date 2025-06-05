#include "library.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Matrix* newMatrix(const int rows, const int cols, const double init)
{
    Matrix* matrix = malloc(sizeof(Matrix));

    matrix->rows = rows;
    matrix->cols = cols;

    matrix->entries = (double**) malloc(rows * sizeof(double*));

    for (int i = 0; i < rows; i++)
    {
        matrix->entries[i] = (double*) malloc(cols * sizeof(double));

        for (int j = 0; j < cols; j++)
        {
            matrix->entries[i][j] = init;
        }
    }

    return matrix;
}

Matrix* newIdentityMatrix(const int size)
{
    Matrix* matrix = malloc(sizeof(Matrix));

    matrix->rows = size;
    matrix->cols = size;

    matrix->entries = (double**) malloc(size * sizeof(double*));

    for (int i = 0; i < size; i++)
    {
        matrix->entries[i] = (double*) malloc(size * sizeof(double));

        for (int j = 0; j < size; j++)
        {
            if (i == j)
            {
                matrix->entries[i][j] = 1;
            }
            else
            {
                matrix->entries[i][j] = 0;
            }
        }
    }

    return matrix;
}

void delMatrix(Matrix* matrix)
{
    for (int i = 0; i < matrix->rows; i++)
    {
        free(matrix->entries[i]);
    }

    free(matrix->entries);
    free(matrix);
}

Matrix* copyMatrix(Matrix* matrix) {
    Matrix* copy = newMatrix(matrix->rows, matrix->cols, 0);

    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++)
        {
            copy->entries[i][j] = matrix->entries[i][j];
        }
    }

    return copy;
}

Matrix* matrixAddition(Matrix* A, Matrix* B)
{
    Matrix* sum = newMatrix(A->rows, A->cols, 0);

    for (int i = 0; i < A->rows; i++)
    {
        for (int j = 0; j < A->cols; j++)
        {
            sum->entries[i][j] = A->entries[i][j] + B->entries[i][j];
        }
    }

    return sum;
}

Matrix* matrixMultiplication(Matrix* A, Matrix* B)
{
    Matrix* product = newMatrix(A->rows, B->cols, 0);

    for (int i = 0; i < A->rows; i++)
    {
        for (int j = 0; j < B->cols; j++)
        {
            for (int k = 0; k < A->cols; k++)
            {
                product->entries[i][j] += A->entries[i][k] * B->entries[k][j];
            }
        }
    }

    return product;
}

double matrixTrace(Matrix* matrix)
{
    double trace = 0;

    const int minDimension = matrix->rows ? matrix->rows < matrix->cols : matrix->cols;

    for (int i = 0; i < minDimension; i++)
    {
        trace += matrix->entries[i][i];
    }

    return trace;
}

Matrix* matrixTranspose(Matrix* matrix)
{
    Matrix* transpose = newMatrix(matrix->cols, matrix->rows, 0);

    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            transpose->entries[j][i] = matrix->entries[i][j];
        }
    }

    return transpose;
}

Matrix* matrixInverse(Matrix* matrix)
{
    if (matrix->rows != matrix->cols)
    {
        printf("Matrix is not invertible!\n");
        return matrix;
    }

    Matrix* original = copyMatrix(matrix);
    Matrix* inverse = newIdentityMatrix(matrix->rows);

    for (int row = 0; row < original->rows; row++)
    {
        const double pivotPointEntry = original->entries[row][row];

        for (int col = 0; col < original->cols; col++)
        {
            original->entries[row][col] /= pivotPointEntry;
            inverse->entries[row][col] /= pivotPointEntry;
        }

        for (int i = 0; i < inverse->rows; i++)
        {
            if (i != row)
            {
                const double pivotColumnEntry = original->entries[i][row];

                for (int j = 0; j < inverse->cols; j++)
                {
                    original->entries[i][j] -= pivotColumnEntry * original->entries[row][j];
                    inverse->entries[i][j] -= pivotColumnEntry * inverse->entries[row][j];
                }
            }
        }
    }

    delMatrix(original);
    return inverse;
}

void randomizeEntries(Matrix* matrix, const double min, const double max)
{
    srand(time(NULL));

    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            matrix->entries[i][j] = max * ((double) rand() / RAND_MAX) + min;
        }
    }
}

void printMatrixGrid(Matrix* matrix)
{
    printf("(%d x %d matrix)\n", matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            printf("%.2lf ", matrix->entries[i][j]);
        }
        printf("\n");
    }
}

void printMatrixList(Matrix* matrix)
{
    printf("(%d x %d matrix)\n", matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            printf("At (%d, %d): %.2lf\n", i + 1, j + 1, matrix->entries[i][j]);
        }
    }
}