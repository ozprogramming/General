#include <tgmath.h>

#include "library.h"
#include "loss.h"

double meanSquaredError(Matrix* prediction, Matrix* actual)
{
    double error = 0.0;

    for (int i = 0; i < actual->rows; ++i)
    {
        error += pow(actual->entries[i][0] - prediction->entries[i][0], 2);
    }

    error /= actual->rows;

    return error;
}

double categoricalCrossEntropyError(Matrix* prediction, Matrix* actual)
{
    double error = 0.0;

    for (int i = 0; i < actual->rows; ++i)
    {
        error += actual->entries[i][0] * (double) logl(prediction->entries[i][0]);
    }

    error *= -1;

    return error;
}

Matrix* d_meanSquaredError(Matrix* prediction, Matrix* actual)
{
    Matrix* error = newMatrix(actual->rows, 1, 0.0);

    for (int i = 0; i < error->rows; ++i)
    {
        error->entries[i][0] = 2 * (prediction->entries[i][0] - actual->entries[i][0]) / actual->rows;
    }

    return error;
}

Matrix* d_categoricalCrossEntropyError(Matrix* prediction, Matrix* actual)
{
    Matrix* error = newMatrix(actual->rows, 1, 0.0);

    for (int i = 0; i < error->rows; ++i)
    {
        error->entries[i][0] = - actual->entries[i][0] / prediction->entries[i][0];
    }

    return error;
}