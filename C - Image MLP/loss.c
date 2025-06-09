#include <tgmath.h>

#include "library.h"
#include "loss.h"

Matrix* meanSquaredError(Matrix* prediction, Matrix* actual)
{
    Matrix* error = newMatrix(actual->rows, 1, 0.0);

    for (int i = 0; i < error->rows; ++i)
    {
        error->entries[i][0] = 0.5 * pow(actual->entries[i][0] - prediction->entries[i][0], 2);
    }

    return error;
}

Matrix* categoricalCrossEntropyError(Matrix* prediction, Matrix* actual)
{
    Matrix* error = newMatrix(actual->rows, 1, 0.0);

    for (int i = 0; i < error->rows; ++i)
    {
        error->entries[i][0] = - actual->entries[i][0] * (double) log10l(prediction->entries[i][0]);
    }

    return error;
}

Matrix* d_meanSquaredError(Matrix* prediction, Matrix* actual)
{
    return matrixSubtraction(prediction, actual);;
}

Matrix* d_categoricalCrossEntropyError(Matrix* prediction, Matrix* actual)
{
    Matrix* error = newMatrix(actual->rows, 1, 0.0);

    for (int i = 0; i < error->rows; ++i)
    {
        error->entries[i][0] = - actual->entries[i][0] * (double) log10l(prediction->entries[i][0]);
    }

    return error;
}