#include <math.h>

#include "activation.h"

void ReLU(Matrix* input)
{
    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];
        input->entries[i][0] = x > 0 ? x : 0;
    }
}

void leakyReLU(Matrix* input)
{
    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];
        input->entries[i][0] = x > 0 ? x : 0.25 * x;
    }
}

void sigmoid(Matrix* input)
{
    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];
        input->entries[i][0] = 1.0 / (1.0 + exp(-x));
    }
}

void softmax(Matrix* input)
{
    double sum = 0.0;

    for (int i = 0; i < input->rows; ++i)
    {
        sum += exp(input->entries[i][0]);
    }

    for (int i = 0; i < input->rows; ++i)
    {
        input->entries[i][0] = exp(input->entries[i][0]) / sum;
    }
}

Matrix* d_ReLU(Matrix* input)
{
    Matrix* d_input = newMatrix(input->rows, 1, 0);

    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];
        d_input->entries[i][0] = x > 0 ? 1 : 0;
    }

    return d_input;
}

Matrix* d_leakyReLU(Matrix* input)
{
    Matrix* d_input = newMatrix(input->rows, 1, 0);

    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];
        d_input->entries[i][0] = x > 0 ? 1 : 0.25;
    }

    return d_input;
}

Matrix* d_sigmoid(Matrix* input)
{
    Matrix* d_input = newMatrix(input->rows, 1, 0);

    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];
        const double s = (1.0 / (1.0 + exp(-x))) * (1 - (1.0 / (1.0 + exp(-x))));
        d_input->entries[i][0] = s;
    }

    return d_input;
}

Matrix* d_softmax(Matrix* input)
{
    for (int i = 0; i < input->rows; ++i)
    {
        double sum = 0.0;

        for (int j = 0; j < input->rows; ++j)
        {
            sum += (j != i) ? exp(input->entries[j][0]) : ;
        }
    }
}