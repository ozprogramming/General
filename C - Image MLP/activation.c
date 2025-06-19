#include <math.h>

#include "activation.h"

#include <stdlib.h>

Matrix* relu(Matrix* input)
{
    Matrix* output = newMatrix(input->rows, 1, 0.0);

    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];

        output->entries[i][0] = x > 0 ? x : 0;
    }

    return output;
}

Matrix* leakyRelu(Matrix* input)
{
    Matrix* output = newMatrix(input->rows, 1, 0.0);

    constexpr double alpha = 0.25;

    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];

        output->entries[i][0] = x > 0 ? x : alpha * x;
    }

    return output;
}

Matrix* sigmoid(Matrix* input)
{
    Matrix* output = newMatrix(input->rows, 1, 0.0);

    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];

        output->entries[i][0] = 1.0 / (1.0 + exp(-x));
    }

    return output;
}

Matrix* hypTan(Matrix* input)
{
    Matrix* output = newMatrix(input->rows, 1, 0.0);

    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];

        output->entries[i][0] = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    return output;
}

Matrix* softmax(Matrix* input)
{
    Matrix* output = newMatrix(input->rows, 1, 0.0);

    double D = 0.0;

    for (int i = 0; i < input->rows; ++i)
    {
        const double d = input->entries[i][0];

        if (d > D)
        {
            D = d;
        }
    }

    D *= -1;

    double sum = 0.0;

    for (int i = 0; i < input->rows; ++i)
    {
        sum += exp(input->entries[i][0] + D);
    }

    for (int i = 0; i < input->rows; ++i)
    {
        output->entries[i][0] = exp(input->entries[i][0] + D) / sum;
    }

    return output;
}

Matrix* d_relu(Matrix* input)
{
    Matrix* d_input = newMatrix(input->rows, 1, 0);

    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];
        d_input->entries[i][0] = x > 0 ? 1 : 0;
    }

    return d_input;
}

Matrix* d_leakyRelu(Matrix* input)
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

Matrix* d_hypTan(Matrix* input)
{
    Matrix* output = newMatrix(input->rows, 1, 0.0);

    for (int i = 0; i < input->rows; ++i)
    {
        const double x = input->entries[i][0];

        output->entries[i][0] = 4.0 / pow(exp(x) + exp(-x), 2.0);
    }

    return output;
}

Matrix* d_softmax(Matrix* input)
{
    Matrix* soft = softmax(input);

    Matrix* output = newMatrix(soft->rows, 1, 0.0);

    for (int i = 0; i < soft->rows; ++i)
    {
        const double s = soft->entries[i][0];

        output->entries[i][0] = s * (1 - s);
    }

    delMatrix(soft);

    return output;
}