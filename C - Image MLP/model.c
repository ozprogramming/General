#include "model.h"

#include <math.h>
#include <stdio.h>

#include "activation.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "loss.h"

void knuthShuffle(int* array, int length)
{
    for (int i = length - 1; i > 0; --i)
    {
        const int randomIndex = rand() % i;
        const int elementOne = array[i];
        const int elementTwo = array[randomIndex];
        array[i] = elementTwo;
        array[randomIndex] = elementOne;
    }
}

MLP* newMLP(const int layerCount, int* layerLengths, double learningRate, char** activation, char* loss, char* optimizer)
{
    MLP* model = malloc(sizeof(MLP));

    model->length = layerCount;

    model->d_weights = (Matrix**) malloc(sizeof(Matrix*) * (model->length - 1));
    model->weights = (Matrix**) malloc(sizeof(Matrix*) * (model->length - 1));

    model->d_bias = (double*) malloc(sizeof(double) * (model->length - 1));
    model->bias = (double*) malloc(sizeof(double) * (model->length - 1));

    for (int i = 0; i < model->length - 1; ++i)
    {
        model->d_weights[i] = newMatrix(layerLengths[i + 1], layerLengths[i], 0.0);
        model->weights[i] = newMatrix(layerLengths[i + 1], layerLengths[i], 0.0);
        randomizeEntries(model->weights[i], -1, 1);

        model->d_bias[i] = 0;
        model->bias[i] = 0;
    }

    model->activation = activation;
    model->loss = loss;
    model->optimizer = optimizer;
    model->learningRate = learningRate;

    return model;
}

void delMLP(MLP* model)
{
    for (int i = 0; i < model->length - 1; ++i)
    {
        delMatrix(model->d_weights[i]);
        delMatrix(model->weights[i]);
    }

    free(model->d_weights);
    free(model->weights);
    free(model->d_bias);
    free(model->bias);
    free(model);
}

static Matrix** feedForward(Matrix* input, MLP* model)
{
    Matrix** activations = malloc(sizeof(Matrix *) * model->length);

    activations[0] = copyMatrix(input);

    for (int i = 1; i < model->length; ++i)
    {
        Matrix* z;

        Matrix* raw = matrixMultiplication(model->weights[i - 1], activations[i - 1]);
        z = matrixScalarOperation(raw, model->bias[i - 1], '+');
        delMatrix(raw);

        if (strcmp(model->activation[i - 1], "ReLU") == 0)
        {
            activations[i] = relu(z);
        }
        else if (strcmp(model->activation[i - 1], "Leaky ReLU") == 0)
        {
            activations[i] = leakyRelu(z);
        }
        else if (strcmp(model->activation[i - 1], "Sigmoid") == 0)
        {
            activations[i] = sigmoid(z);
        }
        else if (strcmp(model->activation[i - 1], "Tanh") == 0)
        {
            activations[i] = hypTan(z);
        }
        else if (strcmp(model->activation[i - 1], "Softmax") == 0)
        {
            activations[i] = softmax(z);
        }

        delMatrix(z);
    }

    return activations;
}

static void propBackward(Matrix** activations, Matrix* expected, MLP* model)
{
    // Get initial error derivatives

    Matrix* result = activations[model->length - 1];
    Matrix* d_error;

    if (strcmp(model->loss, "Categorical Cross Entropy") == 0)
    {
        d_error = d_categoricalCrossEntropyError(result, expected);
    }
    else if (strcmp(model->loss, "Mean Squared") == 0)
    {
        d_error = d_meanSquaredError(result, expected);
    }

    /* BACK PROPAGATION */

    const int numWeights = model->length - 1;

    for (int w = numWeights - 1; w >= 0; --w)
    {
        /* Get activation-to-input derivative vector of current layer */

        // Compute input

        Matrix* raw = matrixMultiplication(model->weights[w], activations[w]);
        Matrix* z = matrixScalarOperation(raw, model->bias[w], '+');
        delMatrix(raw);

        Matrix* d_activation;

        // Enter input into differentiated activation function

        if (strcmp(model->activation[w], "ReLU") == 0)
        {
            d_activation = d_relu(z);
        }
        else if (strcmp(model->activation[w], "Leaky ReLU") == 0)
        {
            d_activation = d_leakyRelu(z);
        }
        else if (strcmp(model->activation[w], "Sigmoid") == 0)
        {
            d_activation = d_sigmoid(z);
        }
        else if (strcmp(model->activation[w], "Tanh") == 0)
        {
            d_activation = d_hypTan(z);
        }
        else if (strcmp(model->activation[w], "Softmax") == 0)
        {
            d_activation = d_softmax(z);
        }

        delMatrix(z);

        // #1: Compute error-to-weight derivatives
        // Equation: dE_dw(i,j)(L) = dE_da(i)(L) * da(i)(L)_dz(i)(L) * dz(i)(L)_dw(i,j)(L)
        // Computation: dE_dw(i,j)(L) = dE_da(i)(L) * activation'(input) * activation(L - 1)
        // Save dE_da(i)(L) * da(i)(L)_dz(i)(L) into vector for back propagation

        Matrix* backPropCalc = hadamardProduct(d_error, d_activation);
        Matrix* backActivations = matrixTranspose(activations[w]);
        Matrix* d_weights = matrixMultiplication(backPropCalc, backActivations);
        Matrix* adjusted_d_weights = matrixAddition(model->d_weights[w], d_weights);

        delMatrix(backActivations);
        delMatrix(d_activation);
        delMatrix(d_weights);
        delMatrix(model->d_weights[w]);

        model->d_weights[w] = copyMatrix(adjusted_d_weights);

        delMatrix(adjusted_d_weights);

        // #2: Compute error-to-activation derivatives
        // Using vector of saved back propagation calculations
        // Equation: dE_da(j)(L - 1) = [SUM(n(L) - 1, i = 0)]{dE_da(i)(L) * da(i)(L)_dz(i)(L) * dz(i)(L)_da(j)(L - 1)}
        // Computation: dE_da(j)(L - 1) = [SUM(n(L) - 1, i = 0)]{bpc(i) * w(i,j)(L)}
        // Save error-to-activation derivatives in error derivative vector

        delMatrix(d_error);

        Matrix* transposeWeights = matrixTranspose(model->weights[w]);
        d_error = matrixMultiplication(transposeWeights, backPropCalc);
        delMatrix(transposeWeights);
        delMatrix(backPropCalc);
    }

    delMatrix(d_error);

    // Delete activations parameter

    for (int i = 0; i < model->length; ++i)
    {
        delMatrix(activations[i]);
        free(activations[i]);
    }

    free(activations);
}

static void applyGradients(MLP* model, const int accumulationSteps)
{
    const double learningRate = model->learningRate;

    for (int w = 0; w < model->length - 1; ++w)
    {
        Matrix* d_weights = model->d_weights[w];
        Matrix* weights = model->weights[w];

        for (int i = 0; i < weights->rows; ++i)
        {
            for (int j = 0; j < weights->cols; ++j)
            {
                const double dw = d_weights->entries[i][j];

                weights->entries[i][j] = weights->entries[i][j] - learningRate * dw / accumulationSteps;
            }
        }

        clearMatrix(d_weights);
    }
}

void fitMLP(Matrix** X_Train, Matrix** y_Train, MLP* model, const int numInputs, const int batchSize, const int epochs, const bool shuffle)
{
    for (int e = 0; e < epochs; ++e)
    {
        int indexArray[numInputs];

        for (int i = 0; i < numInputs; ++i)
        {
            indexArray[i] = i;
        }

        if (shuffle)
        {
            knuthShuffle(indexArray, numInputs);
        }

        const int numFullBatches = numInputs / batchSize;
        const int remainingBatchSize = numInputs % batchSize;

        for (int b = 0; b < numFullBatches; ++b)
        {
            int index = b * batchSize;

            for (int i = 0; i < batchSize; ++i)
            {
                const int randomIndex = indexArray[index];

                Matrix** activations = feedForward(X_Train[randomIndex], model);

                propBackward(activations, y_Train[randomIndex], model);

                ++index;
            }

            applyGradients(model, batchSize);
        }

        for (int i = 0; i < remainingBatchSize; ++i)
        {
            const int index = numFullBatches * batchSize + i;
            const int randomIndex = indexArray[index];

            Matrix** activations = feedForward(X_Train[randomIndex], model);
            propBackward(activations, y_Train[randomIndex], model);

            if (i == remainingBatchSize - 1)
            {
                applyGradients(model, remainingBatchSize);
            }
        }
    }
}

Matrix* outputMLP(Matrix* input, MLP* model)
{
    Matrix** activations = feedForward(input, model);
    Matrix* output = activations[model->length - 1];

    for (int i = 0; i < model->length - 1; ++i)
    {
        delMatrix(activations[i]);
    }

    free(activations);

    return output;
}