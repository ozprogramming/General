#include <stdlib.h>
#include <math.h>

#include "activation.h"
#include "loss.h"
#include "model.h"

#include <string.h>

/* LAYER OPERATION */

static Layer* newLayer(const int inputLength, const int outputLength, char* activation)
{
    Layer* layer = malloc(sizeof(Layer));

    layer->bias = 1.0;
    layer->d_bias = 0.0;
    layer->weights = newMatrix(inputLength, outputLength, 1.0);
    layer->d_weights = newMatrix(inputLength, outputLength, 0.0);
    layer->activation = activation;
    randomizeEntries(layer->weights, -1.0, 1.0);

    return layer;
}

static void clearLayer(Layer* layer)
{
    delMatrix(layer->input);
    delMatrix(layer->output);

    layer->input = nullptr;
    layer->output = nullptr;
}

static void delLayer(Layer* layer)
{
    delMatrix(layer->weights);
    delMatrix(layer->input);
    delMatrix(layer->output);

    free(layer);
}

static Matrix* layerFeed(Matrix* input, Layer* layer)
{
    layer->input = input;

    layer->output = matrixMultiplication(layer->weights, layer->input);

    for (int i = 0; i < layer->output->rows; ++i)
    {
        layer->output->entries[i][0] = layer->output->entries[i][0] + layer->bias;
    }

    leakyReLU(layer->output);

    return layer->output;
}

static Matrix* layerPropagate(Matrix* input, Layer* layer)
{

}

/* MODEL OPERATION */

Model* newModel(const int* layerLengths, const int layerCount, const double learningRate, char* loss)
{
    Model* model = malloc(sizeof(Model));

    model->length = layerCount - 1;
    model->learningRate = learningRate;
    model->layers = (Layer**) malloc(sizeof(Layer*) * layerCount);

    for (int i = 0; i < model->length - 2; ++i)
    {
        model->layers[i] = (Layer*) malloc(sizeof(Layer));
        model->layers[i] = newLayer(layerLengths[i], layerLengths[i + 1], "Leaky ReLU");
    }

    model->layers[model->length - 1] = (Layer*) malloc(sizeof(Layer));
    model->layers[model->length - 1] = newLayer(layerLengths[model->length - 1], layerLengths[model->length], "Softmax");

    model->d_Error = (Matrix*) malloc(sizeof(Matrix));

    model->loss = loss;

    return model;
}

static void clearModel(Model* model)
{
    for (int i = 0; i < model->length; ++i)
    {
        clearLayer(model->layers[i]);
    }
}

void delModel(Model* model)
{
    for (int i = 0; i < model->length; ++i)
    {
        delLayer(model->layers[i]);
    }

    free(model);
}

static Matrix* feedForward(Matrix* input, Model* model)
{
    Matrix* output = layerFeed(input, model->layers[0]);

    for (int i = 1; i < model->length; ++i)
    {
        output = layerFeed(output, model->layers[i]);
    }

    return output;
}

static void propagateBackward(Matrix* actual, Model* model)
{
    Matrix* prediction = model->layers[model->length - 1]->output;

    if (model->d_Error != nullptr)
    {
        delMatrix(model->d_Error);
        model->d_Error = nullptr;
    }

    if (strcmp("Mean Squared", model->loss) != 0)
    {
        model->d_Error = d_meanSquaredError(prediction, actual);
    }
    else if (strcmp("Categorical Cross Entropy", model->loss) != 0)
    {
        model->d_Error = d_categoricalCrossEntropyError(prediction, actual);
    }

    for (int i = 0; i < model->length; ++i)
    {
        layerPropagate(prediction, model->layers[i]);
    }
}

/* MODEL USAGE */

void train()
{}

double test()
{}

void predict()
{}

/* MODEL DISPLAY */

static void printLayer(Layer* layer)
{
    Matrix* horizontalLayer = matrixTranspose(layer->input);

    printMatrixGrid(horizontalLayer);

    delMatrix(horizontalLayer);
}

void printModel(Model* model)
{
    for (int i = 0; i < model->length; ++i)
    {
        printLayer(model->layers[i]);
    }
}