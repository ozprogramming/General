#ifndef MODEL_H
#define MODEL_H

#include "library.h"

typedef struct
{
    double learningRate;
    double momentum;
    double decay;
    int length;
    Matrix** weights;
    Matrix** d_weights;
    double* bias;
    double* d_bias;
    char** activation;
    char* loss;
    char* optimizer;
}
MLP;

// Code Operation
MLP* newMLP(const int layerCount, int* layerLengths, double learningRate, char** activation, char* loss, char* optimizer);
void delMLP(MLP* model);

// Train
void fitMLP(Matrix** inputs, Matrix** expected, MLP* model, const int numInputs, const int batchSize, const int epochs, const bool shuffle);

// Single Input/Output
Matrix* outputMLP(Matrix* input, MLP* model);

#endif //MODEL_H
