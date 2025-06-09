#ifndef MODEL_H
#define MODEL_H

#include "library.h"

// Layer Structure
typedef struct
{
    double bias;
    double d_bias;
    Matrix* d_weights;
    Matrix* weights;
    Matrix* input;
    Matrix* output;
    char* activation;

} Layer;

// Model Structure
typedef struct
{
    int length;
    double learningRate;
    Layer** layers;
    Matrix* d_Error;
    char* loss;
} Model;

// Model Code Operation
Model* newModel(const int* layerLengths, int layerCount, double learningRate, char* loss);
void delModel(Model* model);

// Model Training, Testing, & Prediction
void train();
double test();
void predict();

// Model Display
void printModel(Model* model);

#endif //MODEL_H
