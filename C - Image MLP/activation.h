#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "library.h"
#include "model.h"

void ReLU(Matrix* rawOutput);
void leakyReLU(Matrix* rawOutput);
void sigmoid(Matrix* rawOutput);
void softmax(Matrix* rawOutput);

Matrix* d_ReLU(Matrix* output);
Matrix* d_leakyReLU(Matrix* output);
Matrix* d_sigmoid(Matrix* output);
Matrix* d_softmax(Layer* layer);

#endif //ACTIVATION_H
