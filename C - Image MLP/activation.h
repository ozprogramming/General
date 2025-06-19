#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "library.h"

Matrix* relu(Matrix* input);
Matrix* leakyRelu(Matrix* input);
Matrix* sigmoid(Matrix* input);
Matrix* hypTan(Matrix* input);
Matrix* softmax(Matrix* input);

Matrix* d_relu(Matrix* output);
Matrix* d_leakyRelu(Matrix* output);
Matrix* d_sigmoid(Matrix* output);
Matrix* d_hypTan(Matrix* output);
Matrix* d_softmax(Matrix* output);

#endif //ACTIVATION_H
