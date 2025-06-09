#include <stdio.h>

#include "model.h"

int main(void) {
    printf("Hello, Neural Networks!\n");

    constexpr int layerLengths[] = {32*32, 200, 100, 10};
    constexpr int layerCount = sizeof(layerLengths) / sizeof(int);
    constexpr double learningRate = 0.5;

    Model* myImageMLP = newModel(layerLengths, layerCount, learningRate, "Categorical Cross Entropy");

    return 0;
}