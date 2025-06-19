#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dataset.h"
#include "model.h"

int main(void)
{
    srand(time(nullptr));

    printf("Hello, Neural Networks!\n");

    printf("\nCreating Image Dataset\n");

    char* datasetLocation = "C:\\Users\\Gabe\\Desktop\\Programming\\Datasets\\Flower_Species";
    constexpr int numCategories = 120;
    char* categories[numCategories];

    for (int i = 0; i < numCategories; i++)
    {
        char category[] = "";
        sprintf(category, "%d", i);

        categories[i] = category;
    }

    char* colorChannels = "RGB";
    constexpr int imageWidth = 224;
    constexpr int imageHeight = 224;

    ImageDataset* dataset = createImageDataset(datasetLocation, categories, numCategories, colorChannels, imageWidth, imageHeight);

    printf("\nLoading Image Data to Dataset\n");

    loadImageData(dataset);

    printf("\nCreating MLP\n");

    const int inputSize = getImageVectorSize(dataset);
    const int outputSize = getCategoryVectorSize(dataset);
    int layerLengths[] = {inputSize, 1000, 500, 200, outputSize};
    constexpr int layerCount = sizeof(layerLengths) / sizeof(int);
    double learningRate = 0.001;
    char* activation[] = {"Leaky ReLU", "Leaky ReLU", "Sigmoid", "Softmax"};
    char* loss = "Categorical Cross Entropy";
    char* optimizer = "Mini-Batch Gradient Descent";

    MLP* myMLP = newMLP(layerCount, layerLengths, learningRate, activation, loss, optimizer);

    printf("\nTraining MLP\n");

    constexpr int batchSize = 32;
    constexpr int epochs = 5;
    constexpr bool shuffle = true;

    //*

    Matrix* input = newMatrix(inputSize, 1, 0.0);
    randomizeEntries(input, -10, 10);
    Matrix* output = loadCategoryVector("120", dataset);

    printf("\nACTUAL\n");
    printMatrixGrid(output);

    Matrix* inputs[1] = {input};
    Matrix* outputs[1] = {output};

    printf("\nFIRST INPUT\n");
    Matrix* originalOutput = outputMLP(input, myMLP);
    printMatrixGrid(originalOutput);
    delMatrix(originalOutput);

    fitMLP(inputs, outputs, myMLP, 1, 1, 50, true);

    printf("\nLAST INPUT\n");
    Matrix* endOutput = outputMLP(input, myMLP);
    printMatrixGrid(endOutput);
    delMatrix(endOutput);

    delMatrix(input);
    delMatrix(output);

    /*/

    fitMLP(dataset->data->X_Train, dataset->data->y_Train, myMLP, dataset->data->trainSize, batchSize, epochs, shuffle);

    printf("\nEvaluating MLP\n");

    const double accuracy = evaluateImageTestSet(dataset, myMLP);

    printf("\nThe model is %d%% accurate\n", (int) round(accuracy * 100));

    /*
    const int numTests = 2;
    char* filePaths[] = {"C:\\Users\\Gabe\\Downloads\\test-nine.png", "C:\\Users\\Gabe\\Downloads\\test-five.png"};
    char* testCategories[] = {"9", "5"};

    for (int i = 0; i < numTests; ++i)
    {
        char* filePath = filePaths[i];
        char* category = testCategories[i];

        const double confidence = evaluateImage(filePath, category, dataset, myMLP);

        printf("\nThe model is %d%% confident the image is a \"%s\"\n", (int) round(confidence * 100), category);
    }

    */
    printf("\nDeleting Dataset and MLP\n");

    delImageDataset(dataset);
    delMLP(myMLP);

    printf("\nComplete. Well done!\n");

    return 0;
}