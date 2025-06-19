#ifndef DATASET_H
#define DATASET_H

#include "library.h"
#include "model.h"

typedef struct
{
    char* name;
    int index;
}
Category;

typedef struct
{
    int trainSize;
    int validateSize;
    int testSize;
    char* colorChannels;
    int imageWidth;
    int imageHeight;
    Matrix** X_Train;
    Matrix** y_Train;
    Matrix** X_Validate;
    Matrix** y_Validate;
    Matrix** X_Test;
    Matrix** y_Test;
}
ImageData;

typedef struct
{
    char* datasetLocation;
    int numCategories;
    Category** categories;
    ImageData* data;
}
ImageDataset;

// Code Operation
ImageDataset* createImageDataset(char* datasetLocation, char** categories, int numCategories, char* colorChannels, int imageWidth, int imageHeight);
void delImageDataset(ImageDataset* dataset);

// Load data
Matrix* loadPixelData(char* filePath, ImageDataset* dataset);
Matrix* loadCategoryVector(char* category, ImageDataset* dataset);
void loadImageData(ImageDataset* dataset);

// Dataset attributes for NN
int getImageVectorSize(ImageDataset* dataset);
int getCategoryVectorSize(ImageDataset* dataset);

// Get Image Accuracy
double evaluateImageTestSet(ImageDataset* dataset, MLP* model);
double evaluateImage(char* filePath, char* category, ImageDataset* dataset, MLP* model);

#endif //DATASET_H
