#include "dataset.h"
#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define MAX_FILE_PATH 500

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static Category* createCategory(char* name, const int index)
{
    Category* category = malloc(sizeof(Category));

    category->name = name;
    category->index = index;

    return category;
}

static void delCategory(Category* category)
{
    free(category);
}

static ImageData* createImageData(char* colorChannels, const int imageWidth, const int imageHeight)
{
    ImageData* data = malloc(sizeof(ImageData));

    data->colorChannels = colorChannels;
    data->imageWidth = imageWidth;
    data->imageHeight = imageHeight;

    data->trainSize = 0;
    data->testSize = 0;

    data->X_Train = (Matrix**) malloc(0);
    data->y_Train = (Matrix**) malloc(0);
    data->X_Validate = (Matrix**) malloc(0);
    data->y_Validate = (Matrix**) malloc(0);
    data->X_Test = (Matrix**) malloc(0);
    data->y_Test = (Matrix**) malloc(0);

    return data;
}

static void delImageData(ImageData* data)
{
    for (int i = 0; i < data->trainSize; ++i)
    {
        delMatrix(data->X_Train[i]);
        delMatrix(data->y_Train[i]);
    }

    free(data->X_Train);
    free(data->y_Train);

    for (int i = 0; i < data->testSize; ++i)
    {
        delMatrix(data->X_Test[i]);
        delMatrix(data->y_Test[i]);
    }

    free(data->X_Test);
    free(data->y_Test);

    free(data);
}

ImageDataset* createImageDataset(char* datasetLocation, char** categories, const int numCategories, char* colorChannels, const int imageWidth, const int imageHeight)
{
    ImageDataset* dataset = malloc(sizeof(ImageDataset));

    dataset->datasetLocation = datasetLocation;

    dataset->categories = (Category**) malloc(sizeof(Category*) * numCategories);

    dataset->numCategories = numCategories;

    for (int i = 0; i < numCategories; ++i)
    {
        dataset->categories[i] = createCategory(categories[i], i);
    }

    dataset->data = createImageData(colorChannels, imageWidth, imageHeight);

    return dataset;
}

void delImageDataset(ImageDataset* dataset)
{
    for (int i = 0; i < dataset->numCategories; ++i)
    {
        delCategory(dataset->categories[i]);
    }

    delImageData(dataset->data);

    free(dataset->categories);

    free(dataset);
}

Matrix* loadPixelData(char* filePath, ImageDataset* dataset)
{
    int desiredChannels;

    if (strcmp(dataset->data->colorChannels, "Grayscale") == 0)
    {
        desiredChannels = 1;
    }
    else if (strcmp(dataset->data->colorChannels, "RGB") == 0)
    {
        desiredChannels = 3;
    }
    else if (strcmp(dataset->data->colorChannels, "RGBA") == 0)
    {
        desiredChannels = 4;
    }
    else
    {
        return nullptr;
    }

    int width, height, channels;

    // Load image data

    unsigned char* img_data = stbi_load(filePath, &width, &height, &channels, desiredChannels);

    // Validate Image Load

    if (img_data == nullptr)
    {
        printf("Failed to load image\n");
        return nullptr;
    }

    // Validate Dimensions

    if (dataset->data->imageWidth != width || dataset->data->imageHeight != height)
    {
        printf("Invalid dimensions\n");
        return nullptr;
    }

    // Getting input pixel data from image

    Matrix* pixelData = newMatrix(height * width * desiredChannels, 1, 0.0);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int baseIndex = (y * width + x) * desiredChannels;

            for (int i = 0; i < desiredChannels; ++i)
            {
                pixelData->entries[baseIndex + i][0] = (double) img_data[baseIndex + i] / 255.0;
            }
        }
    }

    stbi_image_free(img_data);

    return pixelData;
}

Matrix* loadCategoryVector(char* category, ImageDataset* dataset)
{
    int c;

    for (int i = 0; i < dataset->numCategories; ++i)
    {
        if (strcmp(category, dataset->categories[i]->name) == 0)
        {
            c = i;
            break;
        }
    }

    Matrix* categoryData = newMatrix(dataset->numCategories, 1, 0.0);
    categoryData->entries[dataset->categories[c]->index][0] = 1.0;

    return categoryData;
}

void loadImageData(ImageDataset* dataset)
{
    char* baseDirectory = dataset->datasetLocation;

    dataset->data->trainSize = 0;
    dataset->data->validateSize = 0;
    dataset->data->testSize = 0;

    Matrix** X_Train = malloc(0);
    Matrix** y_Train = malloc(0);
    Matrix** X_Validate = malloc(0);
    Matrix** y_Validate = malloc(0);
    Matrix** X_Test = malloc(0);
    Matrix** y_Test = malloc(0);

    for (int c = 0; c < dataset->numCategories; ++c)
    {
        char directory[MAX_FILE_PATH] = "";
        char search[MAX_FILE_PATH] = "";

        strcat(directory, baseDirectory);
        strcat(directory, "\\");
        strcat(directory, dataset->categories[c]->name);
        strcat(directory, "\\");

        strcat(search, directory);
        strcat(search, "*.*");

        WIN32_FIND_DATA findFileData;

        HANDLE hFind = FindFirstFile(search, &findFileData);

        if (hFind != INVALID_HANDLE_VALUE)
        {
            do
            {
                if (strcmp(".", findFileData.cFileName) == 0 || strcmp("..", findFileData.cFileName) == 0)
                {
                    continue;
                }

                char filePath[MAX_FILE_PATH] = "";

                strcat(filePath, directory);
                strcat(filePath, findFileData.cFileName);

                Matrix* pixelData  = loadPixelData(filePath, dataset);

                // Corresponding category vector

                Matrix* categoryData = newMatrix(dataset->numCategories, 1, 0.0);
                categoryData->entries[dataset->categories[c]->index][0] = 1.0;

                // Save pixel data vector and category vector

                const double dist = (double) rand() / RAND_MAX;

                if (dist < 0.7)
                {
                    ++dataset->data->trainSize;

                    Matrix** X_Temp = realloc(X_Train,sizeof(Matrix*) * dataset->data->trainSize);

                    if (X_Temp != nullptr)
                    {
                        Matrix** y_Temp = realloc(y_Train,sizeof(Matrix*) * dataset->data->trainSize);

                        if (y_Temp != nullptr)
                        {
                            X_Train = X_Temp;
                            y_Train = y_Temp;
                            X_Train[dataset->data->trainSize - 1] = pixelData;
                            y_Train[dataset->data->trainSize - 1] = categoryData;
                        }
                        else
                        {
                            --dataset->data->trainSize;

                            Matrix** X_Revert = realloc(X_Train,sizeof(Matrix*) * dataset->data->trainSize);

                            if (X_Revert != nullptr)
                            {
                                X_Train = X_Revert;
                            }
                            else
                            {
                                printf("SEVERE ERROR: Memory Reallocation Failure.\n");
                                printf("Deleting image dataset.\n");
                                delImageDataset(dataset);
                                return;
                            }
                        }
                    }
                }
                else if (dist <= 0.7 && dist < 0.85)
                {
                    ++dataset->data->validateSize;

                    Matrix** X_Temp = realloc(X_Validate,sizeof(Matrix*) * dataset->data->validateSize);

                    if (X_Temp != nullptr)
                    {
                        Matrix** y_Temp = realloc(y_Validate,sizeof(Matrix*) * dataset->data->validateSize);

                        if (y_Temp != nullptr)
                        {
                            X_Validate = X_Temp;
                            y_Validate = y_Temp;
                            X_Validate[dataset->data->validateSize - 1] = pixelData;
                            y_Validate[dataset->data->validateSize - 1] = categoryData;
                        }
                        else
                        {
                            --dataset->data->validateSize;

                            Matrix** X_Revert = realloc(X_Validate,sizeof(Matrix*) * dataset->data->validateSize);

                            if (X_Revert != nullptr)
                            {
                                X_Validate = X_Revert;
                            }
                            else
                            {
                                printf("SEVERE ERROR: Memory Reallocation Failure.\n");
                                printf("Deleting image dataset.\n");
                                delImageDataset(dataset);
                                return;
                            }
                        }
                    }
                }
                else
                {
                    ++dataset->data->testSize;

                    Matrix** X_Temp = realloc(X_Test,sizeof(Matrix*) * dataset->data->testSize);

                    if (X_Test != nullptr)
                    {
                        Matrix** y_Temp = realloc(y_Test,sizeof(Matrix*) * dataset->data->testSize);

                        if (y_Temp != nullptr)
                        {
                            X_Test = X_Temp;
                            y_Test = y_Temp;
                            X_Test[dataset->data->testSize - 1] = pixelData;
                            y_Test[dataset->data->testSize - 1] = categoryData;
                        }
                        else
                        {
                            --dataset->data->testSize;

                            Matrix** X_Revert = realloc(X_Test,sizeof(Matrix*) * dataset->data->testSize);

                            if (X_Revert != nullptr)
                            {
                                X_Test = X_Revert;
                            }
                            else
                            {
                                printf("SEVERE ERROR: Memory Reallocation Failure.\n");
                                printf("Deleting image dataset.\n");
                                delImageDataset(dataset);
                                return;
                            }
                        }
                    }
                }
            }
            while (FindNextFile(hFind, &findFileData) != 0);
        }
    }

    dataset->data->X_Train = X_Train;
    dataset->data->X_Test = X_Test;
    dataset->data->X_Validate = X_Validate;
    dataset->data->y_Validate = y_Validate;
    dataset->data->y_Train = y_Train;
    dataset->data->y_Test = y_Test;
}

int getImageVectorSize(ImageDataset* dataset)
{
    int size = 0;

    int channels = 0;

    if (strcmp(dataset->data->colorChannels, "Grayscale") == 0)
    {
        channels = 1;
    }
    else if (strcmp(dataset->data->colorChannels, "RGB") == 0)
    {
        channels = 3;
    }
    else if (strcmp(dataset->data->colorChannels, "RGBA") == 0)
    {
        channels = 4;
    }

    size = dataset->data->imageWidth * dataset->data->imageHeight * channels;

    return size;
}

int getCategoryVectorSize(ImageDataset* dataset)
{
    return dataset->numCategories;
}

double evaluateImageTestSet(ImageDataset* dataset, MLP* model)
{
    int correctPredictions = 0;

    for (int i = 0; i < dataset->data->testSize; ++i)
    {
        Matrix* pixelData = dataset->data->X_Test[i];
        Matrix* categoryData = dataset->data->y_Test[i];

        Matrix* prediction = outputMLP(pixelData, model);

        int categoryIndex = 0;

        for (int j = 0; j < dataset->numCategories; ++j)
        {
            if (categoryData->entries[j][0] == 1.0)
            {
                categoryIndex = j;
                break;
            }
        }

        int predictionIndex = 0;

        for (int j = 0; j < dataset->numCategories; ++j)
        {
            if (prediction->entries[j][0] > prediction->entries[predictionIndex][0])
            {
                predictionIndex = j;
            }
        }

        if (predictionIndex == categoryIndex)
        {
            ++correctPredictions;
        }
    }

    const double accuracy = (double) correctPredictions / dataset->data->testSize;

    return accuracy;
}

double evaluateImage(char* filePath, char* category, ImageDataset* dataset, MLP* model)
{
    int c;

    for (int i = 0; i < dataset->numCategories; ++i)
    {
        if (strcmp(category, dataset->categories[i]->name) == 0)
        {
            c = i;
            break;
        }
    }

    Matrix* pixelData = loadPixelData(filePath, dataset);

    Matrix* result = outputMLP(pixelData, model);

    delMatrix(pixelData);

    const double accuracy = result->entries[c][0];

    delMatrix(result);

    return accuracy;
}