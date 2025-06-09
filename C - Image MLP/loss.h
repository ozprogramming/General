#ifndef LOSS_H
#define LOSS_H

#include "library.h"

Matrix* meanSquaredError(Matrix* prediction, Matrix* actual);

Matrix* categoricalCrossEntropyError(Matrix* prediction, Matrix* actual);

Matrix* d_meanSquaredError(Matrix* prediction, Matrix* actual);

Matrix* d_categoricalCrossEntropyError(Matrix* prediction, Matrix* actual);

#endif //LOSS_H
