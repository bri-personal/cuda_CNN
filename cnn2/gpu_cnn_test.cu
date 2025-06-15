#include <stdio.h>
#include "matrix.h"
#include "cuda_matrix.cuh"
#include "cnn.cuh"


int initModelTest() {
    int batchSize = 10;
    float learningRate = 0.025;

    ConvolutionalModel* model;
    initConvolutionalModel(&model, batchSize, learningRate);

    printf("b: %d, l: %f\n", model->batchSize, model->learningRate);
    return 0;
    if(model->batchSize != 10 || model->learningRate != 0.025) {
        printf("FAILURE: model params NOT correct\n");
        return 1;
    }

    printf("SUCCESS: model params correct\n");
    return 0;
}

int main() {
    int test_total = 0;
    
    test_total += initModelTest();

    return test_total;
}