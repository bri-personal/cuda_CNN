#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "cuda_matrix.cuh"
#include "cnn.cuh"


int initModelTest() {
    int batchSize = 10;
    float learningRate = 0.025;

    int inChannels = 3;
    int inRows = 7;
    int inCols = 9;

    int hiddenChannels = 5;
    int hiddenRows = 5;
    int hiddenCols = 7;

    int outChannels = 2;
    int outRows = 3;
    int outCols = 5;

    ConvolutionalModel* model;
    initConvolutionalModel(&model, batchSize, learningRate);
    addInputLayer(model, inChannels, inRows, inCols);
    addConvLayer(model, hiddenChannels, hiddenRows, hiddenCols);
    addConvLayer(model, outChannels, outRows, outCols);

    if(model->batchSize != 10 || fabsf(model->learningRate - 0.025) > 0.000001 ||
        model->inChannels != inChannels || model->inHeight != inRows || model->inWidth != inChannels ||
        model->outChannels != outChannels || model->outHeight != outRows || model->outWidth != outCols) {
        printf("FAILURE: model params NOT correct\n");
        return 1;
    }

    ConvolutionalNetwork* net = model->network;

    printf("SUCCESS: model params correct\n");
    return 0;
}

int main() {
    int test_total = 0;
    
    test_total += initModelTest();

    return test_total;
}