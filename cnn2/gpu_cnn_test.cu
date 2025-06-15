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

    curandState_t* state = createCurandStates(batchSize*hiddenChannels*inRows*inCols); // more than needed

    ConvolutionalModel* model;
    initConvolutionalModel(&model, batchSize, learningRate);
    addInputLayer(model, inChannels, inRows, inCols, state);
    addConvLayer(model, hiddenChannels, hiddenRows, hiddenCols, state);
    addConvLayer(model, outChannels, outRows, outCols, state);

    cleanupCurandStates(state);

    if(model->batchSize != 10) {
        printf("FAILURE: model batch size NOT correct\n");
        return 1;
    }
    if (fabsf(model->learningRate - 0.025) > 0.000001) {
        printf("FAILURE: model learning rate NOT correct\n");
        return 1;
    }
    if (model->inChannels != inChannels) {
        printf("FAILURE: model in channels NOT correct\n");
        return 1;
    }
    if(model->inHeight != inRows) {
        printf("FAILURE: model in height NOT correct\n");
        return 1;
    }
    if (model->inWidth != inChannels) {
        printf("FAILURE: model in width NOT correct\n");
        return 1;
    } 
    if (model->outChannels != outChannels) {
        printf("FAILURE: model out channels NOT correct\n");
        return 1;
    }
    if(model->outHeight != outRows) {
        printf("FAILURE: model out height NOT correct\n");
        return 1;
    }
    if(model->outWidth != outCols) {
        printf("FAILURE: model out width NOT correct\n");
        return 1;
    }

    //ConvolutionalNetwork* net = model->network;

    printf("SUCCESS: model params correct\n");
    return 0;
}

int main() {
    int test_total = 0;
    
    test_total += initModelTest();

    return test_total;
}