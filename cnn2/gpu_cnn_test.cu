#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "cnn.h"
#include "cuda_matrix.cuh"
#include "cuda_cnn.cuh"


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
    if (model->inWidth != inCols) {
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

    ConvolutionalNetwork* net = model->network;
    ConvolutionalLayer* inLayer = net->input;
    ConvolutionalLayer* hiddenLayer = inLayer->next;
    ConvolutionalLayer* outLayer = net->output;
    if(inLayer->outChannels != inChannels) {
        printf("FAILURE: in layer out channels NOT correct\n");
        return 1;
    }
    if(inLayer->outputRows != inRows) {
        printf("FAILURE: in layer out rows NOT correct\n");
        return 1;
    }
    if(inLayer->outputCols != inCols) {
        printf("FAILURE: in layer out channels NOT correct\n");
        return 1;
    }

    if(hiddenLayer->inChannels != inChannels) {
        printf("FAILURE: hidden layer in channels NOT correct\n");
        return 1;
    }
    if(hiddenLayer->imgRows != inRows) {
        printf("FAILURE: hidden layer in rows NOT correct\n");
        return 1;
    }
    if(hiddenLayer->imgCols != inCols) {
        printf("FAILURE: hidden layer in cols NOT correct\n");
        return 1;
    }
    if(hiddenLayer->outChannels != hiddenChannels) {
        printf("FAILURE: hidden layer out channels NOT correct\n");
        return 1;
    }
    if(hiddenLayer->outputRows != hiddenRows) {
        printf("FAILURE: hidden layer out rows NOT correct\n");
        return 1;
    }
    if(hiddenLayer->outputCols != hiddenCols) {
        printf("FAILURE: hidden layer out cols NOT correct\n");
        return 1;
    }
    if(hiddenLayer->kernelRows != 3) {
        printf("FAILURE: hidden layer kernel rows NOT correct\n");
        return 1;
    }
    if(hiddenLayer->kernelCols != 3) {
        printf("FAILURE: hidden layer kernel cols NOT correct\n");
        return 1;
    }

    if(outLayer->inChannels != hiddenChannels) {
        printf("FAILURE: out layer in channels NOT correct\n");
        return 1;
    }
    if(outLayer->imgRows != hiddenRows) {
        printf("FAILURE: out layer in rows NOT correct\n");
        return 1;
    }
    if(outLayer->imgCols != hiddenCols) {
        printf("FAILURE: out layer in cols NOT correct\n");
        return 1;
    }
    if(outLayer->outChannels != outChannels) {
        printf("FAILURE: out layer out channels NOT correct\n");
        return 1;
    }
    if(outLayer->outputRows != outRows) {
        printf("FAILURE: out layer out rows NOT correct\n");
        return 1;
    }
    if(outLayer->outputCols != outCols) {
        printf("FAILURE: out layer out cols NOT correct\n");
        return 1;
    }
    if(outLayer->kernelRows != 3) {
        printf("FAILURE: out layer kernel rows NOT correct\n");
        return 1;
    }
    if(outLayer->kernelCols != 3) {
        printf("FAILURE: out layer kernel cols NOT correct\n");
        return 1;
    }

    printf("SUCCESS: model params correct\n");
    return 0;
}

int modelForwardTest() {
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

    Tensor4D* deviceInput;
    initRandomTensor4D(&deviceInput, batchSize, inChannels, inRows, inCols, state);

    int total = batchSize*inChannels*inRows*inCols;
    Tensor4D input = {batchSize, inChannels, inRows, inCols, (elem_t*) calloc(total, sizeof(elem_t))};
    getDeviceTensor4DData(input.data, deviceInput, total);

    forward(model, &input);

    cleanupCurandStates(state);

    printf("SUCCESS: model forward didn't crash\n");

    return 0;
}

int main() {
    int test_total = 0;
    
    test_total += initModelTest();
    test_total += modelForwardTest();

    return test_total;
}