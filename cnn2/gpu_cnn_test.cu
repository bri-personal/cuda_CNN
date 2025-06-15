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

    printf("Begin GPU initModelTest\n");

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

    printf("SUCCESS: GPU model params correct\n");
    return 0;
}

int initModelTestCPU() {
    int batchSize = 10;
    float learningRate = 0.025;

    int inChannels = 3;
    int inRows = 7;
    int inCols = 9;

    int hiddenChannels = 5;
    int hiddenRows = 5;
    int hiddenCols = 7;
    int hiddenFilterRows = inRows + 1 - hiddenRows;
    int hiddenFilterCols = inCols + 1 - hiddenCols;

    int outChannels = 2;
    int outRows = 3;
    int outCols = 5;
    int outFilterRows = hiddenRows + 1 - outRows;
    int outFilterCols = hiddenCols + 1 - outCols;

    ConvolutionalModel* model;
    initConvolutionalModel(&model, batchSize, learningRate);
    
    addInputLayerCPU(model, inChannels, inRows, inCols);

    Tensor4D hiddenFilter = {hiddenChannels, inChannels, hiddenFilterRows, hiddenFilterCols,
        (elem_t*) malloc(sizeof(elem_t)*hiddenChannels*inChannels*hiddenFilterRows*hiddenFilterCols)};
    Vector hiddenBiases = {hiddenChannels, (elem_t*) malloc(sizeof(elem_t)*hiddenChannels)};
    addConvLayerCPU(model, hiddenChannels, hiddenRows, hiddenCols, &hiddenFilter, &hiddenBiases);
  
    Tensor4D outFilter = {outChannels, hiddenChannels, outFilterRows, outFilterCols,
        (elem_t*) malloc(sizeof(elem_t)*outChannels*hiddenChannels*outFilterRows*outFilterCols)};
    Vector outBiases = {outChannels, (elem_t*) malloc(sizeof(elem_t)*outChannels)};
    addConvLayerCPU(model, outChannels, outRows, outCols, &outFilter, &outBiases);

    printf("Begin CPU initModelTest\n");

    if(model->batchSize != 10) {
        printf("FAILURE: model batch size NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if (fabsf(model->learningRate - 0.025) > 0.000001) {
        printf("FAILURE: model learning rate NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if (model->inChannels != inChannels) {
        printf("FAILURE: model in channels NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(model->inHeight != inRows) {
        printf("FAILURE: model in height NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if (model->inWidth != inCols) {
        printf("FAILURE: model in width NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    } 
    if (model->outChannels != outChannels) {
        printf("FAILURE: model out channels NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(model->outHeight != outRows) {
        printf("FAILURE: model out height NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(model->outWidth != outCols) {
        printf("FAILURE: model out width NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }

    ConvolutionalNetwork* net = model->network;
    ConvolutionalLayer* inLayer = net->input;
    ConvolutionalLayer* hiddenLayer = inLayer->next;
    ConvolutionalLayer* outLayer = net->output;
    if(inLayer->outChannels != inChannels) {
        printf("FAILURE: in layer out channels NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(inLayer->outputRows != inRows) {
        printf("FAILURE: in layer out rows NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(inLayer->outputCols != inCols) {
        printf("FAILURE: in layer out channels NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }

    if(hiddenLayer->inChannels != inChannels) {
        printf("FAILURE: hidden layer in channels NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(hiddenLayer->imgRows != inRows) {
        printf("FAILURE: hidden layer in rows NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(hiddenLayer->imgCols != inCols) {
        printf("FAILURE: hidden layer in cols NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(hiddenLayer->outChannels != hiddenChannels) {
        printf("FAILURE: hidden layer out channels NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(hiddenLayer->outputRows != hiddenRows) {
        printf("FAILURE: hidden layer out rows NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(hiddenLayer->outputCols != hiddenCols) {
        printf("FAILURE: hidden layer out cols NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(hiddenLayer->kernelRows != 3) {
        printf("FAILURE: hidden layer kernel rows NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(hiddenLayer->kernelCols != 3) {
        printf("FAILURE: hidden layer kernel cols NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }

    if(outLayer->inChannels != hiddenChannels) {
        printf("FAILURE: out layer in channels NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(outLayer->imgRows != hiddenRows) {
        printf("FAILURE: out layer in rows NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(outLayer->imgCols != hiddenCols) {
        printf("FAILURE: out layer in cols NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(outLayer->outChannels != outChannels) {
        printf("FAILURE: out layer out channels NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(outLayer->outputRows != outRows) {
        printf("FAILURE: out layer out rows NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(outLayer->outputCols != outCols) {
        printf("FAILURE: out layer out cols NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(outLayer->kernelRows != 3) {
        printf("FAILURE: out layer kernel rows NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }
    if(outLayer->kernelCols != 3) {
        printf("FAILURE: out layer kernel cols NOT correct\n");
        free(hiddenBiases.data);
        free(hiddenFilter.data);
        free(outBiases.data);
        free(outFilter.data);
        return 1;
    }

    printf("SUCCESS: CPU model params correct\n");

    free(hiddenBiases.data);
    free(hiddenFilter.data);
    free(outBiases.data);
    free(outFilter.data);
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

int modelForwardTestCPU() {
    int batchSize = 10;
    float learningRate = 0.025;

    int inChannels = 3;
    int inRows = 7;
    int inCols = 9;

    int hiddenChannels = 5;
    int hiddenRows = 5;
    int hiddenCols = 7;
    int hiddenFilterRows = inRows + 1 - hiddenRows;
    int hiddenFilterCols = inCols + 1 - hiddenCols;

    int outChannels = 2;
    int outRows = 3;
    int outCols = 5;
    int outFilterRows = hiddenRows + 1 - outRows;
    int outFilterCols = hiddenCols + 1 - outCols;

    ConvolutionalModel* model;
    initConvolutionalModel(&model, batchSize, learningRate);
    
    addInputLayerCPU(model, inChannels, inRows, inCols);

    Tensor4D hiddenFilter = {hiddenChannels, inChannels, hiddenFilterRows, hiddenFilterCols,
        (elem_t*) malloc(sizeof(elem_t)*hiddenChannels*inChannels*hiddenFilterRows*hiddenFilterCols)};
    Vector hiddenBiases = {hiddenChannels, (elem_t*) malloc(sizeof(elem_t)*hiddenChannels)};
    addConvLayerCPU(model, hiddenChannels, hiddenRows, hiddenCols, &hiddenFilter, &hiddenBiases);
  
    Tensor4D outFilter = {outChannels, hiddenChannels, outFilterRows, outFilterCols,
        (elem_t*) malloc(sizeof(elem_t)*outChannels*hiddenChannels*outFilterRows*outFilterCols)};
    Vector outBiases = {outChannels, (elem_t*) malloc(sizeof(elem_t)*outChannels)};
    addConvLayerCPU(model, outChannels, outRows, outCols, &outFilter, &outBiases);

    int total = batchSize*inChannels*inRows*inCols;
    Tensor4D input = {batchSize, inChannels, inRows, inCols, (elem_t*) calloc(total, sizeof(elem_t))};

    forwardCPU(model, &input);

    printf("SUCCESS: CPU model forward didn't crash\n");

    return 0;
}

int modelForwardTestOutputCPU() {
    const int batchSize = 1;
    const int learningRate = 0.01;

    const int inRows = 3;
    const int inCols = 3;
    const int inChannels = 1;

    const int outRows = 2;
    const int outCols = 2;
    const int outChannels = 1;
    int outFilterRows = inRows + 1 - outRows;
    int outFilterCols = inCols + 1 - outCols;
    
    ConvolutionalModel* model;
    initConvolutionalModel(&model, batchSize, learningRate);
    addInputLayerCPU(model, inChannels, inRows, inCols);

    elem_t filterData[] = {1, 0, 0, 1};
    Tensor4D filter = {outChannels, inChannels, outFilterRows, outFilterCols, filterData};

    elem_t biasData[] = {1};
    Vector biases = {outChannels, biasData};
    addConvLayerCPU(model, outChannels, outRows, outCols, &filter, &biases);
    
    elem_t inputData[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    Tensor4D input = {batchSize, inChannels, inRows, inCols, inputData};
    forwardCPU(model, &input);

    elem_t expectedData[] = {
        SIGMOID(1.6f), SIGMOID(1.8f), SIGMOID(2.2f), SIGMOID(2.4f)
    };
    Tensor4D expectedOut = {batchSize, outChannels, outRows, outCols, expectedData};
    
    if(!tensor4DEquals(&expectedOut, model->network->output->outputs, 0.000001)) {
        printf("FAILURE: CPU model forward does NOT have correct output\n");
        return 1;
    }

    printf("SUCCESS: CPU model forward has correct output\n");
    return 0;
}

int modelForwardTestOutput() {
    const int batchSize = 1;
    const int learningRate = 0.01;

    const int inRows = 3;
    const int inCols = 3;
    const int inChannels = 1;

    const int outRows = 2;
    const int outCols = 2;
    const int outChannels = 1;

    /* input */
    curandState_t* state = createCurandStates(batchSize*inChannels*inRows*inCols);

    Tensor4D* deviceInput;
    initRandomTensor4D(&deviceInput, batchSize, inChannels, inRows, inCols, state);

    int inputSize = batchSize*inChannels*inRows*inCols;
    Tensor4D input = {batchSize, inChannels, inRows, inCols, (elem_t*) calloc(inputSize, sizeof(elem_t))};
    getDeviceTensor4DData(input.data, deviceInput, inputSize);

    /* GPU */
    ConvolutionalModel* model;
    initConvolutionalModel(&model, batchSize, learningRate);
    addInputLayer(model, inChannels, inRows, inCols, state);
    addConvLayer(model, outChannels, outRows, outCols, state);

    forward(model, &input);

    int outSize = batchSize*outChannels*outRows*outCols;
    elem_t gpuModelOutData[outSize];
    Tensor4D gpuModelOut = {batchSize, outChannels, outRows, outCols,gpuModelOutData};
    getDeviceTensor4DData(gpuModelOut->data, model->network->output->outputs, outSize);

    cleanupCurandStates(state);

    /* CPU */
    int outFilterRows = inRows + 1 - outRows;
    int outFilterCols = inCols + 1 - outCols;
    int filterSize = outChannels*inChannels*outFilterRows*outFilterCols;
    
    ConvolutionalModel* modelCPU;
    initConvolutionalModel(&modelCPU, batchSize, learningRate);
    addInputLayerCPU(modelCPU, inChannels, inRows, inCols);

    elem_t filterData[filterSize];
    Tensor4D filter = {outChannels, inChannels, outFilterRows, outFilterCols, filterData};
    getDeviceTensor4DData(filter.data, model->network->input->next->filters, filterSize);

    elem_t biasData[outChannels];
    Vector biases = {outChannels, biasData};
    getDeviceVectorData(biases->data, model->network->input->next->biases, outChannels);

    addConvLayerCPU(modelCPU, outChannels, outRows, outCols, &filter, &biases);
    
    forwardCPU(modelCPU, &input);
    
    /* compare */
    if(!tensor4DEquals(gpuModelOut, modelCPU->network->output->outputs, 0.000001)) {
        printf("FAILURE: GPU model forward and CPU model forward do NOT have equal output\n");
        return 1;
    }

    printf("SUCCESS: GPU model forward and CPU model forward have equal output\n");
    return 0;
}

int main() {
    int test_total = 0;
    
    test_total += initModelTest();
    test_total += initModelTestCPU();
    test_total += modelForwardTest();
    test_total += modelForwardTestCPU();
    test_total += modelForwardTestOutputCPU();
    test_total += modelForwardTestOutput();

    return test_total;
}