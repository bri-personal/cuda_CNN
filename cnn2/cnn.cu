#include "cuda_cnn.cuh"
#include "matrix.h"
#include "cuda_matrix.cuh"
#include "util.h"
#include "cnn.h"
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <curand_kernel.h>


ConvolutionalLayer* createConvolutionalLayer(int batch_size, int outChannels,
        int outputRows, int outputCols, ConvolutionalLayer* prev, curandState_t* state) {
    ConvolutionalLayer* layer = (ConvolutionalLayer*) calloc(1, sizeof(ConvolutionalLayer));
    if (!layer) {perror("malloc"); exit(1);}
    layer->outChannels = outChannels;
    layer->outputRows = outputRows;
    layer->outputCols = outputCols;
    layer->prev = prev;
    if (prev != NULL) {
        prev->next = layer;
        layer->inChannels = prev->outChannels;
        layer->imgRows = prev->outputRows;
        layer->imgCols = prev->outputCols;
    }

    // TODO: this is assuming stride = 1 and padding = 0
    layer->kernelRows = layer->imgRows + 1 - outputRows;
    layer->kernelCols = layer->imgCols + 1 - outputCols;

    if(layer->inChannels > 0) {
        initRandomTensor4D(&layer->filters, outChannels, layer->inChannels,
            layer->kernelRows, layer->kernelCols, state);
    }
    initRandomVector(&layer->biases, outChannels, state);
    initTensor4D(&layer->outputs, batch_size, outChannels, outputRows, outputCols);

    return layer;
}

void addInputLayer(ConvolutionalModel *model, int channels, int rows, int cols, curandState_t* state) {
    model->inChannels = channels;
    model->inHeight = rows;
    model->inWidth = cols;
    model->outChannels = channels;
    model->outHeight = rows;
    model->outWidth = cols;
    ConvolutionalLayer* layer = createConvolutionalLayer(model->batchSize, channels,
        rows, cols, NULL, state);
    model->network->input = layer;
    model->network->layers = layer;
    model->network->output = layer;
}

void addConvLayer(ConvolutionalModel *model, int channels, int rows, int cols, curandState_t* state) {
    ConvolutionalLayer* prev = model->network->output;
    ConvolutionalLayer* layer = createConvolutionalLayer(model->batchSize, channels,
        rows, cols, prev, state);
    model->network->numLayers++;
    model->network->output = layer;
    model->outChannels = channels;
    model->outHeight = rows;
    model->outWidth = cols;
}

void layerForward(ConvolutionalLayer *layer, int batchSize) {
    /* for each channel of this input sample, do forward pass */
    int outChannels = layer->outChannels;
    int outRows = layer->outputRows;
    int outCols = layer->outputCols;

    int im2colOutRows = batchSize * outRows * outCols;
    int im2colOutArea = im2colOutRows * outChannels;

    Matrix* temp;
    initMatrix(&temp, im2colOutRows, outChannels);

    deviceConvolve(
        layer->prev->outputs,
        layer->filters,
        temp,
        0, 1, //TODO: actually use padding and stride
        layer->inChannels,
        layer->imgRows,
        layer->imgCols,
        outChannels,
        layer->kernelRows,
        layer->kernelCols,
        im2colOutRows,
        outChannels
    );

    /* add bias for each out channel to every element in that channel */
    deviceMatrixAddScalarColumnwise(temp, temp, layer->biases, im2colOutRows, outChannels);

    /* apply sigmoid activation to every element */
    deviceSigmoid(temp, temp, im2colOutArea);

    /* put temp contents into layer's output Tensor4D */
    deviceReorderIm2ColToConv(temp, layer->outputs, im2colOutArea);

    freeMatrix(temp);
  }
  
/**
 * :param input: list of input samples (size of minibatch).
 * Each input sample is an image with inChannels channels, or a list of inChannels lists of floats.
 */
void forward(ConvolutionalModel *model, Tensor4D* input) {
    ConvolutionalNetwork* net = model->network;
    int batchSize = model->batchSize;
    int inputChannels = model->inChannels;
    int imageSize = model->inHeight * model->inWidth;
    int inputSize = batchSize*inputChannels*imageSize;

    /* initialize 4D tensor of input images */
    setDeviceTensor4DData(net->input->outputs, input->data, inputSize);
    
    ConvolutionalLayer *curr = net->input->next; /* first hidden layer */
    while (curr != NULL) {
        layerForward(curr, batchSize);
        curr = curr->next;
    }
  }


void initLayerGradients(ConvolutionalLayer *layer, int batchSize) {
    int outChannels = layer->outChannels;
    int rows = layer->outputRows;
    int cols = layer->outputCols;

    initTensor4D(&layer->gradient, batchSize, outChannels, rows, cols);

    initTensor4D(&layer->delta, batchSize, outChannels, rows, cols);

    if (layer->prev) {
        initTensor4D(&layer->error, batchSize, outChannels, rows, cols);
    } else {
        layer->error = NULL;
    }
}

void compileModel(ConvolutionalModel *model) {
    int batchSize = model->batchSize;

    ConvolutionalLayer* curr = model->network->layers->next;
    while (curr != NULL) {
        initLayerGradients(curr, batchSize);
        curr = curr->next;
    }
}

void layerBackward(ConvolutionalLayer* layer) {
    int outputSize = layer->outputRows * layer->outputCols;
    
    /* gradient of y^ wrt z */
    deviceTensor4DSigmoidOutputDerivative(layer->outputs, layer->gradient, outputSize);

    /* gradient of L wrt z */
    deviceTensor4DHadamardProd(layer->gradient, layer->error, layer->gradient, outputSize);
    
    // TODO: continue
}

void layerUpdate(ConvolutionalLayer* layer, int batchSize) {
    // TODO
    return;
}

void backward(ConvolutionalModel* model, Tensor4D* targets) {
    ConvolutionalNetwork* net = model->network;
    int batchSize = model->batchSize;
    ConvolutionalLayer* curr = net->output;
    int outputSize = curr->outputRows * curr->outputCols;

    /* get gradient of MSE loss wrt predicted output */
    setDeviceTensor4DData(curr->error, targets, outputSize);
    deviceTensor4DSub(curr->outputs, curr->error, curr->error, outputSize);
    deviceTensor4DDivideScalarElementwise(curr->error, curr->error, outputSize, outputSize);

    /* backprop to calc gradients and update params */
    while (curr != NULL) {
        layerBackward(curr, model);
        layerUpdate(curr, batchSize);
        curr = curr->prev;
    }
}

int modelAccuracy(ConvolutionalModel *model, Tensor4D* images, uint8_t *labels) {
    // TODO
    return 0;
}