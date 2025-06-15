#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "cnn.h"

void initConvolutionalModel(ConvolutionalModel** model, int batchSize, float learningRate) {  
    *model = (ConvolutionalModel*) calloc(1, sizeof(ConvolutionalModel));
    if (!(*model)) { perror("calloc model"); exit(1); }

    ConvolutionalNetwork *cnn = (ConvolutionalNetwork*) malloc(sizeof(ConvolutionalNetwork));
    if (!cnn) { perror("malloc network"); exit(1); }
    cnn->numLayers = 0;
  
    (*model)->network = cnn;
    (*model)->learningRate = learningRate;
    (*model)->batchSize = batchSize;
}

/**
 * filters should be Tensor4D(outChannels, layer->inChannels, (layer->imgRows + 1 - outputRows), (layer->imgCols + 1 - outputCols))
 * assuming a stride of 1 and padding of 0
 * 
 * biases should be Vector(outChannels)
 */
ConvolutionalLayer* createConvolutionalLayerCPU(int batch_size, int outChannels,
    int outputRows, int outputCols, ConvolutionalLayer* prev, Tensor4D* filters, Vector* biases
) {
    ConvolutionalLayer* layer = (ConvolutionalLayer*) calloc(1, sizeof(ConvolutionalLayer));
    if (!layer) {perror("calloc"); exit(1);}
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
    layer->kernelRows = filters->height; //layer->imgRows + 1 - outputRows;
    layer->kernelCols = filters->width; //layer->imgCols + 1 - outputCols;

    layer->filters = filters;
    layer->biases = biases;
    layer->outputs = (Tensor4D*) malloc(sizeof(Tensor4D));
    if (layer->outputs == NULL) {perror("malloc"); exit(1);}
    layer->outputs->dim4 = batch_size;
    layer->outputs->depth = outChannels;
    layer->outputs->height = outputRows;
    layer->outputs->width = outputCols;
    layer->outputs->data = (elem_t*) malloc(sizeof(elem_t)*batch_size*outChannels*outputRows*outputCols);
    if (layer->outputs->data == NULL) {perror("malloc"); exit(1);}

    return layer;
}

void addInputLayerCPU(ConvolutionalModel *model, int channels, int rows, int cols) {
    model->inChannels = channels;
    model->inHeight = rows;
    model->inWidth = cols;
    model->outChannels = channels;
    model->outHeight = rows;
    model->outWidth = cols;
    ConvolutionalLayer* layer = createConvolutionalLayerCPU(model->batchSize, channels,
        rows, cols, NULL, NULL, NULL);
    model->network->input = layer;
    model->network->layers = layer;
    model->network->output = layer;
}

/**
 * filters should be Tensor4D(channels, model->outChannels, (layer->imgRows + 1 - outputRows), (layer->imgCols + 1 - outputCols))
 * assuming a stride of 1 and padding of 0
 * 
 * biases should be Vector(channels)
 */
void addConvLayerCPU(ConvolutionalModel *model, int channels, int rows, int cols, Tensor4D* filters, Vector* biases) {
    ConvolutionalLayer* prev = model->network->output;
    ConvolutionalLayer* layer = createConvolutionalLayerCPU(model->batchSize, channels,
        rows, cols, prev, filters, biases);
    model->network->numLayers++;
    model->network->output = layer;
    model->outChannels = channels;
    model->outHeight = rows;
    model->outWidth = cols;
}

void layerForwardCPU(ConvolutionalLayer *layer, int batchSize) {
    /* for each channel of this input sample, do forward pass */
    
    // TODO: padding and stride are hardcoded to 0 and 1 right now
    conv_CPU(layer->outputs, layer->prev->outputs, layer->filters, 0, 1);

    /* add bias for each out channel to every element in that channel */
    addScalarToEachColumnOfTensor4D_CPU(layer->outputs, layer->outputs, layer->biases);

    /* apply sigmoid activation to every element */
    tensor4DSigmoid_CPU(layer->outputs, layer->outputs);
}

void forwardCPU(ConvolutionalModel *model, Tensor4D* input) {
    ConvolutionalNetwork* net = model->network;
    int batchSize = model->batchSize;
    int inputChannels = model->inChannels;
    int imageSize = model->inHeight * model->inWidth;
    int inputSize = batchSize*inputChannels*imageSize;

    /* initialize 4D tensor of input images */
    net->input->outputs = input;
    
    ConvolutionalLayer *curr = net->input->next; /* first hidden layer */
    while (curr != NULL) {
        layerForwardCPU(curr, batchSize);
        curr = curr->next;
    }
}