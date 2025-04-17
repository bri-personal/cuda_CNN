#include "cnn.h"
#include "util.h"
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>

ConvolutionalModel *initConvolutionalModel(int batchSize, float learningRate) {
    ConvolutionalModel *model = (ConvolutionalModel *)malloc(sizeof(ConvolutionalModel));
    if (!model) { perror("malloc"); exit(1); }
  
    ConvolutionalNetwork *cnn = (ConvolutionalNetwork *)malloc(sizeof(ConvolutionalNetwork));
    if (!cnn) { perror("malloc"); exit(1); }
  
    model->network = cnn;
    model->learningRate = learningRate;
    model->batchSize = batchSize;
  
    checkError("Init CNN");
    return model;
  }

void addInputLayer(ConvolutionalModel *model, int channels, int rows, int cols) {
    model->input_c = channels;
    model->input_h = rows;
    model->input_w = cols;
    model->output_c = channels;
    model->output_h = rows;
    model->output_w = cols;
    ConvolutionalLayer* layer = createConvolutionalLayer(model->batchSize, 0, channels,
        rows, cols, NULL);
    model->network->input = layer;
    model->network->layers = layer;
    model->network->output = layer;
}

void addConvLayer(ConvolutionalModel *model, int channels, int rows, int cols) {
    ConvolutionalLayer* prev = model->network->output;
    ConvolutionalLayer* layer = createConvolutionalLayer(model->batchSize, prev->k, channels,
        rows, cols, prev);
    model->network->numLayers++;
    model->network->output = layer;
    model->output_c = channels;
    model->output_h = rows;
    model->output_w = cols;
}

ConvolutionalLayer* createConvolutionalLayer(int batch_size, int c_in, int k,
        int outputRows, int outputCols, ConvolutionalLayer* prev) {
    ConvolutionalLayer* layer = (ConvolutionalLayer*) calloc(1, sizeof(ConvolutionalLayer));
    if (!layer) {perror("malloc"); exit(1);}
    layer->c_in = c_in;
    layer->k = k;
    layer->outputRows = outputRows;
    layer->outputCols = outputCols;
    layer->prev = prev;
    if (prev != NULL) {
        prev->next = layer;
        layer->imgRows = prev->outputRows;
        layer->imgCols = prev->outputCols;
    }
    layer->kernelRows = layer->imgRows + 1 - outputRows;
    layer->kernelCols = layer->imgCols + 1 - outputCols;

    int i, j;

    /* filters needs k arrays of c_in arrays of pointers to Matrix on the device*/
    layer->filters = (Matrix***) malloc(sizeof(Matrix**) * k);
    if (!(layer->filters)) {perror("malloc layer filters"); exit(1);}
    for (i = 0; i < k; i++) {
        layer->filters[i] = (Matrix**) malloc(sizeof(Matrix*) * c_in);
        if (!(layer->filters[i])) {perror("malloc layer filters"); exit(1);}
        for (j = 0; j < c_in; j++) {
            initRandomMatrix(layer->filters[i] + j, layer->kernelRows, layer->kernelCols);
        }
    }

    /* biases is array of k floats */
    layer->biases = (float *) calloc(k, sizeof(float));
    if (!(layer->biases)) {perror("malloc layer biases"); exit(1);}

    /* outputs needs batch_size arrays of k arrays of pointers to Matrix on the device */
    layer->outputs = (Matrix***) malloc(sizeof(Matrix**) * batch_size);
    if (!(layer->outputs)) {perror("malloc layer outputs"); exit(1);}
    for (i = 0; i < batch_size; ++i) {
        layer->outputs[i] = (Matrix**) malloc(sizeof(Matrix*) * k);
        if (!(layer->outputs[i])) {perror("malloc layer outputs"); exit(1);}
        for (j = 0; j < k; ++j) {
            initMatrix(layer->outputs[i] + j, outputRows, outputCols);
        }
    }

    return layer;
  }

void layerForward(ConvolutionalLayer *layer, int sampleNo) {
    /* for each channel of this input sample, do forward pass */

    // TODO: change size when we have image and kernel dimensions
    int imgRows = layer->imgRows;
    int imgCols = layer->imgCols;
    int imgSize = imgRows * imgCols;
    int kernelRows = layer->kernelRows;
    int kernelCols = layer->kernelCols;
    
    int output_channels = layer->k;
    int input_channels = layer->c_in;

    int c, k;
    Matrix *temp;
    initMatrix(&temp, layer->outputRows, layer->outputCols);

    Matrix **inputImages, *outputImageK, **filtersK;
    inputImages = (layer->prev->outputs)[sampleNo];

    for (k = 0; k < output_channels; k++) {
        outputImageK = (layer->outputs)[sampleNo][k];
        filtersK = (layer->filters)[k];

        // TODO: change to convolution
        /* convolve first input channel image with first filter */
        deviceConvolve(inputImages[0], imgRows, imgCols, 
            filtersK[0], kernelRows, kernelCols,
            outputImageK, 1, 0);
        for (c = 1; c < input_channels; c++) {
            /* for each remaining channel, add the convolution of the image 
             * and filter to the running total
             */
            deviceConvolve(inputImages[c], imgRows, imgCols,
                filtersK[c], kernelRows, kernelCols,
                temp, 1, 0);

            deviceMatrixAdd(
                outputImageK,
                temp,
                outputImageK,
                imgSize
            );
        }
        
        /* add bias to every element */
        deviceMatrixAddScalarElementwise(outputImageK, outputImageK, (layer->biases)[k], imgSize);

        /* apply sigmoid activation to every element */
        deviceSigmoid(outputImageK, outputImageK, imgSize);
    }

    freeMatrix(temp);
  }
  
/**
 * :param input: list of input samples (size of minibatch).
 * Each input sample is an image with c_in channels, or a list of c_in lists of floats.
 */
void forward(ConvolutionalModel *model, float ***input) {
    ConvolutionalNetwork net = *(model->network);
    int batchSize = model->batchSize;
    int inputChannels = model->input_c;
    int imageSize = model->input_h * model->input_w;

    int i, j;

    /* initialize 4D tensor of input images */
    for (i = 0; i < batchSize; ++i) {
        for (j = 0; j < inputChannels; ++j) {
            setDeviceMatrixData((net.layers->outputs)[i][j], input[i][j], imageSize);
        }
    }
    
    ConvolutionalLayer *curr = net.layers->next; /* first hidden layer */
    for (i = 0; i < net.numLayers; ++i) {
        if (!curr) break;

        /* for each sample in minibatch, go forward */
        for (j = 0; j < batchSize; ++j) {
            layerForward(curr, j);
        }
        curr = curr->next;
    }
  }


void initLayerGradients(ConvolutionalLayer *layer, int batchSize) {
    int i, j;
    int k = layer->k;
    int r = layer->outputRows;
    int c = layer->outputCols;

    /* backprop fields needs batchsize arrays of k arrays of pointers to Matrix on the device */
    layer->gradient = (Matrix***) malloc(sizeof(Matrix**) * batchSize);
    if (!(layer->gradient)) {perror("malloc layer g"); exit(1);}

    layer->delta = (Matrix***) malloc(sizeof(Matrix**) * batchSize);
    if (!(layer->delta)) {perror("malloc layer d"); exit(1);}
    if (layer->prev) {
        layer->error = (Matrix***) malloc(sizeof(Matrix**) * batchSize);
        if (!(layer->error)) {perror("malloc layer e"); exit(1);}
    }
    
    for (i = 0; i < batchSize; i++) {
        layer->gradient[i] = (Matrix**) malloc(sizeof(Matrix*) * k);
        if (!(layer->gradient[i])) {perror("malloc layer g"); exit(1);}

        layer->delta[i] = (Matrix**) malloc(sizeof(Matrix*) * k);
        if (!(layer->delta[i])) {perror("malloc layer d"); exit(1);}

        layer->error[i] = (Matrix**) malloc(sizeof(Matrix*) * k);
        if (!(layer->error[i])) {perror("malloc layer e"); exit(1);}
        
        for (j = 0; j < c_in; j++) {
            initMatrix(layer->gradient[i] + j, r, c);
            initMatrix(layer->delta[i] + j, r, c);
            initMatrix(layer->error[i] + j, r, c);
        }
    }
}

void compileModel(ConvolutionalModel *model) {
    ConvolutionalLayer* curr = model->network->layers->next;
    for (int i = 0; i < model->network->numLayers; ++i) {
        if (!curr) break;
        initLayerGradients(curr, model->batchSize);
        curr = curr->next;
    }
}

float backward(ConvolutionalModel* model, float*** targets) {
    Network* net = model->network;
    int batchSize = model->batchSize;
    ConvolutionalLayer* curr = net->output;
}