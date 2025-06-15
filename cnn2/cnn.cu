#include "cnn.cuh"
#include "matrix.h"
#include "cuda_matrix.cuh"
#include "util.h"
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <curand_kernel.h>

void initConvolutionalModel(ConvolutionalModel** model, int batchSize, float learningRate) {  
    *model = (ConvolutionalModel*) calloc(1, sizeof(ConvolutionalModel));
    if (!(*model)) { perror("calloc model"); exit(1); }

    ConvolutionalNetwork *cnn = (ConvolutionalNetwork*) malloc(sizeof(ConvolutionalNetwork));
    if (!cnn) { perror("malloc network"); exit(1); }
    cnn->numLayers = 0;
  
    (*model)->network = cnn;
    (*model)->learningRate = learningRate;
    (*model)->batchSize = batchSize;
  
    checkError("Init CNN");
  }

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

    // TODO: this is assuming stride = 0
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

void layerForward(ConvolutionalLayer *layer) {
    /* for each channel of this input sample, do forward pass */
    int outChannels = layer->outChannels;
    int outRows = layer->outputRows;
    int outCols = layer->outputCols;

    int im2colOutRows = layer->outputs->dim4 * outRows * outCols;
    int im2colOutArea = im2colOutRows * outChannels;

    Matrix* temp;
    printf("%p\n", temp);
    exit(0);

    initMatrix(&temp, im2colOutRows, outChannels);
    exit(0);

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
        outRows,
        outCols
    );

    /* add bias for each out channel to every element in that channel */
    deviceMatrixAddScalarColumnwise(temp, temp, layer->biases, im2colOutRows, outChannels);

    /* apply sigmoid activation to every element */
    deviceSigmoid(temp, temp, im2colOutArea);

    /* put temp  contents into layer's output Tensor4D */
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
        layerForward(curr);
        curr = curr->next;
    }
  }


// void initLayerGradients(ConvolutionalLayer *layer, int batchSize) {
//     int i, j;
//     int outChannels = layer->outChannels;
//     int r = layer->outputRows;
//     int c = layer->outputCols;

//     /* backprop fields needs batchsize arrays of outChannels arrays of pointers to Matrix on the device */
//     layer->gradient = (Matrix***) malloc(sizeof(Matrix**) * batchSize);
//     if (!(layer->gradient)) {perror("malloc layer g"); exit(1);}

//     layer->delta = (Matrix***) malloc(sizeof(Matrix**) * batchSize);
//     if (!(layer->delta)) {perror("malloc layer d"); exit(1);}
//     if (layer->prev) {
//         layer->error = (Matrix***) malloc(sizeof(Matrix**) * batchSize);
//         if (!(layer->error)) {perror("malloc layer e"); exit(1);}
//     }
    
//     for (i = 0; i < batchSize; i++) {
//         layer->gradient[i] = (Matrix**) malloc(sizeof(Matrix*) * outChannels);
//         if (!(layer->gradient[i])) {perror("malloc layer g"); exit(1);}

//         layer->delta[i] = (Matrix**) malloc(sizeof(Matrix*) * outChannels);
//         if (!(layer->delta[i])) {perror("malloc layer d"); exit(1);}

//         layer->error[i] = (Matrix**) malloc(sizeof(Matrix*) * outChannels);
//         if (!(layer->error[i])) {perror("malloc layer e"); exit(1);}
        
//         for (j = 0; j < outChannels; j++) {
//             initMatrix(layer->gradient[i] + j, r, c);
//             initMatrix(layer->delta[i] + j, r, c);
//             initMatrix(layer->error[i] + j, r, c);
//         }
//     }
// }

// void compileModel(ConvolutionalModel *model) {
//     ConvolutionalLayer* curr = model->network->layers->next;
//     for (int i = 0; i < model->network->numLayers; ++i) {
//         if (!curr) break;
//         initLayerGradients(curr, model->batchSize);
//         curr = curr->next;
//     }
// }

// void layerBackward(ConvolutionalLayer* layer, ConvolutionalModel* model) {
//     int batchSize = model->batchSize;
//     int outChannels = layer->outChannels;
//     int r = layer->outputRows;
//     int c = layer->outputCols;
//     int outputSize = r * c;
//     int i, j;

//     for (i = 0; i < batchSize; ++i) {
//         for (j = 0; j < outChannels; ++j) {
//             deviceSigmoidOutputDerivative(layer->outputs[i][j], layer->gradient[i][j], outputSize);
//             deviceHadamardProd(layer->gradient[i][j], layer->error[i][j], layer->gradient[i][j], outputSize);
//         }
//     }
    
// }

// void layerUpdate(ConvolutionalLayer* layer, int batchSize) {
//     return;
// }

// void backward(ConvolutionalModel* model, float*** targets) {
//     ConvolutionalNetwork* net = model->network;
//     int batchSize = model->batchSize;
//     ConvolutionalLayer* curr = net->output;
//     int outputSize = curr->outputRows * curr->outputCols;
//     int i, j;
//     for (i = 0; i < batchSize; ++i) {
//         for (j = 0; j < curr->outChannels; ++j) {
//             setDeviceMatrixData(curr->error[i][j], targets[i][j], outputSize);
//             deviceMatrixSub(curr->outputs[i][j], curr->error[i][j], curr->error[i][j], outputSize);
//             deviceMatrixDivideScalarElementwise(curr->error[i][j], curr->error[i][j], outputSize, outputSize);
//         } 
//     }

//     for (int i = 0; i < net->numLayers; ++i) {
//         if (!curr->prev) break;
//         layerBackward(curr, model);
//         curr = curr->prev;
//       }
//       curr = net->output;
//       for (int i = 0; i < net->numLayers; ++i) {
//         if (!curr->prev) break;
//         layerUpdate(curr, batchSize);
//         curr = curr->prev;
//       }
// }