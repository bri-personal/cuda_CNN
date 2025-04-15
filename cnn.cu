#include "cnn.h"
#include "util.h"
#include <stdio.h>
#include <stdint.h>

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

  void layerForward(ConvolutionalLayer *layer, int sampleNo) {
    /* for each channel of this input sample, do forward pass */

    // TODO: change size when we have image and kernel dimensions
    int size = 0;
    
    int output_channels = layer->k;
    int input_channels = layer->c_in;
    int c, k;
    Matrix temp;
    Matrix *inputImage0, *outputImageK;
    inputImage0 = (layer->prev->outputs)[sampleNo];

    for (k = 0; k < output_channels; k++) {
        outputImageK = (layer->outputs)[sampleNo] + k;

        // TODO: change to convolution
        /* convolve first channel image with first filter */
        deviceMatrixMult(inputImage0, (layer->filters)[k],
            outputImageK, size);
        for (c = 1; c < input_channels; c++) {
            /* for each remaining channel, add the convolution of the image 
             * and filter to the running total
             */
            deviceMatrixMult(inputImage0 + c,
                    (layer->filters)[k] + c, &temp, size);

            deviceMatrixAdd(
                outputImageK,
                &temp,
                outputImageK,
                size
            );
        }
        
        // TODO: change to add scalar to every element
        deviceMatrixAddVec(outputImageK, layer->biases, outputImageK, size);

        deviceSigmoid(outputImageK, outputImageK, size);
    }
    
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
        for (j = 0; j < inputChannels; j++) {
            setDeviceMatrixData((net.layers->outputs)[i] + j, input[i][j], imageSize);
        }
    }
    
    ConvolutionalLayer *curr = net.layers->next; /* first hidden layer */
    for (i = 0; i < net.numLayers; ++i) {
        if (!curr) break;

        /* for each sample in minibatch, go forward */
        for (j = 0; j < batchSize; ++j) {
            layerForward(curr, i);
        }
        curr = curr->next;
    }
  }