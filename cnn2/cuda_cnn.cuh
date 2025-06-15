#ifndef CUDA_CNN_H
#define CUDA_CNN_H
#include <stdint.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "matrix.h"
#include "cuda_matrix.cuh"
#include "cnn.h"

void initConvolutionalModel(ConvolutionalModel** model, int batchSize, float learningRate);

ConvolutionalLayer* createConvolutionalLayer(int batch_size, int outChannels,
        int outputRows, int outputCols, ConvolutionalLayer* prev, curandState_t* state);
void addInputLayer(ConvolutionalModel *model, int channels, int rows, int cols, curandState_t* state);
void addConvLayer(ConvolutionalModel *model, int channels, int rows, int cols, curandState_t* state);

void layerForward(ConvolutionalLayer *layer, int batchSize);
void forward(ConvolutionalModel *model, Tensor4D* input);

void backward(ConvolutionalModel *model, Tensor4D* targets);
void compileModel(ConvolutionalModel *model);
int modelAccuracy(ConvolutionalModel *model, float **images, uint8_t *labels);

#endif