#ifndef CNN_H
#define CNN_H
#include <stdint.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "matrix.h"
#include "cuda_matrix.cuh"

typedef struct ConvolutionalLayer {
  int inChannels; /* input channels */
  int outChannels; /* output channels */
  int imgRows, imgCols, kernelRows, kernelCols, outputRows, outputCols; /* size params */
  struct ConvolutionalLayer *prev;

  /* for outChannels output channels, have a kernel for each of the inChannels input channels*/
  Tensor4D* filters;

  /* for outChannels output channels, have a constant to add to each element of the conv result */
  Vector* biases;

  /* for each sample of minibatch, have outChannels output images */
  Tensor4D* outputs;

  /* for each sample of minibatch, have inChannels input image gradients */
  Tensor4D* delta;

  Tensor4D* gradient;

  Tensor4D* error;

  struct ConvolutionalLayer *next;
} ConvolutionalLayer;

typedef struct {
  ConvolutionalLayer *input;
  ConvolutionalLayer *layers;
  ConvolutionalLayer *output;
  int numLayers;
} ConvolutionalNetwork;

typedef struct {
  ConvolutionalNetwork *network;
  /* input image dimensions */
  int batchSize;
  int inChannels;
  int inHeight;
  int inWidth;

  /* output image dimensions */
  int outChannels;
  int outHeight;
  int outWidth;

  float learningRate;
} ConvolutionalModel;

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