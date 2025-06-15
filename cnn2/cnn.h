#ifndef CNN_H
#define CNN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "matrix.h"

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

ConvolutionalLayer* createConvolutionalLayerCPU(int batch_size, int outChannels,
        int outputRows, int outputCols, ConvolutionalLayer* prev, Tensor4D* filters, Vector* biases);
void addInputLayerCPU(ConvolutionalModel *model, int channels, int rows, int cols);
void addConvLayerCPU(ConvolutionalModel *model, int channels, int rows, int cols, Tensor4D* filters, Vector* biases);

void layerForwardCPU(ConvolutionalLayer *layer, int batchSize);
void forwardCPU(ConvolutionalModel *model, Tensor4D* input);

#ifdef __cplusplus
}
#endif

#endif