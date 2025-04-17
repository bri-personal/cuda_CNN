#ifndef CNN_H
#define CNN_H
#include "matrix.h"
#include <stdint.h>

typedef struct ConvolutionalLayer {
  int c_in; /* input channels */
  int k; /* output channels */
  int imgRows, imgCols, kernelRows, kernelCols, outputRows, outputCols; /* size params */
  struct ConvolutionalLayer *prev;

  /* for k output channels, have a kernel for each of the c_in input channels*/
  Matrix*** filters;

  /* for k output channels, have a constant to add to each element of the conv result */
  float* biases;

  /* for each sample of minibatch, have k output images */
  Matrix*** outputs;

  /* for each sample of minibatch, have c_in input image gradients */
  Matrix*** delta;

  //Matrix *gradient;
  //Matrix *error;

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
  int input_c;
  int input_h;
  int input_w;

  int hidden;

  /* output image dimensions */
  int output_c;
  int output_h;
  int output_w;

  float learningRate;
  int batchSize;
} ConvolutionalModel;

ConvolutionalModel *initModel(int batchSize, float learningRate);
ConvolutionalLayer* createConvolutionalLayer(int batch_size, int c_in, int k,
  int outputRows, int outputCols, ConvolutionalLayer* prev);

void addInputLayer(ConvolutionalModel *model, int size);
void addDenseLayer(ConvolutionalModel *model, int size);

void layerForward(ConvolutionalLayer *layer, int sampleNo);
void forward(ConvolutionalModel *model, float ***input);

float backward(ConvolutionalModel *model, float *target);
void compileModel(ConvolutionalModel *model);
int modelAccuracy(ConvolutionalModel *model, float **images, uint8_t *labels);

#endif