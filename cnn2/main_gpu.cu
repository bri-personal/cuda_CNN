#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "cuda_matrix.cuh"

#define BATCH_SIZE 2
#define IN_CHANNELS 2
#define IMG_HEIGHT 3
#define IMG_WIDTH 3

#define OUT_CHANNELS 2
#define FILTER_HEIGHT 2
#define FILTER_WIDTH 2

#define PADDING 0
#define STRIDE 1


int main() {
  const int I = 8;
  const int K = 6;
  const int J = 9;

  Matrix hostA = {I, K, (elem_t*) calloc(I*K, sizeof(elem_t))};
  Matrix hostB = {K, J, (elem_t*) calloc(K*J, sizeof(elem_t))};
  Matrix hostC = {I, J, (elem_t*) calloc(I*J, sizeof(elem_t))};
  Matrix hostC2 = {I, J, (elem_t*) calloc(I*J, sizeof(elem_t))};
  
  Matrix *deviceA, *deviceB, *deviceC;
  initRandomMatrix(&deviceA, I, K);
  initRandomMatrix(&deviceB, K, J);
  initZerosMatrix(&deviceC, I, J);

  getDeviceMatrixData(hostA.data, deviceA, I*K);
  getDeviceMatrixData(hostB.data, deviceB, K*J);
  gemm_CPU(hostC, hostA, hostB);

  deviceMatrixMult(deviceA, deviceB, deviceC, I*J);
  getDeviceMatrixData(hostC2.data, deviceC, I*J);

  if(matrixEquals(hostC2, hostC)) {
    printf("CPU and GPU GEMM are equal\n");
  } else {
    printf("CPU and GPU GEMM are NOT equal\n");
    return 1;
  }

  freeMatrix(deviceA);
  freeMatrix(deviceB);
  freeMatrix(deviceC);
  
  return 0;



  // elem_t iData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9, 11, 12, 13, 14, 15, 16, 17, 18, 19, -11, -12, -13, -14, -15, -16, -17, -18, -19};
  // Tensor4D i = {BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH, iData};
  // printImage4D(&i);

  // elem_t kData[] = {0, 1, 1, 0, -1, 0, 0, -1, 2, 0, 0, 2, 0, -2, -2, 0};
  // Tensor4D k = {OUT_CHANNELS, IN_CHANNELS, FILTER_HEIGHT, FILTER_WIDTH, kData};
  // printFilter4D(&k);

  // int outputWidth = OUTPUT_DIM(IMG_WIDTH, FILTER_WIDTH, PADDING, STRIDE);
  // int outputHeight = OUTPUT_DIM(IMG_HEIGHT, FILTER_HEIGHT, PADDING, STRIDE);
  // int outputArea = outputWidth*outputHeight;
  // int kernelArea = k.height*k.width;
  // int imageUnfoldedHeight = outputArea*i.dim4;
  // int imageUnfoldedWidth = i.depth*kernelArea;

  // Matrix iUnfolded = {imageUnfoldedHeight, imageUnfoldedWidth, (elem_t*) calloc(imageUnfoldedHeight*imageUnfoldedWidth, sizeof(elem_t))};
  // Matrix kFlattened = {imageUnfoldedWidth, k.dim4, (elem_t*) calloc(imageUnfoldedWidth*k.dim4, sizeof(elem_t))};

  // im2colUnfold4D_CPU(&iUnfolded, &i, k.width, kernelArea, outputWidth, outputArea);
  // im2colFlatten4D_CPU(&kFlattened, &k);

  // printf("I unfolded\n");
  // printMatrix(&iUnfolded);

  // printf("K flattened\n");
  // printMatrix(&kFlattened);

  // Matrix im2colConvOutput = {iUnfolded.height, kFlattened.width,
  //   (elem_t*) calloc(imageUnfoldedHeight*imageUnfoldedWidth, sizeof(elem_t))};
  // gemm_CPU(&im2colConvOutput, &iUnfolded, &kFlattened);

  // printf("Im2Col Conv Output\n");
  // printMatrix(&im2colConvOutput);

  // Tensor4D convOutput = {BATCH_SIZE, OUT_CHANNELS, outputHeight, outputWidth,
  //   (elem_t*) calloc(BATCH_SIZE*OUT_CHANNELS*outputArea, sizeof(elem_t))};

  // conv_CPU(&convOutput, &i, &k);

  // printf("Naive Conv Output\n");
  // printImage4D(&convOutput);

  // printf("im2colConvOutput == itself: %d\n", matrixEquals(&im2colConvOutput, &im2colConvOutput, 0.000001));
  // printf("im2colConvOutput == convOutput: %d\n", im2colMatrixEqualsConvTensor4D(&im2colConvOutput, &convOutput, 0.000001));

  // free(iUnfolded.data);
  // free(kFlattened.data);
  // free(im2colConvOutput.data);
  // free(convOutput.data);

  return 0;
}