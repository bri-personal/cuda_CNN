#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

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
  elem_t iData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9, 11, 12, 13, 14, 15, 16, 17, 18, 19, -11, -12, -13, -14, -15, -16, -17, -18, -19};
  Tensor4D i = {BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH, iData};
  printImage4D(&i);

  elem_t kData[] = {0, 1, 1, 0, -1, 0, 0, -1, 2, 0, 0, 2, 0, -2, -2, 0};
  Tensor4D k = {OUT_CHANNELS, IN_CHANNELS, FILTER_HEIGHT, FILTER_WIDTH, kData};
  printFilter4D(&k);

  int outputWidth = OUTPUT_DIM(IMG_WIDTH, FILTER_WIDTH, PADDING, STRIDE);
  int outputHeight = OUTPUT_DIM(IMG_HEIGHT, FILTER_HEIGHT, PADDING, STRIDE);
  int outputArea = outputWidth*outputHeight;
  int kernelArea = k.height*k.width;
  int imageUnfoldedHeight = outputArea*i.dim4;
  int imageUnfoldedWidth = i.depth*kernelArea;

  Matrix iUnfolded = {imageUnfoldedHeight, imageUnfoldedWidth, calloc(imageUnfoldedHeight*imageUnfoldedWidth, sizeof(elem_t))};
  Matrix kFlattened = {imageUnfoldedWidth, k.dim4, calloc(imageUnfoldedWidth*k.dim4, sizeof(elem_t))};

  im2colUnfold4D_CPU(&iUnfolded, &i, k.width, kernelArea, outputWidth, outputArea);
  im2colFlatten4D_CPU(&kFlattened, &k);

  printf("I unfolded\n");
  printMatrix(&iUnfolded);

  printf("K flattened\n");
  printMatrix(&kFlattened);

  Matrix im2colConvOutput = {iUnfolded.height, kFlattened.width,
    calloc(imageUnfoldedHeight*imageUnfoldedWidth, sizeof(elem_t))};
  gemm_CPU(&im2colConvOutput, &iUnfolded, &kFlattened);

  printf("Im2Col Conv Output\n");
  printMatrix(&im2colConvOutput);

  Tensor4D convOutput = {BATCH_SIZE, OUT_CHANNELS, outputHeight, outputWidth,
    calloc(BATCH_SIZE*OUT_CHANNELS*outputArea, sizeof(elem_t))};

  conv_CPU(&convOutput, &i, &k);

  printf("Naive Conv Output\n");
  printImage4D(&convOutput);

  printf("convOutput == itself: %d", matrixIsEqual(&convOutput, &convOutput));

  free(iUnfolded.data);
  free(kFlattened.data);
  free(im2colConvOutput.data);
  free(convOutput.data);

  return 0;
}