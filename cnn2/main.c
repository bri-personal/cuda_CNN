#include <stdio.h>
#include "matrix.h"

int main() {
  elem_t iData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9, 11, 12, 13, 14, 15, 16, 17, 18, 19, -11, -12, -13, -14, -15, -16, -17, -18, -19};
  Tensor4D i = {2, 2, 3, 3, iData};
  printImage4D(&i);

  elem_t kData[] = {0, 1, 1, 0, -1, 0, 0, -1, 2, 0, 0, 2, 0, -2, -2, 0};
  Tensor4D k = {2, 2, 2, 2, kData};
  printFilter4D(&k);

  int outputWidth = OUTPUT_DIM(3, 2, 0, 1);
  int outputHeight = outputWidth;
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

  elem_t oData[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Matrix im2colConvOutput = {8, 2, oData};
  gemm_CPU(&im2colConvOutput, &iUnfolded, &kFlattened);

  printf("Im2Col Conv Output\n");
  printMatrix(&im2colConvOutput);

  elem_t oData2[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Tensor4D convOutput = {2, 2, 2, 2, oData2};

  conv_CPU(&convOutput, &i, &k);

  printf("Naive Conv Output\n");
  printImage4D(&convOutput);

  return 0;
}