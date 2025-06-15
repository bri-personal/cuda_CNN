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


int gemmTest() {
  const int I = 8;
  const int K = 6;
  const int J = 9;

  Matrix hostA = {I, K, (elem_t*) calloc(I*K, sizeof(elem_t))};
  Matrix hostB = {K, J, (elem_t*) calloc(K*J, sizeof(elem_t))};
  Matrix hostC = {I, J, (elem_t*) calloc(I*J, sizeof(elem_t))};
  Matrix hostC2 = {I, J, (elem_t*) calloc(I*J, sizeof(elem_t))};
  
  Matrix *deviceA, *deviceB, *deviceC;

  curandState_t* state = createCurandStates(I * J * K); // more than are needed
  initRandomMatrix(&deviceA, I, K, state);
  initRandomMatrix(&deviceB, K, J, state);
  cleanupCurandStates(state);

  initMatrix(&deviceC, I, J);

  getDeviceMatrixData(hostA.data, deviceA, I*K);
  getDeviceMatrixData(hostB.data, deviceB, K*J);
  gemm_CPU(&hostC, &hostA, &hostB);

  deviceMatrixMult(deviceA, deviceB, deviceC, I*J);
  getDeviceMatrixData(hostC2.data, deviceC, I*J);

  if(!matrixEquals(&hostC2, &hostC, 0.000001)) {
    printf("FAILURE: CPU and GPU GEMM are NOT equal\n");
    free(hostA.data);
    free(hostB.data);
    free(hostC.data);
    free(hostC2.data);
    freeMatrix(deviceA);
    freeMatrix(deviceB);
    freeMatrix(deviceC);
    return 1;
  }

  printf("SUCCESS: CPU and GPU GEMM are equal\n");
  free(hostA.data);
  free(hostB.data);
  free(hostC.data);
  free(hostC2.data);
  freeMatrix(deviceA);
  freeMatrix(deviceB);
  freeMatrix(deviceC);
  return 0;
}

int im2colUnfoldTest() {
  const int batchSize = 10;
  const int inChannels = 3;
  const int inHeight = 5;
  const int inWidth = 7;

  const int filterHeight = 3;
  const int filterWidth = 5;

  const int outHeight = OUTPUT_DIM(inHeight, filterHeight, 0, 1);
  const int outWidth = OUTPUT_DIM(inWidth, filterWidth, 0, 1);

  const int unfoldedHeight = batchSize*outHeight*outWidth;
  const int unfoldedWidth = inChannels*filterHeight*filterWidth;

  Tensor4D hostInput = {batchSize, inChannels, inHeight, inWidth,
    (elem_t*) calloc(batchSize*inChannels*inHeight*inWidth, sizeof(elem_t))};
  Matrix hostInputUnfolded = {unfoldedHeight, unfoldedWidth,
    (elem_t*) calloc(unfoldedHeight*unfoldedWidth, sizeof(elem_t))};
  Matrix hostInputUnfolded2 = {unfoldedHeight, unfoldedWidth,
    (elem_t*) calloc(unfoldedHeight*unfoldedWidth, sizeof(elem_t))};

  Tensor4D* deviceInput;
  Matrix* deviceInputUnfolded;

  curandState_t* state = createCurandStates(unfoldedHeight*unfoldedWidth);
  initRandomTensor4D(&deviceInput, batchSize, inChannels, inHeight, inWidth, state);
  cleanupCurandStates(state);

  initMatrix(&deviceInputUnfolded, unfoldedHeight, unfoldedWidth);

  getDeviceTensor4DData(hostInput.data, deviceInput, batchSize*inChannels*inHeight*inWidth);
  im2colUnfold4D_CPU(&hostInputUnfolded, &hostInput, filterWidth, filterWidth*filterHeight,
    outWidth, outWidth*outHeight);

  deviceUnfoldImage(deviceInput, deviceInputUnfolded, filterWidth, filterHeight*filterWidth,
    outWidth, outWidth*outHeight, unfoldedWidth, unfoldedWidth*unfoldedHeight);

  getDeviceMatrixData(hostInputUnfolded2.data, deviceInputUnfolded, unfoldedHeight*unfoldedWidth);

  if(!matrixEquals(&hostInputUnfolded2, &hostInputUnfolded, 0.000001)) {
    printf("FAILURE: CPU and GPU im2col unfold are NOT equal\n");
    free(hostInput.data);
    free(hostInputUnfolded.data);
    free(hostInputUnfolded2.data);
    freeTensor4D(deviceInput);
    freeMatrix(deviceInputUnfolded);
    return 1;
  }

  printf("SUCCESS: CPU and GPU im2col unfold are equal\n");
  free(hostInput.data);
  free(hostInputUnfolded.data);
  free(hostInputUnfolded2.data);
  freeTensor4D(deviceInput);
  freeMatrix(deviceInputUnfolded);
  return 0;
}

int im2colFlattenTest() {
  const int inChannels = 3;
  const int outChannels = 4;
  const int filterHeight = 3;
  const int filterWidth = 5;

  const int flattenedHeight = inChannels*filterHeight*filterWidth;
  const int flattenedWidth = outChannels;

  Tensor4D hostKernel = {outChannels, inChannels, filterHeight, filterWidth,
    (elem_t*) calloc(outChannels*inChannels*filterHeight*filterWidth, sizeof(elem_t))};
  Matrix hostKernelFlattened = {flattenedHeight, flattenedWidth,
    (elem_t*) calloc(flattenedHeight*flattenedWidth, sizeof(elem_t))};
  Matrix hostKernelFlattened2 = {flattenedHeight, flattenedWidth,
    (elem_t*) calloc(flattenedHeight*flattenedWidth, sizeof(elem_t))};

  Tensor4D* deviceKernel;
  Matrix* deviceKernelFlattened;

  curandState_t* state = createCurandStates(flattenedHeight*flattenedWidth);
  initRandomTensor4D(&deviceKernel, outChannels, inChannels, filterHeight, filterWidth, state);
  cleanupCurandStates(state);

  initMatrix(&deviceKernelFlattened, flattenedHeight, flattenedWidth);

  getDeviceTensor4DData(hostKernel.data, deviceKernel, outChannels*inChannels*filterHeight*filterWidth);
  im2colFlatten4D_CPU(&hostKernelFlattened, &hostKernel);

  deviceFlattenKernel(deviceKernel, deviceKernelFlattened, flattenedWidth, flattenedWidth*flattenedHeight);

  getDeviceMatrixData(hostKernelFlattened2.data, deviceKernelFlattened, flattenedHeight*flattenedWidth);

  if(!matrixEquals(&hostKernelFlattened2, &hostKernelFlattened, 0.000001)) {
    printf("FAILURE: CPU and GPU im2col flatten are NOT equal\n");
    free(hostKernel.data);
    free(hostKernelFlattened.data);
    free(hostKernelFlattened2.data);
    freeTensor4D(deviceKernel);
    freeMatrix(deviceKernelFlattened);
    return 1;
  }

  printf("SUCCESS: CPU and GPU im2col flatten are equal\n");
  free(hostKernel.data);
  free(hostKernelFlattened.data);
  free(hostKernelFlattened2.data);
  freeTensor4D(deviceKernel);
  freeMatrix(deviceKernelFlattened);
  return 0;
}

int cpuConvTest() {
  // broken right now
  elem_t iData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9, 11, 12, 13, 14, 15, 16, 17, 18, 19, -11, -12, -13, -14, -15, -16, -17, -18, -19};
  Tensor4D i = {BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH, iData};

  elem_t kData[] = {0, 1, 1, 0, -1, 0, 0, -1, 2, 0, 0, 2, 0, -2, -2, 0};
  Tensor4D k = {OUT_CHANNELS, IN_CHANNELS, FILTER_HEIGHT, FILTER_WIDTH, kData};

  int outputWidth = OUTPUT_DIM(IMG_WIDTH, FILTER_WIDTH, PADDING, STRIDE);
  int outputHeight = OUTPUT_DIM(IMG_HEIGHT, FILTER_HEIGHT, PADDING, STRIDE);
  int outputArea = outputWidth*outputHeight;
  int kernelArea = k.height*k.width;
  int imageUnfoldedHeight = outputArea*i.dim4;
  int imageUnfoldedWidth = i.depth*kernelArea;

  Matrix iUnfolded = {imageUnfoldedHeight, imageUnfoldedWidth, (elem_t*) calloc(imageUnfoldedHeight*imageUnfoldedWidth, sizeof(elem_t))};
  Matrix kFlattened = {imageUnfoldedWidth, k.dim4, (elem_t*) calloc(imageUnfoldedWidth*k.dim4, sizeof(elem_t))};

  im2colUnfold4D_CPU(&iUnfolded, &i, k.width, kernelArea, outputWidth, outputArea);
  im2colFlatten4D_CPU(&kFlattened, &k);

  Matrix im2colConvOutput = {iUnfolded.height, kFlattened.width,
    (elem_t*) calloc(imageUnfoldedHeight*imageUnfoldedWidth, sizeof(elem_t))};
  gemm_CPU(&im2colConvOutput, &iUnfolded, &kFlattened);

  Tensor4D convOutput = {BATCH_SIZE, OUT_CHANNELS, outputHeight, outputWidth,
    (elem_t*) calloc(BATCH_SIZE*OUT_CHANNELS*outputArea, sizeof(elem_t))};
  conv_CPU(&convOutput, &i, &k);

  if(!im2colMatrixEqualsConvTensor4D(&im2colConvOutput, &convOutput, 0.000001)) {
    printf("CPU im2col GEMM and CPU conv are NOT equal\n");
    free(iUnfolded.data);
    free(kFlattened.data);
    free(im2colConvOutput.data);
    free(convOutput.data);
    return 1;
  }

  printf("CPU im2col GEMM and CPU conv are equal\n");
  free(iUnfolded.data);
  free(kFlattened.data);
  free(im2colConvOutput.data);
  free(convOutput.data);
  return 0;
}

int convTest() {
  /* test params */
  const int padding = 0;
  const int stride = 1;

  const int batchSize = 10;
  const int inChannels = 3;
  const int inHeight = 5;
  const int inWidth = 7;

  const int outChannels = 4;
  const int filterHeight = 3;
  const int filterWidth = 5;

  const int outHeight = OUTPUT_DIM(inHeight, filterHeight, padding, stride);
  const int outWidth = OUTPUT_DIM(inWidth, filterWidth, padding, stride);

  const int unfoldedHeight = batchSize*outHeight*outWidth;
  const int unfoldedWidth = inChannels*filterHeight*filterWidth;

  const int im2colOutHeight = unfoldedHeight;
  const int im2colOutWidth = outChannels;
  const int im2colOutArea = im2colOutHeight * im2colOutWidth;

  /* set up input image on host */
  Tensor4D hostInput = {batchSize, inChannels, inHeight, inWidth,
    (elem_t*) calloc(batchSize*inChannels*inHeight*inWidth, sizeof(elem_t))};
  if(hostInput.data == NULL) {
    perror("Failed to alloc hostInput data");
    exit(1);
  }
  printf("1\n");

  /* set up filter kernels on host*/
  Tensor4D hostKernel = {outChannels, inChannels, filterHeight, filterWidth,
    (elem_t*) calloc(outChannels*inChannels*filterHeight*filterWidth, sizeof(elem_t))};
  if(hostKernel.data == NULL) {
    perror("Failed to alloc hostKernel data");
    exit(1);
  }
  printf("2\n");

  /* set up input image and filter kernels on device */
  Tensor4D* deviceInput;
  Tensor4D* deviceKernel;
  curandState_t* state = createCurandStates(unfoldedHeight*unfoldedWidth);
  initRandomTensor4D(&deviceInput, batchSize, inChannels, inHeight, inWidth, state);
  initRandomTensor4D(&deviceKernel, outChannels, inChannels, filterHeight, filterWidth, state);
  cleanupCurandStates(state);
  printf("3\n");

  /* set up output feature map on host */
  Tensor4D hostResultTensor4D = {
    batchSize, outChannels, outHeight, outWidth,
    (elem_t*) calloc(batchSize*outChannels*outHeight*outWidth, sizeof(elem_t))
  };
  if(hostResultTensor4D.data == NULL) {
    perror("Failed to alloc hostResultTensor4D data");
    exit(1);
  }
  printf("4\n");
  return 0;
  Matrix hostResultMatrix = {im2colOutHeight, im2colOutWidth,
    (elem_t*) calloc(im2colOutArea, sizeof(elem_t))};
  if(hostResultMatrix.data == NULL) {
    perror("Failed to alloc hostResultMatrix data");
    exit(1);
  }
  printf("5\n");

  /* set up output feature map on device */
  Matrix* deviceResult;
  initMatrix(&deviceResult, im2colOutHeight, im2colOutWidth);
  printf("6\n");

  /* get input and filter elements on host */
  getDeviceTensor4DData(hostInput.data, deviceInput, batchSize*inChannels*inHeight*inWidth);
  getDeviceTensor4DData(hostKernel.data, deviceKernel, outChannels*inChannels*filterHeight*filterWidth);
  printf("7\n");

  /* CPU convolution for comparison */
  conv_CPU(&hostResultTensor4D, &hostInput, &hostKernel);
  printf("8\n");

  /* convolution on device */
  deviceConvolve(deviceInput, deviceKernel, deviceResult, padding, stride);
  printf("9\n");
  getDeviceMatrixData(hostResultMatrix.data, deviceResult, im2colOutArea);
  printf("10\n");

  /* equality check */
  if(!im2colMatrixEqualsConvTensor4D(&hostResultMatrix, &hostResultTensor4D, 0.000001)) {
    printf("FAILURE: CPU and GPU conv are NOT equal\n");
    free(hostInput.data);
    freeTensor4D(deviceInput);
    free(hostKernel.data);
    freeTensor4D(deviceKernel);
    free(hostResultTensor4D.data);
    free(hostResultMatrix.data);
    freeMatrix(deviceResult);
    return 1;
  }

  printf("SUCCESS: CPU and GPU conv are equal\n");
  free(hostInput.data);
  freeTensor4D(deviceInput);
  free(hostKernel.data);
  freeTensor4D(deviceKernel);
  free(hostResultTensor4D.data);
  free(hostResultMatrix.data);
  freeMatrix(deviceResult);
  return 0;
}

int main() {
  int test_total = 0;

  // test_total += gemmTest();
  // test_total += im2colUnfoldTest();
  // test_total += im2colFlattenTest();
  // test_total += cpuConvTest();
  test_total += convTest();

  return test_total;
}