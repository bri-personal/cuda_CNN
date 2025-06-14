#ifndef MATRIX_H
#define MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

/* data type used in all tensors */
typedef float elem_t;


/* structs for data structures */
typedef struct {
  int length;
  elem_t* data;
} Vector;

typedef struct {
  int height;
  int width;
  elem_t* data;
} Matrix;

typedef struct {
  int depth;
  int height;
  int width;
  elem_t* data;
} Tensor3D;

typedef struct {
  int dim4;
  int depth;
  int height;
  int width;
  elem_t* data;
} Tensor4D;


/* CPU general matrix functions */
void printMatrix(Matrix* m);
void printTensor4D(Tensor4D* t, char* dim4Text, char* depthText);
void printImage4D(Tensor4D* i);
void printFilter4D(Tensor4D* f);
int matrixEquals(Matrix* m1, Matrix* m2, elem_t delta);
void gemm_CPU(Matrix* C, Matrix* A, Matrix* B);


/* CPU im2col functions */
#define OUTPUT_DIM(imageDim, kernelDim, padding, stride) ((imageDim - kernelDim + (padding << 1)) / stride + 1);
int im2colMatrixEqualsConvTensor4D(Matrix* im2col, Tensor4D* conv, elem_t delta);
void im2colUnfold4D_CPU(Matrix* imageUnfolded, Tensor4D* image, int kernelWidth, int kernelArea, int outputWidth, int outputArea);
void im2colFlatten4D_CPU(Matrix* kernelFlattened, Tensor4D* kernel);


/* CPU convolution functions */
void conv_CPU(Tensor4D* result, Tensor4D* input, Tensor4D* filter);

#ifdef __cplusplus
}
#endif

#endif