#ifndef MATRIX_CUDA_H
#define MATRIX_CUDA_H

#include <cuda.h>
#include <curand_kernel.h>
#include "matrix.h"

/* GPU params and macros */
#define BLOCKDIM 512
#define CONSTRAIN(x, min, max) (x < min ? min : (x > max ? max : x))


/* GPU functions */
/* random stuff */
curandState_t* createCurandStates(int num_elements);
void cleanupCurandStates(curandState_t* state);

/** MEMORY management **/
/* for Matrix */
void initMatrix(Matrix **mat, int height, int width);
void freeMatrix(Matrix *mat);
void initRandomMatrix(Matrix **mat, int height, int width, curandState_t* state);
void initZerosMatrix(Matrix **mat, int height, int width);
void getDeviceMatrixData(float *dest, Matrix *source, int n);
void setDeviceMatrixData(Matrix *dest, float *source, int n);

/* for Tensor4D */
void initTensor4D(Tensor4D **tensor, int dim4, int depth, int height, int width);
void freeTensor4D(Tensor4D *tensor);
__global__ void initRandomDataTensor4D(Tensor4D *tensor, float range, curandState_t* state);
void initRandomTensor4D(Tensor4D **tensor, int dim4, int depth, int height, int width, curandState_t* state);
__global__ void initZerosDataTensor4D(Tensor4D *tensor);
void initZerosTensor4D(Tensor4D **tensor, int dim4, int depth, int height, int width);
void getDeviceTensor4DData(elem_t *dest, Tensor4D *source, int n);
void setDeviceTensor4DData(Tensor4D *dest, elem_t *source, int n);

/* for Vector */
void initVector(Vector **v, int width);
void freeVector(Vector *v);
void initRandomVector(Vector **v, int width, curandState_t* state);
void getDeviceVectorData(float *dest, Vector *source, int n);

/** HELPER **/
__device__ int size(Matrix *mat);
__device__ int tensor4DSize(Tensor4D *t);

/** MATH **/
void deviceMatrixMult(Matrix *a, Matrix *b, Matrix *ab, int N);
void deviceMatrixAdd(Matrix *a, Matrix *b, Matrix *c, int N);
void deviceMatrixSub(Matrix *a, Matrix *b, Matrix *c, int N);
void deviceTensor4DAdd(Tensor4D *a, Tensor4D *b, Tensor4D *c, int N);
void deviceTensor4DSub(Tensor4D *a, Tensor4D *b, Tensor4D *c, int N);
void deviceMatrixScale(Matrix *a, float scale, Matrix *b, int N);
void deviceMatrixAddScalarElementwise(Matrix *src, Matrix *dest, float scalar, int N);
void deviceMatrixAddScalarColumnwise(Matrix* src, Matrix *dest, Vector* scalars, int rows, int cols);
void deviceMatrixDivideScalarElementwise(Matrix *src, Matrix *dest, float scalar, int N);
void deviceTensor4DDivideScalarElementwise(Tensor4D *src, Tensor4D *dest, elem_t scalar, int N);
void deviceHadamardProd(Matrix *a, Matrix *b, Matrix *c, int N);
void deviceTensor4DHadamardProd(Tensor4D *a, Tensor4D *b, Tensor4D *c, int N);
void deviceSigmoid(Matrix *a, Matrix *b, int N);
void deviceSigmoidOutputDerivative(Matrix *a, Matrix *b, int N);
void deviceTensor4DSigmoidOutputDerivative(Tensor4D *src, Tensor4D *dest, int N);
void deviceMatrixAddVec(Matrix *a, Matrix *b, Matrix *c, int N);
void deviceMatrixReduceRows(Matrix *x, Matrix *y, int height, int width);

/** TRANSPOSE **/
void matrixTranpose(Matrix *a, Matrix **b, int arows, int acols);

/** LOSS **/
void squareLoss(Matrix *x, float *result, int height, int width);

/** CONVOLUTION **/
void deviceUnfoldImage(Tensor4D* img, Matrix* imgUnfolded,
    int kernelWidth, int kernelArea,
    int outWidth, int outArea,
    int unfoldedWidth, int unfoldedArea,
    int padding, int stride
);
void deviceFlattenKernel(Tensor4D* kernel, Matrix* kernelFlattened,
    int flattenedWidth, int flattenedArea
);
void deviceConvolve(
    Tensor4D* img, Tensor4D* kernel, Matrix* result,
    int padding, int stride,
    int inChannels, int imgHeight, int imgWidth,
    int outChannels, int kernelHeight, int kernelWidth,
    int resHeight, int resWidth
);
void deviceReorderIm2ColToConv(Matrix* src, Tensor4D* dest, int n);
  
#endif