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
void cleanupCurandStates(curandState_t* state)

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

/** HELPER **/
__device__ int size(Matrix *mat);

/** MATH **/
void deviceMatrixMult(Matrix *a, Matrix *b, Matrix *ab, int N);
void deviceMatrixAdd(Matrix *a, Matrix *b, Matrix *c, int N);
void deviceMatrixSub(Matrix *a, Matrix *b, Matrix *c, int N);
void deviceMatrixScale(Matrix *a, float scale, Matrix *b, int N);
void deviceMatrixAddScalarElementwise(Matrix *src, Matrix *dest, float scalar, int N);
void deviceMatrixDivideScalarElementwise(Matrix *src, Matrix *dest, float scalar, int N);
void deviceHadamardProd(Matrix *a, Matrix *b, Matrix *c, int N);
void deviceSigmoid(Matrix *a, Matrix *b, int N);
void deviceSigmoidOutputDerivative(Matrix *a, Matrix *b, int N);
void deviceMatrixAddVec(Matrix *a, Matrix *b, Matrix *c, int N);
void deviceMatrixReduceRows(Matrix *x, Matrix *y, int height, int width);

/** TRANSPOSE **/
void matrixTranpose(Matrix *a, Matrix **b, int arows, int acols);

/** LOSS **/
void squareLoss(Matrix *x, float *result, int height, int width);

/** CONVOLUTION **/
void deviceUnfoldMatrix(Matrix* img, Matrix** imgUnfolded, int kernelRows, int kernelCols, int resRows, int resCols);
void deviceConvolve(Matrix* img, int imgRows, int imgCols,
  Matrix* kernel, int kernelRows, int kernelCols,
  Matrix* result, int stride, int padding);
  
#endif