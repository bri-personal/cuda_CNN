#include <iostream>
#include <string.h>
#include "util.h"
#include "matrix.h"
#include "matrix.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#define SIGMOID(x) (1/(1+exp(x * -1)))

/** MEMORY management **/
void initMatrix(Matrix **mat, int height, int width) {
  Matrix temp;
  temp.height = height;
  temp.width = width;

  CERROR( cudaMalloc(&(temp.data), height * width * sizeof(float)) );
  CERROR( cudaMalloc(mat, sizeof(Matrix)) );
  CERROR( cudaMemcpy(*mat, &temp, sizeof(Matrix), cudaMemcpyHostToDevice) );
}
void freeMatrix(Matrix *mat) {
  Matrix temp;
  CERROR( cudaMemcpy(&temp, mat, sizeof(Matrix), cudaMemcpyDeviceToHost) );
  CERROR( cudaFree(temp.data) );
  CERROR( cudaFree(mat) );
}

__global__ void initRandomData(Matrix *mat, float range) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < mat->height * mat->width) {
    curandState_t state;
    curand_init(1234, i, 0, &state);
    mat->data[i] = (float)(range + -((range * 2) * curand_uniform(&state)));
  }
}

void initRandomMatrix(Matrix **mat, int height, int width) {
  initMatrix(mat, height, width);
  initRandomData<<<(height*width + 511) / 512, 512>>>(*mat, 1.0f);
}

__global__ void initZerosData(Matrix *mat) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < mat->height * mat->width)
    mat->data[i] = 0;
}

void initZerosMatrix(Matrix **mat, int height, int width) {
  initMatrix(mat, height, width);
  initZerosData<<<(height*width + 511) / 512, 512>>>(*mat);
}

void getDeviceMatrixData(float *dest, Matrix *source, int n) {
  Matrix temp;
  CERROR( cudaMemcpy(&temp, source, sizeof(Matrix), cudaMemcpyDeviceToHost) );
  CERROR( cudaMemcpy(dest, temp.data, n * sizeof(float), cudaMemcpyDeviceToHost) );
}

void setDeviceMatrixData(Matrix *dest, float *source, int n) {
  Matrix temp;
  CERROR( cudaMemcpy(&temp, dest, sizeof(Matrix), cudaMemcpyDeviceToHost) );
  CERROR( cudaMemcpy(temp.data, source, n * sizeof(float), cudaMemcpyHostToDevice) );
}



/** HELPER **/
__device__ int size(Matrix *mat) {
  return mat->height * mat->width;
}



/** MATH **/
__global__ void matrixMult(Matrix *a, Matrix *b, Matrix *ab) {
  // calculate the row & col index of the element
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= a->height * b->width) return;

  int row = i / b->width;
  int col = i % b->width;
  float result = 0;
  // do dot product between row of a and col of b
  for(int k = 0; k < a->width; ++k)
    result += a->data[row*(a->width)+k] * b->data[k*(b->width)+col];
  ab->data[row * b->width + col] = result;   // (n,m) * (m,p) = (n,p)
}
void deviceMatrixMult(Matrix *a, Matrix *b, Matrix *ab, int N) {
  matrixMult<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b, ab);
  cudaDeviceSynchronize();
  checkError("Matrix mult");
}

__global__ void matrixAdd(Matrix *a, Matrix *b, Matrix *c, int negate) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(c))
    c->data[i] = a->data[i] + (b->data[i] * negate);
}
void deviceMatrixAdd(Matrix *a, Matrix *b, Matrix *c, int N) {
  matrixAdd<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b, c, 1);
  cudaDeviceSynchronize();
  checkError("Matrix add");
}
void deviceMatrixSub(Matrix *a, Matrix *b, Matrix *c, int N) {
  matrixAdd<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b, c, -1);
  cudaDeviceSynchronize();
  checkError("Matrix sub");
}
__global__ void matrixAddVec(Matrix *a, Matrix *b, Matrix *c) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(c)) {
    int row = i / a->width;
    int col = i % a->width;
    c->data[row * a->width + col] = a->data[row * a->width + col] + b->data[col];
  }
}
void deviceMatrixAddVec(Matrix *a, Matrix *b, Matrix *c, int N) {
  matrixAddVec<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b, c);
  cudaDeviceSynchronize();
  checkError("Matrix add vector");
}

__global__ void matrixAddScalarElementwise(Matrix * src, Matrix *dest, float scalar) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(dest))
    dest->data[i] =  src->data[i] + scalar;
}
void deviceMatrixAddScalarElementwise(Matrix *src, Matrix *dest, float scalar, int N) {
  matrixAddScalarElementwise<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(src, dest, scalar);
  cudaDeviceSynchronize();
  checkError("Matrix add scalar elementwise");
}

__global__ void matrixDivideScalarElementwise(Matrix * src, Matrix *dest, float scalar) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(dest))
    dest->data[i] =  src->data[i] / scalar;
}
void deviceMatrixDivideScalarElementwise(Matrix *src, Matrix *dest, float scalar, int N) {
  matrixDivideScalarElementwise<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(src, dest, scalar);
  cudaDeviceSynchronize();
  checkError("Matrix divide scalar elementwise");
}

__global__ void reduceRows(Matrix *x, Matrix *y) {
  int row = threadIdx.x;
  int col = blockIdx.x;
  if (col >= x->width) return;

  extern __shared__ float shared[];

  float result = 0.0f;
  for (int i = row; i < x->height; i += blockDim.x) {
    result += x->data[i * x->width + col];
  }
  shared[row] = result;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (row < s && (row + s) < blockDim.x) {
      shared[row] += shared[row + s];
    }
    __syncthreads();
  }

  if (row == 0) {
    y->data[col] = shared[0];
  }
}
void deviceMatrixReduceRows(Matrix *x, Matrix *y, int height, int width) {
  int blockSize = CONSTRAIN(height / 2, 1, 1024);
  int blockNum = width;
  reduceRows<<<blockNum, blockSize, blockSize>>>(x, y);
  cudaDeviceSynchronize();
  checkError("Reduce Rows");
}

__global__ void matrixScale(Matrix *a, float scale, Matrix *b) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(b))
    b->data[i] = a->data[i] * scale;
}
void deviceMatrixScale(Matrix *a, float scale, Matrix *b, int N) {
  matrixScale<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, scale, b);
  cudaDeviceSynchronize();
  checkError("Matrix scale");
}

__global__ void hadamardProd(Matrix *a, Matrix *b, Matrix *c) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(c))
    c->data[i] = a->data[i] * b->data[i];
}
void deviceHadamardProd(Matrix *a, Matrix *b, Matrix *c, int N) {
  hadamardProd<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b, c);
  cudaDeviceSynchronize();
  checkError("Hadamard");
}

__global__ void sigmoid(Matrix *a, Matrix *b) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(b))
    b->data[i] =  SIGMOID(a->data[i]);
}
void deviceSigmoid(Matrix *a, Matrix *b, int N) {
  sigmoid<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b);
  cudaDeviceSynchronize();
  checkError("Sigmoid");
}

__global__ void sigmoidOutputDerivative(Matrix *a, Matrix *b) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(b)) {
    float x = a->data[i];
    b->data[i] = x * (1 - x);
  }
}
void deviceSigmoidOutputDerivative(Matrix *a, Matrix *b, int N) {
  sigmoidOutputDerivative<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b);
  cudaDeviceSynchronize();
  checkError("Derivative");
}


/** TRANSPOSE **/
__global__ void transpose(Matrix *a, Matrix *b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int height = a->height, width = a->width;
  if (i >= height * width) return;

  int new_i = (i % width) * height + (i / width);  // (curr_col, curr_row)
  b->data[new_i] = a->data[i];
}
void matrixTranpose(Matrix *a, Matrix **b, int arows, int acols) {
  initMatrix(b, acols, arows); // Create matrix with switched height/width
  transpose<<<(arows*acols + 511) / 512, 512>>>(a, *b);
  cudaDeviceSynchronize();
  checkError("Transpose");
}

/** LOSS **/
__global__ void _squareLoss(Matrix *x, float *result) {
  int row = threadIdx.x;
  int col = blockIdx.x;
  if (col >= x->width) return;

  extern __shared__ float shared[];

  float sum = 0.0f;
  for (int i = row; i < x->height; i += blockDim.x) {
    float err = x->data[i * x->width + col];
    sum += err * err;
  }
  shared[row] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (row < s && (row + s) < blockDim.x) {
      shared[row] += shared[row + s];
    }
    __syncthreads();
  }

  if (row == 0) {
    atomicAdd(result, shared[0]);
  }
}
void squareLoss(Matrix *x, float *result, int height, int width) {
  float *y;
  CERROR( cudaMalloc(&y, sizeof(float)) );
  int blockSize = CONSTRAIN(height / 2, 1, 1024);
  int blockNum = width;
  _squareLoss<<<blockNum, blockSize, blockSize>>>(x, y);
  cudaDeviceSynchronize();
  checkError("Loss");
  CERROR( cudaMemcpy(result, y, sizeof(float), cudaMemcpyDeviceToHost) );
}


__global__ void unfoldMatrix(Matrix* m, Matrix* mUnfolded, int kernelCols, int resCols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int height = mUnfolded->height, width = mUnfolded->width;
    int imgCols = m->width;

    while (i < height * width) {
        int r = i / width;
        int c = i % width;
        int j = imgCols * (r / resCols + c / kernelCols) + r % resCols + c % kernelCols;
        mUnfolded->data[i] = m->data[j];
        i += gridDim.x;
    }
}
void deviceUnfoldMatrix(Matrix* img, Matrix** imgUnfolded, int kernelRows, int kernelCols, int resRows, int resCols) {
    int unfoldedRows = resRows * resCols;
    int unfoldedCols = kernelRows * kernelCols;
    initMatrix(imgUnfolded, unfoldedRows, unfoldedCols);
    
    unfoldMatrix<<<(unfoldedRows * unfoldedCols + 511) / 512, 512>>>(img, *imgUnfolded, kernelRows, resCols);
    cudaDeviceSynchronize();
}

void deviceConvolve(Matrix* img, int imgRows, int imgCols,
      Matrix* kernel, int kernelRows, int kernelCols,
      Matrix* result, int stride, int padding) {
    /* im 2 col */
    int resRows = (imgRows - kernelRows + (padding << 1)) / stride + 1;
    int resCols = (imgCols - kernelCols + (padding << 1)) / stride + 1;

    /* unfold image */
    Matrix* imgUnfolded;
    deviceUnfoldMatrix(img, &imgUnfolded, kernelRows, kernelCols, resRows, resCols);

    /* flatten kernel */
    int newKernelRows = kernelRows * kernelCols;
    int newKernelCols = 1;
    // TODO: can we do this better?
    CERROR( cudaMemcpy(&(kernel->height), &newKernelRows, sizeof(int), cudaMemcpyHostToDevice) );
    CERROR( cudaMemcpy(&(kernel->width), &newKernelCols, sizeof(int), cudaMemcpyHostToDevice) );

    /* convolve */
    deviceMatrixMult(imgUnfolded, kernel, result, resRows * resCols);
    freeMatrix(imgUnfolded);

    /* fix matrix dimensions */
    // TODO: can we do this better?
    CERROR( cudaMemcpy(&(kernel->height), &kernelRows, sizeof(int), cudaMemcpyHostToDevice) );
    CERROR( cudaMemcpy(&(kernel->width), &kernelRows, sizeof(int), cudaMemcpyHostToDevice) );
}