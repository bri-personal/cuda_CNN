#include <iostream>
#include <string.h>
#include "util.h"
#include "matrix.h"
#include "cuda_matrix.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#define SIGMOID(x) (1/(1+exp(x * -1)))

/* Randomness stuff */
/* CLAUDE */
__global__ void setup_kernel(curandState_t *state, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        curand_init(1234, i, 0, state + i);
    }
}

/* CLAUDE */
curandState_t* createCurandStates(int num_elements) {
    curandState_t *state;
    CERROR(cudaMalloc(&state, num_elements * sizeof(curandState_t)));
    
    setup_kernel<<<BLOCKS(num_elements, BLOCKDIM), BLOCKDIM>>>(state, num_elements);
    checkError("setup_kernel failed");
    CERROR(cudaDeviceSynchronize());
    
    return state;
}

/* CLAUDE */
void cleanupCurandStates(curandState_t* state) {
    CERROR(cudaFree(state));
}


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

__global__ void initRandomData(Matrix *mat, float range, curandState_t* state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < mat->height * mat->width) {
    mat->data[i] = (curand_uniform(state + i) * 2.0f - 1.0f) * range;
  }
}

void initRandomMatrix(Matrix **mat, int height, int width, curandState_t* state) {
  initMatrix(mat, height, width);
  initRandomData<<<BLOCKS(height*width, BLOCKDIM), BLOCKDIM>>>(*mat, 1.0f, state);
  checkError("initRandomData kernel failed");
}

__global__ void initZerosData(Matrix *mat) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < mat->height * mat->width)
    mat->data[i] = 0;
}

void initZerosMatrix(Matrix **mat, int height, int width) {
  initMatrix(mat, height, width);
  initZerosData<<<BLOCKS(height*width, BLOCKDIM), BLOCKDIM>>>(*mat);
  checkError("initZerosData kernel failed");
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

/* for Tensor4D */
void initTensor4D(Tensor4D **tensor, int dim4, int depth, int height, int width) {
    Tensor4D temp;
    temp.dim4 = dim4;
    temp.depth = depth;
    temp.height = height;
    temp.width = width;
    
    int total_elements = dim4 * depth * height * width;
    CERROR(cudaMalloc(&(temp.data), total_elements * sizeof(elem_t)));
    CERROR(cudaMalloc(tensor, sizeof(Tensor4D)));
    CERROR(cudaMemcpy(*tensor, &temp, sizeof(Tensor4D), cudaMemcpyHostToDevice));
}

void freeTensor4D(Tensor4D *tensor) {
    Tensor4D temp;
    CERROR(cudaMemcpy(&temp, tensor, sizeof(Tensor4D), cudaMemcpyDeviceToHost));
    CERROR(cudaFree(temp.data));
    CERROR(cudaFree(tensor));
}

__global__ void initRandomDataTensor4D(Tensor4D *tensor, float range, curandState_t* state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int total_elements = tensor->dim4 * tensor->depth * tensor->height * tensor->width;
    if (i < total_elements) {
        tensor->data[i] = (curand_uniform(state + i) * 2.0f - 1.0f) * range;
    }
}

void initRandomTensor4D(Tensor4D **tensor, int dim4, int depth, int height, int width, curandState_t* state) {
    initTensor4D(tensor, dim4, depth, height, width);
    int total_elements = dim4 * depth * height * width;
    initRandomDataTensor4D<<<BLOCKS(total_elements, BLOCKDIM), BLOCKDIM>>>(*tensor, 1.0f, state);
    checkError("initRandomDataTensor4D kernel failed");
}

__global__ void initZerosDataTensor4D(Tensor4D *tensor) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int total_elements = tensor->dim4 * tensor->depth * tensor->height * tensor->width;
    if (i < total_elements) {
        tensor->data[i] = 0;
    }
}

void initZerosTensor4D(Tensor4D **tensor, int dim4, int depth, int height, int width) {
    initTensor4D(tensor, dim4, depth, height, width);
    int total_elements = dim4 * depth * height * width;
    initZerosDataTensor4D<<<BLOCKS(total_elements, BLOCKDIM), BLOCKDIM>>>(*tensor);
    checkError("initZerosDataTensor4D kernel failed");
}

void getDeviceTensor4DData(elem_t *dest, Tensor4D *source, int n) {
    Tensor4D temp;
    CERROR(cudaMemcpy(&temp, source, sizeof(Tensor4D), cudaMemcpyDeviceToHost));
    CERROR(cudaMemcpy(dest, temp.data, n * sizeof(elem_t), cudaMemcpyDeviceToHost));
}

void setDeviceTensor4DData(Tensor4D *dest, elem_t *source, int n) {
    Tensor4D temp;
    CERROR(cudaMemcpy(&temp, dest, sizeof(Tensor4D), cudaMemcpyDeviceToHost));
    CERROR(cudaMemcpy(temp.data, source, n * sizeof(elem_t), cudaMemcpyHostToDevice));
}

/* for vector */
void initVector(Vector **v, int width) {
  Vector temp;
  temp.width = width;

  CERROR( cudaMalloc(&(temp.data), width * sizeof(elem_t)) );
  CERROR( cudaMalloc(v, sizeof(Vector)) );
  CERROR( cudaMemcpy(*v, &temp, sizeof(Vector), cudaMemcpyHostToDevice) );
}
void freeVector(Vector *v) {
  Vector temp;
  CERROR( cudaMemcpy(&temp, v, sizeof(Vector), cudaMemcpyDeviceToHost) );
  CERROR( cudaFree(temp.data) );
  CERROR( cudaFree(v) );
}

__global__ void initRandomDataVector(Vector *v, float range, curandState_t* state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < v->width) {
    v->data[i] = (curand_uniform(state + i) * 2.0f - 1.0f) * range;
  }
}

void initRandomVector(Vector **v, int width, curandState_t* state) {
  initVector(v, width);
  initRandomDataVector<<<BLOCKS(width, BLOCKDIM), BLOCKDIM>>>(*v, 1.0f, state);
  checkError("initRandomDataVector kernel failed");
}

void getDeviceVectorData(float *dest, Vector *source, int n) {
    Vector temp;
    CERROR(cudaMemcpy(&temp, source, sizeof(Vector), cudaMemcpyDeviceToHost));
    CERROR(cudaMemcpy(dest, temp.data, n * sizeof(elem_t), cudaMemcpyDeviceToHost));
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
  checkError("Matrix add scalar elementwise each col");
}

__global__ void matrixAddScalarColumnwise(Matrix* src, Matrix *dest, Vector* scalars, int rows, int cols) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(dest))
    dest->data[i] =  src->data[i] + scalars->data[i%cols];
}
void deviceMatrixAddScalarColumnwise(Matrix* src, Matrix *dest, Vector* scalars, int rows, int cols) {;
  matrixAddScalarColumnwise<<<BLOCKS(rows*cols, BLOCKDIM), BLOCKDIM>>>(src, dest, scalars, rows, cols);
  cudaDeviceSynchronize();
  checkError("Matrix add scalar elementwise each col");
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
  transpose<<<BLOCKS(arows*acols, BLOCKDIM), BLOCKDIM>>>(a, *b);
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

/* Convolution */
__global__ void flattenKernel(Tensor4D* kernel, Matrix* kernelFlattened,
    int flattenedWidth, int flattenedArea
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int inChannels = kernel->depth;
    int kernelHeight = kernel->height;
    int kernelWidth = kernel->width;

    int kernelAreaPerInputChannel = kernelHeight * kernelWidth;
    int kernelAreaPerOutputChannel = inChannels * kernelAreaPerInputChannel;

    while (i < flattenedArea) {
        int h = i / flattenedWidth;
        int w = i % flattenedWidth;

        int h_mod_kernel_area = h % kernelAreaPerInputChannel;

        kernelFlattened->data[i] = kernel->data[
            w*kernelAreaPerOutputChannel +
            (h / kernelAreaPerInputChannel)*kernelAreaPerInputChannel +
            (h_mod_kernel_area / kernelWidth)*kernelWidth +
            (h_mod_kernel_area % kernelWidth)
        ];

        i += gridDim.x*blockDim.x;
    }
}
void deviceFlattenKernel(Tensor4D* kernel, Matrix* kernelFlattened,
    int flattenedWidth, int flattenedArea
) {
    flattenKernel<<<BLOCKS(flattenedArea, BLOCKDIM), BLOCKDIM>>>(
        kernel, kernelFlattened, flattenedWidth, flattenedArea);
    cudaDeviceSynchronize();
}

__global__ void unfoldImage(Tensor4D* img, Matrix* imgUnfolded,
        int kernelWidth, int kernelArea, int outputWidth, int outputArea,
        int unfoldedWidth, int unfoldedArea, int padding, int stride
) {
    // TODO: padding and stride not accounted for
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int imageHeight = img->height;
    int imageWidth = img->width;
    int imageChannels = img->depth;
    int imageAreaPerChannel = imageHeight*imageWidth;
    int imageAreaPerSample = imageAreaPerChannel*imageChannels;

    while (i < unfoldedArea) {
        int h = i / unfoldedWidth;
        int w = i % unfoldedWidth;

        int h_mod_output_area = h % outputArea;
        int w_mod_kernel_area = w % kernelArea;

        int inputRow = h_mod_output_area / outputWidth + w_mod_kernel_area / kernelWidth;
        int inputCol = h_mod_output_area % outputWidth + w_mod_kernel_area % kernelWidth;

        if(inputRow >=0 && inputRow < imageHeight && inputCol >= 0 && inputCol <= imageWidth) {
            imgUnfolded->data[i] = img->data[
                (h / outputArea)*imageAreaPerSample +
                (w / kernelArea)*imageAreaPerChannel +
                inputRow*imageWidth +
                inputCol
            ];
        } else {
            imgUnfolded->data[i] = 0;
        }

        i += gridDim.x*blockDim.x;
    }
}
void deviceUnfoldImage(Tensor4D* img, Matrix* imgUnfolded,
    int kernelWidth, int kernelArea,
    int outWidth, int outArea,
    int unfoldedWidth, int unfoldedArea,
    int padding, int stride
) {
    unfoldImage<<<BLOCKS(unfoldedArea, BLOCKDIM), BLOCKDIM>>>(
        img, imgUnfolded, kernelWidth, kernelArea, outWidth, outArea,
        unfoldedWidth, unfoldedArea, padding, stride);
    cudaDeviceSynchronize();
}

void deviceConvolve(
    Tensor4D* img, Tensor4D* kernel, Matrix* result,
    int padding, int stride,
    int inChannels, int imgHeight, int imgWidth,
    int outChannels, int kernelHeight, int kernelWidth,
    int resHeight, int resWidth
) {
    /* get dimensions */
    int kernelArea = kernelHeight * kernelWidth;

    int outWidth = OUTPUT_DIM(imgWidth, kernelWidth, padding, stride);
    int outArea = outWidth*OUTPUT_DIM(imgHeight, kernelHeight, padding, stride);

    /* im 2 col */
    int resArea = resHeight * resWidth;

    /* unfold image */
    int unfoldedWidth = inChannels*kernelArea;
    int unfoldedArea = resHeight * unfoldedWidth;

    Matrix* imgUnfolded;
    initMatrix(&imgUnfolded, resHeight, unfoldedWidth);

    deviceUnfoldImage(img, imgUnfolded, kernelWidth, kernelArea,
        outWidth, outArea, unfoldedWidth, unfoldedArea, padding, stride);

    /* flatten kernel */
    int flattenedArea = unfoldedWidth*outChannels;

    Matrix* kernelFlattened;
    initMatrix(&kernelFlattened, unfoldedWidth, outChannels);
    deviceFlattenKernel(kernel, kernelFlattened, outChannels, flattenedArea);

    /* GEMM */
    deviceMatrixMult(imgUnfolded, kernelFlattened, result, resArea);

    /* cleanup */
    freeMatrix(imgUnfolded);
    freeMatrix(kernelFlattened);
}

__global__ void reorderIm2ColToConv(Matrix* src, Tensor4D* dest, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int channels = dest->depth;
  int height = dest->height;
  int width = dest->width;
  
  int destAreaPerChannel = height*width;
  int destAreaPerSample = channels*destAreaPerChannel;

  int srcWidth = src->width;

  while(i < N) {
    int n = i / destAreaPerSample;
    int inSampleIdx = i % destAreaPerSample;
    int k = inSampleIdx / destAreaPerChannel;
    int inChannelIdx = inSampleIdx % destAreaPerChannel;
    int h = inChannelIdx / width;
    int w = inChannelIdx % width;

    dest->data[i] = src->data[(n*destAreaPerChannel + h*width + w)*srcWidth + k];

    i += gridDim.x*blockDim.x;
  }
}
void deviceReorderIm2ColToConv(Matrix* src, Tensor4D* dest, int N) {
  reorderIm2ColToConv<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(src, dest, N);
  cudaDeviceSynchronize();
}