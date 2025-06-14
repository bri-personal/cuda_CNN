#ifndef MATRIX_H
#define MATRIX_H

#define SIGMOID(x) (1/(1+exp(x * -1)))

/** STRUCTS **/
typedef struct {
  float *data;
  int rows;
  int cols;
} Matrix;

/** MEMORY management **/
void initMatrix(Matrix **mat, int rows, int cols);
void freeMatrix(Matrix *mat);
void initRandomMatrix(Matrix **mat, int rows, int cols);
void initZerosMatrix(Matrix **mat, int rows, int cols);
void getDeviceMatrixData(float *dest, Matrix *source, int n);
void setDeviceMatrixData(Matrix *dest, float *source, int n);

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
void deviceMatrixReduceRows(Matrix *x, Matrix *y, int rows, int cols);

/** TRANSPOSE **/
void matrixTranpose(Matrix *a, Matrix **b, int arows, int acols);

/** LOSS **/
void squareLoss(Matrix *x, float *result, int rows, int cols);

/** CONVOLUTION **/
void deviceUnfoldMatrix(Matrix* img, Matrix** imgUnfolded, int kernelRows, int kernelCols, int resRows, int resCols);
void deviceConvolve(Matrix* img, int imgRows, int imgCols,
  Matrix* kernel, int kernelRows, int kernelCols,
  Matrix* result, int stride, int padding);

#endif