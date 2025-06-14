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

/* general matrix functions */
void printMatrix(Matrix* m);
void printTensor4D(Tensor4D* t);
void gemm_CPU(Matrix* C, Matrix* A, Matrix* B);

/* im2col functions */
#define OUTPUT_DIM(imageDim, kernelDim, padding, stride) ((imageDim - kernelDim + (padding << 1)) / stride + 1);

void im2colUnfold4D_CPU(Matrix* imageUnfolded, Tensor4D* image, int kernelWidth, int kernelArea, int outputWidth, int outputArea);
void im2colFlatten4D_CPU(Matrix* kernelFlattened, Tensor4D* kernel);

/* convolution functions */
void conv_CPU(Tensor4D* result, Tensor4D* input, Tensor4D* filter);