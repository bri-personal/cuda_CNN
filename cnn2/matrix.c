#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"


void printMatrix(Matrix* m) {
  for(int h = 0; h < m->height; ++h) {
    for(int w = 0; w < m->width; ++w) {
      printf("%f ", m->data[h*m->width + w]);
    }
    printf("\n");
  }
}

void printImage4D(Tensor4D* i) {
  printTensor4D(i, "Sample", "Channel");
}

void printFilter4D(Tensor4D* f) {
  printTensor4D(f, "Out channel", "In channel");
}

void printTensor4D(Tensor4D* t, char* dim4Text, char* depthText) {
  int tWidth = t->width;
  int areaPerDim3 = t->height * tWidth;
  int volumePerDim4 = t->depth * areaPerDim3;

  for(int n = 0; n < t->dim4; ++n) {
    printf("%s=%d\n", dim4Text, n);
    for(int d = 0; d < t->depth; ++d) {
      printf("- %s=%d\n", depthText, d);
      for(int h = 0; h < t->height; ++h) {
        for(int w = 0; w < tWidth; ++w) {
          printf("%f ", t->data[n*volumePerDim4 + d*areaPerDim3 + h*tWidth + w]);
        }
        printf("\n");
      }
    }
  }
}

void gemm_CPU(Matrix* C, Matrix* A, Matrix* B) {
  /* inner product formulation */
  int cRows = C->height;
  int cCols = C->width;
  int aCols = A->width;
  for (int i = 0; i < cRows; ++i) {
    elem_t* cDataI = C->data + i*cCols;
    elem_t* aDataI = A->data + i*aCols;
    for (int j = 0; j < cCols; ++j) {
      elem_t* bDataJ = B->data + j;
      for (int k = 0; k < aCols; ++k) {
        *(cDataI + j) += *(aDataI + k) * *(bDataJ + k*cCols);
      }
    }
  }
}

void addScalarToEachMatrixOfTensor4D_CPU(Tensor4D* dest, Tensor4D* src, Vector* scalars) {
  /* width of scalar must equal DEPTH of src and dest */
  int dim4 = src->dim4;
  int depth = src->depth;
  int height = src->height;
  int width = src->width;

  for(int n = 0; n < dim4; ++n) {
    for(int k = 0; k < depth; ++k) {
      for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
          int idx = n*depth*height*width + k*height*width + i*width + j;
          dest->data[idx] = src->data[idx] + scalars->data[k];
        }
      }
    }
  }
}

void tensor4DSigmoid_CPU(Tensor4D* dest, Tensor4D* src) {
  int size = src->dim4*src->depth*src->height*src->width;
  for (int i = 0; i < size; ++i) {
    dest->data[i] = SIGMOID(src->data[i]);
  }
}

int matrixEquals(Matrix* m1, Matrix* m2, elem_t delta) {
  if(m1->height!=m2->height || m1->width!=m2->width) {
    return 0;
  }

  int ret = 1;
  for(int i = 0; i < m1->height; ++i) {
    for(int j = 0; j < m1->width; ++j) {
      if(fabsf(m1->data[i*m1->width + j] - m2->data[i*m2->width + j]) > delta) {
        printf("NOT EQUAL: m1[%d][%d] (%f) != m2[%d][%d] (%f) within %f\n",
          i, j, m1->data[i*m1->width + j], i, j, m2->data[i*m2->width + j], delta);
        ret = 0; // keep going to see all differences
      }
    }
  }
  return ret;
}

int tensor4DEquals(Tensor4D* t1, Tensor4D* t2, elem_t delta) {
  int dim4 = t1->dim4;
  int depth = t1->depth;
  int height = t1->height;
  int width = t1->width;

  if(dim4 != t2->dim4 || depth != t2->depth || height!=t2->height || width!=t2->width) {
    printf("NOTE EQUAL: t1 and t2 have different dimensions\n");
    return 0;
  }

  for(int n = 0; n < dim4; ++n) {
    for(int k = 0; k < depth; ++k) {
      for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
          int idx = n*depth*height*width + k*height*width + i*width + j;
          if(fabsf(t1->data[idx] - t2->data[idx]) > delta) {
            printf("NOT EQUAL: t1[%d][%d][%d][%d] (%f) != t2[%d][%d][%d][%d] (%f) within %f\n",
              n, k, i, j, t1->data[idx], n, k, i, j, t2->data[idx], delta);
            return 0;
          }
        }
      }
    }
  }
  return 1;
}

int im2colMatrixEqualsConvTensor4D(Matrix* im2col, Tensor4D* conv, elem_t delta) {
  if(im2col->height != conv->dim4*conv->height*conv->width ||
      im2col->width != conv->depth) {
    return 0;
  }

  int convArea = conv->height*conv->width;
  int convAreaPerSample = convArea * conv->depth;

  for(int n = 0; n < conv->dim4; ++n) {
    for(int k = 0; k < conv->depth; ++k) {
      for(int h = 0; h < conv->height; ++h) {
        for(int w = 0; w < conv->width; ++w) {
          if (fabsf(im2col->data[(n*convArea + h*conv->width + w)*im2col->width + k] - 
                conv->data[n*convAreaPerSample + k*convArea + h*conv->width + w]) > delta) {
            return 0;
          }
        }
      }
    }
  }
  return 1;
}

void im2colUnfold4D_CPU(Matrix* imageUnfolded, Tensor4D* image, int kernelWidth,
  int kernelArea, int outputWidth, int outputArea, int padding, int stride
) {
  //TODO: padding and stride not accounted for
  int imageUnfoldedHeight = imageUnfolded->height;
  int imageUnfoldedWidth = imageUnfolded->width;

  int imageDepth = image->depth;
  int imageHeight = image->height;
  int imageWidth = image->width;

  int imageAreaPerChannel = imageHeight * imageWidth;
  int imageAreaPerSample = imageDepth * imageAreaPerChannel;

  for(int h = 0; h < imageUnfoldedHeight; ++h) {
    int h_mod_output_area = h % outputArea;
    int h_div_output_area = h / outputArea;

    int hImageUnfoldedRows = h * imageUnfoldedWidth;

    for(int w = 0; w < imageUnfoldedWidth; ++w) {
        int w_mod_kernel_area = w % kernelArea;
        imageUnfolded->data[hImageUnfoldedRows + w] = image->data[
            h_div_output_area*imageAreaPerSample +
            (w / kernelArea)*imageAreaPerChannel +
            (h_mod_output_area / outputWidth + w_mod_kernel_area / kernelWidth)*imageWidth +
            (h_mod_output_area % outputWidth + w_mod_kernel_area % kernelWidth)
        ];
    }
  }
}

void im2colFlatten4D_CPU(Matrix* kernelFlattened, Tensor4D* kernel) {
  int outChannels = kernel->dim4;
  int kernelHeight = kernel->height;
  int kernelWidth = kernel->width;

  int kernelAreaPerInputChannel = kernelHeight * kernelWidth;
  int kernelAreaPerOutputChannel = kernelFlattened->height;

  for(int h = 0; h < kernelAreaPerOutputChannel; ++h) {
    int h_div_kernel_area = h / kernelAreaPerInputChannel;
    int h_mod_kernel_area = h % kernelAreaPerInputChannel;

    int hKernelFlattenedRows = h * outChannels;
    for(int w = 0; w < outChannels; ++w) {
      kernelFlattened->data[hKernelFlattenedRows + w] = kernel->data[
        w*kernelAreaPerOutputChannel +
        h_div_kernel_area*kernelAreaPerInputChannel +
        (h_mod_kernel_area / kernelWidth)*kernelWidth +
        (h_mod_kernel_area % kernelWidth)
      ];
    }
  }
}

void conv_CPU(Tensor4D* result, Tensor4D* input, Tensor4D* filter, int padding, int stride) {
  int batchSize = input->dim4;
  int inChannels = input->depth;
  int imageHeight = input->height;
  int imageWidth = input->width;

  int outChannels = filter->dim4;
  int kernelHeight = filter->height;
  int kernelWidth = filter->width;

  int inputAreaPerChannel = imageHeight*imageWidth;
  int inputAreaPerSample = inChannels*inputAreaPerChannel;

  int filterAreaPerInputChannel = kernelHeight*kernelWidth;
  int filterAreaPerOutputChannel = filter->depth*filterAreaPerInputChannel;

  int outputHeight = result->height;
  int outputWidth = result->width;
  int outputAreaPerChannel = outputHeight*outputWidth;
  int outputAreaPerSample = outChannels*outputAreaPerChannel;

  for(int n = 0; n < batchSize; ++n) {
    for(int k = 0; k < outChannels; ++k) {
      for(int h = 0; h + kernelHeight <= imageHeight; h += stride) {
        for(int w = 0; w + kernelWidth <= imageWidth; w += stride) {
          elem_t acc = 0;
          for(int c = 0; c < inChannels; ++c) {
            for(int hk = 0; hk < kernelHeight; ++hk) {
              for(int wk = 0; wk < kernelWidth; ++wk) {
                  acc += input->data[n*inputAreaPerSample + c*inputAreaPerChannel + (h + hk)*imageWidth + (w + wk)] * 
                    filter->data[k*filterAreaPerOutputChannel + c*filterAreaPerInputChannel + hk*kernelWidth + wk];
              }
            }
            result->data[n*outputAreaPerSample + k*outputAreaPerChannel + h*outputWidth + w] = acc;
          }
        } 
      }
    }
  }
}