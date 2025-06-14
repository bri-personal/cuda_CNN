#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main() {
  elem_t iData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9, 11, 12, 13, 14, 15, 16, 17, 18, 19, -11, -12, -13, -14, -15, -16, -17, -18, -19};
  Tensor4 i = {2, 2, 3, 3, iData};

  elem_t kData[] = {0, 1, 1, 0, -1, 0, 0, -1, 2, 0, 0, 2, 0, -2, -2, 0};
  Tensor4 k = {2, 2, 2, 2, kData};

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
  for(int h = 0; h < iUnfolded.height; ++h) {
    for(int w = 0; w < iUnfolded.width; ++w) {
      printf("%f ", iUnfolded.data[h*iUnfolded.width + w]);
    }
    printf("\n");
  }

  printf("K flattened\n");
  for(int h = 0; h < kFlattened.height; ++h) {
    for(int w = 0; w < kFlattened.width; ++w) {
      printf("%f ", kFlattened.data[h*kFlattened.width + w]);
    }
    printf("\n");
  }

  elem_t oData[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Matrix o = {8, 2, oData};
  gemm_CPU(&o, &iUnfolded, &kFlattened);

  printf("O\n");
  for(int h = 0; h < o.height; ++h) {
    for(int w = 0; w < o.width; ++w) {
      printf("%f ", o.data[h*o.width + w]);
    }
    printf("\n");
  }

  return 0;
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

void im2colUnfold4D_CPU(Matrix* imageUnfolded, Tensor4* image, int kernelWidth,
  int kernelArea, int outputWidth, int outputArea) {
  int imageUnfoldedHeight = imageUnfolded->height;
  int imageUnfoldedWidth = imageUnfolded->width;

  int imageDim4 = image->dim4;
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

void im2colFlatten4D_CPU(Matrix* kernelFlattened, Tensor4* kernel) {
  int outChannels = kernel->dim4;
  int inChannels = kernel->depth;
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
