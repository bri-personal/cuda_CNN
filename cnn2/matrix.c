#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main() {
  elem_t iData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9, 11, 12, 13, 14, 15, 16, 17, 18, 19, -11, -12, -13, -14, -15, -16, -17, -18, -19};
  Tensor4D i = {2, 2, 3, 3, iData};
  printImage4D(&i);

  elem_t kData[] = {0, 1, 1, 0, -1, 0, 0, -1, 2, 0, 0, 2, 0, -2, -2, 0};
  Tensor4D k = {2, 2, 2, 2, kData};
  printFilter4D(&k);

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
  printMatrix(&iUnfolded);

  printf("K flattened\n");
  printMatrix(&kFlattened);

  elem_t oData[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Matrix im2colConvOutput = {8, 2, oData};
  gemm_CPU(&im2colConvOutput, &iUnfolded, &kFlattened);

  printf("Im2Col Conv Output\n");
  printMatrix(&im2colConvOutput);

  elem_t oData2[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Tensor4D convOutput = {2, 2, 2, 2, oData2};

  conv_CPU(&convOutput, &i, &k);

  printf("Naive Conv Output\n");
  printImage4D(&convOutput);

  return 0;
}

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
    printf("Dim4=%d\n", n);
    for(int d = 0; d < t->depth; ++d) {
      printf("- Depth=%d\n", d);
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

void im2colUnfold4D_CPU(Matrix* imageUnfolded, Tensor4D* image, int kernelWidth,
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

void im2colFlatten4D_CPU(Matrix* kernelFlattened, Tensor4D* kernel) {
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

void conv_CPU(Tensor4D* result, Tensor4D* input, Tensor4D* filter) {
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
      for(int h = 0; h < imageHeight; ++h) {
        for(int w = 0; w < imageWidth; ++w) {
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