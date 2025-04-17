#include "util.h"
#include "matrix.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

void printArray(float *a, int n) {
  printf("[ ");
  for (int i = 0; i < n; ++i)
    printf("%d ", (int)a[i]);
  printf("]\n");
}

void test_matrixMult() {
  // (4,2)*(2,3) = (4,3)
  float a[8] = {-2,-1,
                 0, 1,
                 2, 3,
                 4, 5};
  float b[6] = {-1,0,1,
                 0,1,0};
  Matrix *A, *B, *C;
  initMatrix(&A, 4, 2);
  initMatrix(&B, 2, 3);
  initMatrix(&C, 4, 3);
  setDeviceMatrixData(A, a, 8);
  setDeviceMatrixData(B, b, 6);

  deviceMatrixMult(A, B, C, 12);

  float c[12];
  getDeviceMatrixData(c, C, 12);

  char result[64];
  char expected[64] = "2 -1 -2 0 1 0 -2 3 2 -4 5 4";
  int offset = 0;
  for (int i = 0; i < 12; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)c[i]);
  }
  printf("Testing matrix mult\n");
  printf("Result: %s\n", result);
  printf("Expect: %s\n", expected);
  if (strncmp(result, expected, 27) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }
  printf("\nPASSED\n\n");

  freeMatrix(A);
  freeMatrix(B);
  freeMatrix(C);
}

void test_matrixElementWise() {
  Matrix *A, *B, *C;
  float data[6] = {0,1,2,3,4,5};

  initMatrix(&A, 2, 3);
  initMatrix(&B, 2, 3);
  initMatrix(&C, 2, 3);
  setDeviceMatrixData(A, data, 6);
  setDeviceMatrixData(B, data, 6);

  deviceMatrixAdd(A, B, C, 6);

  float c[6];
  getDeviceMatrixData(c, C, 6);

  char result[64];
  int offset = 0;
  for (int i = 0; i < 6; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)c[i]);
  }
  printf("Testing matrix add\n");
  printf("Result: %s\n", result);
  printf("Expect: 0 2 4 6 8 10\n");
  if (strncmp(result, "0 2 4 6 8 10", 12) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }

  deviceMatrixSub(A, B, C, 6);
  getDeviceMatrixData(c, C, 6);

  offset = 0;
  for (int i = 0; i < 6; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)c[i]);
  }
  printf("Testing matrix sub\n");
  printf("Result: %s\n", result);
  printf("Expect: 0 0 0 0 0 0\n");
  if (strncmp(result, "0 0 0 0 0 0", 11) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }

  deviceHadamardProd(A, B, C, 6);
  getDeviceMatrixData(c, C, 6);

  offset = 0;
  for (int i = 0; i < 6; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)c[i]);
  }
  printf("Testing hadamardProd \n");
  printf("Result: %s\n", result);
  printf("Expect: 0 1 4 9 16 25\n");
  if (strncmp(result, "0 1 4 9 16 25", 13) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }

  deviceSigmoid(C, C, 6);
  getDeviceMatrixData(c, C, 6);

  offset = 0;
  for (int i = 0; i < 6; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%.2f ", c[i]);
  }
  printf("Testing sigmoid \n");
  printf("Result: %s\n", result);
  printf("Expect: 0.50 0.73 0.98 1.00 1.00 1.00\n");
  if (strncmp(result, "0.50 0.73 0.98 1.00 1.00 1.00", 28) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }
  printf("\nPASSED\n\n");


  deviceMatrixScale(A, 2, C, 6);
  getDeviceMatrixData(c, C, 6);

  offset = 0;
  for (int i = 0; i < 6; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)c[i]);
  }
  printf("Testing scalar \n");
  printf("Result: %s\n", result);
  printf("Expect: 0 2 4 6 8 10\n");
  if (strncmp(result, "0 2 4 6 8 10", 12) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }
  printf("\nPASSED\n\n");

  freeMatrix(A);
  freeMatrix(B);
  freeMatrix(C);
}

void test_transpose() {
  Matrix *A, *tA, *C;
  float a[8] = {
    0,1,2,3,
    4,5,6,7
  };
  initMatrix(&A, 2, 4);     // A (2,4)
  setDeviceMatrixData(A, a, 8);
  matrixTranpose(A, &tA, 2, 4);   // tA (4,2)

  initMatrix(&C, 2, 2);
  deviceMatrixMult(A, tA, C, 4);  // (2,4)(4,2) = (2,2)

  float c[12];
  getDeviceMatrixData(c, C, 4);

  char result[32];
  char expected[32] = "14 38 38 126";
  int offset = 0;
  for (int i = 0; i < 4; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)c[i]);
  }
  printf("Testing matrix transpose\n");
  printf("Result: %s\n", result);
  printf("Expect: %s\n", expected);
  if (strncmp(result, expected, strlen(expected)) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }
  printf("\nPASSED\n\n");

  freeMatrix(A);
  freeMatrix(tA);
  freeMatrix(C);
}

void test_acrossRows() {
  Matrix *A, *B, *C;
  float a[6] = {1,2,3,
                4,5,6};
  float b[3] = {5,5,5};
  printf("Testing matrix add vec\n");
  initMatrix(&A, 2, 3);
  setDeviceMatrixData(A, a, 6);
  initMatrix(&B, 1, 3);
  setDeviceMatrixData(B, b, 3);
  initMatrix(&C, 2, 3);
  deviceMatrixAddVec(A, B, C, 6);
  float c[6];
  getDeviceMatrixData(c, C, 6);
  float exp[6] = {6,7,8,9,10,11};
  printArray(exp, 6);
  printArray(c, 6);
  for (int i = 0; i < 6; ++i)
    assert(c[i] == (a[i] + 5));
  printf("\nPASSED\n\n");
  printf("Testing matrix reduce rows\n");
  deviceMatrixReduceRows(A, B, 2, 3);
  getDeviceMatrixData(b, B, 3);
  float exp2[3] = {5,7,9};
  printArray(exp2, 3);
  printArray(b, 3);
  for (int i = 0; i < 3; ++i)
    assert(b[i] == (a[i] + a[i+3]));
  printf("\nPASSED\n\n");
}

void test_unfold() {
    Matrix *Img, *Kernel, *Unfolded;
    float img[9] = {
      0,1,2,
      3,4,5,
      6,7,8
    };
    initMatrix(&Img, 3, 3);
    setDeviceMatrixData(Img, img, 9);

    float kernel[4] = {
        0, 1,
        2, 3
    };
    initMatrix(&Kernel, 2, 2);
    setDeviceMatrixData(Kernel, kernel, 4);

    int resRows = 3 - 2 + 1;
    int resCols = 3 - 2 + 1;

    deviceUnfoldMatrix(Img, &Unfolded, 2, 2, resRows, resCols);
  
    float unfolded[16];
    getDeviceMatrixData(unfolded, Unfolded, 16);
  
    char result[64];
    char expected[64] = "0 1 3 4 1 2 4 5 3 4 6 7 4 5 7 8";
    int offset = 0;
    for (int i = 0; i < 16; ++i) {
      offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)unfolded[i]);
    }
    printf("Testing matrix unfold\n");
    printf("Result: %s\n", result);
    printf("Expect: %s\n", expected);
    if (strncmp(result, expected, strlen(expected)) != 0) {
      printf("FAILED\n");
      exit(EXIT_FAILURE);
    }

    printf("\nPASSED\n\n");
  
    freeMatrix(Img);
    freeMatrix(Kernel);
    freeMatrix(Unfolded);
  }

  void test_convolve() {
    Matrix *Img, *Kernel, *Convolved;
    float img[9] = {
      0,1,2,
      3,4,5,
      6,7,8
    };
    initMatrix(&Img, 3, 3);
    setDeviceMatrixData(Img, img, 9);

    float kernel[4] = {
        0, 1,
        1, 0
    };
    initMatrix(&Kernel, 2, 2);
    setDeviceMatrixData(Kernel, kernel, 4);

    deviceConvolve(Img, 3, 3, Kernel, 2, 2, &Convolved, 1, 0);
  
    float convolved[4];
    getDeviceMatrixData(convolved, Convolved, 4);
  
    char result[32];
    char expected[32] = "4 6 10 12";
    int offset = 0;
    for (int i = 0; i < 4; ++i) {
      offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)convolved[i]);
    }
    printf("Testing matrix unfold\n");
    printf("Result: %s\n", result);
    printf("Expect: %s\n", expected);
    if (strncmp(result, expected, strlen(expected)) != 0) {
      printf("FAILED\n");
      exit(EXIT_FAILURE);
    }

    printf("\nPASSED\n\n");
  
    freeMatrix(Img);
    freeMatrix(Kernel);
    freeMatrix(Convolved);
  }

int main() {

  test_matrixMult();
  test_matrixElementWise();
  test_transpose();
  test_acrossRows();
  test_unfold();
  test_convolve();

  return 0;
}