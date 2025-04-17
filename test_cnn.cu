#include "cnn.h"
#include "util.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>

void test_create_conv_layer() {
    const int INPUT_ROWS = 3;
    const int INPUT_COLS = 3;
    const int FILTER_ROWS = 2;
    const int FILTER_COLS = 2;
    const int OUTPUT_ROWS = 2;
    const int OUTPUT_COLS = 2;

    ConvolutionalLayer* input = createConvolutionalLayer(1, 0, 1, INPUT_ROWS, INPUT_COLS, NULL);
    printf("Testing create convolutional layer 1\n");
    printf("Output rows:\nExpect: %d\nActual: %d\n", INPUT_ROWS, input->outputRows);
    if (INPUT_ROWS != input->outputRows) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }
    printf("Output cols:\nExpect: %d\nActual: %d\n", INPUT_COLS, input->outputCols);
    if (INPUT_COLS != input->outputCols) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }
    Matrix temp;
    CERROR(cudaMemcpy(&temp, input->outputs[0][0], sizeof(Matrix), cudaMemcpyDeviceToHost));
    printf("Testing create convolutional layer 1 matrix\n");
    printf("Output rows:\nExpect: %d\nActual: %d\n", INPUT_ROWS, temp.rows);
    if (INPUT_ROWS != temp.rows) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }
    printf("Output cols:\nExpect: %d\nActual: %d\n", INPUT_COLS, temp.cols);
    if (INPUT_COLS != temp.cols) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }

    ConvolutionalLayer* layer = createConvolutionalLayer(1, 1, 1, 2, 2, input);
    printf("Testing create convolutional layer 2\n");
    printf("Input rows:\nExpect: %d\nActual: %d\n", INPUT_ROWS, layer->imgRows);
    if (INPUT_ROWS != layer->imgRows) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }
    printf("Input cols:\nExpect: %d\nActual: %d\n", INPUT_COLS, layer->imgCols);
    if (INPUT_COLS != layer->imgCols) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }
    printf("Filter rows:\nExpect: %d\nActual: %d\n", FILTER_ROWS, layer->kernelRows);
    if (FILTER_ROWS != layer->kernelRows) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }
    printf("Filter rows:\nExpect: %d\nActual: %d\n", FILTER_COLS, layer->kernelCols);
    if (FILTER_COLS != layer->kernelCols) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }
    printf("Output rows:\nExpect: %d\nActual: %d\n", OUTPUT_ROWS, layer->outputRows);
    if (OUTPUT_ROWS != layer->outputRows) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }
    printf("Output cols:\nExpect: %d\nActual: %d\n", OUTPUT_COLS, layer->outputCols);
    if (OUTPUT_COLS != layer->outputCols) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }

    CERROR(cudaMemcpy(&temp, layer->filters[0][0], sizeof(Matrix), cudaMemcpyDeviceToHost));
    printf("Testing create convolutional layer 2 matrices\n");
    printf("Filter rows:\nExpect: %d\nActual: %d\n", FILTER_ROWS, temp.rows);
    if (FILTER_ROWS != temp.rows) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }
    printf("Filter cols:\nExpect: %d\nActual: %d\n", FILTER_COLS, temp.cols);
    if (FILTER_COLS != temp.cols) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }

    CERROR(cudaMemcpy(&temp, layer->outputs[0][0], sizeof(Matrix), cudaMemcpyDeviceToHost));
    printf("Output rows:\nExpect: %d\nActual: %d\n", OUTPUT_ROWS, temp.rows);
    if (OUTPUT_ROWS != temp.rows) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }
    printf("Output cols:\nExpect: %d\nActual: %d\n", OUTPUT_COLS, temp.cols);
    if (OUTPUT_COLS != temp.cols) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }

    printf("\nPASSED\n\n");
}

void test_layer_forward() {
    const int INPUT_SIZE = 9;
    const int FILTER_SIZE = 4;
    const int OUTPUT_SIZE = 4;

    ConvolutionalLayer* input = createConvolutionalLayer(1, 0, 1, 3, 3, NULL);
    float inputData[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    setDeviceMatrixData((input->outputs)[0][0], inputData, INPUT_SIZE);

    ConvolutionalLayer* layer = createConvolutionalLayer(1, 1, 1, 2, 2, input);
    float filterData[] = {1, 0, 0, 1};
    setDeviceMatrixData((layer->filters)[0][0], filterData, FILTER_SIZE);
    layer->biases[0] = 1;
    
    layerForward(layer, 0);
    
    float res[OUTPUT_SIZE];
    getDeviceMatrixData(res, (layer->outputs)[0][0], OUTPUT_SIZE);

    float expected[OUTPUT_SIZE] = {
        SIGMOID(1.6f), SIGMOID(1.8f), SIGMOID(2.2f), SIGMOID(2.4f)
    };
    
    printf("Testing layer forward\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Result: %f\n", res[i]);
        printf("Expect: %f\n", expected[i]);
        if (fabs(res[i] - expected[i]) > 0.000001) {
            printf("FAILED\n");
            exit(EXIT_FAILURE);
          }
    }

    printf("\nPASSED\n\n");
}

void test_layer_forward_2k() {
    const int INPUT_SIZE = 9;
    const int FILTER_SIZE = 4;
    const int OUTPUT_SIZE = 4;

    ConvolutionalLayer* input = createConvolutionalLayer(1, 0, 1, 3, 3, NULL);
    float inputData[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    setDeviceMatrixData((input->outputs)[0][0], inputData, INPUT_SIZE);

    ConvolutionalLayer* layer = createConvolutionalLayer(1, 1, 2, 2, 2, input);
    float filterData[] = {1, 0, 0, 1};
    setDeviceMatrixData((layer->filters)[0][0], filterData, FILTER_SIZE);
    layer->biases[0] = 1;
    filterData[0] = 0;
    filterData[1] = 2;
    filterData[2] = 2;
    filterData[3] = 0;
    setDeviceMatrixData((layer->filters)[1][0], filterData, FILTER_SIZE);
    layer->biases[1] = 2;
    
    layerForward(layer, 0);
    
    float res0[OUTPUT_SIZE];
    getDeviceMatrixData(res0, (layer->outputs)[0][0], OUTPUT_SIZE);

    float expected0[OUTPUT_SIZE] = {
        SIGMOID(1.6f), SIGMOID(1.8f), SIGMOID(2.2f), SIGMOID(2.4f)
    };
    
    printf("Testing layer forward\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Result: %f\n", res0[i]);
        printf("Expect: %f\n", expected0[i]);
        if (fabs(res0[i] - expected0[i]) > 0.000001) {
            printf("FAILED\n");
            exit(EXIT_FAILURE);
          }
    }

    float res1[OUTPUT_SIZE];
    getDeviceMatrixData(res1, (layer->outputs)[0][1], OUTPUT_SIZE);

    float expected1[OUTPUT_SIZE] = {
        SIGMOID(3.2f), SIGMOID(3.6f), SIGMOID(4.4f), SIGMOID(4.8f)
    };

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Result: %f\n", res1[i]);
        printf("Expect: %f\n", expected1[i]);
        if (fabs(res1[i] - expected1[i]) > 0.000001) {
            printf("FAILED\n");
            exit(EXIT_FAILURE);
          }
    }

    printf("\nPASSED\n\n");
}

int main() {
    test_create_conv_layer();
    test_layer_forward();
    //test_layer_forward_2k();
    return 0;
}