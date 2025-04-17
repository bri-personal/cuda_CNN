#include "cnn.h"
#include "util.h"
#include <stdio.h>
#include <stdint.h>

void test_layer_forward() {
    ConvolutionalLayer* input = createConvolutionalLayer(1, 0, 1, 3, 3, NULL);
    ConvolutionalLayer* layer = createConvolutionalLayer(1, 1, 1, 2, 2, input);
    layerForward(layer, 0);
}

int main() {
    test_layer_forward();
    return 0;
}