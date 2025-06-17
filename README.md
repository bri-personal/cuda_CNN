# CUDA CNN
## From "Acceleration of Neural Network Computations: Software and Hardware Approaches"
### Part of Brian Slonim's Senior Project 2025, advised by Dr. Maria Pantoja

Rudimentary implementation of a convolutional neural network using CUDA. Convolution is implemented on GPU via explicit, channel-last Im2Col, and on CPU with the naive method.

This code was adapted from [this Jupyter notebook](https://colab.research.google.com/drive/1Ynj9EVbEGMZsl5hPUcaLrlV5nNL51B74?usp=sharing), which contains a neural network implementation with only fully-connected layers.
