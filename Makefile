matrixtest: test_matrix
	./test_matrix
networktest: test_network
	./test_network
cnntest: test_cnn
	./test_cnn
mnisttrain: mnist
	./mnist

test_matrix: test_matrix.o matrix.o
	nvcc test_matrix.o matrix.o -o test_matrix
test_network: test_network.o matrix.o network.o
	nvcc test_network.o network.o matrix.o -o test_network
test_cnn: test_cnn.o matrix.o cnn.o
	nvcc test_cnn.o cnn.o matrix.o -o test_cnn
mnist: mnist.o matrix.o network.o
	nvcc mnist.o network.o matrix.o -o mnist

test_matrix.o: test_matrix.cu
	nvcc -c test_matrix.cu -o test_matrix.o
test_network.o: test_network.cu
	nvcc -c test_network.cu -o test_network.o
test_cnn.o: test_cnn.cu
	nvcc -c test_cnn.cu -o test_cnn.o
network.o: network.cu
	nvcc -c network.cu -o network.o
cnn.o: cnn.cu
	nvcc -c cnn.cu -o cnn.o
matrix.o: matrix.cu
	nvcc -c matrix.cu -o matrix.o
mnist.o: mnist.cu
	nvcc -c mnist.cu -o mnist.o

clean:
	rm -f test_matrix.o matrix.o test_matrix test_network.o network.o test_network test_cnn.o cnn.o test_cnn mnist.o mnist job.* core.*
