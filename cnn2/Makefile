cpu_matrix_test:
	gcc -o cpu_matrix_test main.c matrix.c matrix.h

gpu_matrix:
	nvcc -c matrix.cu matrix.c -o matrix.o

gpu_matrix_test: gpu_matrix
	nvcc matrix.o gpu_matrix_test.cu -o gpu_matrix_test

gpu_cnn: gpu_matrix
	nvcc -c matrix.o cnn.cu cnn.c -o cnn.o

gpu_cnn_test: gpu_cnn
	nvcc matrix.o cnn.o gpu_cnn_test.cu -o gpu_cnn_test

batch:
	sbatch BatchJobscript

clean:
	rm *_matrix_test *_cnn_test *.o

job_clean:
	rm job.* core.*
