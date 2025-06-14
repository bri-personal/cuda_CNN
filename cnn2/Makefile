cpu_matrix_test:
	gcc -o cpu_matrix_test main.c matrix.c matrix.h

gpu_matrix_test:
	nvcc matrix.cu matrix.c main_gpu.cu -o gpu_test_matrix

clean:
	rm *_matrix_test
