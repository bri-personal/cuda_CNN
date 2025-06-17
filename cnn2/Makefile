cpu_matrix_test:
	gcc -o cpu_matrix_test main.c matrix.c matrix.h

gpu_matrix_test:
	nvcc matrix.cu matrix.c gpu_matrix_test.cu -o gpu_matrix_test

gpu_cnn_test:
	nvcc matrix.cu matrix.c cnn.cu cnn.c gpu_cnn_test.cu -o gpu_cnn_test

batch:
	sbatch BatchJobscriptCnn

clean:
	rm *_matrix_test *_cnn_test

job_clean:
	rm job.* core.*
