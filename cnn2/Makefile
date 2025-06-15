cpu_matrix_test:
	gcc -o cpu_matrix_test main.c matrix.c matrix.h

gpu_matrix_test:
	nvcc matrix.cu matrix.c main_gpu.cu -o gpu_matrix_test

batch:
	sbatch BatchJobscript

clean:
	rm *_matrix_test

job_clean:
	rm job.* core.*
