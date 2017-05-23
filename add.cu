#include<stdio.h>

//Device code

__global__ void addvec (float* a, float* b, float* c, int N)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i<N)
	 c[i] = a[i]+b[i];
}

//Host code

int main()
{
	int N = 10;
	size_t size = N*sizeof(float);

	//Allocate input vectors h_A and h_B in host memory
	float* h_a = (float*)malloc(size);
	float* h_b = (float*)malloc(size);
	float* h_c = (float*)malloc(size);

        //Initialize input vectors
	int i;
	
        for (i=0;i<N;i++){
		h_a[i] = i+1;
	}

	for (i=0;i<N;i++){
		h_b[i] = i+1;
	}

	//Allocate vectors in device memory
	float* d_a;
	cudaMalloc(&d_a, size);
	float* d_b;
	cudaMalloc(&d_b,size);
	float* d_c;
	cudaMalloc(&d_c,size); 

	//Copy vectors from host memory to device memory
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	//Invoke kernel
	int threads_per_block = 256;
	int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
	addvec<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, N);

	//Copy result from device memory to host memory
	//h_c contains the result in host memory
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	for (i=0;i<N;i++){
		printf("%f \n",h_c[i]);
	}

	//Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	//Free host memory
	cudaFree(h_a);
	cudaFree(h_b);
}

