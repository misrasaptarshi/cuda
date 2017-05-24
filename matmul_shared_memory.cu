

// Matrices are stored in row-major order
// M(row,col) - *(M.elements + row*M.width + col)

typedef struct{
	int width;
	int height;
	float* elements;
}Matrix;

//Thread block size
#define BLOCK_SIZE 16

//Declaration of the matrix multiplication kernel
__global__ void MatMulkernel(Matrix A,  Matrix B, Matrix C)
{
	//Each thread computes one element of C by accumulating results into Cvalue

	float Cvalue=0;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	for (int e=0; e<A.width; ++e)
		Cvalue+= A.elements[row*A.width+e] * B.elements[e*B.width+col];
	C.elements[row*C.width+col] = Cvalue;
}

// Matrix multiplication - Host code
//Matrix dimensions are assumed to be multiples of BLOCK_SIZE


void MatMul(const Matrix A, const Matrix B, Matrix C)
{

	//Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements,size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width*B.height*sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	//Allocate c in device memory
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = C.width*C.height*sizeof(float);
	cudaMalloc(&d_C.elements, size);

	//Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width/dimBlock.x, A.height/dimBlock.y);
	MatMulkernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
}


#include<stdio.h>
int main()
{
	Matrix a;
	Matrix b;
	Matrix c;
	a.width = 20; a.height = 20;
	float size = 400*sizeof(float*);
	
	b.width = 20; b.height = 20;
	c.width = 20; c.height = 20;
	a.elements = (float*)malloc(size);
        b.elements = (float*)malloc(size);
	c.elements = (float*)malloc(size);

	//a.elements[400];
	//b.elements[400];
	//c.elements[400];
	
	int i,j;
	for (i=0;i<20;i++){
		for (j=0;j<20;j++){
			*(a.elements + i*20 + j) = (i+j);
		}
	}

	for (i=0;i<20;i++){
		for (j=0;j<20;j++){
			*(b.elements + i*20 + j) = (i+j);
		}
	}

	MatMul(a,b,c);

	for (i=0;i<20;i++){
		for (j=0;j<20;j++){
			printf ("%f \n", *(c.elements + i*20 + j));
		}
	}
}
