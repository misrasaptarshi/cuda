#include<stdio.h>
#include <stdlib.h>
#include <math.h>
#define TILE_DIM 32
#define BLOCK_ROWS 8
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>

/*void init(float *x, int d)
{
        //mat->data = new float[height * width];
        //mat->w = width;
        //mat->h = height;
        for (int j = 0; j < d; j++) {
                        x[j] = 0.0;
                }
        
}*/

void min_max_normalize(double *xs, int m, int d)
{
        for (int x = 0; x < d-1; ++x) {
                // calculate std for each column
                double min = xs[x*d + 0];
                double max = xs[x*d + 0];
                for (int y = d; y < m*d; ++y) {
                        double val = xs[x*d + y];
                        if (val < min) {
                                min = val;
                        } else if (val > max) {
                                max = val;
                        }
                }

                for (int y = 0; y < m*d; ++y) {
                        double val = xs[x*d + y];
                        xs[x*d + y] = (val - min) / (max-min);
                }
        }
}

__global__ void calhyp(double *xs, double *ys, double *params, double *h, int m, int d){   // m is the no. of samples and d is the number of features in xs(input)
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index<m){
		double accum = params[0];
		//double accum = 0.0;
                for (int j=0; j<d; j++){
                        accum += xs[index*(d-1)+j] * params[j+1];
                }
		
		h[index] = 1.0/ (1.0 + exp(-accum));
	}
}


__global__ void calgrad (double *xs, double *ys, double *h, double *gradvec, int m, int d, double alpha){ // alpha is the learning rate
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		 gradvec[index*d+0] =  (h[index] - ys[index]) * 1;
		for (int j = 1; j < d; j++){
        	//gradvec[index*d+0] = alpha * (h[index] - ys[index]) * xs[index*d+0];
		//gradvec[index*d+1] = alpha * (h[index] - ys[index]) * xs[index*d+1];
		gradvec[index*d+j] =  (h[index] - ys[index]) * xs[index*d+j];
        	}

		}
	
}

/*
__global__ void calhyp(double *xs, double *ys, double *params, double *h, int m, int d){   // m is the no. of samples and d is the number of features in xs(input)
      int index = blockIdx.x * blockDim.x + threadIdx.x;

      if (index<m*d){
              //double accum = params[0];
              //double accum = 0.0;
              //for (int j=0; j<d; j++){
              //        accum += xs[index*(d-1)+j] * params[j+1];
             // }

              h[index] = 1.0/ (1.0 + exp(-xs[index]));
      }
}


__global__ void calgrad (double *xs, double *ys, double *h, double *gradvec, int m, int d, double alpha){ // alpha is the learning rate
      int index = blockIdx.x * blockDim.x + threadIdx.x;

      if (index < m*d) {
               //gradvec[index*d+0] =  (h[index] - ys[index]) * 1;
              for (int j = 0; j < d; j++){
              //gradvec[index*d+0] = alpha * (h[index] - ys[index]) * xs[index*d+0];
              //gradvec[index*d+1] = alpha * (h[index] - ys[index]) * xs[index*d+1];
              gradvec[index*d+j] =  (h[index] - ys[index]) * xs[index*d+j];
              }
	}	

	}
*/


#define WARPSIZE  32

__device__ inline double atomicAddDouble(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed  = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + 
                      __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ inline double __shfl_double(double d, int lane) {
  // Split the double number into 2 32b registers.
  int lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(d));

  // Shuffle the two 32b registers.
  lo = __shfl(lo, lane);
  hi = __shfl(hi, lane);

  // Recreate the 64b number.
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(d) : "r"(lo), "r"(hi));
  return d;
}

__device__ inline double warpReduceSum(double val) {
  int i = blockIdx.x  * blockDim.x + threadIdx.x;
#pragma unroll
  for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
     val += __shfl_double(val, (i + offset) % WARPSIZE);
  }
  return val;
}

__device__ inline double4 __shfl_double4(double4 d, int lane) {
  // Split the double number into 2 32b registers.
  int lox, loy, loz, low, hix, hiy, hiz, hiw;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lox), "=r"(hix) : "d"(d.x));
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(loy), "=r"(hiy) : "d"(d.y));
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(loz), "=r"(hiz) : "d"(d.z));
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(low), "=r"(hiw) : "d"(d.w));

  // Shuffle the two 32b registers.
  lox = __shfl(lox, lane);
  hix = __shfl(hix, lane);
  loy = __shfl(loy, lane);
  hiy = __shfl(hiy, lane);
  loz = __shfl(loz, lane);
  hiz = __shfl(hiz, lane);
  low = __shfl(low, lane);
  hiw = __shfl(hiw, lane);

  // Recreate the 64b number.
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(d.x) : "r"(lox), "r"(hix));
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(d.y) : "r"(loy), "r"(hiy));
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(d.z) : "r"(loz), "r"(hiz));
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(d.w) : "r"(low), "r"(hiw));
  return d;
}

__device__ inline double4 warpReduceVSum(double4 val4) {
  int i = blockIdx.x  * blockDim.x + threadIdx.x;
#pragma unroll
  for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
     double4 shiftedVal4 = __shfl_double4(val4, (i + offset) % WARPSIZE);
     val4.x += shiftedVal4.x;
     val4.y += shiftedVal4.y;
     val4.z += shiftedVal4.z;
     val4.w += shiftedVal4.w;
  }
  return val4;
}



__device__ double* deviceReduceKernelj(double * inArray, double *out, long i, long n, long length) {
    double sum = 0;
    int count = 0;
    double *inArrayBody;
    int index =  blockIdx.x * blockDim.x + threadIdx.x;
    for (long idx = index; idx < n; idx += blockDim.x * gridDim.x) {
        inArrayBody = &inArray[idx*length];
	//i = idx*length;
        sum += inArrayBody[i];
    }

    sum = warpReduceSum(sum);

    if ((threadIdx.x & (WARPSIZE -1)) == 0){
            //printf("%lf :: %lf \n", out,sum);
        atomicAddDouble(out, sum);
    }
    return out;
}




__device__ void deviceReduceArrayKernelj(double * inArray, double *outputArrayBody, long length, long n) {
    long i = 0;
    double *inArrayBody;

    // unrolled version
    while ((length - i) >= 4) {
        double4 sum4;
        sum4.x = 0; sum4.y = 0; sum4.z = 0; sum4.w = 0;
        for (long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
            inArrayBody = &inArray[idx*length];
	    //i = idx*length;
            sum4.x += inArrayBody[i];
            sum4.y += inArrayBody[i+1];
            sum4.z += inArrayBody[i+2];
            sum4.w += inArrayBody[i+3];
        }

        sum4 = warpReduceVSum(sum4);

        if ((threadIdx.x & (WARPSIZE - 1)) == 0) {

        double *outx = &outputArrayBody[i];
        double *outy = &outputArrayBody[i+1];
        double *outz = &outputArrayBody[i+2];
        double *outw = &outputArrayBody[i+3];
            atomicAddDouble(outx, sum4.x);
            atomicAddDouble(outy, sum4.y);
            atomicAddDouble(outz, sum4.z);
            atomicAddDouble(outw, sum4.w);
        }
        i += 4;
    }

    for (; i < length; i++) {
        deviceReduceKernelj(inArray, &outputArrayBody[i], i, n, length);
    }
}

extern "C"
__global__
void reducegrad(double *gradvec, double * sumgradvec, int m, int d) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < m)
       deviceReduceArrayKernelj(gradvec, sumgradvec, d, m);

}



void reducegrad1 (double *gradvec, double *sumgradvec, int m, int d){
	/*int index = blockIdx.x * blockDim.x + threadIdx.x;

	sumgradvec[0] = 0.0;
	//for (int index=0; index<m; index++){
	if (index<m){
		sumgradvec[0] += gradvec[index];
	}*/

	sumgradvec[0]= 0.0;
	sumgradvec[1] = 0.0;
	sumgradvec[2] = 0.0;

	for (int index=0; index<m*d; index+=2){
		sumgradvec[0] += gradvec[index];
		sumgradvec[1] += gradvec[index+1];
		sumgradvec[2] += gradvec[index+2];
	}

	/*for (int i=0; i<d; i++){
		sumgradvec[i] = 0;
	}

	if (index<m*d){
		int k = index%m;
		sumgradvec[k] += gradvec[index];
	}*/

}


void updateweight (double *params, double *sumgradvec, int m, int d, float alpha, float theta){
	for (int i=0; i<d; i++){
		//params[0] = params[0] - sumgradvec[0];
		//params[1] = params[1] - sumgradvec[1];
		params[i] = params[i] - alpha * (sumgradvec[i]) - theta * alpha * params[i];
	}
}


#define round(a) (int) (a+0.5)

int main(){

	//Declare variables in host memory and scan the variable values from file
	int m=  100;
	int d = 3;

	//float data[m][d];
	size_t size1 = m*d*sizeof(double);
	size_t size2 = m*sizeof(double);
	size_t size3 = d*sizeof(double);

	FILE *fp, *fp1;
        fp = fopen ("input", "r");

	double *xs;
	double *ys;
	double *params;
	double *sumgradvechost;
	xs = (double*)malloc(m*d*sizeof(double));
        ys = (double*)malloc(m*sizeof(double));
	params = (double*)malloc(size3);
	sumgradvechost = (double*)malloc(sizeof(double)*d);
	//float sumgradvechost;

	double *h1;
	h1 = (double*)malloc(size2);
	double *gradvec1;
	gradvec1 = (double*)malloc(size1);
	
	for (int i=0; i<d; i++){
        	//params[i] = -1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2)));
		params[i] = 0.0;
	}

	if (!fp){
                printf ("Unable to open file!");
                return 1;
        }

        for (int i=0; i<m; i++){
		for (int j=0; j<d-1; j++){
                //fscanf (fp, "%lf %lf %lf", &xs[i*2+0], &xs[i*2+1], &ys[i]);
		fscanf(fp, "%lf", &xs[i*(d-1) + j]);
		//fscanf(fp, "%lf", &xs[i*d + j]);
		}
		fscanf(fp, "%lf", &ys[i]);
        }


        fclose(fp);
	
	/*for (int i=0; i<m; i++){
		ys[i] = 2*(xs[i*2+0] + xs[i*2+1]);
	}
	*/
		
	/*for (int i=0; i<2; i++){
		printf("%f \n", xs[i]);
	}*/	

	 for (int i=0; i<8; i+=2) {
         printf("%lf %lf => %lf \n", xs[i], xs[i+1], ys[i/2]);
 }

	min_max_normalize(xs, m, d);	

	for (int i=0; i<m; i++){
		if(ys[i]==-1){
			ys[i] = 0;
		}
	}

	for (int i=0; i<8; i+=2) {
      		printf("%lf %lf => %lf \n", xs[i], xs[i+1], ys[i/2]);
	}

	//Allocate data in device memory
	double *gpu_params;
	//params = (float*)malloc(m*sizeof(float));
	cudaMalloc (&gpu_params, size3);

	double *gpu_xs;
	double *gpu_ys;
	double *h;
	double *gradvec;
	cudaMalloc(&gpu_xs, size1);
	cudaMalloc(&gpu_ys, size2);
	cudaMalloc(&h, size2);
	cudaMalloc(&gradvec, size1);

	//Copy vectors from host memory to device memory
	cudaMemcpy(gpu_xs, xs, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_ys, ys, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_params, params, size3, cudaMemcpyHostToDevice);
	

	//Invoke kernel 1
	double *sumgradvec;
	cudaMalloc(&sumgradvec, sizeof(double)*d);
	//sumgradvec = (float*)malloc(sizeof(float));

	for (int i=0; i<400; i++){
	calhyp<<<1,100>>>(gpu_xs, gpu_ys, gpu_params, h,  m, d);
	
	 cudaMemcpy(h1, h, size2, cudaMemcpyDeviceToHost);
	
	/*for (int j=0; j<m; j++){
	printf("%lf \t", h1[j]);
	}*/

	calgrad<<<1,100>>>(gpu_xs, gpu_ys, h, gradvec, m, d, 0.03);

	cudaMemcpy(gradvec1, gradvec, size1, cudaMemcpyDeviceToHost);
 	// Sum of all grad vector
 	reducegrad<<<1,100>>>(gradvec, sumgradvec, m, d);

	/*float x = 0.0;
 for (int j=0; j<m*d; j++){
         printf("%lf \n", gradvec1[j]);
 }
*/

/*
 for (int i=0; i<m*d; i++){
         x +=  gradvec1[i];}


 printf("%f \n", x);
*/
	//float sumgradvechost;
	//cudaMemset(sumgradvec, 0, d*sizeof(double));
	//reducegrad1(gradvec1, sumgradvechost, m, d);
	//reducegrad1(gradvec, sumgradvechost, m, d);

 	// copyout grad's vector from GPU to CPU
 	cudaMemcpy (sumgradvechost, sumgradvec, sizeof(double)*d, cudaMemcpyDeviceToHost);

 	// update weight - cpu function
	updateweight(params, sumgradvechost, m, d, 0.001, 0);
	
	/* for (int i=0; i<d; i++){
         	printf("%lf \n", sumgradvechost[i]);
 	 }*/

	 for (int j=0; j<d; j++){
         printf("%lf \t", params[j]); }
         printf("\n");
         
 	
	//cudaMemcpy(params, gpu_params, size2, cudaMemcpyDeviceToHost);
	

	//printf("%f \n", sumgradvechost[0]);
	 // Copyin the updated weights back to GPU
 	cudaMemcpy (gpu_params, params, sizeof(double) * d, cudaMemcpyHostToDevice);
	}

	//for (int i=0; i<d; i++){
        //	printf("%lf \n", sumgradvechost[i]);
	//}		

	//for (int i=0; i<d; i++){
	//	printf("%lf \n", params[i]);
	//}

	 double predict[m];
	 for (int index=0; index<m; index++){	
		 predict[index] = params[0];
		//predict[index] = 0.0;
		 for (int j=0; j<d; j++){
			//predict[index]  += xs[index*d+j] * params[j];
			predict[index]  += xs[index*(d-1)+j] * params[j+1];
		}
	}
	
		

	double error = 0.0;
	for (int i=0; i<m; i++){
                int tmp = 0;
                if ((1/( 1 + exp(-predict[i]))) >= 0.5) tmp = 1; else tmp = 0;
                if (tmp != ys[i])
			error ++;
	}
		//error += ((1/ 1 + exp(-predict[i])) - ys[i])*(1 [i]) - ys[i]);
	

	fp1 = fopen("output", "w");
	for (int i=0; i<m; i++){
		fprintf(fp1, "%lf  \t %lf \t %lf \n", predict[i],  1 / (1 + exp(-predict[i])),  exp(-predict[i]) / (1 + exp(-predict[i]))); 
	}

	
	error = error / m;	
	printf("%lf \n", error);
	
} 


