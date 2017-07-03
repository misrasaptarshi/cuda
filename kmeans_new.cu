#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#define square(x) x*x


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



 

__global__ void map(int n, double *xs, double *c, int k, int *s0, int *cluster_index, int d){ //xs indicates datapoints, c indicates centroids, k indicates no. of clusters
        
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index<n){
                double dist;
                double prevBest = 10000000000.0;
                int centroidIndex = 0;
                
                for (int j1 = 0; j1 < k; j1++) {
			dist = 0.0;
                	for (int j=0; j<d; j++){
                        	dist += ((xs[index*d + j]) - (c[j1*d+j])) * ((xs[index*d + j]) - (c[j1*d+j]));
				//if(index == 0 ) printf("%lf \t %lf \t", xs[index*d + j], c[j1*d+j]);
                	}
			      //if(index == 0 ) printf("%d: %d -  %d :: %f == %f \n", j1, centroidIndex, j1, dist, prevBest);
                	if (dist<prevBest) {
			      prevBest = dist; centroidIndex = j1; 
			}
			
                }

		//atomicAddDouble(&s0[centroidIndex], 1.0);
		cluster_index[index] = centroidIndex;
	}
}


/*__global__ void map1(int n, double *xs, double *c, int k, int *cluster_index, double *s1, double *s2, int d){

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int index1 = blockIdx.y * blockDim.y + threadIdx.y;

	if ((index<n1) && (index1<n2)){
        	int ind = cluster_index[index];

		for (int j=0; j<d; j++){
			c[index*d + j] += xs[index*d+j];		
		}
		
	}
	
	
	

		
		
}


__global__ void map2(int n, double *xs, f (j == 0) {
double *c, int k, int *cluster_index, double *s1, double *s2, int d){

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int index1 = blockIdx.y * blockDim.y + threadIdx.y;

        if ((index<n1) && (index1<n2)){
                int ind = cluster_index[index];

                for (int j=0; j<d; j++){
                        c[index*d + j] += xs[index*d+j];
                }

        }


}
*/


__global__ void map2(int n, double *xs, double *c, int k, int *cluster_index, int *intermediates0, double *intermediates1, double *intermediates2, int d){
     int blocksize = n / 450 + 1;
     int start = blockIdx.x * blocksize;
     int end = start + blocksize;

		// loop for every K 
		for (int i = threadIdx.y; i < k; i+= blockDim.y){
			// loop for every dimension(features)
			for (int j = threadIdx.x; j < d; j+= blockDim.x) {

				// Calculate intermediate S0
				// for counts we don't have dimensions
				if (j ==0) {
					int count = 0;
					for(int z=start; z<end; z++)
					{
						if(cluster_index[z] == i) {
						count ++;
						}
					}
					intermediates0[blockIdx.x*k+i] = count;
				}

				// Calculate intermediate S1 and S2
				double sum1 = 0.0;
				double sum2 = 0.0;
				for (int z=start; z<end; z++) {
					if(cluster_index[z] == i) {
						sum1 += xs[z * d + j];
						sum2 += xs[z * d + j] * xs[z * d + j];

					}
				}
				int index = (blockIdx.x*k*d + i*d + j);
				intermediates1[index] = sum1;
				intermediates2[index] = sum2;
			}
	}
}



__global__ void map3(int n, double *xs, double *c, int k, int *cluster_index, int *intermediates0, double *intermediates1, double *intermediates2, int *s0, double *s1, double *s2, int d){		// Only block is invoked.		

	// loop for every K 
	for (int i = threadIdx.y; i < k; i+= blockDim.y){
		// loop for every dimension(features)
		for (int j = threadIdx.x; j < d; j+= blockDim.x) {

			// Calculate  S0
			// for counts we don't have dimensions
			if (j == 0) {
				//count = 0;
				for(int z = i; z < 450*k; z+=k){
					{
						s0[i] += intermediates0[z];
					}
				}

			// Calculate S1 and S2
            		int start = i * k + j;
            		int kd    = k * d;
            		double *s1end = &intermediates1[450 * kd];
            		double *s1cur = &intermediates1[start];
            		double *s2cur = &intermediates2[start];

	           	for (; s1cur < s1end; s1cur += kd, s2cur += kd)
	           	{
				s1[start] += *s1cur;
				s2[start] += *s2cur;
	           	}

			}	
		}
	}
}		


/*__global__ void map1(int n, double *xs, double *c, int k, int *cluster_index, double *s1, double *s2, int d){ //xs indicates datapoints, c indicates centroids, k indicates no. of cluster

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index<n){
                int ind = cluster_index[index];

                for (int j = 0; j < d; j++){
                                //s1[ind*d + j] += xs[index*d + j];
				//s2[ind*d + j] += xs[index*d + j] * xs[index*d + j];
				double x = xs[ind*d+j];
				double x1 = xs[ind*d + j] * xs[ind*d+j];
				atomicAddDouble(&s1[ind*d+j] , x);
				atomicAddDouble(&s2[ind*d+j] , x1);
				//if(index == 0 ) printf("%d \t %lf \t %lf \n ", cluster_index[index], xs[index*d + j], s1[ind*d + j], xs[index*d+j]);
                       }
                	 //if(index == 0 ) printf("%d \t %lf \t %lf \n ", cluster_index[index], s1[ind*d + 0], xs[index*d+0]);
         }
}
*/

void calculate_centroids (double *c1, int *s0, double *s1, double *s2, int k, int d, double cost){
	
	cost = 0.0;
	for (int i = 0; i < k; i++){
		for (int j = 0; j < d; j++){
		if (s0[i] >= 1){
			c1 [i*d + j] = s1[i*d + j]/ s0[i];
		}
		else{
			c1 [i*d + j] = s1[i*d + j]/ 1;
		}
		cost += (c1[i*d + j] * ((c1[i*d + j] * s0[i]) - 2*s1[i])) + s2[i];
	}
	}
	//printf("%lf \n", cost);
} 



#define num_iterations 1
#include <time.h>

int main(){
	clock_t start, end;
	double time_used;

	int n = 150;
	int d = 4;
	int k = 3;
	
	 //Allocate host memory variables
 	size_t size1 = n*d*sizeof(double);
 	size_t size2 = n*sizeof(double);
	size_t size3 = d*sizeof(double);
	size_t size4 = k*sizeof(int);
	size_t size5 = k*k*sizeof(double);
	size_t size6 = n*sizeof(int);
	size_t size7 = k*sizeof(int);
	size_t size8 = k*sizeof(int);
	size_t size9 = k*d*d*sizeof(double);



	double *xs;
	double *ys;
	int *cluster_index_host;
	int *s0_host;
	double *s1_host;
	double *s2_host;
	double *c_host;
	double *c1_host;
	double cost;

	double *gpu_xs;
	double *gpu_ys;
	int *cluster_index;
	double *c;
	int *s0;
 	double *s1;
	double *s2;
	int *intermediates0;
        double *intermediates1;
        double *intermediates2;
	


 	xs = (double*)malloc(size1);
 	ys = (double*)malloc(size2);
 	cluster_index_host = (int*)malloc(size6);
	c_host = (double*)malloc(size5);
	c1_host = (double*)malloc(size5);
	s0_host = (int*)malloc(size4);
	s1_host = (double*)malloc(size5);
	s2_host = (double*)malloc(size5);
	//cost = (double*)malloc(sizeof(float));
	//c = (double*)malloc(size4);
	//s0 = (double*)malloc(size4);
	//s1 = (double*)malloc(size5);
	//s2 = (double*)malloc(size5);

	cudaMalloc(&gpu_xs, size1);
	cudaMalloc(&gpu_ys, size2);
	cudaMalloc(&cluster_index, size6);
	cudaMalloc(&c, size5);
	cudaMalloc(&s0, size4);
	cudaMalloc(&s1, size5);
	cudaMalloc(&s2, size5);
	cudaMalloc(&intermediates0, size8);
	cudaMalloc(&intermediates1, size9);
	cudaMalloc(&intermediates2, size9);


	for (int i=0; i<k; i++){
		s0_host[i] = 0;
	}

	for (int i=0; i<k*d; i++){
		s1_host[i] = 0;
		s2_host[i] = 0;
	}


	//Copy vectors from host memory to device memory
	/*cudaMemcpy(gpu_xs, xs, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_ys, ys, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(s0, s0_host, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(s1, s1_host, size5, cudaMemcpyHostToDevice);
	cudaMemcpy(s2, s2_host, size5, cudaMemcpyHostToDevice);
	*/

	//double xs[600], xs1[12], ys[150];

	//Read input data from file
	FILE *fp;
	fp = fopen ("iris", "r");

	if (!fp){
        	printf ("Unable to open file!");
		return 1;
	}

	for (int i=0; i<n; i++){
        	for (int j=0; j<d; j++){
                	fscanf(fp, "%lf", &xs[i*d + j]);
        	}	
        fscanf(fp, "%lf", &ys[i]);
	}


	fclose(fp);

	printf("HI1");

	/*for (int i=0; i<5; i++){
         	for (int j=0; j<d; j++){
                	printf("%lf \n", xs[i*d + j]);
        	}
	}*/
	

	//Randomly select k datapoints as centroids

	int ind[3];

	for (int i=0; i<k; i++){
		ind[i] = rand()%n;
		//printf ("%d \t", ind[i]);
	}
	


	for (int i=0; i<k; i++){
		for (int j=0; j<d; j++){
			int r = ind[i];
			c_host[i*d + j] = xs[r*d + j];
		}
	}

	/*for (int i=0; i<k; i++){
        for (int j=0; j<d; j++){
                printf("%lf \n", c_host[i*d + j]);
        }
}
*/	




	cudaMemcpy(c, c_host, size5, cudaMemcpyHostToDevice);
	
	//cudaMemcpy(c_host, c, size5, cudaMemcpyDeviceToHost);


	
/*      for (int i=0; i<k; i++){
        for (int j=0; j<d; j++){
                printf("%lf \n", c_host[i*d + j]);
        }
}
*/
	printf("HI"); 

	cudaMemcpy(gpu_xs, xs, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_ys, ys, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(s0, s0_host, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(s1, s1_host, size5, cudaMemcpyHostToDevice);
	cudaMemcpy(s2, s2_host, size5, cudaMemcpyHostToDevice);

	//double cost = 10000000.0;
	//double cost1 = 0.0;
	//int changed = 1;
	
	//while (changed==1){
	for (int i=0; i<num_iterations; i++){
          	start = clock();
		cudaMemset((void*)s0, 0, size4);
                //Compute hypothesis function and element-wise gradients
                map<<<2,75>>>(n, gpu_xs, c, k, s0, cluster_index, d);


                //Copy the element wise gradients from GPU to CPU
                //cudaMemcpy(gradvec1, gradvec, size1, cudaMemcpyDeviceToHost);

                //Compute sum of all grad vector in GPU
		cudaMemset((void*)s1, 0, size5);
                end = clock();
                time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        	printf("Time taken for map : %f \n", time_used);

		cudaMemset((void*)s2, 0, size5);

                /*start = clock();
                map1<<<2000,512>>>(n, gpu_xs, c, k, cluster_index, s1, s2, d);
		*/

                /*//Update cost
		for (int i=0; i<k; i++){
			for (int j=0; j<d; j++){
				cost1 += (c[i*d + j] * ((c[i*d + j] * s0[i]) - 2*s1[i])) + s2[i];
			}
		}*/

		start = clock();
		dim3 nthreads(4,3);
		map2<<<450,nthreads>>>(n, gpu_xs, c, k, cluster_index, intermediates0, intermediates1, intermediates2, d);		

		//cudaMemcpy(s0_host, s0, size4, cudaMemcpyDeviceToHost);
		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("Time taken for map1 : %f \n", time_used);


		start = clock();
		dim3 nthreads1(4,3);
		map3<<<1,nthreads1>>>(n, gpu_xs, c, k, cluster_index, intermediates0, intermediates1, intermediates2, s0, s1, s2, d);

		cudaMemcpy(s0_host, s0, size4, cudaMemcpyDeviceToHost);
		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("Time taken for map1 : %f \n", time_used);
		
	
		start = clock();
		cudaMemcpy(s1_host, s1, size5, cudaMemcpyDeviceToHost);
		end = clock();
 		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%f \n", time_used);
 		

		start = clock();
		cudaMemcpy(s2_host, s2, size5, cudaMemcpyDeviceToHost);
		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%f \n", time_used);

		/*for (int i=0; i<k; i++){
			for (int j=0; j<d; j++){
				printf("%lf \n", s1_host[i*d+j]);
			}
		} 
		*/
		start = clock();
		calculate_centroids (c1_host, s0_host, s1_host, s2_host, k, d, cost);
		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%f \n", time_used);
		/*for (int i=0; i<k; i++){
			for (int j=0; j<d; j++){
        			printf("%lf \n", c1_host[i*d + j]);
			}
		}*/

		start = clock();
		double maxdelta = 0.0;

		for (int i=0; i<k; i++){
			for (int j=0; j<d; j++){
				maxdelta += (c1_host[i*d+j] - c_host[i*d+j]) * (c1_host[i*d+j] - c_host[i*d+j]);
			}
		}
		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%f \n", time_used);



		start = clock();	
		memcpy(c_host, c1_host, size5);
		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%f \n", time_used);

		//printf("%lf \n", cost);
		//printf("%lf \n", maxdelta);

		//changed = maxdelta>0.5;	
		start = clock();
		cudaMemcpy(c, c1_host, size5, cudaMemcpyHostToDevice);
		//cudaMemcpy(s0, s0_host, size6, cudaMemcpyHostToDevice);

		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%lf \n", time_used);
				
	}
	//}
	

	cudaMemcpy(cluster_index_host, cluster_index, size6, cudaMemcpyDeviceToHost);


	for (int i=0; i<5; i++){
		printf("%d \n", cluster_index_host[i]);
	}

	for (int i=140; i<150; i++){
        printf("%d \n", cluster_index_host[i]);
	}

	for (int i=0; i<k; i++){
		printf("%d \n", s0_host[i]);
	}


	FILE *fp1;
	//Dump the prediction output to a file
        fp1 = fopen("output_kmeans", "w");
        for (int i=0; i<n; i++){
                fprintf(fp1, "%d  \n", cluster_index_host[i]);
        }
}

			


	






















