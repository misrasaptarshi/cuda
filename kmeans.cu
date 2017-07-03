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



 

__global__ void map(int n, double *xs, double *c, int k, double *s0, int *cluster_index, int d){ //xs indicates datapoints, c indicates centroids, k indicates no. of clusters
        
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

		atomicAddDouble(&s0[centroidIndex], 1.0);
		cluster_index[index] = centroidIndex;
	}
}




__global__ void map1(int n, double *xs, double *c, int k, int *cluster_index, double *s1, double *s2, int d){ //xs indicates datapoints, c indicates centroids, k indicates no. of cluster

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



// A function to randomly select k items from stream[0..n-1].
void selectKItems(double *stream, double *reservoir, int n, int k)
{
    int i;  // index for elements in stream[]
 
    // reservoir[] is the output array. Initialize it with
    // first k elements from stream[]
    //int reservoir[k];
    for (i = 0; i < k; i++)
        reservoir[i] = stream[i];
 
    // Use a different seed value so that we don't get
    // same result each time we run this program
    srand(time(NULL));
 
    // Iterate from the (k+1)th element to nth element
    for (; i < n; i++)
    {
        // Pick a random index from 0 to i.
        int j = rand() % (i+1);
 
        // If the randomly  picked index is smaller than k, then replace
        // the element present at the index with new element from stream
        if (j < k)
          reservoir[j] = stream[i];
    }
 
    //printf("Following are k randomly selected items \n");
    //printArray(reservoir, k);
}


void calculate_centroids (double *c1, double *s0, double *s1, double *s2, int k, int d, double cost){
	
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

	int n = 1000000;
	int d = 400;
	int k = 2;
	
	 //Allocate host memory variables
 	size_t size1 = n*d*sizeof(double);
 	size_t size2 = n*sizeof(double);
	size_t size3 = d*sizeof(double);
	size_t size4 = k*sizeof(double);
	size_t size5 = k*d*sizeof(double);
	size_t size6 = n*sizeof(int);
	size_t size7 = k*sizeof(int);


	double *xs;
	double *ys;
	int *cluster_index_host;
	double *s0_host;
	double *s1_host;
	double *s2_host;
	double *c_host;
	double *c1_host;
	double cost;

	double *gpu_xs;
	double *gpu_ys;
	int *cluster_index;
	double *c;
	double *s0;
 	double *s1;
	double *s2;


 	xs = (double*)malloc(size1);
 	ys = (double*)malloc(size2);
 	cluster_index_host = (int*)malloc(size6);
	c_host = (double*)malloc(size5);
	c1_host = (double*)malloc(size5);
	s0_host = (double*)malloc(size4);
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
	fp = fopen ("input", "r");

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

	int ind[2];

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
                map<<<2000,512>>>(n, gpu_xs, c, k, s0, cluster_index, d);


                //Copy the element wise gradients from GPU to CPU
                //cudaMemcpy(gradvec1, gradvec, size1, cudaMemcpyDeviceToHost);

                //Compute sum of all grad vector in GPU
		cudaMemset((void*)s1, 0, size5);
                end = clock();
                time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        	printf("Time taken for map : %f \n", time_used);

		cudaMemset((void*)s2, 0, size5);

                start = clock();
                map1<<<2000,512>>>(n, gpu_xs, c, k, cluster_index, s1, s2, d);
                /*//Update cost
		for (int i=0; i<k; i++){
			for (int j=0; j<d; j++){
				cost1 += (c[i*d + j] * ((c[i*d + j] * s0[i]) - 2*s1[i])) + s2[i];
			}
		}*/

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
		//cudaMemcpy(c_host, c, size5, cudaMemcpyDeviceToHost);

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

	for (int i=90000; i<90005; i++){
        printf("%d \n", cluster_index_host[i]);
	}

	for (int i=0; i<k; i++){
		printf("%lf \n", s0_host[i]);
	}


	FILE *fp1;
	//Dump the prediction output to a file
        fp1 = fopen("output_kmeans", "w");
        for (int i=0; i<n; i++){
                fprintf(fp1, "%d  \n", cluster_index_host[i]);
        }
}

			


	






















