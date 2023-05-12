#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h> //I added this to print my silly little dumb kernel.


__global__ 
void dumb_kernel(int size, int* matrix) {
	//printf("%d is the current thread\n", threadIdx.x);
	for (int i=0; i<size*10; i++) {
		matrix[i] = i;
		/*
		for (int j=0; j<size; j++) {
			matrix[i*size+j] = i*size+j+threadIdx.x;
		}
		*/
	}
	
}

/*

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]=hVel[i][k]*INTERVAL;
		}
	}
	free(accels);
	free(values);
}


__global__
void sum_rows(int n, vector3* accel_sum, vector3** accels) {
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i=0; i<n; i+=stride) {
		accel_sum[n] += 0;
	}
}
*/


void do_thing() {
	printf("starting the function\n");
	int my_matrix[100];
	int example_matrix[100];
	for (int i = 0; i < 100; i++) {
			my_matrix[i] = 0;
			printf("%d\t", my_matrix[i]);
		}
	printf("initialized local matrix\n");
	int d_my_matrix[100];
	//cudaMallocManaged(&my_matrix, 100*sizeof(int));
	printf("allocating matrix on device\n");
	cudaMalloc((void**) &d_my_matrix, 10*10*sizeof(int));
	printf("copying matrix to device\n");
	cudaMemcpy(d_my_matrix, my_matrix, 10*10*sizeof(int), cudaMemcpyHostToDevice);
	//cudaDeviceSynchronize();
	printf("cuda malloc completedd\n");
	dumb_kernel<<<1, 1>>>(10, d_my_matrix);
	printf("launched kernel\n");
	cudaDeviceSynchronize();
	printf("sync'd up\n");
	cudaMemcpy(example_matrix, d_my_matrix, 10*10*sizeof(int), cudaMemcpyDeviceToHost);
	printf("copied back\n");
	cudaFree(d_my_matrix);
	printf("freed device matrix\n");
	for (int i = 0; i < 100; i++) {
		printf("%d\n", example_matrix[i]);
		/*
		printf("\n");
		for (int j = 0; j < 10; j++) {
			printf("%d\t", my_matrix[i*10+j]);
		}
		*/
	}
	printf("done\n");

}
