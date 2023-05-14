#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>


__global__ void print_from_kernel(vector3* d_accels, vector3* d_hPos, vector3* d_hVel, dim3 dimBlock, dim3 dimGrid, double* dev_mass) {
	int i,j;
	printf("num entities: %d\n", NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++){
		printf("pos=(");
		for (j=0;j<3;j++){
			printf("%lf,",d_hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			printf("%lf,",d_hVel[i][j]);
		}
		printf("),m=%lf\n",dev_mass[i]);
	}
}
//My Kernel
//Whatever I called it
__global__ void pairwise_accel(vector3* d_accels, vector3* d_hPos, vector3* d_hVel, double* d_mass) {
	int k;
	//Assuming we spawn enough blocks+threads to cover the whole NUMENTITIESxNUMENTITIES matrix, each thread does 1 calculation
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i==10 && j==0) {
	printf("Hello from thread coordinates %d, %d with args bidx.x: %d, bdim.x: %d, tidx.x: %d, bidx.y: %d, bdim.y: %d, tidx.y: %d, mass: %d\n", 
	i, j, blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y, blockDim.y, threadIdx.y, d_mass[j]);
	}
	//HELLO? WHAT THE FRICK?
	if (i > NUMENTITIES || j > NUMENTITIES) {
		return;
	}
	if (i==j) {
		FILL_VECTOR(d_accels[i*NUMENTITIES+j],0,0,0);
	}
	else{
		vector3 distance;
		for (k=0;k<3;k++) distance[k]=d_hPos[i][k]-d_hPos[j][k];
		double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
		double magnitude=sqrt(magnitude_sq);
		double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
		FILL_VECTOR(d_accels[i*NUMENTITIES+j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
		if (i==10 && j==0) printf("accels at %d, %d: %f\t%f\t%f\n", i, j, d_accels[i*NUMENTITIES+j][0],d_accels[i*NUMENTITIES+j][1],d_accels[i*NUMENTITIES+j][2]);
	}
}


__global__ void sum_rows_and_compute(vector3* d_accels, vector3* d_hPos, vector3* d_hVel, double* d_mass) {
	int k;
	//Assuming we spawn enough blocks+threads to cover the whole NUMENTITIESxNUMENTITIES matrix, each thread does 1 calculation
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
		vector3 accel_sum={0,0,0};
		for (k=0;k<3;k++)
			accel_sum[k]+=d_accels[i*NUMENTITIES+j][k];

		for (k=0;k<3;k++){
			d_hVel[i][k]+=accel_sum[k]*INTERVAL;
			d_hPos[i][k]=d_hVel[i][k]*INTERVAL;
		}
}



//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(vector3* d_accels, vector3* d_hPos, vector3* d_hVel, dim3 dimBlock, dim3 dimGrid, double* dev_mass){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	//int i,j,k;
	/**
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
	Commented out original code */


	//MY CODE SECTION (1st attempt):
	//cudaMalloc(&d_values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	print_from_kernel<<<1,1>>>(d_accels, d_hPos, d_hVel, dev_mass);
	pairwise_accel<<<dimGrid, dimBlock>>>(d_accels, d_hPos, d_hVel, dev_mass);
	cudaDeviceSynchronize();
	sum_rows_and_compute<<<dimGrid, dimBlock>>>(d_accels, d_hPos, d_hVel, dev_mass);
	cudaDeviceSynchronize();
	
	//END MY CODE SECTION


	/*COMMENTED OUT
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
	*/

	//Parallel Frees
	//cudaFree(d_accels);
}