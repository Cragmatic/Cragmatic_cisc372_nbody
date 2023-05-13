#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//My Kernel
//Whatever I called it
__global__ void pairwise_accel(vector3** accels, vector3* hPos, double* mass) {
	int k;
	//Assuming we spawn enough blocks+threads to cover the whole NUMENTITIESxNUMENTITIES matrix, each thread does 1 calculation
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
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
	/**
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
	vector3** d_accels;
	cudaMalloc(&d_accels, sizeof(vector3*)*NUMENTITIES);
	cudaMalloc(&d_hPos, sizeof(vector3) * NUMENTITIES);
	cudaMalloc(&d_hVel, sizeof(vector3) * NUMENTITIES);

	cudaMemcpy(d_accels, accels, sizeof(vector3*) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	dim3 dimBlock(16, 16);
	dim3 dimGrid(NUMENTITIES/dimBlock.x, NUMENTITIES/dimBlock.y);
	pairwise_accel<<<dimGrid, dimBlock>>>(d_accels, d_hPos, mass);
	cudaMemcpy(accels, d_accels, sizeof(vector3*) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

	//END MY CODE SECTION


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

	//Parallel Frees
	cudaFree(d_accels);
	cudaFree(d_hPos);
	cudaFree(d_hVel);
}