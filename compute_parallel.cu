#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	dim3 dimBlock(16, 16);
	dim3 dimGrid((NUMENTITIES+dimBlock.x-1)/dimBlock.x, (NUMENTITIES+dimBlock.y-1)/dimBlock.y);
	compute_pairwise<<<dimGrid, dimBlock>>>(d_accels, d_hPos, d_mass);
	dim3 dimBlock(256);
	dim3 dimGrid((NUMENTITIES+dimBlock.x-1)/dimBlock.x);
	update_bodies<<<dimGrid, dimBlock>>>(d_accels, d_hVel, d_mass)
}

__global__ void compute_pairwise(vector3* accels, vector3* hPos, double* mass) {
	//Assuming this is going to be called with enough thread blocks + threads that each is doing 1 calculation.
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= NUMENTITIES || j >= NUMENTITIES) {
		return; //avoids touching memory not in use (extending past the # of entities we have)
	}

	//first compute the pairwise accelerations.  Effect is on the first argument.
	if (i==j) {
		FILL_VECTOR(accels[i*NUMENTITIES+j],0,0,0);
	}
	else{
		vector3 distance;
		for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
		double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
		double magnitude=sqrt(magnitude_sq);
		double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
		FILL_VECTOR(accels[i*NUMENTITIES+j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
	}
}

__global__ void update_bodies(vector3* accels, vector3* hVel, vector3* hPos) {
	//Assuming that each call to update_bodies has enough blocks/threads that each is summing one column.
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	vector3 accel_sum={0,0,0};
	for (j=0;j<NUMENTITIES;j++){
		for (k=0;k<3;k++)
			accel_sum[k]+=accels[i*NUMENTITIES+j][k];
	}
	//compute the new velocity based on the acceleration and time interval
	//compute the new position based on the velocity and time interval
	for (k=0;k<3;k++){
		hVel[i][k]+=accel_sum[k]*INTERVAL;
		hPos[i][k]=hVel[i][k]*INTERVAL;
	}
}