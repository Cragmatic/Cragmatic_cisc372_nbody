#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute_parallel.h"

// represents the objects in the system.  Global variables
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
double *mass;

//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	mass = (double *)malloc(sizeof(double) * numObjects);
}

//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(mass);
}

//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(){
	int i,j;
	double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
	for (i=0;i<=NUMPLANETS;i++){
		for (j=0;j<3;j++){
			hPos[i][j]=data[i][j];
			hVel[i][j]=data[i][j+3];
		}
		mass[i]=data[i][6];
	}
}

//randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
//Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
//Returns: None
//Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j, c = start;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

//printSystem: Prints out the entire system to the supplied file
//Parameters: 	handle: A handle to an open file with write access to prnt the data to
//Returns: 		none
//Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE* handle){
	int i,j;
	for (i=0;i<NUMENTITIES;i++){
		fprintf(handle,"pos=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hVel[i][j]);
		}
		fprintf(handle,"),m=%lf\n",mass[i]);
	}
}

//Debug Function
__global__ void print_from_kernel(vector3* d_accels, vector3* d_hPos, vector3* d_hVel, double* dev_mass) {
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
//End Debug Function

int main(int argc, char **argv)
{
	clock_t t0=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);
	initHostMemory(NUMENTITIES);
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	//now we have a system.
	#ifdef DEBUG
		//printSystem(stdout);
	#endif

    //start block
	vector3* d_accels;
	double *dev_mass;
	cudaMalloc(&d_hPos, sizeof(vector3) * NUMENTITIES);
	cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMalloc(&d_hVel, sizeof(vector3) * NUMENTITIES);
	cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMalloc(&dev_mass, sizeof(double) * NUMENTITIES);
	cudaMemcpy(dev_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice); //copy mass up into d_mass
	cudaMalloc(&d_accels, sizeof(vector3*)*NUMENTITIES*NUMENTITIES);

	dim3 dimBlock(16, 16);
	//Spawns in enough 16x16 blocks arranged in NxN to coveer the whole matrix
	dim3 dimGrid((NUMENTITIES+dimBlock.x-1)/dimBlock.x, (NUMENTITIES+dimBlock.y-1)/dimBlock.y);
	//printf("\n\ndimGrid: %d, %d",dimGrid.x, dimGrid.y);
    //end block
	//cudaDeviceSynchronize();
	//print_from_kernel<<<1,1>>>(d_accels, d_hPos, d_hVel, dev_mass);

	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){
		compute(d_accels, d_hPos, d_hVel, dimBlock, dimGrid, dev_mass); //Altered
		//compute<<<1, 1>>>(d_accels, d_hPos, d_hVel, dimBlock, dimGrid, dev_mass);
		//printf("I am about to call pairwise. wow!!!");
		//pairwise_accel<<<dimGrid, dimBlock, dimBlock.x*dimBlock.y*sizeof(vector3)>>>(d_hPos, d_hVel, mass);
		cudaDeviceSynchronize();
	}

    //start block 2
	//printf("hVel[0] before: %f, %f, %f\n", hVel[0][0], hVel[0][1], hVel[0][2]);
	//printf("hPos[0] before: %f, %f, %f\n", hPos[0][0], hPos[0][1], hPos[0][2]);
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	//printf("hVel[0] after: %f, %f, %f\n", hVel[0][0], hVel[0][1], hVel[0][2]);
	//printf("hPos[0] after: %f, %f, %f\n", hPos[0][0], hPos[0][1], hPos[0][2]);
    
    cudaFree(d_hPos);
	cudaFree(d_hVel);
	cudaFree(dev_mass);
    //end block 2

	clock_t t1=clock()-t0;
#ifdef DEBUG
	//printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	freeHostMemory();
}