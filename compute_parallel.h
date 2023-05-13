void compute(vector3* d_accels, vector3* d_hPos, vector3* d_hVel, dim3 dimBlock, dim3 dimGrid, double* d_mass);
//__global__ void pairwise_accel(vector3* d_hPos, vector3* d_hVel, double* mass);