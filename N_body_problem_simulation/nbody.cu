#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 9999     // number of bodies
#define MASS 0     // row in array for mass
#define X_POS 1    // row in array for x position
#define Y_POS 2    // row in array for y position
#define Z_POS 3    // row in array for z position
#define X_VEL 4    // row in array for x velocity
#define Y_VEL 5    // row in array for y velocity
#define Z_VEL 6    // row in array for z velocity
#define G 200      // "gravitational constant" (not really)
#define MU 0.001   // "frictional coefficient"
#define BOXL 100.0 // periodic boundary box length

__device__ void norm_device(float *x, float *y, float *z) {
  float mag = sqrt((*x) * (*x) + (*y) * (*y) + (*z) * (*z));
  *x /= mag;
  *y /= mag;
  *z /= mag;
}

__device__ void crossProduct_device(float *vect_A, float *vect_B, float *cross_P){
    cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1]; 
    cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2]; 
    cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0]; 
}

// kernal to intiailize the random states
__global__ void init (unsigned int seed, curandState_t *states) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // compute global thread id
    int skip = blockDim.x * gridDim.x; //return the total number of threads availble
    for (int i = index; i < N; i += skip) {
      if (i < N)
      {
        curand_init(seed, i, 0, &states[index]);
      }
    }
}

// kernal to intiailize the position and velocity
__global__ void initialize (curandState_t *states, float* dev_body){
    int index = blockIdx.x * blockDim.x + threadIdx.x; // compute global thread id
    int skip = blockDim.x * gridDim.x; //return the total number of threads availble

    float vect_A[3];
    float vect_B[3];
    float cross_P[3];

    for (int i = index; i < N; i += skip) {
      if (i < N)
      {
        dev_body[i * 7 + MASS] = 0.001;

        // TODO: initial coordinates centered on origin, ranging -150.0 to +150.0
        dev_body[i* 7 +X_POS] = curand_uniform(&states[index]) * 300 - 150;
        dev_body[i* 7 +Y_POS] = curand_uniform(&states[index]) * 300 - 150;
        dev_body[i* 7 +Z_POS] = curand_uniform(&states[index]) * 300 - 150;

        // initial velocities directions around z-axis
        vect_A[0]= dev_body[i* 7 +X_POS];
        vect_A[1]= dev_body[i* 7 +Y_POS];
        vect_A[2]= dev_body[i* 7 +Z_POS];
        norm_device(&vect_A[0], &vect_A[1], &vect_A[2]);
        vect_B[0]= 0.0; vect_B[1]= 0.0; vect_B[2]= 1.0;
        cross_P[0] = 0.0; cross_P[1] = 0.0; cross_P[2] = 0.0; 
        crossProduct_device(vect_A, vect_B, cross_P);

        // random initial velocities magnitudes
        dev_body[i* 7 +X_VEL] = curand_uniform(&states[index]) * 100 * cross_P[0];
        dev_body[i* 7 +Y_VEL] = curand_uniform(&states[index]) * 100 * cross_P[1];
        dev_body[i* 7 +Z_VEL] = curand_uniform(&states[index]) * 100 * cross_P[2];
      }
    }

}

__global__ void calculateForcesUpdatePosition(float* dev_body, float dt) {
  int index = blockIdx.x * blockDim.x + threadIdx.x; // compute global thread id
  int skip = blockDim.x * gridDim.x; //return the total number of threads availble

  for (int i = index; i < N; i += skip) { // if threads are not enough for total number of bodies, some of the threads would compute more than one body
    if (i < N) // compute if x is smaller than N
    {
        float Fx = 0.0, Fy = 0.0, Fz = 0.0;
        for (int j = 0; j < N; j++)
        {
            if (j != i) 
            {
                float x_diff = dev_body[j * 7 + X_POS] - dev_body[i * 7 + X_POS];
                float y_diff = dev_body[j * 7 + Y_POS] - dev_body[i * 7 + Y_POS];
                float z_diff = dev_body[j * 7 + Z_POS] - dev_body[i * 7 + Z_POS];

                float rr = (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
                float r = sqrt(rr);

                if (r > 50) 
                {
                    float F = -1.0 * (G * dev_body[j * 7 + MASS] * dev_body[i * 7 + MASS]) / (r * r);

                    norm_device(&x_diff, &y_diff, &z_diff); // call norm function that could be accessed by device

                    Fx += x_diff * F;
                    Fy += y_diff * F;
                    Fz += z_diff * F;
                }
            }
        }

        __syncthreads();
        // update velocity
        dev_body[i * 7 + X_VEL] = dev_body[i * 7 + X_VEL] + (Fx * dt) / dev_body[i * 7 + MASS];
        dev_body[i * 7 + Y_VEL] = dev_body[i * 7 + Y_VEL] + (Fy * dt) / dev_body[i * 7 + MASS];
        dev_body[i * 7 + Z_VEL] = dev_body[i * 7 + Z_VEL] + (Fz * dt) / dev_body[i * 7 + MASS];

	      // pdate positions
        dev_body[i * 7 + X_POS] = dev_body[i * 7 + X_POS] + dev_body[i * 7 + X_VEL] * dt;
        dev_body[i * 7 + Y_POS] = dev_body[i * 7 + Y_POS] + dev_body[i * 7 + Y_VEL] * dt;
        dev_body[i * 7 + Z_POS] = dev_body[i * 7 + Z_POS] + dev_body[i * 7 + Z_VEL] * dt;
    }
  }    
}

void crossProduct(float vect_A[], float vect_B[], float cross_P[]) { 
    cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1]; 
    cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2]; 
    cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0]; 
}

void norm(float &x, float &y, float &z) {
  float mag = sqrt(x*x+y*y+z*z);
  x/=mag; y/=mag; z/=mag;
}

int main(int argc, char **argv) {
  float body[9999][7]; // data array of bodies
  float dt = 0.05;
  float diff;
  clock_t total_start, total_end;
  clock_t start, end;
  float totalTime, serialTime, parallelTime;
  FILE *fptr;
  int tmax = 0;
  totalTime = 0; serialTime = 0; parallelTime = 0;
  fptr = fopen("nbody.pdb","w");

  if (argc != 2) {
    fprintf(stderr, "Format: %s { number of timesteps }\n", argv[0]);
    exit (-1);
  }
 
  tmax = atoi(argv[1]);
  // black hole at the center
  body[0][MASS] = 4000.0;
  body[0][X_POS] = 0.0;
  body[0][Y_POS] = 0.0;
  body[0][Z_POS] = 0.0;
  body[0][X_VEL] = 0.0;
  body[0][Y_VEL] = 0.0;
  body[0][Z_VEL] = 0.0;
  
  total_start = clock();
  int totalThreads = 9000;
  int threadsPerBlock = 512;
  int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
  float *dev_body;
  int bodySize = N * 7 * sizeof(float);
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  // assign each body a random initial positions and velocities
  curandState_t* states;
  cudaMalloc((void**) &states, N * sizeof(curandState_t));
  cudaEventRecord(startEvent);
  init<<<numBlocks, threadsPerBlock>>>(time(NULL), states);
  cudaDeviceSynchronize();
  cudaEventRecord(stopEvent);
  cudaEventElapsedTime(&diff, startEvent, stopEvent);
  parallelTime += diff/1000;

  cudaMalloc((void**)&dev_body, bodySize);

  cudaEventRecord(startEvent);
  initialize<<<numBlocks, threadsPerBlock>>>(states, dev_body);
  cudaDeviceSynchronize();
  cudaEventRecord(stopEvent);
  cudaEventElapsedTime(&diff, startEvent, stopEvent);
  parallelTime += diff/1000;

  // print out initial positions in PDB format
  start = clock();
  cudaMemcpy(body, dev_body, bodySize, cudaMemcpyDeviceToHost);
  fprintf(fptr,"MODEL %8d\n", 0);
  for (int i = 0; i < N; i++) {
    fprintf(fptr, "%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
           "ATOM", i+1, "CA ", "GLY", "A", i+1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
  }
  fprintf(fptr,"TER\nENDMDL\n");
  end = clock();
  serialTime += ((double) end - start) / CLOCKS_PER_SEC;

  // step through each time step
  for (int t = 0; t < tmax; t++) {
    // force calculation
    cudaEventRecord(startEvent);
    calculateForcesUpdatePosition<<<numBlocks, threadsPerBlock>>>(dev_body, dt);
    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaMemcpy(body, dev_body, bodySize, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&diff, startEvent, stopEvent);
    parallelTime += diff/1000;

    // print out positions in PDB format
    start = clock();
    fprintf(fptr,"MODEL %8d\n", t+1);
    for (int i = 0; i < N; i++) {
	      fprintf(fptr,"%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
               "ATOM", i+1, "CA ", "GLY", "A", i+1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
    }
    fprintf(fptr,"TER\nENDMDL\n");
    end = clock();
    serialTime += ((double) end - start) / CLOCKS_PER_SEC;
  }  // end of time period loop
  total_end = clock();
  totalTime = ((double) total_end - total_start) / CLOCKS_PER_SEC;
  printf("Simulation of %d rounds with %d bodies with %d processors\n", tmax, N, totalThreads);
  printf("Total time, Serial time, Parallel time: %f %f %f seconds\n", totalTime, serialTime, parallelTime);
}
