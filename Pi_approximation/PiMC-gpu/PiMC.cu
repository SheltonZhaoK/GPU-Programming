#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#define RADIUS 1

// kernal to intiailize the random states
__global__ void init (unsigned int seed, curandState_t *states) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, index, 0, &states[index]);
}

// kernal to check and initiate counter using hierarchical atomics strategy
__global__ void approximate_pi (curandState_t *states, int *counter, int numIteration) {
    // initialize shared memory for each block
    __shared__ int partial_Count[2];
    if(threadIdx.x == 0){
        partial_Count[0] = 0;
        partial_Count[1] = 0;
    }
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numIteration){
        float x = curand_uniform(&states[index]) * 2;
        float y = curand_uniform(&states[index]) * 2;

        float distance = sqrt((x - 1) * (x - 1) + (y - 1) * (y - 1));

        // determine the position of the point 0->circle 1->square
        if (distance < RADIUS){
            atomicAdd(&partial_Count[0], 1);
            atomicAdd(&partial_Count[1], 1);
        }
        else{
            atomicAdd(&partial_Count[1], 1);
        }
        __syncthreads();

        if(threadIdx.x == 0){
            atomicAdd(&counter[0], partial_Count[0]);
            atomicAdd(&counter[1], partial_Count[1]);
        }
    }
}

int main(int argc, char *argv[]){
    int numIteration;

    // check appropriate number of command line arguments and positive input
    if (argc > 2){
        perror("Too many arguments are supplied. Program terminated.\n");
        exit(1);
    }
    else if (argc < 2){
        perror("One argument is expected. Program terminated.\n");
        exit(1);
    }
    else if (atof(argv[1]) < 0){
        perror("A positive size should be provided. Program terminated.\n");
        exit(1);
    }

    numIteration = (int) atof(argv[1]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // initialize random states 
    
    curandState_t* states;
    cudaMalloc((void**) &states, numIteration * sizeof(curandState_t));
    dim3 dimBlock(32, 1);
    dim3 dimGrid((int) ceil((float) numIteration / dimBlock.x), 1);
    init<<<dimGrid, dimBlock>>>(time(NULL), states);

    // approximate pi
    float pi;
    int counter[2] = {0, 0};
    int *dev_counter;
    cudaMalloc((void**) &dev_counter, 2 * sizeof(int));
    cudaMemcpy(dev_counter, counter, 2*sizeof(int), cudaMemcpyHostToDevice);

    approximate_pi<<<dimGrid, dimBlock>>>(states, dev_counter, numIteration);

    // run kernel function twice to ensure accurate comparison
    cudaEventRecord(start);
    init<<<dimGrid, dimBlock>>>(time(NULL), states);
    approximate_pi<<<dimGrid, dimBlock>>>(states, dev_counter, numIteration);
    cudaEventRecord(stop);
    cudaMemcpy(counter, dev_counter, 2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    pi = (counter[0] / (float) counter[1]) * 4;
    printf("pi approximated is %f\n", pi);

    float diff;
    cudaEventElapsedTime(&diff, start, stop);
    printf("Time taken: %f s\n", diff/1000);

    cudaFree(dev_counter);
    cudaFree(states);
}