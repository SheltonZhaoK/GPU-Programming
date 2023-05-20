/*
 * Sample program that uses CUDA to perform element-wise add of two
 * vectors.  Each element is the responsibility of a separate thread.
 *
 * compile with:
 *    nvcc -o addVectors addVectors.cu
 * run with:
 *    ./addVectors
 */

#include <stdio.h>

// problem size (vector length):
#define N 10

__global__ void kernel(int* res, int* a, int* b) {
  // function that runs on GPU to do the addition
  // sets res[i] = a[i] + b[i]; each thread is responsible for one value of i

  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if(thread_id < N) {
    res[thread_id] = a[thread_id] + b[thread_id];
  }
}

int main() {
  for (int i = 0; i < 100; i++){
    printf("Iteration %d\n", i);
    int* a;       //input arrays (on host)
    int* b;
    int* res;     //output array (on host)

    int* dev_a;   //input arrays (on GPU)
    int* dev_b;
    int* dev_res; //output array (on GPU) 

    // allocate memory
    a = (int*) malloc(N*sizeof(int));
    b = (int*) malloc(N*sizeof(int));
    res = (int*) malloc(N*sizeof(int));
    cudaMalloc((void**) &dev_a, N*sizeof(int));
    cudaMalloc((void**) &dev_b, N*sizeof(int));
    cudaMalloc((void**) &dev_res, N*sizeof(int));

    // set up contents of a and b
    for(int i=0; i < N; i++)
      a[i] = b[i] = i;

    // allocate timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start timer
    cudaEventRecord(start);

    // transfer a and b to the GPU
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

    // call the kernel
    int threads = 512;                   // # threads per block
    int blocks = (N+threads-1)/threads;  // # blocks (N/threads rounded up)
    kernel<<<blocks,threads>>>(dev_res, dev_a, dev_b);

    // transfer res to the host
    cudaMemcpy(res, dev_res, N*sizeof(int), cudaMemcpyDeviceToHost);

    // stop timer and print time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float diff;
    cudaEventElapsedTime(&diff, start, stop);
    printf("time: %f ms\n", diff);

    // deallocate timers
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // verify results
    for(int i=0; i < N; i++)
      printf("%d ", res[i]);
    printf("\n");

    // free the memory (because I care)
    free(a);
    free(b);
    free(res);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);
  }
  


}
