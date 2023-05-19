#include <stdio.h>
#include <stdlib.h>

__global__ void matrixMul(float *m1, float *m2, float *m3, int N);

/* naive matrix multiplication by GPU using global memory */
__global__ void matrixMul(float *m1, float *m2, float *m3, int N){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    if (row < N && column < N){
        for (int i = 0; i < N; i++){
            sum += m1[row * N + i] * m2[i * N + column];
        }
        m3[row * N + column] = sum;
    }
}

/*
    Main function
*/
int main(int argc, char *argv[]){
    int width;
    FILE *outfile; 

    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    /* check correct number of arguments and the positive input size */
    if (argc > 2){
        perror("Too many arguments are supplied. Program terminated.\n");
        exit(1);
    }
    else if (argc < 2){
        perror("One argument is expected. Program terminated.\n");
        exit(1);
    }
    else if (atof(argv[1]) < 0){
        perror("A positive matrix size should be provided. Program terminated.\n");
        exit(1);
    }

    width = (int) atof(argv[1]);
    int size = width * width * sizeof(float);

    // allocate memory for matrices
    a = (float*) malloc(size);
    b = (float*) malloc(size);
    c = (float*) malloc(size);

    // randomly generate float from 0 to 999
    for(int i = 0; i < width * width; i++){
        a[i] = ((float)rand()/RAND_MAX)* 1000.0;
        b[i] = ((float)rand()/RAND_MAX)* 1000.0;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate memory for cuda devices
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    // copy memory from host to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid( (int) ceil((float) width / dimBlock.x), (int) ceil((float) width / dimBlock.y));

    matrixMul<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, width);

    // Call kernal twice for accurate timing information
    cudaEventRecord(start);
    matrixMul<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, width);
    cudaEventRecord(stop);

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    /* output results to product.dat */
    outfile = fopen("product.dat", "w");
    for(int i = 0; i < width * width; i++){
        if( (i != 0) && (i % width == 0) ){
            fputs("\n", outfile);
        }
        fprintf(outfile, "%f\t", c[i]);
    }

    /* calcualte the time for GPU execution */
    float diff;
    cudaEventElapsedTime(&diff, start, stop);
    printf("time to multiply two %d x %d matrices: %f s\n", width, width, diff/1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    exit(0);
}