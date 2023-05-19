#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// kernal to compute frequency using hierarchical atomics strategy
__global__ void compute_frequency (int *dev_digits, int *dev_frequency, int numDigits) {
    __shared__ int partial_frequency[10];

    if (threadIdx.x == 0){
        for (int i = 0; i < 10; i ++){
            partial_frequency[i] = 0;
        }
    }
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numDigits){
        atomicAdd(&partial_frequency[dev_digits[index]], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0){
        for (int i = 0; i < 10; i ++){
            atomicAdd(&dev_frequency[i], partial_frequency[i]);
        }
    }
    
}

int main(int argc, char *argv[]){
    int numDigits;
    FILE *pi_input, *output;
    int *digits;
    int frequency[10] = {0,0,0,0,0,0,0,0,0,0};

    int *dev_digits, *dev_frequency;

    // check appropriate number of command line arguments and positive input
    if (argc > 3){
        perror("Too many arguments are supplied. Program terminated.\n");
        exit(1);
    }
    else if (argc < 3){
        perror("Two arguments are expected. Program terminated.\n");
        exit(1);
    }
    else if (atof(argv[2]) < 0 ){
        perror("A positive size should be provided. Program terminated.\n");
        exit(1);
    }

    numDigits = (int) atof(argv[2]);
    pi_input = fopen(argv[1], "r");
    if (pi_input == NULL)
    {
        perror("File does not exist. Program terminated.\n");
    }

    digits = (int*) malloc(numDigits * sizeof(int));

    // parse digits and store them into a int type array
    char *temp;
    temp = (char*) malloc((numDigits + 1) * sizeof(char));
    fgets(temp, numDigits + 1, pi_input);

    for (int i = 0; i < numDigits; i++){
        digits[i] = temp[i] - '0';
    }

    cudaMalloc((void**) &dev_digits, numDigits * sizeof(int));
    cudaMalloc((void**) &dev_frequency, 10 * sizeof(int));

    cudaMemcpy(dev_digits, digits, numDigits * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_frequency, frequency, 10 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 1);
    dim3 dimGrid((int) ceil((float) numDigits / dimBlock.x), 1);
    compute_frequency<<<dimGrid, dimBlock>>>(dev_digits, dev_frequency, numDigits);
    cudaDeviceSynchronize();
    cudaMemcpy(frequency, dev_frequency, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    // output frequency
    output = fopen("frequency_gpu.txt","w");
    for (int i = 0; i < 10; i++){
        fprintf(output, "%d: %d\n", i, frequency[i]);
    }

    printf("The frequency is outputed to 'frequency_gpu.txt'.\n");

    free(digits);
    free(temp);
    cudaFree(dev_digits);
    cudaFree(dev_frequency);

    fclose(pi_input);
    fclose(output);

}