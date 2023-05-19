#include <stdio.h>

__global__ void decode(char *text, int length);

/*
    Decrease each character in the character array by 1
*/
__global__ void decode(char *text, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length){
        text[i] = text[i] - 1;
    }
}

/*
    Main function
*/
int main(int argc, char *argv[] ){
    FILE* filePtr;
    char *text;
    char *dev_a;

    //check appropriate number of command line arguments
    if (argc == 2){
        printf("The encryption file is %s\n", argv[1]);
    }
    else if (argc > 2){
        perror("Too many arguments are supplied. Program terminated.\n");
        exit(1);
    }
    else{
        perror("One argument is expected. Program terminated.\n");
        exit(1);
    }

    //check whether the file exists
    filePtr = fopen(argv[1], "r");
    if (filePtr == NULL)
    {
        perror("File is not exist. Program terminated.\n");
        exit(1);
    }
    
    //dynamically allocate memory for character read from a file
    int index = 0;
    char character = fgetc(filePtr);
    while(character != EOF){
        text = (char *) realloc(text, (index+1) * sizeof(char));
        text[index] = character;
        index += 1;
        character = fgetc(filePtr);
    }       
    text[index] = '\0'; //null terminated

    int length = strlen(text);
    int size = length * sizeof(char);

    cudaMalloc((void**)&dev_a, size); //allocate memory for cuda device
    cudaMemcpy(dev_a, text, size, cudaMemcpyHostToDevice); //copy the encrypted text

    dim3 dimBlock(1024, 1, 1); 
    dim3 dimGrid((int) ceil((float)length/1024), 1, 1);
    decode<<< dimGrid, dimBlock>>>(dev_a, length); //Running with "length" threads, handle text that is longer than 1024.

    cudaDeviceSynchronize();

    cudaMemcpy(text, dev_a, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_a);

    for(int i = 0; i < size; i++){
        printf("%c", text[i]);
    }
    printf("\n");

    fclose(filePtr);

    exit(0);
}
