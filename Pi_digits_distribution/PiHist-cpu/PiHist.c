#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void compute_frequency(int *digits, int *frequency, int numDigits);

// function to compute frequency
void compute_frequency(int *digits, int *frequency, int numDigits){
    for (int i = 0; i < numDigits; i++){
        frequency[digits[i]] ++;
    } 
}

int main(int argc, char *argv[]){
    int numDigits;
    FILE *pi_input, *output;
    int *digits;
    int frequency[10] = {0,0,0,0,0,0,0,0,0,0};

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

    digits = (int*) malloc((numDigits) * sizeof(int));

    // parse digits and store them into a int type array
    char *temp = (char*) malloc((numDigits + 1) * sizeof(char));
    fgets(temp, numDigits + 1, pi_input);
    for (int i = 0; i < numDigits; i++){
        digits[i] = temp[i] - '0';
    }

    compute_frequency(digits, frequency, numDigits);

    // output frequency
    output = fopen("frequency_cpu.txt","w");
    for (int i = 0; i < 10; i++){
        fprintf(output, "%d: %d\n", i, frequency[i]);
    }

    printf("The frequency is outputed to 'frequency_cpu.txt'.\n");

    free(digits);
    free(temp);
    fclose(pi_input);
    fclose(output);

}