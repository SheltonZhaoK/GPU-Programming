#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define RADIUS 1

float approximate_pi(int N, int* counter);

// function to approximate pi
float approximate_pi(int N, int* counter){
    float pi, distance, x, y;
    for (int i = 0; i < N; i++){
        x = ((float) rand() / RAND_MAX) * 2; // generate random x coordinate from 0 to 2
        y = ((float) rand() / RAND_MAX) * 2; // generate random y coordinate from 0 to 2
        
        distance = sqrt((x - 1) * (x - 1) + (y - 1) * (y - 1)); // compute distance to center

        // determine the position of the point 0->circle 1->square
        if (distance < RADIUS){
            counter[0] ++; 
            counter[1] ++;
        }
        else{
            counter[1] ++;
        }
        
    }

    pi = (counter[0] / (float) counter[1]) * 4;
    return pi;
}   


int main(int argc, char *argv[]){
    int numIteration;
    float pi;
    int counter[2];
    clock_t start, end;

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

    start = clock();
    pi = approximate_pi(numIteration, counter);
    end = clock();

    printf("pi approximated is %f\n", pi);
    printf("Time taken: %f s\n", ((double) end - start) / CLOCKS_PER_SEC);


}