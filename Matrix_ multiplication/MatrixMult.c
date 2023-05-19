#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrixMult(float *m_a, float *m_b, float *m_c, int width);

/* sequential matrix multiplication */
void matrixMult(float *m_a, float *m_b, float *m_c, int width) {
  for (int row = 0; row < width; row++) {
    for (int col = 0; col < width; col++) {
        float temp = 0;
        for (int i = 0; i < width; i++) {
            temp += m_a[row * width + i] * m_b[i * width + col];
        }
        m_c[row * width + col] = temp;
    }
  }
}

/* main function */
int main(int argc, char *argv[]){
    int width;
    float *m_a, *m_b, *m_c;
    clock_t start, end;
    FILE *outfile; 

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
        perror("A positive matrix size should be provided. Program terminated.\n");
        exit(1);
    }

    width = (int) atof(argv[1]);
    int size = width * width * sizeof(float);

    // allocate memory for matrices
    m_a = (float*) malloc(size);
    m_b = (float*) malloc(size);
    m_c = (float*) malloc(size);

    // randomly generate float from 0 to 999
    for(int i = 0; i < width * width; i++){
        m_a[i] = ((float)rand()/RAND_MAX)* 1000.0;
        m_b[i] = ((float)rand()/RAND_MAX)* 1000.0;
    }
    
    start = clock();
    matrixMult(m_a, m_b, m_c, width);
    end = clock();

    double time = ((double) end - start) / CLOCKS_PER_SEC;
    printf("Time to multiply two %d x %d matrices: %f s\n", width, width, time);

    /* store results to product.dat */
    outfile = fopen("product.dat", "w");
    for(int i = 0; i < width * width; i++){
        if( (i != 0) && (i % width == 0) ){
            fputs("\n", outfile);
        }
        fprintf(outfile, "%f\t", m_c[i]);
    }

    free(m_a); free(m_b); free(m_c);
}