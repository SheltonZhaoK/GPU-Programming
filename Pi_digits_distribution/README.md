# Pi Digits Distribution
The PiHist programs use two command line arguments that corresponds to 1) the filename that contains 10 million digits of pi and 2) the number of digits to be evaluated. All of the digits are on a single line of the file. The programs then output the frequencies of the numbers 0-9 that are encountered in the file. The C program implements a serial code via a function call, and the CUDA
program implements a parallel code via a kernel call. The hierarchical atomics strategy is used.
