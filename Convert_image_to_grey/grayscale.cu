#include <iostream>
#include <vector>

#include <cuda.h>
#include <vector_types.h>

#include "bitmap_image.hpp"

using namespace std;

__global__ void color_to_grey(uchar3 *d_in, uchar3 *d_out, int width, int height);

/*
    convert color to grayscale
*/
__global__ void color_to_grey(uchar3 *d_in, uchar3 *d_out, int width, int height)
{    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int pos = row * width + col;
   
    if (col < width && row < height){
        int color = d_in[pos].x * 0.299 + d_in[pos].y * 0.578 + d_in[pos].z * 0.114;
        d_out[pos].x = color; 
        d_out[pos].y = color;
        d_out[pos].z = color;
    }
    
    // TODO: Convert color to grayscale by mapping components of uchar3 to RGB
    // x -> R; y -> G; z -> B
    // Apply the formula:
    // output = 0.299f * R + 0.578f * G + 0.114f * B
    // Hint: First create a mapping from 2D block and grid locations to an
    // absolute 2D location in the image then use that to calculate a 1D offset
}

/*
    main function
*/
int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << "format: " << argv[0] << " { 24-bit BMP Image Filename }" << endl;
        exit(1);
    }
    
    bitmap_image bmp(argv[1]);

    if(!bmp)
    {
        cerr << "Image not found" << endl;
        exit(1);
    }

    int height = bmp.height();
    int width = bmp.width();

    cout << "Image dimensions:" << endl;
    cout << "height: " << height << " width: " << width << endl;

    cout << "Converting " << argv[1] << " from color to grayscale..." << endl;
    //Transform image into vector of doubles
    vector<uchar3> input_image;
    rgb_t color;

    //read pixels row by row
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){  
            bmp.get_pixel(x, y, color);
            input_image.push_back( {color.red, color.green, color.blue} );
        }
    }
    vector<uchar3> output_image(input_image.size());

    uchar3 *d_in, *d_out;
    int img_size = (input_image.size() * sizeof(char) * 3);
    cudaMalloc(&d_in, img_size);  //allocate memory for cuda device
    cudaMalloc(&d_out, img_size); //allocate memory for cuda device

    cudaMemcpy(d_in, input_image.data(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, input_image.data(), img_size, cudaMemcpyHostToDevice);

    
    // TODO: Fill in the correnct blockSize and gridSize
    // currently only one block with one thread is being launched

    dim3 dimBlock(32, 32, 1); //use block with 2 dimensions and 1024 threads for each block in total
    dim3 dimGrid((int) ceil((float)width/32), (int) ceil((float)height/32), 1); //use grid with 2 dimensions and width/32 * height/32 blocks for each grid in total
    color_to_grey<<<dimGrid, dimBlock>>> (d_in, d_out, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(output_image.data(), d_out, img_size, cudaMemcpyDeviceToHost);
    
    //set pixels line by line
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++)
        {
            int pos = y * width + x;
            bmp.set_pixel(x, y, output_image[pos].x, output_image[pos].y, output_image[pos].z);
        }
    }

    cout << "Conversion complete." << endl;
    
    char output[] = "test_gray_konghaoZhao.bmp";
    bmp.save_image(output);
    printf("Converted picture is %s\n", output);

    cudaFree(d_in);
    cudaFree(d_out);
}

