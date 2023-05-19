# Convert Image to grey scale
A color image can be converted into a grayscale image by converting the value of each pixel so that it only carries intensity information. One possible calculating formula is: output = 0.299f * R + 0.578f * G + 0.114f * B.

The code grayscale.cu is to read in a 24-bit BMP image, with its name as a command line input, convert it into a grayscale image, and output it in a BMP format with the filename “grayscaled.bmp”.