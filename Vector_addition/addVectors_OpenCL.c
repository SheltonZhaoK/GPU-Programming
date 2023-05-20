#include <CL/cl.h>
__kernel void vec_add (__global const float *a,__global const float *b, __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
}

void main(){
  int N = 64; // Array length
  // Get the first available platform
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL); 
  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_context context = clCreateContext(0, 1, &device,NULL, NULL, NULL); 
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL); 
  cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL); 
  clBuildProgram(program,1, &device, NULL, NULL, NULL); 

  cl_kernel kernel = clCreateKernel(program, "vec_add", NULL); 

  cl_float *a = (cl_float *) malloc(N*sizeof(cl_float));
  cl_float *b = (cl_float *) malloc(N*sizeof(cl_float));

  for(int i=0;i<N;i++){
    a[i] = i;
    b[i] = N-i;
  }

  cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*sizeof(cl_float), a, NULL);
  cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*sizeof(cl_float), b, NULL);
  cl_mem c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N*sizeof(cl_float), NULL, NULL);

  size_t global_work_size = N;
  // Set the kernel arguments
  clSetKernelArg(kernel, 0, sizeof(a_buffer), (void*) &a_buffer);
  clSetKernelArg(kernel, 1, sizeof(b_buffer), (void*) &b_buffer);
  clSetKernelArg(kernel, 2, sizeof(c_buffer), (void*) &c_buffer);

  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL); 

  // Block until all commands in command-queue have completed
  clFinish(queue);
  cl_float *c = (cl_float *) malloc(N*sizeof(cl_float));
  clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, N * sizeof(cl_float), c, 0, NULL, NULL); 
  
  free(a); free(b); free(c);
  clReleaseMemObject(a_buffer); clReleaseMemObject(b_buffer); clReleaseMemObject(c_buffer);
  clReleaseKernel(kernel); clReleaseProgram(program); clReleaseContext(context); clReleaseCommandQueue(queue);
