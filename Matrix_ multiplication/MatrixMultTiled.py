import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

# Define the size of the matrices and tile size
M = 1024
N = 1024
K = 1024
TILE_SIZE = 16

# Generate two random matrices
a = np.random.randn(M, K).astype(np.float32)
b = np.random.randn(K, N).astype(np.float32)

# Define the CUDA kernel function
mod = SourceModule("""
  #define TILE_SIZE %(tile_size)d

  __global__ void matmul(float *a, float *b, float *c, int m, int n, int k) {
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0;
    for (int i = 0; i < (k-1)/TILE_SIZE+1; i++) {
      if (row < m && i*TILE_SIZE+tx < k) {
        s_a[ty][tx] = a[row*k + i*TILE_SIZE+tx];
      } else {
        s_a[ty][tx] = 0.0;
      }
      if (i*TILE_SIZE+ty < k && col < n) {
        s_b[ty][tx] = b[(i*TILE_SIZE+ty)*n + col];
      } else {
        s_b[ty][tx] = 0.0;
      }
      __syncthreads();

      for (int j = 0; j < TILE_SIZE; j++) {
        sum += s_a[ty][j] * s_b[j][tx];
      }
      __syncthreads();
    }
    if (row < m && col < n) {
      c[row*n+col] = sum;
    }
  }
""" % {"tile_size": TILE_SIZE})

# Get the kernel function from the module
matmul = mod.get_function("matmul")

# Allocate memory on the GPU
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(M*N * np.dtype(np.float32).itemsize)

# Copy the matrices to the GPU
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Launch the kernel function
grid = ((N-1)//TILE_SIZE+1, (M-1)//TILE_SIZE+1, 1)
block = (TILE_SIZE, TILE_SIZE, 1)
matmul(a_gpu, b_gpu, c_gpu, np.int32(M), np.int32(N), np.int32(K), block=block, grid=grid)

# Copy the result from the GPU to the CPU
c = np.empty((M, N), dtype=np.float32)
cuda.memcpy_dtoh(c, c_gpu)

# Print the result
print(c)
