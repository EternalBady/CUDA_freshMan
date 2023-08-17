#### 1. In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them

a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.

    ```C++
    void MatrixMul(float* M, float * N, float *P, int32_t width){
        for(uint32_t i = 0; i < width; ++i){    // turn out to be row
            for(uint32_t j = 0; j < width; ++j){
                float Pvalue = 0.0F;
                for(uint32_t k = 0; k < width; ++k){
                    Pvalue += M[i * width + k] * N[k * width + j];
                }
                P[i*width + j] = Pvalue;
            }
        }
    }

    // 由上面的遍历改造而成，把对 i的遍历改成对row的遍历；
    __global__ void MatrixMulKernel(float* M, float * N, float *P, int32_t width){
        int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
        if(row < width){
            for(uint32_t j = 0; j < width; ++j){
                float Pvalue = 0.0F;
                for(uint32_t k = 0; k < width; ++k){
                    Pvalue += M[row * width + k] * N[k * width + j];
                }
                P[row*width + j] = Pvalue;
            }
        }
    }
    ```
b. Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design.

    ```C++
    __global__ void MatrixMulKernel(float* M, float * N, float *P, int32_t width){
        int32_t col = blockIdx.y * blockDim.y + threadIdx.y;
        if(col < width){
            for(uint32_t i = 0; i < width; ++i){
                float Pvalue = 0.0F;
                for(uint32_t j = 0; j < width; ++j){
                    // 第二层for 遍历计算每一行的第j 个元素的值；
                    Pvalue += M[i * width + j] * N[col + j * width];
                }
                // 将Pvalue 的值放到P[col + i*width]，也就是P[col][i]
                P[col + i * width] = Pvalue;
            }
        }
    }
    ```

c. Analyze the pros and cons of each of the two kernel designs.

    a: 按照行输出，可以利用空间局部性 
    b: 

#### 2. A matrix-vector multiplication takes an input matrix B and a vector C and produces one output vector A. Each element of the output vector A is the dot product of one row of the input matrix B and C, that is, $\mathbf{A[i]=\sum ^i B[i][j]+C[j]}$ . For simplicity we will handle only square matrices whose elements are singleprecision floating-point numbers. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. Use one thread to calculate an output vector element

#### 3. Consider the following CUDA kernel and the corresponding host function that calls it\

    ```C++
    __global__
    void foo_kernel(float * a, float * b, uint32_t M, uint32_t N){
        uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

        if(row < M && col < N){
            b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
        }
    }
    void foo(float * a_d, float * b_d){
        uint32_t M = 150;
        uint32_t N = 300;
        dim3 bd(16, 32);
        dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);
        foo_kernel <<<gd, bd>>>(a_d, b_d, M, N);
    }
    ```

a. What is the number of threads per block?

    16*32 = 512

b. What is the number of threads in the grid?

    ((N-1)/16+1) * ((M-1)/32+1) * 512 = ((300-1)/16+1) * ((150-1)/32+1) * 512 = 120 * 512 = 61440

c. What is the number of blocks in the grid?

    ((300-1)/16+1) * ((150-1)/32+1) = 20*6 = 120

d. What is the number of threads that execute the code on line 05?

    ((N-1)/16+1) * ((M-1)/32+1) * 512 = ((300-1)/16+1) * ((150-1)/32+1) * 512 = 57015

#### 4. Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10

a. If the matrix is stored in row-major order.

    400 * 19 + 10 = 7610
b. If the matrix is stored in column-major order.

    500 * 9 + 20 = 4520

#### 5. Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and z = 5

    (5-1)*(400*500) + (20-1)*400 + 5 = 4*200000 + 7600 + 5 = 807605