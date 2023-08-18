# Excercises-4

### 1. Consider the following CUDA kernel and the corresponding host function that calls it:

    ```C++
        __global__ void foo_kernel(int32_t * a, int32_t *b){
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if(threadIdx.x < 40 || threadIdx.x >= 104){
                b[i] = a[i] + 1;
            }
            if(i%2 == 0){
                a[i] = b[i] * 2;
            }
            for(uint32_t j = 0; j < 5 - (i%3); ++j){
                b[i] += j;
            }
        }

        void foo(int32_t * a_d, int32_t * b_d){
            uint32_t N = 1024;
            foo_kernel <<< (N + 128 -1)/128, 128 >>>(a_d, b_d);
        }
    ```

    a. What is the number of warps per block?
        128 / 32 = 4;
    b. What is the number of warps in the grid?
        (N + 128 - 1)/128 * 4 = 1151 / 128 * 4 = 9 * 4 = 36;
    c. For the statement on line 04: b[i] = a[i] + 1;

        i. How many warps in the grid are active?
            128 分为四个warp，1：0-31，2：32-63，3：64-95；4：96-127
            因此有效的warp包括 1，2，4；
            所以有 3 * 9 = 27 warps active；
        ii. How many warps in the grid are divergent?
            四个warps中只有 1 处于完全执行，3 处于完全不执行，因此有两组 warps处于分化状态；
            所以有 2 * 9 = 18 warps divergent
        iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
            warp0, 32/32 * 100% = 100%；
        iv. What is the SIMD efficiency (in %) of warp 1 of block 0?
            warp1, (40-32) / 32 * 100% = 8/32 * 100% = 25%;
        v. What is the SIMD efficiency (in %) of warp 3 of block 0?
            warp3, (128-104)/32 * 100% = 24/32 * 100% = 75%;

    d. For the statement on line 07:

        i. How many warps in the grid are active?
            128 分为四个warp，1：0-31，2：32-63，3：64-95；4：96-127
            因此有效的warp包括 1，2，3，4；
            所以有 4 * 9 = 36 warps active；
        ii. How many warps in the grid are divergent?
            All of warps
        iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
            16/32 * 100% = 50%;

    e. For the loop on line 09:
        i. How many iterations have no divergence?
            一次循环中有 0 1 2 3 4 五种可能，而前三次循环是每次都会进行的，则有3 iterations 没有divergence
        ii. How many iterations have divergence?
            from i，2 iterations have divergence

### 2. For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

    2000 / 512 ≈ 4 blocks 4*512 = 2048 threads

### 3. For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?

    According to 1 warps = 32 threads, warps index nearby 2000 is [1983, 2015], so there is one warp have divergence due to the boundary check

### 4. Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads’ total execution time is spent waiting for the barrier?

    Assume that all threads follow the maximum time-consuming：3.0*8 = 24.0 s 
    The ideal running time is：2.0+2.3+3.0+2.8+2.4+1.9+2.6+2.9 = 19.9 s
    So percentage of the threads’ total execution time is spent waiting for the barrier is: 19.9/24.0 * 100% = 82.91%

### 5. A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the __syncthreads() instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain

    It's not a good idea, Because we can't guarantee all the threads Simultaneous execution completed.

### 6. If a CUDA device’s SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM?

    a. 128 threads per block
    b. 256 threads per block
    c. 512 threads per block
    d. 1024 threads per block

    a 128 * 4 = 512
    b 256 * 4 = 1024
    c 512 * 3 = 1536
    d 1024 * 1 = 1024
    So c is the best answer

### 7. Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level

    a. 8 blocks with 128 threads each
    b. 16 blocks with 64 threads each
    c. 32 blocks with 32 threads each
    d. 64 blocks with 32 threads each
    e. 32 blocks with 64 threads each
    
    a 8*128 = 1024 
    b 16*64 = 1024
    c 32*32 = 1024
    d 64*32 = 2048
    e 32*64 = 2048

### 8. Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64K (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor

    a. The kernel uses 128 threads per block and 30 registers per thread.
       2048 / 128 = 16, 
       So the maximum number of threads is 2048, 
       2048 * 30 = 61440 < 65536 
       So a kernel can reach full occupancy.
    b. The kernel uses 32 threads per block and 29 registers per thread.
       2048 / 32 = 64 > 32,
       So the maximum number of threads is 32*32 = 1024<2048
       So b kernel cannot reach full occupancy, the limitation is threads per block.
    c. The kernel uses 256 threads per block and 34 registers per thread.
       2048 / 256 = 8,
       So the maximum number of threads is 2048, 
       2048 * 34 = 69630 > 65536 
       So b kernel cannot reach full occupancy, the limitation is registers.

### 9. A student mentions that they were able to multiply two 1024\*1024 matrices using a matrix multiplication kernel with 32\*32 thread blocks. The student is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?

    1024*1024 / (32*32) = 1024 threads > 512, it means the minimum number of threads per block is 1024.
    So it is impossible to use this device to implement this multiplication.
