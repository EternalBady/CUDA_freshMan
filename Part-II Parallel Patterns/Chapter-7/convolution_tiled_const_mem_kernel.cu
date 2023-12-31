#define IN_TILE_DIM 32
#define FILTER_RADIUS 2
#define OUT_TILE_DIM ((IN_TILE_DIM)-2 * (FILTER_RADIUS))

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P, int32_t width, int32_t height)
{
    // Setting indices
    int32_t col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int32_t row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // loading input file
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width)
    { // Store 8*8 data
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    }
    else
    { // Setting Ghost cell
        N_s[threadIdx.y][threadIdx.x] = 0.0F;
    }
    __syncthreads();

    // Calculating output elements
    int32_t tileCol = threadIdx.x - FILTER_RADIUS;
    int32_t tileRow = threadIdx.y - FILTER_RADIUS;

    // turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height)
    {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM)
        {
            float Pvalue = 0.0F;
            for (int32_t fRow = 0; fRow < 2 * FILTER_RADUIS + 1; fRow++)
            {
                for (int32_t fCol = 0; fCol < 2 * FILTER_RADUIS + 1; fCol++)
                {
                    Pvalue += F[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

__global__ void convolution_tiled_2D_const_mem_kernel_b(float *N, float *P, int32_t width, int32_t height)
{
}

#define TILE_DIM 32
#define FILTER_RADIUS 2
__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N, float *P, int32_t width, int32_t height)
{
    // Setting indices
    int32_t col = blockIdx.x * TILE_DIM + threadIdx.x;
    int32_t row = blockIdx.y * TILE_DIM + threadIdx.y;

    // loading input file
    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width)
    { // Store 8*8 data
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    }
    else
    { // Setting Ghost cell
        N_s[threadIdx.y][threadIdx.x] = 0.0F;
    }
    __syncthreads();

    // Calculating output elements
    // turning off the threads at the edges of the block
    if (col < width && row < height)
    {
        float Pvalue = 0.0F;
        for (int32_t fRow = 0; fRow < 2 * FILTER_RADUIS + 1; fRow++)
        {
            for (int32_t fCol = 0; fCol < 2 * FILTER_RADUIS + 1; fCol++)
            {
                if (threadIdx.x - FILTER_RADIUS + fCol >= 0 &&
                    threadIdx.x - FILTER_RADIUS + fCol < TILE_DIM &&
                    threadIdx.y - FILTER_RADIUS + fRow >= 0 &&
                    threadIdx.y - FILTER_RADIUS + fRow < TILE_DIM)
                {
                    Pvalue += F[fRow][fCol] * N_s[threadIdx.y + fRow][threadIdx.x + fCol];
                }
                else
                {
                    if (row - FILTER_RADIUS + fRow >= 0 &&
                        row - FILTER_RADIUS + fRow < height &&
                        col - FILTER_RADIUS + fCol >= 0 &&
                        col - FILTER_RADIUS + fCol < width)
                    {
                        Pvalue += F[fRow][fCol] * N_s[(row - FILTER_RADIUS + fRow) * width + (col - FILTER_RADIUS + fCol)];
                    }
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
__global__ void convolution_tiled_3D_const_mem_kernel(float *N, float *P, int32_t width, int32_t height, int32_t length)
{
    // Setting indices
    int32_t col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int32_t row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int32_t level = blockIdx.z * OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;

    // loading input file
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if ((row >= 0 && row < height) && (col >= 0 && col < width) && (level >= 0 && level < length))
    { // Store 8*8 data
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = N[(level * height + row) * width + col];
    }
    else
    { // Setting Ghost cell
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0F;
    }
    __syncthreads();

    // Calculating output elements
    int32_t tileCol = threadIdx.x - FILTER_RADIUS;
    int32_t tileRow = threadIdx.y - FILTER_RADIUS;
    int32_t tileLevel = threadIdx.z - FILTER_RADIUS;

    // turning off the threads at the edges of the block
    if ((row >= 0 && row < height) && (col >= 0 && col < width) && (level >= 0 && level < length))
    {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM && tileLevel >= 0 && tileLevel < OUT_TILE_DIM)
        {
            float Pvalue = 0.0F;
            for (int32_t fRow = 0; fRow < 2 * FILTER_RADUIS + 1; fRow++)
            {
                for (int32_t fCol = 0; fCol < 2 * FILTER_RADUIS + 1; fCol++)
                {
                    for (int32_t fLevel = 0; fLevel < 2 * FILTER_RADUIS + 1; fLevel++)
                    {
                        Pvalue += F[fRow][fCol][fLevel] * N_s[tileRow + fRow][tileCol + fCol][tileCol + fLevel];
                    }
                }
            }
            P[(level * height + row) * width + col] = Pvalue;
        }
    }
}