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

__global__ void
convolution_3D_basic_kernel(float *N, float *F, float *P, int32_t r, int32_t width, int32_t height, int32_t level)
{
    // Output index also input center index
    int32_t outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t outLevel = blockIdx.z * blockDim.z + threadIdx.z;
    // Store calculate value
    float Pvalue = 0.0F;

    // For each row;
    for (uint32_t fRow = 0; fRow < 2 * r + 1; fRow++)
    {
        // For each col value;
        for (uint32_t fCol = 0; fCol < 2 * r + 1; fCol++)
        {
            // For each level value;
            for (uint32_t fLevel = 0; fLevel < 2 * r + 1; fLevel++)
            {
                // Current input data array index(inRow, inCol, inLevel)
                inRow = outRow - r + fRow;
                inCol = outCol - r + fCol;
                inLevel = outLevel - r + fLevel;
                // Setting bandary conditions include col and row;
                if ((inRow >= 0 && inRow < height) && (inCol >= 0 && inCol < width) && (inLevel >= 0 && inLevel < level))
                {
                    // Pvalue = Fwith output index * N width current index;
                    Pvalue += F[fRow][fCol][fLevel] * N[(level * height + inRow) * width + inCol];
                }
            }
        }
    }
    // Output Pvalue
    P[outRow][outCol][outLevel] = Pvalue;
}

__global__ void
convolution_2D_basic_kernel(float *N, float *F, float *P, int32_t r, int32_t width, int32_t height)
{
    // Output index also input center index
    int32_t outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t outRow = blockIdx.y * blockDim.y + threadIdx.y;
    // Store calculate value
    float Pvalue = 0.0F;

    // For each row;
    for (uint32_t fRow = 0; fRow < 2 * r + 1; fRow++)
    {
        // For each col value;
        for (uint32_t fCol = 0; fCol < 2 * r + 1; fCol++)
        {
            // Current input data array index(inRow, inCol)
            inRow = outRow - r + fRow;
            inCol = outCol - r + fCol;
            // Setting bandary conditions include col and row;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
            {
                // Pvalue = Fwith output index * N width current index;
                Pvalue += F[fRow][fCol] * N[inRow * width + inCol];
            }
        }
    }
    // Output Pvalue
    P[outRow][outCol] = Pvalue;
}