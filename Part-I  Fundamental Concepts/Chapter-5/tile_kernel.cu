#define TILE_WIDTH 16
__global__ void matrixMulKernel(float* M, float * N, float * P, int32_t Width){

    // Define shared_memory to store 
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int32_t bx = blockIdx.x;int32_t by = blockIdx.y;
    int32_t tx = threadIdx.x;int32_t ty = threadIdx.y;

    //Identify the row and column of the P element to work on
    int32_t Row = by * TILE_WIDTH + ty;
    int32_t Col = bx * TILE_WIDTH + tx;

    //Loop over the M & N tiles required to compute P element
    float Pvalue = 0;
    for (uint32_t ph = 0; ph < Width/TILE_WIDTH; ++ph){

        // Collaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH +tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();

        for(uint32_t k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[Row*Width + Col] = Pvalue;

}