#include <iostream>
#include <cstdint>
#include <cuda.h>

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
convolution_2D_const_mem_kernel(float *N, float *P, int32_t r, int32_t width, int32_t height)
{

    int32_t outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0.0F;

    for (uint32_t fRow = 0; fRow < 2 * r + 1; fRow++)
    {
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

__global__ void
convolution_3D_const_mem_kernel(float *N, float *P, int32_t r, int32_t width, int32_t height, int32_t level)
{

    int32_t outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t outLevel = blockIdx.z * blockDim.z + threadIdx.z;

    float Pvalue = 0.0F;

    for (uint32_t fRow = 0; fRow < 2 * r + 1; fRow++)
    {
        for (uint32_t fCol = 0; fCol < 2 * r + 1; fCol++)
        {
            // For each level value;
            for (uint32_t fLevel = 0; fLevel < 2 * r + 1; fLevel++)
            {
                // Current input data array index(inRow, inCol)
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
    P[outRow][outCol] = Pvalue;
}