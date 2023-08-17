#include <iostream>
#include <cstdint>

__global__
void colortoGrayscaleConvertion(u_char * Pout, u_char *Pin, int32_t width, int32_t height){
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < width && row < height){
        int32_t grayOffset = row*width + col;

        int32_t rgbOffset = grayOffset * CHANNELS;
        u_char r = Pin[rgbOffset ];
        u_char g = Pin[rgbOffset + 1];
        u_char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

__global__
void blurKernel(u_char * in, u_char * out, int32_t w, int32_t h){
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < w && row < h){
        int32_t pixVal = 0;
        int32_t pixels = 0;

        for(int32_t blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow){
            for(int32_t blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol){
                int32_t curRow = row + blurRow;
                int32_t curCol = col + blurCol;

                if(((curRow >= 0) && (curRow < h) && ((curCol >= 0) && (curCol < width)))){
                    pixVal += in[curRow *w +curCol];
                    ++pixels;
                }
            }
        }

        out[row*w + col] = static_cast<u_char>(pixVal/pixels);
    }
}

/**
 * @brief Each Thread Output one value of matrix
 * 
 * @param M 
 * @param N 
 * @param P 
 * @param width 
 * @return __global__ 
 */
__global__ void MatrixMulKernel(float* M, float * N, float *P, int32_t width){
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if((row < width) && (col < width)){
        float Pvalue = 0;
        for(uint32_t k = 0; k < width; ++k){
            Pvalue += M[row*width + k] * N[k*width + col];
        }
        P[row*width + col] = Pvalue;
    }
}

__global__ void MatrixMulKernel(float* M, float * N, float *P, int32_t width){
    int32_t col = blockIdx.x * blockDim.x;// 遍历每一行的首元素；
    if(col < width){
        for(uint32_t i = 0; i < width; ++i){
            // 第一层for 遍历第col行的每一个元素
            float Pvalue = 0;
            for(uint32_t j = 0; j < width; ++j){
                // 第二层for 遍历计算每一行的第i 个元素的值；
                // M[col + j] 遍历第col行；
                // N[j*width + i] 遍历第i 列；
                Pvalue += M[col + j] * N[j*width + i];
            }
            // 将Pvalue 的值放到P[col + i]，也就是P[col][i]
            P[col + i] = Pvalue;
        }
    }
}

__global__ void MatrixMulKernel(float* M, float * N, float *P, int32_t width){
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;// 遍历N每一列的首元素；
    if(row < width){
        for(uint32_t i = 0; i < width; ++i){
            // 第一层for 遍历第row 列的每一个元素
            float Pvalue = 0;
            for(uint32_t j = 0; j < width; ++j){
                // 第二层for 遍历计算每一行的第i 个元素的值；
                for(uint32_t k = 0; k < width; ++k){
                    Pvalue += M[j*width + k] * N[k*width + row];
                }
            }
            // 将Pvalue 的值放到P[i*width + row]，也就是P[i*width][row]
            P[i*width + row] = Pvalue;
        }
    }
}