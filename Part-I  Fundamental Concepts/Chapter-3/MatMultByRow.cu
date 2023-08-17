#include <cuda.h>
#include <iostream>

void MatrixMul(float* M, float * N, float *P, int32_t width){
    for(uint32_t i = 0; i < width; ++i){    // turn out to be row
        for(uint32_t j = 0; j < width; ++j){
            float Pvalue = 0.0F;
            for(uint32_t k = 0; k < width; ++k){
                Pvalue += M[i*width + k] * N[k*width + j];
            }
            P[i*width + j] = Pvalue;
        }
    }
}

__global__ void MatrixMulKernel(float* M, float * N, float *P, int32_t width){
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;    // 遍历N 的第一列的首元素；
    if(row < width){
        for(uint32_t i = 0; i < width; ++i){
            float Pvalue = 0.0F;
            for(uint32_t j = 0; j < width; ++j){
                Pvalue += M[row*width + j] * N[j*width + i];
            }
            P[row*width + i] = Pvalue;
        }
    }
}

void initialData(float * arr, uint32_t arr_size){
    // for(uint32_t i = 0; i < arr_size; i++){
    //     for(uint32_t j = 0; j < arr_size; j++){
    //         arr[j*arr_size + i] = i+1;
    //     }
    // }
    for(uint32_t i = 0; i < arr_size*arr_size; i++){
        arr[i] = i;
    }
}

int main(){
    uint32_t dev = 0;
    cudaSetDevice(dev);

    uint32_t nElem = 4;
    uint64_t nByte = sizeof(float) * nElem * nElem;
    float * M_h = static_cast<float *>(malloc(nByte));
    float * N_h = static_cast<float *>(malloc(nByte));
    float * P_h = static_cast<float *>(malloc(nByte));
    float * res_from_gpu_h = static_cast<float *>(malloc(nByte));

    initialData(M_h, nElem);
    initialData(N_h, nElem);
    memset(P_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    // for(uint32_t i = 0; i < nElem; i++){
    //     for(uint32_t j = 0; j < nElem; j++){
    //         std::cout<<M_h[i*nElem + j]<<" ";
    //     }
    //     printf("\n");
    // }

    float * M_d;
    float * N_d;
    float * P_d;
    cudaMalloc((float**)&M_d, nByte);
    cudaMalloc((float**)&N_d, nByte);
    cudaMalloc((float**)&P_d, nByte);

    cudaMemcpy(M_d, M_h, nByte, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, nByte, cudaMemcpyHostToDevice);

    dim3 block(nElem/4, nElem/4, 1);
    dim3 grid(4, 4, 1);
    MatrixMulKernel<<<grid, block>>>(M_d, N_d, P_d, nElem);

    cudaMemcpy(res_from_gpu_h, P_d, nByte, cudaMemcpyDeviceToHost);

    for(uint32_t i = 0; i < nElem; i++){
        for(uint32_t j = 0; j < nElem; j++){
            std::cout<<res_from_gpu_h[i*nElem + j]<<" ";
        }
        printf("\n");
    }
    MatrixMul(M_h, N_h, P_h, nElem);
    printf("The True Value is :\n");
    for(uint32_t i = 0; i < nElem; i++){
        for(uint32_t j = 0; j < nElem; j++){
            std::cout<<P_h[i*nElem + j]<<" ";
        }
        printf("\n");
    }

    return 0;
}
