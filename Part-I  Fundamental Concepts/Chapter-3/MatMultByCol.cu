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
    int32_t col = blockIdx.y * blockDim.y + threadIdx.y;// 遍历每一行的首元素；
    if(col < width){
        for(uint32_t i = 0; i < width; ++i){
            // 第一层for 遍历第col行的每一个元素
            float Pvalue = 0.0F;
            for(uint32_t j = 0; j < width; ++j){
                // 第二层for 遍历计算每一行的第i 个元素的值；
                // M[col + j] 遍历第col行；
                // N[j*width + i] 遍历第i 列；
                Pvalue += M[i * width + j] * N[col + j * width];
            }
            // 将Pvalue 的值放到P[col + i]，也就是P[col][i]
            P[col + i * width] = Pvalue;
        }
    }
}

void initialData(float * arr, uint32_t arr_size){
    for(uint32_t i = 0; i < arr_size*arr_size; i++){
        arr[i] = i;
    }
}

void initialData2(float * arr, uint32_t arr_size){
    // for(uint32_t i = 0; i < arr_size; i++){
    //     for(uint32_t j = 0; j < arr_size; j++){
    //         arr[j*arr_size + i] = i+1;
    //     }
    // }
    for(uint32_t i = 0; i < arr_size*arr_size; i++){
        arr[i] = (arr_size*arr_size - i);
    }
}

int main(){
    uint32_t dev = 0;
    cudaSetDevice(dev);

    uint32_t nElem = 8;
    uint64_t nByte = sizeof(float) * nElem * nElem;
    float * M_h = static_cast<float *>(malloc(nByte));
    float * N_h = static_cast<float *>(malloc(nByte));
    float * P_h = static_cast<float *>(malloc(nByte));
    float * res_from_gpu_h = static_cast<float *>(malloc(nByte));

    initialData(M_h, nElem);
    initialData2(N_h, nElem);
    memset(P_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    for(uint32_t i = 0; i < nElem; i++){
        for(uint32_t j = 0; j < nElem; j++){
            std::cout<<M_h[i*nElem + j]<<" ";
        }
        printf("\n");
    }
    printf("--------------------------\n");
    for(uint32_t i = 0; i < nElem; i++){
        for(uint32_t j = 0; j < nElem; j++){
            std::cout<<N_h[i*nElem + j]<<" ";
        }
        printf("\n");
    }
    printf("--------------------------\n");
    float * M_d;
    float * N_d;
    float * P_d;
    cudaMalloc((float**)&M_d, nByte);
    cudaMalloc((float**)&N_d, nByte);
    cudaMalloc((float**)&P_d, nByte);

    cudaMemcpy(M_d, M_h, nByte, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, nByte, cudaMemcpyHostToDevice);

    dim3 block(nElem/4, nElem/2, 1);
    dim3 grid(4, 2, 1);
    MatrixMulKernel<<<grid, block>>>(M_d, N_d, P_d, nElem);

    cudaMemcpy(res_from_gpu_h, P_d, nByte, cudaMemcpyDeviceToHost);

    for(uint32_t i = 0; i < nElem; i++){
        for(uint32_t j = 0; j < nElem; j++){
            std::cout<<res_from_gpu_h[i*nElem + j]<<" ";
        }
        printf("\n");
    }
    printf("--------------------------\n");
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
