#include <cuda.h>
#include <iostream>

__global__ void MatMultVecKernel(float* B, float * C, float *A, int32_t width){
    int32_t col = threadIdx.x + blockIdx.x*blockDim.x;// 遍历每一行的首元素；
    if(col < width){
        float AValue = 0.0F;
        for(uint32_t i = 0; i < width; i++){
            AValue += B[col*width + i] * C[i];
        }
        A[col] = AValue;
    }
}

void initialMatData(float * arr, uint32_t arr_size){
    for(uint32_t i = 0; i < arr_size*arr_size; i++){
        arr[i] = i;
    }
}

void initialVecData(float * arr, uint32_t arr_size){
    for(uint32_t i = 0; i < arr_size; i++){
        arr[i] = arr_size - i;
    }
}

int main(){
    uint32_t dev = 0;
    cudaSetDevice(dev);

    uint32_t nElem = 8;
    uint64_t nByte = sizeof(float) * nElem;
    float * B_h = static_cast<float *>(malloc(nByte * nElem));
    float * C_h = static_cast<float *>(malloc(nByte));
    float * A_h = static_cast<float *>(malloc(nByte));
    float * res_from_gpu_h = static_cast<float *>(malloc(nByte));

    initialMatData(B_h, nElem);
    initialVecData(C_h, nElem);
    memset(A_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    for(uint32_t i = 0; i < nElem; i++){
        for(uint32_t j = 0; j < nElem; j++){
            std::cout<<B_h[i*nElem + j]<<" ";
        }
        printf("\n");
    }
    printf("--------------------------\n");
    for(uint32_t i = 0; i < nElem; i++){
        std::cout<<C_h[i]<<" ";
        printf("\n");
    }
    printf("--------------------------\n");

    float * B_d;
    float * C_d;
    float * A_d;
    cudaMalloc((float**)&B_d, nByte*nElem);
    cudaMalloc((float**)&C_d, nByte);
    cudaMalloc((float**)&A_d, nByte);

    cudaMemcpy(B_d, B_h, nByte*nElem, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, nByte, cudaMemcpyHostToDevice);

    dim3 block(2, 2, 1);
    dim3 grid((nElem-1)/block.x+1, (nElem-1)/block.y+1, 1);
    MatMultVecKernel<<<grid, block>>>(B_d, C_d, A_d, nElem);

    cudaMemcpy(res_from_gpu_h, A_d, nByte, cudaMemcpyDeviceToHost);

    for(uint32_t i = 0; i < nElem; i++){
        std::cout<<res_from_gpu_h[i]<<" ";
        printf("\n");
    }
    printf("--------------------------\n");

    return 0;
}
