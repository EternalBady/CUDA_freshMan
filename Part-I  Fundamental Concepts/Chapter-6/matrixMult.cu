#include <cuda.h>
#include <ctime>
#include <iostream>
#include <time.h>
#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif
#ifdef _WIN32
int gettimeofday(struct timeval *tp, void *tzp)
{
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;
  GetLocalTime(&wtm);
  tm.tm_year   = wtm.wYear - 1900;
  tm.tm_mon   = wtm.wMonth - 1;
  tm.tm_mday   = wtm.wDay;
  tm.tm_hour   = wtm.wHour;
  tm.tm_min   = wtm.wMinute;
  tm.tm_sec   = wtm.wSecond;
  tm. tm_isdst  = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;
  return (0);
}
#endif
double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

static const uint32_t TILE_SIZE = 1;

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

__global__ void MatrixMulKernelWithoutCT(float* M, float * N, float *P, int32_t width){
    __shared__ float M_t[TILE_SIZE][TILE_SIZE];
    __shared__ float N_t[TILE_SIZE][TILE_SIZE];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    uint32_t Row = by * TILE_SIZE + ty;
    uint32_t Col = bx * TILE_SIZE + tx;

    // Loop
    float Pvalue = 0.0F;
    for(uint32_t ph = 0; ph < width/TILE_SIZE; ++ph){

        //
        M_t[ty][tx] = M[Row*width + ph*TILE_SIZE + tx];
        N_t[ty][tx] = N[(ph*TILE_SIZE + ty)*width + Col];
        __syncthreads();

        for(uint32_t k = 0; k < TILE_SIZE; ++k){
            Pvalue += M_t[ty][k] * N_t[k][tx];
        }
        __syncthreads();
    }
    P[Row*width+Col] = Pvalue;
}

template<typename T>
__global__ void MatrixMulKernelWithCT(T* M, T * N, T *P, int32_t width){
    __shared__ T M_t[TILE_SIZE][TILE_SIZE];
    __shared__ T N_t[TILE_SIZE][TILE_SIZE];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    uint32_t Row = by * TILE_SIZE + ty;
    uint32_t Col = bx * TILE_SIZE + tx;

    // Loop
    T Pvalue = 0.0F;
    for(uint32_t ph = 0; ph < width/TILE_SIZE; ++ph){

        //
        M_t[ty][tx] = M[Row*width + ph*TILE_SIZE + tx];
        N_t[tx][ty] = N[(ph*TILE_SIZE + ty)*width + Col];
        __syncthreads();

        for(uint32_t k = 0; k < TILE_SIZE; ++k){
            Pvalue += M_t[ty][k] * N_t[tx][k];
        }
        __syncthreads();
    }
    P[Row*width+Col] = Pvalue;
}

void initialData(float * arr, uint32_t arr_size){
    for(uint32_t i = 0; i < arr_size*arr_size; i++){
        arr[i] = i;
    }
}

int main(int argc, char* argv[]){
    uint32_t dev = 0;
    cudaSetDevice(dev);

    uint32_t nElem = 2<<14;
    uint64_t nByte = sizeof(float) * nElem * nElem;
    float * M_h = static_cast<float *>(malloc(nByte));
    float * N_h = static_cast<float *>(malloc(nByte));
    float * P_h = static_cast<float *>(malloc(nByte));
    float * res_CT_h = static_cast<float *>(malloc(nByte));
    float * res_NCT_h = static_cast<float *>(malloc(nByte));

    initialData(M_h, nElem);
    initialData(N_h, nElem);
    memset(P_h, 0, nByte);
    memset(res_CT_h, 0, nByte);
    memset(res_NCT_h, 0, nByte);

    float * M_d;
    float * N_d;
    float * P_d;
    cudaMalloc((float**)&M_d, nByte);
    cudaMalloc((float**)&N_d, nByte);
    cudaMalloc((float**)&P_d, nByte);

    cudaMemcpy(M_d, M_h, nByte, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, nByte, cudaMemcpyHostToDevice);

    double iStart, iElaps;

    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((nElem-1)/block.x+1, (nElem-1)/block.y+1, 1);
    iStart = cpuSecond();
    MatrixMulKernelWithCT<float><<<grid, block>>>(M_d, N_d, P_d, nElem);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    std::cout<<"Kernel With corner turning Time cost = "<<iElaps<<std::endl;

    // cudaMemcpy(res_CT_h, P_d, nByte, cudaMemcpyDeviceToHost);

    iStart = cpuSecond();
    MatrixMulKernelWithoutCT<<<grid, block>>>(M_d, N_d, P_d, nElem);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    std::cout<<"Kernel Without corner turning Time cost = "<<iElaps<<std::endl;

    cudaMemcpy(res_NCT_h, P_d, nByte, cudaMemcpyDeviceToHost);

    // for(uint32_t i = 0; i < nElem; i++){
    //     for(uint32_t j = 0; j < nElem; j++){
    //         std::cout<<res_NCT_h[i*nElem + j]<<" ";
    //     }
    //     printf("\n");
    // }

    return 0;
}
