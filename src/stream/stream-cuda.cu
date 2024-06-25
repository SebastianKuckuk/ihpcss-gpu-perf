#include <chrono>
#include <iostream>

#include "../util.h"
#include "../cuda-util.h"
#include "stream-util.h"


__global__ void stream(size_t nx, const tpe *__restrict__ src, tpe *__restrict__ dest) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nx; i += gridDim.x * blockDim.x)
        dest[i] = src[i] + 1;
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    tpe *src, *dest;
    checkCudaError(cudaMallocHost((void **) &src, sizeof(tpe) * nx));
    checkCudaError(cudaMallocHost((void **) &dest, sizeof(tpe) * nx));

    // init
    initStream(src, nx);

    tpe *d_src, *d_dest;
    checkCudaError(cudaMalloc((void **) &d_src, sizeof(tpe) * nx));
    checkCudaError(cudaMalloc((void **) &d_dest, sizeof(tpe) * nx));

    checkCudaError(cudaMemcpy(d_src, src, sizeof(tpe) * nx, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_dest, dest, sizeof(tpe) * nx, cudaMemcpyHostToDevice));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t numSM = deviceProp.multiProcessorCount;

    dim3 blockSize(256);
    dim3 numBlocks(std::min(32 * numSM, ceilingDivide(nx, blockSize.x)));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream<<<numBlocks, blockSize>>>(nx, d_src, d_dest);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream<<<numBlocks, blockSize>>>(nx, d_src, d_dest);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nx, nIt, 2 * sizeof(tpe), 1);

    checkCudaError(cudaMemcpy(src, d_src, sizeof(tpe) * nx, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(dest, d_dest, sizeof(tpe) * nx, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionStream(src, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_src));
    checkCudaError(cudaFree(d_dest));

    checkCudaError(cudaFreeHost(src));
    checkCudaError(cudaFreeHost(dest));

    return 0;
}
