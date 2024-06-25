#include <chrono>
#include <iostream>

#include "../util.h"
#include "../cuda-util.h"
#include "strided-fma-util.h"


__global__ void fma(size_t nx, const tpe *__restrict__ src, tpe *__restrict__ dest, unsigned int stride) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < stride * nx; i += gridDim.x * blockDim.x) {
        if (0 == i % stride) {
            auto acc = src[i / stride];

            for (auto j = 0; j < numRepetitions; ++j)
                acc = (tpe)0.5 * acc + (tpe)1;

            dest[i / stride] = acc;
        }
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    unsigned int stride;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt, stride);

    tpe *src, *dest;
    checkCudaError(cudaMallocHost((void **) &src, sizeof(tpe) * nx));
    checkCudaError(cudaMallocHost((void **) &dest, sizeof(tpe) * nx));

    // init
    initFMA(src, nx);

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
        fma<<<numBlocks, blockSize>>>(nx, d_src, d_dest, stride);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        fma<<<numBlocks, blockSize>>>(nx, d_src, d_dest, stride);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nx, nIt, 2 * sizeof(tpe), 2 * numRepetitions);

    checkCudaError(cudaMemcpy(src, d_src, sizeof(tpe) * nx, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(dest, d_dest, sizeof(tpe) * nx, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionFMA(src, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_src));
    checkCudaError(cudaFree(d_dest));

    checkCudaError(cudaFreeHost(src));
    checkCudaError(cudaFreeHost(dest));

    return 0;
}
