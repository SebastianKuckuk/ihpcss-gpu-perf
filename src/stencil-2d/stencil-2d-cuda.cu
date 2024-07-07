#include <chrono>
#include <iostream>

#include "../util.h"
#include "../cuda-util.h"
#include "stencil-2d-util.h"


__global__ void jacobi(size_t nx, size_t ny, const tpe *__restrict__ u, tpe *__restrict__ uNew) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || j < 1 || i >= nx - 1 || j >= ny - 1)
        return;

    uNew[j * nx + i] = 0.25 * (u[j * nx + i - 1] + u[j * nx + i + 1] + u[(j - 1) * nx + i] + u[(j + 1) * nx + i]);
}


int main(int argc, char *argv[]) {
    constexpr bool useCudaMallocHost = true;

    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    tpe *u, *uNew;
    if (useCudaMallocHost) {
        checkCudaError(cudaMallocHost((void **) &u, sizeof(tpe) * nx * ny));
        checkCudaError(cudaMallocHost((void **) &uNew, sizeof(tpe) * nx * ny));
    } else {
        u = new tpe[nx * ny];
        uNew = new tpe[nx * ny];
    }

    // init
    initStencil2D(u, nx, ny);
    initStencil2D(uNew, nx, ny);

    tpe *d_u, *d_uNew;
    checkCudaError(cudaMalloc((void **) &d_u, sizeof(tpe) * nx * ny));
    checkCudaError(cudaMalloc((void **) &d_uNew, sizeof(tpe) * nx * ny));

    checkCudaError(cudaMemcpy(d_u, u, sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_uNew, uNew, sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x), ceilingDivide(ny, blockSize.y));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        jacobi<<<numBlocks, blockSize>>>(nx, ny, d_u, d_uNew);
        std::swap(d_u, d_uNew);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        jacobi<<<numBlocks, blockSize>>>(nx, ny, d_u, d_uNew);
        std::swap(d_u, d_uNew);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nx * ny, nIt, 2 * sizeof(tpe), 4);

    checkCudaError(cudaMemcpy(u, d_u, sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(uNew, d_uNew, sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionStencil2D(u, nx, ny);

    checkCudaError(cudaFree(d_u));
    checkCudaError(cudaFree(d_uNew));

    if (useCudaMallocHost) {
        checkCudaError(cudaFreeHost(u));
        checkCudaError(cudaFreeHost(uNew));
    } else {
        delete[] u, uNew;
    }

    return 0;
}
