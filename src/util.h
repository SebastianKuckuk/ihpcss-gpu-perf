#pragma once

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>


#ifdef __NVCC__
#   define FCT_DECORATOR __host__ __device__
#else
#   define FCT_DECORATOR
#endif


template<typename tpe>
void printStats(const std::chrono::duration<double> elapsedSeconds, size_t nIt, size_t nCells, char* tpeName, size_t numBytesPerCell, size_t numFlopsPerCell) {
    std::cout << "  #cells / #it:  " << nCells << " / " << nIt << "\n";
    std::cout << "  type:          " << tpeName << "\n";
    std::cout << "  elapsed time:  " << 1e3 * elapsedSeconds.count() << " ms\n";
    std::cout << "  per iteration: " << 1e3 * elapsedSeconds.count() / nIt << " ms\n";
    std::cout << "  MLUP/s:        " << 1e-6 * nCells * nIt / elapsedSeconds.count() << "\n";
    std::cout << "  bandwidth:     " << 1e-9 * numBytesPerCell * nCells * nIt / elapsedSeconds.count() << " GB/s\n";
    std::cout << "  compute:       " << 1e-9 * numFlopsPerCell * nCells * nIt / elapsedSeconds.count() << " GFLOP/s\n";
}

FCT_DECORATOR size_t ceilingDivide(size_t a, size_t b) {
    return (a + b - 1) / b;
}

FCT_DECORATOR size_t ceilToMultipleOf(size_t a, size_t b) {
    return ceilingDivide(a, b) * b;
}
