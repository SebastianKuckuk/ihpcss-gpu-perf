#pragma once

#include <cstdlib>
#include <iostream>
#include <chrono>


void parseCLA_1d(int argc, char *const *argv, size_t &nx, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 1024 * 1024;
    nItWarmUp = 2;
    nIt = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i) nx = atoi(argv[i]);
    ++i;
    if (argc > i) nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i) nIt = atoi(argv[i]);
    ++i;
}

void parseCLA_1d(int argc, char *const *argv, size_t &nx, size_t &nItWarmUp, size_t &nIt, unsigned int &stride) {
    // default values
    stride = 1;

    // override with command line arguments
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);
    int i = 4;
    if (argc > i) stride = atoi(argv[i]);
    ++i;
}

void parseCLA_2d(int argc, char *const *argv, size_t &nx, size_t &ny, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 1024;
    ny = nx;
    nItWarmUp = 2;
    nIt = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i) nx = atoi(argv[i]);
    ++i;
    if (argc > i) ny = atoi(argv[i]);
    ++i;
    if (argc > i) nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i) nIt = atoi(argv[i]);
    ++i;
}

void parseCLA_3d(int argc, char *const *argv, size_t &nx, size_t &ny, size_t &nz, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 128;
    ny = nx;
    nz = ny;
    nItWarmUp = 2;
    nIt = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i) nx = atoi(argv[i]);
    ++i;
    if (argc > i) ny = atoi(argv[i]);
    ++i;
    if (argc > i) nz = atoi(argv[i]);
    ++i;
    if (argc > i) nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i) nIt = atoi(argv[i]);
    ++i;
}


void printStats(const std::chrono::duration<double> elapsedSeconds, size_t nCells, size_t nIt, size_t numBytes, size_t numFlops) {
    std::cout << "  #cells / #it:  " << nCells << " / " << nIt << "\n";
    std::cout << "  elapsed time:  " << 1e3 * elapsedSeconds.count() << " ms\n";
    std::cout << "  per iteration: " << 1e3 * elapsedSeconds.count() / nIt << " ms\n";
    std::cout << "  MLUP/s:        " << 1e-6 * nCells * nIt / elapsedSeconds.count() << "\n";
    std::cout << "  bandwidth:     " << 1e-9 * numBytes * nCells * nIt / elapsedSeconds.count() << " GB/s\n";
    std::cout << "  compute:       " << 1e-9 * numFlops * nCells * nIt / elapsedSeconds.count() << " GFLOP/s\n";
}


#ifdef __NVCC__
#   define FCT_DECORATOR __host__ __device__
#else
#   define FCT_DECORATOR
#endif


FCT_DECORATOR size_t ceilingDivide(size_t a, size_t b) {
    return (a + b - 1) / b;
}


FCT_DECORATOR size_t ceilToMultipleOf(size_t a, size_t b) {
    return ceilingDivide(a, b) * b;
}
