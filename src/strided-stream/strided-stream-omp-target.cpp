#include <chrono>

#include <omp.h>

#include "../util.h"
#include "strided-stream-util.h"


inline void stream(size_t nx, const tpe *__restrict__ src, tpe *__restrict__ dest, unsigned int stride) {
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < nx; ++i)
        dest[stride * i] = src[stride * i] + 1;
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    unsigned int stride;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt, stride);

    auto src = new tpe[stride * nx];
    auto dest = new tpe[stride * nx];

    // init
    initStream(src, nx, stride);

#pragma omp target enter data map(to : src[0 : stride * nx], dest[0 : stride * nx])

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream(nx, src, dest, stride);
        std::swap(src, dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream(nx, src, dest, stride);
        std::swap(src, dest);
    }

    auto end = std::chrono::steady_clock::now();

#pragma omp target exit data map(from : src[0 : stride * nx], dest[0 : stride * nx])

    printStats(end - start, nx, nIt, 2 * sizeof(tpe), 1);

    // check solution
    checkSolutionStream(src, nx, nIt + nItWarmUp, stride);

    delete[] src;
    delete[] dest;

    return 0;
}
