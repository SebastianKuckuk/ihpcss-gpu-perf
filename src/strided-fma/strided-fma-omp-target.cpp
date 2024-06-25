#include <chrono>

#include <omp.h>

#include "../util.h"
#include "strided-fma-util.h"


inline void fma(size_t nx, const tpe *__restrict__ src, tpe *__restrict__ dest, unsigned int stride) {
#pragma omp target teams distribute parallel for schedule(static)
    for (size_t i = 0; i < stride * nx; ++i) {
        auto acc = src[i / stride];

        if (0 == i % stride) {
            for (auto j = 0; j < numRepetitions; ++j)
                acc = (tpe)0.5 * acc + (tpe)1;
        }

        dest[i / stride] = acc;
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    unsigned int stride;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt, stride);

    auto src = new tpe[nx];
    auto dest = new tpe[nx];

    // init
    initFMA(src, nx);

#pragma omp target enter data map(to : src[0 : nx], dest[0 : nx])

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        fma(nx, src, dest, stride);
        std::swap(src, dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        fma(nx, src, dest, stride);
        std::swap(src, dest);
    }

    auto end = std::chrono::steady_clock::now();

#pragma omp target exit data map(from : src[0 : nx], dest[0 : nx])

    printStats(end - start, nx, nIt, 2 * sizeof(tpe), 2 * numRepetitions);

    // check solution
    checkSolutionFMA(src, nx, nIt + nItWarmUp);

    delete[] src;
    delete[] dest;

    return 0;
}
