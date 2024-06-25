#include <chrono>

#include "../util.h"
#include "fma-util.h"


inline void fma(size_t nx, tpe *__restrict__ src, tpe *__restrict__ dest) {
#pragma omp parallel for schedule (static)
    for (size_t i = 0; i < nx; ++i) {
        auto acc = src[i];
        for (auto j = 0; j < numRepetitions; ++j)
            acc = (tpe)0.5 * acc + (tpe)1;
 
        dest[i] = acc;
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    auto src = new tpe[nx];
    auto dest = new tpe[nx];

    // init
    initFMA(src, nx);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        fma(nx, src, dest);
        std::swap(src, dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        fma(nx, src, dest);
        std::swap(src, dest);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nx, nIt, 2 * sizeof(tpe), 2 * numRepetitions);

    // check solution
    checkSolutionFMA(src, nx, nIt + nItWarmUp);

    delete[] src;
    delete[] dest;

    return 0;
}
