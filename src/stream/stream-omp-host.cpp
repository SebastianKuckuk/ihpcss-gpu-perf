#include <chrono>

#include "../util.h"
#include "stream-util.h"


inline void stream(size_t nx, const tpe *__restrict__ src, tpe *__restrict__ dest) {
#pragma omp parallel for schedule (static)
    for (size_t i = 0; i < nx; ++i)
        dest[i] = src[i] + 1;
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    auto src = new tpe[nx];
    auto dest = new tpe[nx];

    // init
    initStream(src, nx);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream(nx, src, dest);
        std::swap(src, dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream(nx, src, dest);
        std::swap(src, dest);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nx, nIt, 2 * sizeof(tpe), 1);

    // check solution
    checkSolutionStream(src, nx, nIt + nItWarmUp);

    delete[] src;
    delete[] dest;

    return 0;
}
