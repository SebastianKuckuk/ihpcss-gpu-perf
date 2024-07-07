#include <chrono>

#include "../util.h"
#include "stencil-2d-util.h"

inline void jacobi(size_t nx, size_t ny, const tpe *__restrict__ u, tpe *__restrict__ uNew) {
#pragma omp parallel for schedule (static) //collapse(2)
    for (size_t j = 1; j < ny - 1; ++j) {
        for (size_t i = 1; i < nx - 1; ++i) {
            uNew[j * nx + i] = 0.25 * (u[j * nx + i - 1] + u[j * nx + i + 1] + u[(j - 1) * nx + i] + u[(j + 1) * nx + i]);
        }
    }
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    auto u = new tpe[nx * ny];
    auto uNew = new tpe[nx * ny];

    // init
    initStencil2D(u, nx, ny);
    initStencil2D(uNew, nx, ny);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        jacobi(nx, ny, u, uNew);
        std::swap(u, uNew);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        jacobi(nx, ny, u, uNew);
        std::swap(u, uNew);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nx * ny, nIt, 2 * sizeof(tpe), 4);

    // check solution
    checkSolutionStencil2D(u, nx, ny);

    delete[] u;
    delete[] uNew;

    return 0;
}
