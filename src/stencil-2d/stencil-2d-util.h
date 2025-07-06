#include "../util.h"


template <typename tpe>
inline void initStencil2D(tpe *__restrict__ u, tpe *__restrict__ uNew, size_t nx, size_t ny) {
    for (size_t i1 = 0; i1 < ny; ++i1) {
        for (size_t i0 = 0; i0 < nx; ++i0) {
            if (0 == i0 || nx - 1 == i0 || 0 == i1 || ny - 1 == i1) {
                u[i0 + i1 * nx] = (tpe)0;
                uNew[i0 + i1 * nx] = (tpe)0;
            } else {
                u[i0 + i1 * nx] = (tpe)1;
                uNew[i0 + i1 * nx] = (tpe)1;
            }
        }
    }
}

template <typename tpe>
inline void checkSolutionStencil2D(const tpe *const __restrict__ u, const tpe *const __restrict__ uNew, size_t nx, size_t ny, size_t nIt) {
    tpe res = 0;
    for (size_t i1 = 1; i1 < ny - 1; ++i1) {
        for (size_t i0 = 1; i0 < nx - 1; ++i0) {
            const tpe localRes = -u[i0 + i1 * nx + 1] - u[i0 + i1 * nx - 1] + 4 * u[i0 + i1 * nx] - u[i0 + nx * (i1 + 1)] - u[i0 + nx * (i1 - 1)];
            res += localRes * localRes;
        }
    }

    res = sqrt(res);

    std::cout << "  Final residual is " << res << std::endl;
}

inline void parseCLA_2d(int argc, char **argv, char *&tpeName, size_t &nx, size_t &ny, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 4096;
    ny = 4096;

    nItWarmUp = 2;
    nIt = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i)
        tpeName = argv[i];
    ++i;
    if (argc > i)
        nx = atoi(argv[i]);
    ++i;
    if (argc > i)
        ny = atoi(argv[i]);
    ++i;

    if (argc > i)
        nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i)
        nIt = atoi(argv[i]);
    ++i;
}
