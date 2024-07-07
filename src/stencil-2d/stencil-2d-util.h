#pragma once

#include <iostream>
#include <cmath>

typedef double tpe;

void initStencil2D(tpe *u, size_t nx, size_t ny) {
    for (size_t j = 0; j < ny; ++j)
        for (size_t i = 0; i < nx; ++i)
            if (0 == i || 0 == j || nx - 1 == i || ny - 1 == j)
                u[j * nx + i] = 0;
            else
                u[j * nx + i] = 1;
}

void checkSolutionStencil2D(const tpe *const u, size_t nx, size_t ny) {
    tpe res = 0;
    for (size_t j = 1; j < ny - 1; ++j) {
        for (size_t i = 1; i < nx - 1; ++i) {
            const tpe localRes = 4 * u[j * nx + i] - (u[j * nx + i - 1] + u[j * nx + i + 1] + u[(j - 1) * nx + i] + u[(j + 1) * nx + i]);
            res += localRes * localRes;
        }
    }

    res = sqrt(res);

    std::cout << "  Final residual is " << res << std::endl;
}
