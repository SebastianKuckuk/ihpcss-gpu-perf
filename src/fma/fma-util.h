#pragma once

#include <iostream>

typedef float tpe;

constexpr auto numRepetitions = 1024 * 1024;

void initFMA(tpe *vec, size_t nx) {
    for (size_t i = 0; i < nx; ++i)
        vec[i] = (tpe) 2;
}

void checkSolutionFMA(const tpe *const vec, size_t nx, size_t nIt) {
    for (size_t i = 0; i < nx; ++i)
        if ((tpe) 2 != vec[i]) {
            std::cerr << "FMA check failed for element " << i << " (expected " << 2 << " but got " << vec[i] << ")" << std::endl;
            return;
        }

    std::cout << "  Passed result check" << std::endl;
}
