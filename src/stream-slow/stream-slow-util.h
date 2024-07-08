#pragma once

#include <iostream>

typedef double tpe;

void initStream(tpe *vec, size_t nx) {
    for (size_t i = 0; i < nx; ++i)
        vec[i] = (tpe) i;
}

void checkSolutionStream(const tpe *const vec, size_t nx, size_t nIt) {
    for (size_t i = 0; i < nx; ++i)
        if ((tpe) (i + nIt) != vec[i]) {
            std::cerr << "Stream check failed for element " << i << " (expected " << i + nIt << " but got " << vec[i] << ")" << std::endl;
            return;
        }

    std::cout << "  Passed result check" << std::endl;
}
