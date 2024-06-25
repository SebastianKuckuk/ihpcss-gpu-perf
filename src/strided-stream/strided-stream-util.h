#pragma once

#include <iostream>

typedef double tpe;

void initStream(tpe *vec, size_t nx, unsigned int stride) {
    for (size_t i = 0; i < nx; ++i)
        vec[stride * i] = (tpe) i;
}

void checkSolutionStream(const tpe *const vec, size_t nx, size_t nIt, unsigned int stride) {
    for (size_t i = 0; i < nx; ++i)
        if ((tpe) (i + nIt) != vec[stride * i]) {
            std::cerr << "Stream check failed for element " << i << " (expected " << i + nIt << " but got " << vec[i] << ")" << std::endl;
            return;
        }

    std::cout << "  Passed result check" << std::endl;
}
