#pragma once

#include "sparse.hpp"
#include "vectors.hpp"

std::vector<REAL> eigen_linear_solve (const SpMat<REAL> &A,
                                        const std::vector<REAL> &b);
