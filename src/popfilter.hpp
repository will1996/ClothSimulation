
#ifndef POPFILTER_HPP
#define POPFILTER_HPP

#include "cloth.hpp"
#include "constraint.hpp"

void apply_pop_filter (Cloth &cloth, const std::vector<Constraint*> &cons,
                       REAL regularization=REAL(1e3));

#endif
