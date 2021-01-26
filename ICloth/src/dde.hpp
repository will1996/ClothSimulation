#pragma once

#include "real.hpp"
#include "vectors.hpp"

typedef Vec<4> Vec4;
struct Edge;

struct StretchingData { Vec4 d[2][5]; };

struct StretchingSamples { Vec4 s[40][40][40]; };

struct BendingData { REAL d[3][5]; };

void evaluate_stretching_samples(StretchingSamples &, const StretchingData &data);
Vec4 stretching_stiffness(const Mat2x2 &G, const StretchingSamples &samples);
REAL bending_stiffness(const Edge *edge, int side, const BendingData &data, REAL initial_angle = 0);
