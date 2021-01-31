#pragma once

#include "mesh.hpp"

struct Morph {
	Mesh *mesh;
	std::vector<Mesh> targets;
	typedef std::vector<REAL> Weights;
	Spline<Weights> weights;
	Spline<REAL> log_stiffness;
	Vec3 pos(REAL t, const Vec2 &u) const;
};

void apply(const Morph &morph, REAL t);

