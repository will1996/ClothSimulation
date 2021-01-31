#pragma once

#include "real.hpp"
#include "vectors.hpp"
#include <vector>

template<typename T>
class Spline {
public:
	// cubic Hermite spline with linear extrapolation
	struct Point { REAL t; T x, v; };
	std::vector<Point> points;
	T pos(REAL t) const;
	T vel(REAL t) const;
};

std::vector<REAL> operator+ (const std::vector<REAL> &x,
	const std::vector<REAL> &y);
std::vector<REAL> operator- (const std::vector<REAL> &x,
	const std::vector<REAL> &y);
std::vector<REAL> operator* (const std::vector<REAL> &x, REAL a);
std::vector<REAL> operator/ (const std::vector<REAL> &x, REAL a);

template <typename T> void fill_in_velocity(Spline<T> &s, int i) {
	if (i - 1 < 0 || i + 1 >= s.points.size())
		s.points[i].v = s.points[i].x * 0.;
	else
		s.points[i].v = (s.points[i + 1].x - s.points[i - 1].x)
		/ (s.points[i + 1].t - s.points[i - 1].t);
}
