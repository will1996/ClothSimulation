#pragma once

#include <string> // aa: win
#include "mesh.hpp"
#include "spline.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#ifndef WIN32
#define FORCEINLINE __inline
#endif

#define EPSILON		1e-7f

// i+1 and i-1 modulo 3
// This way of computing it tends to be faster than using %
#define NEXT(i) ((i)<2 ? (i)+1 : (i)-2)
#define PREV(i) ((i)>0 ? (i)-1 : (i)+2)

typedef unsigned int uint;

struct Transformation;

// Easy reporting of vertices and faces

std::ostream &operator<< (std::ostream &out, const Vert *vert);
std::ostream &operator<< (std::ostream &out, const Node *node);
std::ostream &operator<< (std::ostream &out, const Edge *edge);
std::ostream &operator<< (std::ostream &out, const Face *face);

// Math utilities

extern const REAL infinity;


template <typename T> T interp(const T &a, const T &b, const T &w) {
	return a*(1 - w) + b*w;
}

template <typename T> T clamp(const T &x, const T &a, const T &b) {
	return std::min(std::max(x, a), b);
}

template <typename T> T min(const T &a, const T &b, const T &c) {
	return std::min(a, std::min(b, c));
}
template <typename T> T min(const T &a, const T &b, const T &c, const T &d) {
	return std::min(std::min(a, b), std::min(c, d));
}

template <typename T> T max(const T &a, const T &b, const T &c) {
	return std::max(a, std::max(b, c));
}
template <typename T> T max(const T &a, const T &b, const T &c, const T &d) {
	return std::max(std::max(a, b), std::max(c, d));
}

template <typename T> T sgn(const T &x) { return x<0 ? -1 : 1; }

int solve_quadratic(REAL a, REAL b, REAL c, REAL x[2]);

// Find, replace, and all that jazz

template <typename T> inline int find(const T *x, T* const *xs, int n = 3) {
	for (int i = 0; i < n; i++) if (xs[i] == x) return i; return -1;
}

template <typename T> inline int find(const T &x, const T *xs, int n = 3) {
	for (int i = 0; i < n; i++) if (xs[i] == x) return i; return -1;
}

template <typename T> inline int find(const T &x, const std::vector<T> &xs) {
	for (int i = 0; i < xs.size(); i++) if (xs[i] == x) return i; return -1;
}

template <typename T> inline bool is_in(const T *x, T* const *xs, int n = 3) {
	return find(x, xs, n) != -1;
}

template <typename T> inline bool is_in(const T &x, const T *xs, int n = 3) {
	return find(x, xs, n) != -1;
}

template <typename T> inline bool is_in(const T &x, const std::vector<T> &xs) {
	return find(x, xs) != -1;
}

template <typename T> inline void include(const T &x, std::vector<T> &xs) {
	if (!is_in(x, xs)) xs.push_back(x);
}

template <typename T> inline void remove(int i, std::vector<T> &xs) {
	xs[i] = xs.back(); xs.pop_back();
}

template <typename T> inline void exclude(const T &x, std::vector<T> &xs) {
	int i = find(x, xs); if (i != -1) remove(i, xs);
}

template <typename T> inline void replace(const T &v0, const T &v1, T vs[3]) {
	int i = find(v0, vs); if (i != -1) vs[i] = v1;
}

template <typename T>
inline void replace(const T &x0, const T &x1, std::vector<T> &xs) {
	int i = find(x0, xs); if (i != -1) xs[i] = x1;
}

template <typename T>
inline bool subset(const std::vector<T> &xs, const std::vector<T> &ys) {
	for (int i = 0; i < xs.size(); i++) if (!is_in(xs[i], ys)) return false;
	return true;
}

template <typename T>
inline void append(std::vector<T> &xs, const std::vector<T> &ys) {
	xs.insert(xs.end(), ys.begin(), ys.end());
}

// Mesh utilities

bool is_seam_or_boundary(const Vert *v);
bool is_seam_or_boundary(const Node *n);
bool is_seam_or_boundary(const Edge *e);
bool is_seam_or_boundary(const Face *f);

// Debugging

void debug_save_mesh(const Mesh &mesh, const std::string &name, int n = -1);
void debug_save_meshes(const std::vector<Mesh*> &meshes,
	const std::string &name, int n = -1);

template <typename T>
std::ostream &operator<< (std::ostream &out, const std::vector<T> &v) {
	out << "[";
	for (int i = 0; i < v.size(); i++)
		out << (i == 0 ? "" : ", ") << v[i];
	out << "]";
	return out;
}

#define ECHO(x) std::cout << #x << std::endl

#define REPORT(x) std::cout << #x << " = " << (x) << std::endl

#define REPORT_ARRAY(x,n) std::cout << #x << "[" << #n << "] = " << vector<REAL>(&(x)[0], &(x)[n]) << std::endl
