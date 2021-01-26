#pragma once

#include "vec3.cuh"
#include "tools.cuh"

#define EPS_N   REAL(0.000001)
#define HALF_PI REAL(0.5*M_PI)

typedef struct __align__(16) _cone3f{
	REAL3 _axis;
	REAL _angle;

	inline __host__ __device__ void empty()
	{
		_axis = zero3f();
		_angle = 0.0;
	}

	inline __host__ __device__ void set(const REAL3 &v)
	{
		_axis = v;
		_angle = EPS_N;
	}

	inline __host__ __device__ void set(const _cone3f &n)
	{
		_axis = n._axis;
		_angle = n._angle;
	}

	inline __host__ __device__ void set(const REAL3 &v1, const REAL3 &v2)
	{
		_axis = normalize(v1 + v2);
		_angle = acos(dot(v1, v2))*REAL(0.5);
	}

	inline __host__ __device__ bool full() const
	{
		return _angle >= HALF_PI;
	}

	inline __host__ __device__ void set_full()
	{
		_angle = HALF_PI + EPS_N;
	}

} g_cone;


inline __host__ __device__
void operator += (g_cone &a, const REAL3 &v)
{
	if (a.full()) return;

	REAL vdot = dot(v, a._axis);

	if (vdot < 0) {
		a.set_full();
		return;
	}

	REAL angle = acos(vdot);
	if (angle <= a._angle)
		return;

	a._axis = normalize(a._axis + v);
	a._angle += angle*REAL(0.5);
}

inline __host__ __device__
void operator+=(g_cone &a, g_cone &b)
{
	if (a.full()) return;

	if (b.full()) {
		a.set_full();
		return;
	}

	REAL vdot = dot(a._axis, b._axis);
	if (vdot < 0) {
		a.set_full();
		return;
	}

	REAL angle = acos(vdot);
	REAL diff_angle = fabs(a._angle - b._angle);
	if (angle <= diff_angle) {
		if (b._angle > a._angle)
			a.set(b);

		return;
	}

	a._axis = normalize(a._axis + b._axis);
	a._angle = angle*REAL(0.5) + fmaxf(a._angle, b._angle);
}

typedef struct __align__(16) _box3f {
	REAL3 _min, _max;

	inline __host__ __device__ void set(const REAL3 &a)
	{
		_min = _max = a;
	}

	inline __host__ __device__ void set(const REAL3 &a, const REAL3 &b)
	{
		_min = fminf(a, b);
		_max = fmaxf(a, b);
	}

	inline __host__ __device__  void set(const _box3f &a, const _box3f &b)
	{
		_min = fminf(a._min, b._min);
		_max = fmaxf(a._max, b._max);
	}

	inline __host__ __device__  void add(const REAL3 &a)
	{
		_min = fminf(_min, a);
		_max = fmaxf(_max, a);
	}

	inline __host__ __device__  void add(const _box3f &b)
	{
		_min = fminf(_min, b._min);
		_max = fmaxf(_max, b._max);
	}

	inline __host__ __device__  void enlarge(REAL thickness)
	{
		_min -= make_REAL3(thickness);
		_max += make_REAL3(thickness);
	}

	inline __host__ __device__ bool overlaps(const _box3f& b) const
	{
		if (_min.x > b._max.x) return false;
		if (_min.y > b._max.y) return false;
		if (_min.z > b._max.z) return false;

		if (_max.x < b._min.x) return false;
		if (_max.y < b._min.y) return false;
		if (_max.z < b._min.z) return false;

		return true;
	}
	
	inline __host__ __device__
		REAL3 maxV() const {
		return _max;
	}

	inline __host__ __device__
		REAL3 minV() const {
		return _min;
	}
} g_box;

