#pragma once

#if 0
inline vec3f norm(vec3f &p1, vec3f &p2, vec3f &p3)
{
	return (p2 - p1).cross(p3 - p1);
}

inline bool
check_abcd(vec3f &a0, vec3f &b0, vec3f &c0, vec3f &d0,
vec3f &a1, vec3f &b1, vec3f &c1, vec3f &d1)
{
	vec3f n0 = norm(a0, b0, c0);
	vec3f n1 = norm(a1, b1, c1);
	vec3f delta = norm(a1 - a0, b1 - b0, c1 - c0);
	vec3f nX = (n0 + n1 - delta)*0.5;

	vec3f pa0 = d0 - a0;
	vec3f pa1 = d1 - a1;

	float A = n0.dot(pa0);
	float B = n1.dot(pa1);
	float C = nX.dot(pa0);
	float D = nX.dot(pa1);
	float E = n1.dot(pa0);
	float F = n0.dot(pa1);

	if (A > 0 && B > 0 && (2 * C + F) > 0 && (2 * D + E) > 0)
		return false;

	if (A < 0 && B < 0 && (2 * C + F) < 0 && (2 * D + E) < 0)
		return false;

	return true;
}

bool
check_vf(unsigned int fid, unsigned int vid)
{
	unsigned v0 = _tris[fid].id0();
	unsigned v1 = _tris[fid].id1();
	unsigned v2 = _tris[fid].id2();

	vec3f &a0 = _prev_vtxs[v0];
	vec3f &b0 = _prev_vtxs[v1];
	vec3f &c0 = _prev_vtxs[v2];
	vec3f &p0 = _prev_vtxs[vid];

	vec3f &a1 = _cur_vtxs[v0];
	vec3f &b1 = _cur_vtxs[v1];
	vec3f &c1 = _cur_vtxs[v2];
	vec3f &p1 = _cur_vtxs[vid];

	return check_abcd(a0, b0, c0, p0, a1, b1, c1, p1);
}


bool
check_ee(unsigned int e1, unsigned int e2)
{
	unsigned v0 = _edges[e1].vid(0);
	unsigned v1 = _edges[e1].vid(1);
	unsigned w0 = _edges[e2].vid(0);
	unsigned w1 = _edges[e2].vid(1);

	vec3f &a0 = _prev_vtxs[v0];
	vec3f &b0 = _prev_vtxs[v1];
	vec3f &c0 = _prev_vtxs[w0];
	vec3f &d0 = _prev_vtxs[w1];

	vec3f &a1 = _cur_vtxs[v0];
	vec3f &b1 = _cur_vtxs[v1];
	vec3f &c1 = _cur_vtxs[w0];
	vec3f &d1 = _cur_vtxs[w1];

	return check_abcd(a0, b0, c0, d0, a1, b1, c1, d1);
}

#endif

#ifdef WIN32
inline __device__ REAL3 norm(REAL3 &p1, REAL3 &p2, REAL3 &p3)
{
	return cross(p2 - p1, p3 - p1);
}
#else
inline __device__ REAL3 norm(REAL3 p1, REAL3 p2, REAL3 p3)
{
	return cross(p2 - p1, p3 - p1);
}
#endif

inline __device__ bool
dnf_filter(REAL3 &a0, REAL3 &b0, REAL3 &c0, REAL3 &d0,
				REAL3 &a1, REAL3 &b1, REAL3 &c1, REAL3 &d1)
{
	REAL3 n0 = norm(a0, b0, c0);
	REAL3 n1 = norm(a1, b1, c1);
	REAL3 delta = norm(a1 - a0, b1 - b0, c1 - c0);
	REAL3 nX = (n0 + n1 - delta)*REAL(0.5);

	REAL3 pa0 = d0 - a0;
	REAL3 pa1 = d1 - a1;

	REAL A = dot(n0, pa0);
	REAL B = dot(n1, pa1);
	REAL C = dot(nX, pa0);
	REAL D = dot(nX, pa1);
	REAL E = dot(n1, pa0);
	REAL F = dot(n0, pa1);

	if (A > 0 && B > 0 && (REAL(2.0) * C + F) > 0 && (REAL(2.0) * D + E) > 0)
		return false;

	if (A < 0 && B < 0 && (REAL(2.0) * C + F) < 0 && (REAL(2.0) * D + E) < 0)
		return false;

	return true;
}
