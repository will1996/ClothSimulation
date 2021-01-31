#define REC_VF_EE_PAIRS
#define USE_DNF_FILTER
#define ONE_PASS_CD

#include "../src/timer.hpp"

// CUDA Runtime
#include <cuda_runtime.h>

#include <cuda_profiler_api.h>
#include <assert.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper for CUDA error checking

#include "vec3.cuh"
#include "mat3.cuh"
#include "tools.cuh"
#include "mat9.cuh"
#include "box.cuh"

#include <math.h>
#include <stdarg.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <string>
using namespace std;

#include "tri3f.cuh"

extern void cudaCSR2COO(int *rows, int n, int m, int *all_rows);
extern void input2(const char *fname, REAL *data, int len);
extern void output2(const char *fname, REAL *data, int len);
extern void output1(const char *fname, REAL *data, int len);
extern void output(char *fname, REAL *data, int len);
extern void output9(char *fname, REAL *data, int len);
extern void output9x9(char *fname, REAL *data, int len);
extern void output12(char *fname, REAL *data, int len);
extern void output12x12(char *fname, REAL *data, int len);

//========================================
extern string outprefix;

#define REAL_infinity 1.0e30

#include "constraint.cuh"
g_constraints Cstrs;
g_handles Hdls;
g_glues Glus;

#include "impact.cuh"
g_impacts Impcts;

__global__ void
kernel_str_apply_cnstrs(EqCon *hdls, REAL3 *dx, int num)
{
	LEN_CHK(num);
	EqCon &cstr = hdls[idx];

	dx[cstr.node] = cstr.x;
}

__global__ void
kernel_project_outside1(g_IneqCon *cons, REAL3 *dx, REAL *w, int num,
REAL *cm, REAL *om, REAL3 *cx, REAL3 *ox, REAL mrt, REAL mpt)
{
	LEN_CHK(num);

	g_IneqCon *c = cons + idx;
	if (false == c->_valid)
		return;

	MeshGrad dxc;
	c->project(dxc, cm, om, cx, ox, mrt, mpt);
	for (int i = 0; i<4; i++) {
		if (c->free[i]) {
			int id = c->nodes[i];

			REAL wn = norm2(dxc[i]);
			atomicAddD(w + id, wn);
			atomicAdd3(dx + id, wn*dxc[i]);
		}
	}
}

__global__ void
kernel_project_outside2(REAL3 *x, REAL3 *dx, REAL *w, int num)
{
	LEN_CHK(num);

	if (w[idx] == 0)
		return;

	x[idx] += dx[idx] / w[idx];
}

void init_constraints_gpu()
{
	Cstrs.init();
}

void init_impacts_gpu()
{
	Impcts.init();
}

void init_handles_gpu(int num)
{
	Hdls.init(num);
}

void init_glues_gpu(int num)
{
	Glus.init(num);
}

inline int getHandleNum()
{
	return Hdls.length();
}

inline EqCon *getHandles()
{
	return Hdls.data();
}

void check_eq_cstrs_gpu()
{
	Hdls.checkData();
}


void push_glue_gpu(int n0, int n1, REAL *x0, REAL *x1, REAL stiff, int id)
{
	g_GlueCon data(n0, n1, (REAL3 *)x0, (REAL3 *)x1, stiff);

	checkCudaErrors(cudaMemcpy(Glus.data() + id, &data, sizeof(g_GlueCon), cudaMemcpyHostToDevice));
	getLastCudaError("52");
}

void push_eq_cstr_gpu(int nid, REAL *x, REAL *n, REAL stiff, int id)
{
	EqCon data(nid, (REAL3 *)x, (REAL3 *)n, stiff);

	checkCudaErrors(cudaMemcpy(Hdls.data() + id, &data, sizeof(EqCon), cudaMemcpyHostToDevice));
	getLastCudaError("52");
}

inline int getGlueNum()
{
	return Glus.length();
}

inline g_GlueCon *getGlues()
{
	return Glus.data();
}

inline int getConstraintNum()
{
	return Cstrs.length();
}

inline g_IneqCon *getConstraints()
{
	return Cstrs.data();
}

struct StretchingSamples { REAL2x2 s[40][40][40]; };
struct BendingData { REAL d[3][5]; };

struct Wind {
	REAL density;
	REAL3 velocity;
	REAL drag;
};

struct Gravity {
	REAL3 _g;
};

StretchingSamples *dMaterialStretching;
BendingData *dMaterialBending;
Wind *dWind;
Gravity *dGravity;

// generic one
extern void gpuSolver(int max_iter, int cooN, int *cooRowInd, int *cooColInd, REAL *cooData, bool bsr,
	REAL *dSolverB, REAL *dSolverX, int vtxN, REAL *);

static const int nsamples = 30;

inline __host__  ostream &operator<< (ostream &out, const tri3f &tri)
{
	//	out << "f " << tri.id0() << " " << tri.id1() << " " << tri.id2() << endl;
	out << "f " << tri._ids.x + 1 << " " << tri._ids.y + 1 << " " << tri._ids.z + 1 << endl;
	return out;
}

inline __device__
int get_indices(int i, int j, int *inc)
{
	int beginIdx = (i == 0) ? 0 : inc[i - 1];
	return (beginIdx + j) * 9;
}

inline __device__
void get_indices(int i, int j, int loc[], int *inc)
{
	int beginIdx = (i == 0) ? 0 : inc[i - 1] * 9;
	int curLoc = j * 3;
	int lineOffset = (i == 0) ? inc[0] : inc[i] - inc[i - 1];
	lineOffset *= 3;

	loc[0] = beginIdx + curLoc;
	loc[1] = loc[0] + 1;
	loc[2] = loc[0] + 2;

	loc[3] = loc[0] + lineOffset;
	loc[4] = loc[3] + 1;
	loc[5] = loc[3] + 2;

	loc[6] = loc[3] + lineOffset;
	loc[7] = loc[6] + 1;
	loc[8] = loc[6] + 2;
}

__global__ void
kernel_generate_idx(int **matIdx, int *rowInc, int *rows, int *cols, int num)
{
	LEN_CHK(num);

	int *here = matIdx[idx];
	int len = (idx == 0) ? rowInc[0] : rowInc[idx] - rowInc[idx - 1];

	for (int l = 0; l<len; l++) {
		int locs[9];
		get_indices(idx, l, locs, rowInc);

		int s = 0, r = idx * 3, c = here[l] * 3;
		rows[locs[s]] = r;
		cols[locs[s++]] = c;
		rows[locs[s]] = r;
		cols[locs[s++]] = c + 1;
		rows[locs[s]] = r;
		cols[locs[s++]] = c + 2;

		rows[locs[s]] = r + 1;
		cols[locs[s++]] = c;
		rows[locs[s]] = r + 1;
		cols[locs[s++]] = c + 1;
		rows[locs[s]] = r + 1;
		cols[locs[s++]] = c + 2;

		rows[locs[s]] = r + 2;
		cols[locs[s++]] = c;
		rows[locs[s]] = r + 2;
		cols[locs[s++]] = c + 1;
		rows[locs[s]] = r + 2;
		cols[locs[s++]] = c + 2;
	}
}

__global__ void
kernel_compress_idx(int **matIdx, int *cItems, int *rowInc, int num)
{
	LEN_CHK(num);

	int loc = rowInc[idx];
	int *src = matIdx[idx];
	int len = (idx == 0) ? loc : loc - rowInc[idx - 1];
	int *dst = cItems + ((idx == 0) ? 0 : rowInc[idx - 1]);

	for (int i = 0; i<len; i++)
		dst[i] = src[i];
}

__global__ void
kernel_update_nodes(REAL3 *v, REAL3 *x, REAL3 *dv, REAL dt, bool update, int num)
{
	LEN_CHK(num);

	v[idx] += dv[idx];

	if (update)
		x[idx] += v[idx] * dt;
}


inline __device__ void
_sort(int *data, int num)
{
	for (int i = 0; i<num - 1; i++) {
		for (int j = num - 1; j>i; j--) {
			if (data[j] < data[j - 1]) {
				int tmp = data[j];
				data[j] = data[j - 1];
				data[j - 1] = tmp;
			}
		}
	}
}

inline __device__ int
_unique(int *data, int num)
{
	int loc = 0;
	for (int i = 1; i<num; i++) {
		if (data[i] == data[loc])
			continue;
		else
			data[++loc] = data[i];
	}

	return loc + 1;
}

inline __device__ int
gpu_sort(int *data, int num)
{
	if (num == 0) return 0;
	_sort(data, num);
	return _unique(data, num);
}

__global__ void
kernel_sort_idx(int **mIdx, int *col, int *inc, int num)
{
	LEN_CHK(num);

	int len = (idx == 0 ? col[0] : col[idx] - col[idx - 1]);
	inc[idx] = gpu_sort(mIdx[idx], len);
}

__global__ void
kernel_set_matIdx(int **mIdx, int *col, int *items, int num)
{
	LEN_CHK(num);

	mIdx[idx] = items + ((idx == 0) ? 0 : col[idx - 1]);
}

__device__ void mat_add(int i, int j, int *rowIdx, int **matIdx)
{
	int index = atomicAdd(&rowIdx[i], 1);
	matIdx[i][index] = j;
}


__device__ void mat_add_submat(int a, int b, int c, int *rowIdx, int **matIdx)
{
	mat_add(a, a, rowIdx, matIdx); mat_add(a, b, rowIdx, matIdx); 	mat_add(a, c, rowIdx, matIdx);
	mat_add(b, a, rowIdx, matIdx); 	mat_add(b, b, rowIdx, matIdx); 	mat_add(b, c, rowIdx, matIdx);
	mat_add(c, a, rowIdx, matIdx); 	mat_add(c, b, rowIdx, matIdx); 	mat_add(c, c, rowIdx, matIdx);
}

__device__ void mat_add_submat(int a, int b, int c, int d, int *rowIdx, int **matIdx)
{
	mat_add(a, a, rowIdx, matIdx); mat_add(a, b, rowIdx, matIdx);
	mat_add(a, c, rowIdx, matIdx); mat_add(a, d, rowIdx, matIdx);
	mat_add(b, a, rowIdx, matIdx); 	mat_add(b, b, rowIdx, matIdx);
	mat_add(b, c, rowIdx, matIdx); mat_add(b, d, rowIdx, matIdx);
	mat_add(c, a, rowIdx, matIdx); 	mat_add(c, b, rowIdx, matIdx);
	mat_add(c, c, rowIdx, matIdx); mat_add(c, d, rowIdx, matIdx);
	mat_add(d, a, rowIdx, matIdx); 	mat_add(d, b, rowIdx, matIdx);
	mat_add(d, c, rowIdx, matIdx); mat_add(d, d, rowIdx, matIdx);
}

__global__ void
kernel_add_face_forces(tri3f *faces, int *colLen, int *rowIdx, int **matIdx, int num, bool counting)
{
	LEN_CHK(num);

	tri3f *t = faces + idx;
	int a = t->id0();
	int b = t->id1();
	int c = t->id2();

	if (counting) {
		atomicAdd(&colLen[a], 3);
		atomicAdd(&colLen[b], 3);
		atomicAdd(&colLen[c], 3);
	}
	else
		mat_add_submat(a, b, c, rowIdx, matIdx);
}

__global__ void
kernel_add_constraint_forces(g_IneqCon *cstrs, int *colLen, int *rowIdx, int **matIdx, int num, bool counting)
{
	LEN_CHK(num);

	g_IneqCon *cstr = cstrs + idx;
	if (false == cstr->_valid)
		return;

	for (int i = 0; i<4; i++) {
		if (!cstr->free[i]) continue;
		int ni = cstr->nodes[i];

		for (int j = 0; j<4; j++) {
			if (!cstr->free[j]) continue;
			int nj = cstr->nodes[j];

			if (counting) {
				atomicAdd(&colLen[ni], 1);
			}
			else
				mat_add(ni, nj, rowIdx, matIdx);
		}
	}

}

inline __device__ int edge_opp_node(tri3f *faces, int ef, int en) {
	tri3f *f = faces + ef;

	if (f->id0() == en) return f->id2();
	if (f->id1() == en) return f->id0();
	if (f->id2() == en) return f->id1();

	assert(0);
	return -1;
}

inline __device__ int edge_vert(tri3f *v, tri3f *n, int en) {
	if (n->id0() == en) return v->id0();
	if (n->id1() == en) return v->id1();
	if (n->id2() == en) return v->id2();

	assert(0);
	return -1;
}

__global__ void
kernel_add_edge_forces(uint2 *ens, uint2 *efs, tri3f *faces, int *colLen, int *rowIdx, int **matIdx, int num, bool counting)
{
	LEN_CHK(num);

	uint2 en = ens[idx];
	uint2 ef = efs[idx];

	if (ef.x == -1 || ef.y == -1)
		return;

	int a = en.x;
	int b = en.y;
	int c = edge_opp_node(faces, ef.x, a);
	int d = edge_opp_node(faces, ef.y, b);

	if (counting) {
		atomicAdd(&colLen[a], 4);
		atomicAdd(&colLen[b], 4);
		atomicAdd(&colLen[c], 4);
		atomicAdd(&colLen[d], 4);
	}
	else
		mat_add_submat(a, b, c, d, rowIdx, matIdx);
}

inline __device__ int
find_location(int **matIdx, int *rowInc, int i, int v)
{
	int len = (i == 0) ? rowInc[0] : rowInc[i] - rowInc[i - 1];

	for (int l = 0; l<len; l++)
		if (matIdx[i][l] == v)
			return l;

	assert(0);
	return -1;
}

inline __device__
void add_val(int loc, REAL *vals, const REAL3x3 &v)
{
	atomicAddD(vals + loc + 0, getIJ(v, 0, 0));
	atomicAddD(vals + loc + 1, getIJ(v, 0, 1));
	atomicAddD(vals + loc + 2, getIJ(v, 0, 2));
	atomicAddD(vals + loc + 3, getIJ(v, 1, 0));
	atomicAddD(vals + loc + 4, getIJ(v, 1, 1));
	atomicAddD(vals + loc + 5, getIJ(v, 1, 2));
	atomicAddD(vals + loc + 6, getIJ(v, 2, 0));
	atomicAddD(vals + loc + 7, getIJ(v, 2, 1));
	atomicAddD(vals + loc + 8, getIJ(v, 2, 2));
}

inline __device__
void add_val(int *locs, REAL *vals, const REAL3x3 &v)
{
	atomicAddD(vals + locs[0], getIJ(v, 0, 0));
	atomicAddD(vals + locs[1], getIJ(v, 0, 1));
	atomicAddD(vals + locs[2], getIJ(v, 0, 2));
	atomicAddD(vals + locs[3], getIJ(v, 1, 0));
	atomicAddD(vals + locs[4], getIJ(v, 1, 1));
	atomicAddD(vals + locs[5], getIJ(v, 1, 2));
	atomicAddD(vals + locs[6], getIJ(v, 2, 0));
	atomicAddD(vals + locs[7], getIJ(v, 2, 1));
	atomicAddD(vals + locs[8], getIJ(v, 2, 2));
}

inline __device__ REAL3
subvec3(const REAL3x3 &b, int i)
{
	if (i == 0)
		return b.c0();
	if (i == 1)
		return b.c1();
	if (i == 2)
		return b.c2();

	assert(0);
	return make_REAL3(0.0);
}

inline __device__ REAL3
subvec4(const REAL12 &b, int i)
{
	return b.m(i);
}

inline __device__ REAL3x3
submat3(const REAL9x9 &A, int i, int j)
{
	int r = i * 3;
	int c = j * 3;

	return make_REAL3x3(
		make_REAL3(A.getIJ(r, c), A.getIJ(r + 1, c), A.getIJ(r+2, c)), 
		make_REAL3(A.getIJ(r, c+1), A.getIJ(r + 1, c+1), A.getIJ(r + 2, c+1)),
		make_REAL3(A.getIJ(r, c+2), A.getIJ(r + 1, c+2), A.getIJ(r + 2, c+2)));
}

inline __device__ REAL3x3
submat4(const REAL12x12 &A, int i, int j)
{
	int r = i * 3;
	int c = j * 3;

	return make_REAL3x3(
		make_REAL3(A.getIJ(r, c), A.getIJ(r + 1, c), A.getIJ(r + 2, c)),
		make_REAL3(A.getIJ(r, c + 1), A.getIJ(r + 1, c + 1), A.getIJ(r + 2, c + 1)),
		make_REAL3(A.getIJ(r, c + 2), A.getIJ(r + 1, c + 2), A.getIJ(r + 2, c + 2)));
}

inline __device__ void
find_submat3(tri3f *f, int *rowInc, int **matIdx, bool bsr, int *locs)
{
	int idx = 0;
	for (int i = 0; i<3; i++)
		for (int j = 0; j<3; j++) {
			int ii = f->id(i);
			int jj = find_location(matIdx, rowInc, ii, f->id(j));

			if (bsr) {
				locs[idx++] = get_indices(ii, jj, rowInc);
			}
			else {
				assert(0);
			}
		}
}


inline __device__ void
update_submat3(const REAL9x9 &asub, REAL *vals, bool bsr, int *locs)
{
	assert(bsr);

	int idx = 0;
	for (int i = 0; i<3; i++)
		for (int j = 0; j<3; j++) {
			int loc = locs[idx++];
			add_val(loc, vals, submat3(asub, i, j));
		}
}

inline __device__ void
add_submat3(const REAL9x9 &asub, tri3f *f, REAL *vals, int *rowInc, int **matIdx, bool bsr)
{
	int locs[9];

	for (int i = 0; i<3; i++)
		for (int j = 0; j<3; j++) {
			int ii = f->id(i);
			int jj = find_location(matIdx, rowInc, ii, f->id(j));

			if (bsr) {
				int loc = get_indices(ii, jj, rowInc);
				add_val(loc, vals, submat3(asub, i, j));
			}
			else {
				get_indices(ii, jj, locs, rowInc);
				add_val(locs, vals, submat3(asub, i, j));
			}
		}
}

inline __device__ uint
get(uint4 &f, int i)
{
	if (i == 0) return f.x;
	else if (i == 1) return f.y;
	else if (i == 2) return f.z;
	else return f.w;
}

inline __device__ void
find_submat4(uint4 &f, int *rowInc, int **matIdx, bool bsr, int *locs)
{
	int idx = 0;
	for (int i = 0; i<4; i++)
		for (int j = 0; j<4; j++) {
			int ii = get(f, i);
			int jj = find_location(matIdx, rowInc, ii, get(f, j));

			if (bsr) {
				locs[idx++] = get_indices(ii, jj, rowInc);
			}
			else {
				assert(0);
			}
		}
}

inline __device__ void
update_submat4(REAL12x12 asub, REAL *vals, bool bsr, int *locs)
{
	assert(bsr);

	int idx = 0;
	for (int i = 0; i<4; i++)
		for (int j = 0; j<4; j++) {
			int loc = locs[idx++];
			add_val(loc, vals, submat4(asub, i, j));
		}
}

inline __device__ void
add_submat4(REAL12x12 asub, uint4 &f, REAL *vals, int *rowInc, int **matIdx, bool bsr)
{
	int locs[9];

	for (int i = 0; i<4; i++)
		for (int j = 0; j<4; j++) {
			int ii = get(f, i);
			int jj = find_location(matIdx, rowInc, ii, get(f, j));

			if (bsr) {
				int loc = get_indices(ii, jj, rowInc);
				add_val(loc, vals, submat4(asub, i, j));
			}
			else {
				get_indices(ii, jj, locs, rowInc);
				add_val(locs, vals, submat4(asub, i, j));
			}
		}
}

inline __device__ void
add_subvec3(const REAL3x3 &bsub, tri3f *f, REAL3 *b)
{
	for (int i = 0; i < 3; i++) {
		atomicAdd3(b + f->id(i), subvec3(bsub, i));
	}
}

inline __device__ void
add_subvec4(const REAL12 &bsub, uint4 &f, REAL3 *b)
{
	for (int i = 0; i<4; i++) {
		REAL3 d = subvec4(bsub, i);
		int id = get(f, i);

		atomicAdd3(b + id, d);
	}
}

inline __device__ REAL2
derivative(REAL a0, REAL a1, REAL a2, const REAL2x2 &idm) {
	return getTrans(idm) * make_REAL2(a1 - a0, a2 - a0);
}

inline __device__ REAL3x2
derivative(REAL3 &w0, REAL3 &w1, REAL3 &w2, const REAL2x2 &idm) {
	return make_REAL3x2(w1 - w0, w2 - w0) * idm;
}

inline __device__ REAL2x3
derivative(const REAL2x2 &idm) {
	return getTrans(idm)*make_REAL2x3(
		make_REAL2(-1, -1),
		make_REAL2(1, 0),
		make_REAL2(0, 1));
}

inline __device__ REAL2x2
stretching_stiffness(const REAL2x2 &G, const StretchingSamples &samples) {
	REAL a = (getIJ(G, 0, 0) + 0.25)*nsamples;
	REAL b = (getIJ(G, 1, 1) + 0.25)*nsamples;
	REAL c = fabsf(getIJ(G, 0, 1))*nsamples;
	a = clamp(a, 0.0, nsamples - 1 - 1e-5);
	b = clamp(b, 0.0, nsamples - 1 - 1e-5);
	c = clamp(c, 0.0, nsamples - 1 - 1e-5);
	int ai = (int)floor(a);
	int bi = (int)floor(b);
	int ci = (int)floor(c);
	if (ai<0)        ai = 0;
	if (bi<0)        bi = 0;
	if (ci<0)        ci = 0;
	if (ai>nsamples - 2)        ai = nsamples - 2;
	if (bi>nsamples - 2)        bi = nsamples - 2;
	if (ci>nsamples - 2)        ci = nsamples - 2;
	a = a - ai;
	b = b - bi;
	c = c - ci;
	REAL weight[2][2][2];
	weight[0][0][0] = (1 - a)*(1 - b)*(1 - c);
	weight[0][0][1] = (1 - a)*(1 - b)*(c);
	weight[0][1][0] = (1 - a)*(b)*(1 - c);
	weight[0][1][1] = (1 - a)*(b)*(c);
	weight[1][0][0] = (a)*(1 - b)*(1 - c);
	weight[1][0][1] = (a)*(1 - b)*(c);
	weight[1][1][0] = (a)*(b)*(1 - c);
	weight[1][1][1] = (a)*(b)*(c);
	REAL2x2 stiffness = zero2x2();
	for (int i = 0; i<2; i++)
		for (int j = 0; j<2; j++)
			for (int k = 0; k<2; k++)
				for (int l = 0; l<4; l++)
				{
					getI(stiffness, l) += getI(samples.s[ai + i][bi + j][ci + k], l)*weight[i][j][k];
				}
	return stiffness;
}

inline __device__ REAL3x9
kronecker(const REAL3 &A, const REAL3x3 &B)
{
	REAL3x9 t;

	for (int i = 0; i < 9; i++)
		t.c[i] = A.x * B.c[i];
	for (int i = 0; i < 9; i++)
		t.c[i + 9] = A.y*B.c[i];
	for (int i = 0; i < 9; i++)
		t.c[i + 18] = A.z * B.c[i];
	return t;
}

inline __device__ REAL
unwrap_angle(REAL theta, REAL theta_ref) {
	if (theta - theta_ref > M_PI)
		theta -= 2 * M_PI;
	if (theta - theta_ref < -M_PI)
		theta += 2 * M_PI;
	return theta;
}

inline __device__ REAL
dihedral_angle(uint4 en, uint2 ef, REAL3 *x, REAL3 *n, REAL ref)
{
	if (ef.x == -1 || ef.y == -1)
		return 0.0;

	REAL3 e = normalize(x[en.x] - x[en.y]);
	if (norm2(e) == 0) return 0.0;

	REAL3 n0 = n[ef.x], n1 = n[ef.y];
	if (norm2(n0) == 0 || norm2(n1) == 0) return 0.0;

	REAL cosine = dot(n0, n1), sine = dot(e, cross(n0, n1));
	REAL theta = atan2(sine, cosine);
	return unwrap_angle(theta, ref);
}

inline __device__ REAL
bending_stiffness(int side, const BendingData &data,
uint2 ef, uint4 en, tri3f *vrts,
REAL eTheta, REAL eLen, REAL A12,
REAL2 *vU, tri3f *nods,
REAL initial_angle)
{
	REAL curv = eTheta * eLen / A12;
	REAL alpha = curv / 2;
	REAL value = alpha*0.2;
	if (value>4) value = 4;
	int  value_i = (int)value;
	if (value_i<0)   value_i = 0;
	if (value_i>3)   value_i = 3;
	value -= value_i;

	int vid1 = edge_vert(vrts + (side == 0 ? ef.x : ef.y), nods + (side == 0 ? ef.x : ef.y), en.y);
	int vid2 = edge_vert(vrts + (side == 0 ? ef.x : ef.y), nods + (side == 0 ? ef.x : ef.y), en.x);
	REAL2 du = vU[vid1] - vU[vid2];

	REAL    bias_angle = (atan2f(du.y, du.x) + initial_angle) * 4 / M_PI;
	if (bias_angle<0)        bias_angle = -bias_angle;
	if (bias_angle>4)        bias_angle = 8 - bias_angle;
	if (bias_angle>2)        bias_angle = 4 - bias_angle;
	int             bias_id = (int)bias_angle;
	if (bias_id<0)   bias_id = 0;
	if (bias_id>1)   bias_id = 1;
	bias_angle -= bias_id;
	REAL actual_ke = data.d[bias_id][value_i] * (1 - bias_angle)*(1 - value)
		+ data.d[bias_id + 1][value_i] * (bias_angle)*(1 - value)
		+ data.d[bias_id][value_i + 1] * (1 - bias_angle)*(value)
		+data.d[bias_id + 1][value_i + 1] * (bias_angle)*(value);
	if (actual_ke<0) actual_ke = 0;
	return actual_ke;
}

inline __device__ void
bending_force(uint4 en, uint2 ef, REAL el, REAL et, REAL eti, REAL eref,
tri3f *vrts, REAL *fa, REAL3 *x, REAL3 *fn, REAL2x2 *idms, REAL2 *vu, tri3f *nods,
REAL12x12 &oJ, REAL12 &oF, BendingData *bd, int *mtrIdx)
{
	uint f0 = ef.x;
	uint f1 = ef.y;

	int m0 = mtrIdx[f0];
	int m1 = mtrIdx[f1];

	REAL theta = dihedral_angle(en, ef, x, fn, eref);
	REAL a = fa[f0] + fa[f1];
	REAL3 x0 = x[en.x], x1 = x[en.y], x2 = x[en.z], x3 = x[en.w];

	REAL h0 = distance(x2, x0, x1), h1 = distance(x3, x0, x1);

	REAL3 n0 = fn[ef.x], n1 = fn[ef.y];

	REAL2	w_f0 = barycentric_weights(x2, x0, x1),
		w_f1 = barycentric_weights(x3, x0, x1);

	REAL12 dtheta = make_REAL3x4(
		-(w_f0.x*n0 / h0 + w_f1.x*n1 / h1),
		-(w_f0.y*n0 / h0 + w_f1.y*n1 / h1),
		n0 / h0,
		n1 / h1);

	REAL d1 = bending_stiffness(0, bd[m0], ef, en, vrts, et, el, a, vu, nods, 0.0);
	REAL d2 = bending_stiffness(1, bd[m1], ef, en, vrts, et, el, a, vu, nods, 0.0);
	REAL ke = std::fminf(d1, d2);

	REAL shape = (el*el) / (2 * a);

	oJ = -ke*shape*outer(dtheta, dtheta)*0.5;
	oF = -ke*shape*(theta - eti)*dtheta*0.5;
}

inline __device__ void
stretching_force(
const tri3f &t, REAL fa, REAL3 *x, const REAL2x2 &idm,
REAL9x9 &oJ, REAL3x3 &oF, StretchingSamples *ss, REAL weakening, REAL damage)
{
	REAL3x2 F = derivative(x[t.id0()], x[t.id1()], x[t.id2()], idm);

	REAL2x2 G = (getTrans(F)*F - make_REAL2x2(1.0))*0.5;
	REAL2x2 k = stretching_stiffness(G, *ss);
	k *= 1 / (1 + weakening*damage);

	REAL2x3 D = derivative(idm);
	REAL3 du = getRow(D, 0), dv = getRow(D, 1);

	REAL3x3 I = identity3x3();
	REAL3x9 Du = kronecker(du, I);
	REAL3x9 Dv = kronecker(dv, I);

	const REAL3 &xu = F.c0();
	const REAL3 &xv = F.c1();

	REAL9 fuu = getTrans(Du)*xu;
	REAL9 fvv = getTrans(Dv)*xv;
	REAL9 fuv = (getTrans(Du)*xv + getTrans(Dv)*xu)*0.5;

	REAL9 grad_e =
		getIJ(k, 0, 0)*getIJ(G, 0, 0)*fuu +
		getIJ(k, 0, 1)*getIJ(G, 1, 1)*fvv +
		getIJ(k, 1, 0)*(getIJ(G, 0, 0)*fvv + getIJ(G, 1, 1)*fuu) +
		2.0*getIJ(k, 1, 1)*getIJ(G, 0, 1)*fuv;

	REAL9x9 hess_e =
		getIJ(k, 0, 0)*(outer(fuu, fuu) + max(getIJ(G, 0, 0), 0.)*(getTrans(Du)*Du)) +
		getIJ(k, 0, 1)*(outer(fvv, fvv) + max(getIJ(G, 1, 1), 0.)*(getTrans(Dv)*Dv)) +
		getIJ(k, 1, 0)*(outer(fuu, fvv) + max(getIJ(G, 0, 0), 0.)*(getTrans(Dv)*Dv)
		+ outer(fvv, fuu) + max(getIJ(G, 1, 1), 0.)*(getTrans(Du)*Du)) +
		2.0*getIJ(k, 1, 1)*outer(fuv, fuv);

	oJ = -fa*hess_e;
	oF = -fa*grad_e;
}

__global__ void
kernel_find_face_forces(tri3f *faces, bool bsr, int num,
		int **matIdx, int *rowInc, int *facLocs)
{
	LEN_CHK(num);

	tri3f *f = faces + idx;
	int *locs = facLocs + 9 * idx;

	find_submat3(f, rowInc, matIdx, bsr, locs);
}

__global__ void
kernel_update_face_forces(
tri3f *faces, REAL3 *v, REAL *fa, REAL3 *x, REAL2x2 *idms,
REAL dt, REAL *vals, bool bsr, int num,
int **matIdx, int *rowInc, REAL3 *b, REAL *m,
REAL3 *Fext, REAL3x3 *Jext, StretchingSamples *ss,
REAL9 *oF, REAL9x9 *oJ, int *facLocs, REAL damping, REAL weakening, REAL damage)
{
	LEN_CHK(num);

	tri3f *f = faces + idx;
	int *locs = facLocs + 9 * idx;

	REAL9 vs = make_REAL3x3(v[f->id0()], v[f->id1()], v[f->id2()]);

	REAL9x9 J;
	REAL3x3 F;
	stretching_force(faces[idx], fa[idx], x, idms[idx], J, F, ss, weakening, damage);

	// for debug
	if (oF != NULL) oF[idx] = F;
	if (oJ != NULL) oJ[idx] = J;

	if (dt == 0) {
		update_submat3(-J, vals, bsr, locs);
		add_subvec3(F, f, b);
	}
	else {
		update_submat3(-dt*(dt + damping)*J, vals, bsr, locs);
		add_subvec3(dt*(F + (dt + damping)*J*vs), f, b);
	}
}


__global__ void
kernel_internal_face_forces(
tri3f *faces, REAL3 *v, REAL *fa, REAL3 *x, REAL2x2 *idms,
REAL dt, REAL *vals, bool bsr, int num,
int **matIdx, int *rowInc, REAL3 *b, REAL *m,
REAL3 *Fext, REAL3x3 *Jext, StretchingSamples *ss, int *mtrIdx,
REAL9 *oF, REAL9x9 *oJ, REAL damping, REAL weakening, REAL damage)
{
	LEN_CHK(num);

	tri3f *f = faces + idx;
	int mIdx = mtrIdx[idx];
	REAL9 vs = make_REAL3x3(v[f->id0()], v[f->id1()], v[f->id2()]);

	REAL9x9 J;
	REAL3x3 F;
	stretching_force(faces[idx], fa[idx], x, idms[idx], J, F, ss+mIdx, weakening, damage);

	// for debug
	if (oF != NULL) oF[idx] = F;
	if (oJ != NULL) oJ[idx] = J;

	if (dt == 0) {
		add_submat3(-J, f, vals, rowInc, matIdx, bsr);
		add_subvec3(F, f, b);
	}
	else {
		add_submat3(-dt*(dt + damping)*J, f, vals, rowInc, matIdx, bsr);
		add_subvec3(dt*(F + (dt + damping)*J*vs), f, b);
	}
}

__global__ void
kernel_find_edge_forces(
uint2 *ens, uint2 *efs, tri3f *nods, bool bsr, int num,
int **matIdx, int *rowInc, int *edgLocs)
{
	LEN_CHK(num);

	uint2 en = ens[idx];
	uint2 ef = efs[idx];

	if (ef.x == -1 || ef.y == -1)
		return;

	int aa = en.x;
	int bb = en.y;
	int cc = edge_opp_node(nods, ef.x, aa);
	int dd = edge_opp_node(nods, ef.y, bb);
	uint4 em = make_uint4(aa, bb, cc, dd);

	int *locs = edgLocs + idx * 16;

	find_submat4(em, rowInc, matIdx, bsr, locs);
}

__global__ void
kernel_update_edge_forces(
uint2 *ens, uint2 *efs, tri3f *nods, REAL3 *v,
REAL *els, REAL *ets, REAL *etis, REAL *eref,
REAL *fas, REAL3 *fns, REAL2x2 *idms, REAL2 *vus, REAL3 *x, tri3f *vrts,
REAL dt, REAL *vals, bool bsr, int num,
int **matIdx, int *rowInc, REAL3 *b, REAL *m, REAL3 *Fext, REAL3x3 *Jext, BendingData *bd, int *mtrIdx,
REAL12 *oF, REAL12x12 *oJ, int *edgLocs, REAL damping)
{
	LEN_CHK(num);

	uint2 en = ens[idx];
	uint2 ef = efs[idx];

	if (ef.x == -1 || ef.y == -1)
		return;

	int *locs = edgLocs + 16 * idx;
	int aa = en.x;
	int bb = en.y;
	int cc = edge_opp_node(nods, ef.x, aa);
	int dd = edge_opp_node(nods, ef.y, bb);
	uint4 em = make_uint4(aa, bb, cc, dd);
	REAL12 vs = make_REAL3x4(v[aa], v[bb], v[cc], v[dd]);

	REAL12x12 J;
	REAL12 F;
	bending_force(em, ef, els[idx], ets[idx], etis[idx], eref[idx],
		vrts, fas, x, fns, idms, vus, nods,
		J, F, bd, mtrIdx);

	// for debug
	if (oF != NULL) oF[idx] = F;
	if (oJ != NULL) oJ[idx] = J;

	if (dt == 0) {
		update_submat4(-J, vals, bsr, locs);
		add_subvec4(F, em, b);
	}
	else {
		update_submat4(-dt*(dt + damping)*J, vals, bsr, locs);

		REAL12 t = dt*(F + (dt + damping)*J*vs);
		add_subvec4(t, em, b);
	}
}

__global__ void
kernel_internal_edge_forces(
uint2 *ens, uint2 *efs, tri3f *nods, REAL3 *v,
REAL *els, REAL *ets, REAL *etis, REAL *eref,
REAL *fas, REAL3 *fns, REAL2x2 *idms, REAL2 *vus, REAL3 *x, tri3f *vrts,
REAL dt, REAL *vals, bool bsr, int num,
int **matIdx, int *rowInc, REAL3 *b, REAL *m, REAL3 *Fext, REAL3x3 *Jext, BendingData *bd, int *mtrIdx,
REAL12 *oF, REAL12x12 *oJ, REAL damping)
{
	LEN_CHK(num);

	uint2 en = ens[idx];
	uint2 ef = efs[idx];

	if (ef.x == -1 || ef.y == -1)
		return;

	int aa = en.x;
	int bb = en.y;
	int cc = edge_opp_node(nods, ef.x, aa);
	int dd = edge_opp_node(nods, ef.y, bb);
	uint4 em = make_uint4(aa, bb, cc, dd);
	REAL12 vs = make_REAL3x4(v[aa], v[bb], v[cc], v[dd]);

	REAL12x12 J;
	REAL12 F;
	bending_force(em, ef, els[idx], ets[idx], etis[idx], eref[idx],
		vrts, fas, x, fns, idms, vus, nods,
		J, F, bd, mtrIdx);

	// for debug
	if (oF != NULL) oF[idx] = F;
	if (oJ != NULL) oJ[idx] = J;

	if (dt == 0) {
		add_submat4(-J, em, vals, rowInc, matIdx, bsr);
		add_subvec4(F, em, b);
	}
	else {
		add_submat4(-dt*(dt + damping)*J, em, vals, rowInc, matIdx, bsr);

		REAL12 t = dt*(F + (dt + damping)*J*vs);
		add_subvec4(t, em, b);
	}
}

__global__ void
kernel_fill_constraint_forces(g_IneqCon *cstrs,
REAL dt, REAL *vals, bool bsr, int num,
int **matIdx, int *rowInc, REAL3 *b,
REAL3 *cx, REAL3 *ox, REAL3 *cv, REAL3 *ov, REAL mrt)
{
	LEN_CHK(num);
	g_IneqCon &cstr = cstrs[idx];

	if (false == cstr._valid)
		return;

	REAL v = cstr.value(cx, ox, mrt);
	REAL g = cstr.energy_grad(v, mrt);
	REAL h = cstr.energy_hess(v, mrt);

	MeshGrad grad;
	cstr.gradient(grad);

	REAL v_dot_grad = 0;
	for (int i = 0; i<4; i++) {
		v_dot_grad += dot(grad[i], cstr.get_x(i, cv, ov));
	}

	for (int i = 0; i<4; i++) {
		if (!cstr.free[i]) continue;
		int ni = cstr.nodes[i];

		for (int j = 0; j<4; j++) {
			if (!cstr.free[j]) continue;
			int nj = cstr.nodes[j];

			int locs[9];
			int k = find_location(matIdx, rowInc, ni, nj);

			if (bsr) {
				int loc = get_indices(ni, k, rowInc);

				if (dt == 0) {
					add_val(loc, vals, h*outer(grad[i], grad[j]));
				}
				else {
					REAL3x3 m = dt*dt*h*outer(grad[i], grad[j]);
					add_val(loc, vals, m);
				}
			}
			else {
				get_indices(ni, k, locs, rowInc);

				if (dt == 0) {
					add_val(locs, vals, h*outer(grad[i], grad[j]));
				}
				else {
					REAL3x3 m = dt*dt*h*outer(grad[i], grad[j]);
					add_val(locs, vals, m);
				}
			}
		}

		if (dt == 0)
			atomicAdd3(b + ni, -g*grad[i]);
		else {
			REAL3 dd = -dt*(g + dt*h*v_dot_grad)*grad[i];
			atomicAdd3(b + ni, dd);
		}
	}
}


__global__ void
kernel_fill_glue_forces(REAL3 *fext, g_GlueCon *glus,
REAL dt, REAL3 *cx, REAL3 *cv, float cur_stitch_ratio, int num)
{
	LEN_CHK(num);

	g_GlueCon &glu= glus[idx];

	int id0 = glu.n0;
	int id1 = glu.n1;

	const float len_init = length(glu.x1 - glu.x0);
	const float len_cur = length(cx[id1] - cx[id0]) + 1e-16f;
	const float ratio = cur_stitch_ratio * len_init / len_cur;

	//REAL3 fg = glu.stiff*(1 - ratio)*(cx[id0] - cx[id1] + dt*(cv[id0] - cv[id1]));
	REAL3 fg = 0.5*(1 - ratio)*(cx[id0] - cx[id1] + dt*(cv[id0] - cv[id1]));

	atomicAdd3(fext + id0, -fg);
	atomicAdd3(fext + id1, fg);
}

__global__ void
kernel_update_handle_forces(EqCon *hdls,
REAL dt, REAL *vals, bool bsr, int num,
int **matIdx, int *rowInc, REAL3 *b,
REAL3 *cx, REAL3 *cv, int *diaLocs)
{
	LEN_CHK(num);
	EqCon &cstr = hdls[idx];

	REAL v = cstr.value(cx);
	REAL g = cstr.energy_grad(v);
	REAL h = cstr.energy_hess();

	REAL3 grad;
	cstr.gradient(grad);

	int ni = cstr.node;
	REAL v_dot_grad = dot(grad, cv[ni]);

	assert(bsr);

	int loc = diaLocs[ni];

	if (dt == 0)
		add_val(loc, vals, h*outer(grad, grad));
	else
		add_val(loc, vals, dt*dt*h*outer(grad, grad));

	if (dt == 0)
		atomicAdd3(b + ni, -g*grad);
	else
		atomicAdd3(b + ni, -dt*(g + dt*h*v_dot_grad)*grad);
}

__global__ void
kernel_fill_handle_forces(EqCon *hdls,
REAL dt, REAL *vals, bool bsr, int num,
int **matIdx, int *rowInc, REAL3 *b,
REAL3 *cx, REAL3 *cv)
{
	LEN_CHK(num);
	EqCon &cstr = hdls[idx];

	REAL v = cstr.value(cx);
	REAL g = cstr.energy_grad(v);
	REAL h = cstr.energy_hess();

	REAL3 grad;
	cstr.gradient(grad);

	int ni = cstr.node;
	REAL v_dot_grad = dot(grad, cv[ni]);

	int locs[9];
	int k = find_location(matIdx, rowInc, ni, ni);

	if (bsr) {
		int loc = get_indices(ni, k, rowInc);

		if (dt == 0)
			add_val(loc, vals, h*outer(grad, grad));
		else
			add_val(loc, vals, dt*dt*h*outer(grad, grad));
	}
	else {
		get_indices(ni, k, locs, rowInc);

		if (dt == 0)
			add_val(locs, vals, h*outer(grad, grad));
		else
			add_val(locs, vals, dt*dt*h*outer(grad, grad));
	}

	if (dt == 0)
		atomicAdd3(b + ni, -g*grad);
	else
		atomicAdd3(b + ni, -dt*(g + dt*h*v_dot_grad)*grad);
}

__global__ void
kernel_fill_friction_forces(g_IneqCon *cstrs,
REAL dt, REAL *vals, bool bsr, int num,
int **matIdx, int *rowInc, REAL3 *b, REAL3 *bf,
REAL3 *cx, REAL3 *ox, REAL3 *cv, REAL3 *ov, REAL *cm, REAL *om, REAL mrt)
{
	LEN_CHK(num);
	g_IneqCon &cstr = cstrs[idx];
	if (false == cstr._valid)
		return;

	MeshHess jac;
	MeshGrad force;
	cstr.friction(dt, jac, force, cm, om, cx, ox, cv, ov, mrt);

	for (int i = 0; i<4; i++) {
		if (!cstr.free[i]) continue;
		int id = cstr.nodes[i];
		atomicAdd3(b + id, dt*force[i]);
		atomicAdd3(bf + id, dt*force[i]);
	}

	for (int i = 0; i<4; i++)
		for (int j = 0; j<4; j++) {
			if (!cstr.free[i] || !cstr.free[j])
				continue;

			int ni = cstr.nodes[i];
			int nj = cstr.nodes[j];

			int locs[9];
			int k = find_location(matIdx, rowInc, ni, nj);

			if (bsr) {
				int loc = get_indices(ni, k, rowInc);
				add_val(loc, vals, -dt*jac[i][j]);
			}
			else {
				get_indices(ni, k, locs, rowInc);
				add_val(locs, vals, -dt*jac[i][j]);
			}
		}
}

__global__ void
kernel_mat_fill(REAL dt, REAL *vals, int num,
int **matIdx, int *rowInc, REAL3 *b, REAL *m, REAL3 *Fext, REAL3x3 *Jext)
{
	LEN_CHK(num);

	int locs[9];
	int j = find_location(matIdx, rowInc, idx, idx);
	get_indices(idx, j, locs, rowInc);
	add_val(locs, vals, make_REAL3x3(m[idx]) - dt*dt*Jext[idx]);
	b[idx] += dt*Fext[idx];
}

__global__ void
kernel_mat_find_bsr(int **matIdx, int *rowInc, int *locs, int num)
{
	LEN_CHK(num);

	int j = find_location(matIdx, rowInc, idx, idx);
	locs[idx] = get_indices(idx, j, rowInc);
}


__global__ void
kernel_mat_update_bsr(REAL dt, REAL *vals, int num,
int *locs, REAL3 *b, REAL *m, REAL3 *Fext, REAL3x3 *Jext)
{
	LEN_CHK(num);

	int loc = locs[idx];
	add_val(loc, vals, make_REAL3x3(m[idx]) - dt*dt*Jext[idx]);
	b[idx] += dt*Fext[idx];
}

__global__ void
kernel_mat_fill_bsr(REAL dt, REAL *vals, int num,
int **matIdx, int *rowInc, REAL3 *b, REAL *m, REAL3 *Fext, REAL3x3 *Jext, int *locs)
{
	LEN_CHK(num);

	int j = find_location(matIdx, rowInc, idx, idx);
	int loc = get_indices(idx, j, rowInc);
	add_val(loc, vals, make_REAL3x3(m[idx]) - dt*dt*Jext[idx]);
	b[idx] += dt*Fext[idx];

	if (locs)
		locs[idx] = loc;
}

__global__ void
kernel_jacobi_idx(int *rows, int *cols, int num)
{
	LEN_CHK(num);

	rows[idx] = cols[idx] = idx;
}

__global__ void
kernel_jacobi_val(REAL *jbs, int *locs, REAL *vals, int num)
{
	LEN_CHK(num);

	REAL *ptr = vals + locs[idx];
	REAL3x3 aa;
	aa.put(ptr);
	REAL3x3 inv = getInverse(aa);
	inv.get(jbs + idx * 9);
}


__global__ void
kernel_jacobi_applyB(REAL3 *bs, REAL *jbs, int num)
{
	LEN_CHK(num);

	bs[idx].x *= jbs[idx * 3];
	bs[idx].y *= jbs[idx*3+1];
	bs[idx].z *= jbs[idx*3+2];
}

__global__ void
kernel_jacobi_applyA(REAL *vals, int *rows, REAL *jbs, int num)
{
	LEN_CHK(num);

	REAL *ptr = vals + idx*9;
	int r = rows[idx];
	REAL *j = jbs + r * 3;

	ptr[0] *= j[0];
	ptr[1] *= j[0];
	ptr[2] *= j[0];
	ptr[3] *= j[1];
	ptr[4] *= j[1];
	ptr[5] *= j[1];
	ptr[6] *= j[2];
	ptr[7] *= j[2];
	ptr[8] *= j[2];
}


__global__ void
kernel_mat_add(int *colLen, int *rowIdx, int **matIdx, int num, bool counting)
{
	LEN_CHK(num);

	if (counting)
		colLen[idx] += 1;
	else
		mat_add(idx, idx, rowIdx, matIdx);
}

__global__ void
kernel_add_gravity(REAL3 *fext, REAL *m, Gravity *g, int num)
{
	LEN_CHK(num);
	fext[idx] += m[idx] * g->_g;
}


inline __device__ REAL3 wind_force(
	int id0, int id1, int id2, const Wind &wind, REAL3 *v, REAL3 &fn, REAL fa)
{
	REAL3 vface = (v[id0] + v[id1] + v[id2]) / 3.0;
	REAL3 vrel = wind.velocity - vface;
	REAL vn = dot(fn, vrel);
	REAL3 vt = vrel - vn*fn;
	return wind.density*fa*abs(vn)*vn*fn + wind.drag*fa*vt;
}

__global__ void
kernel_add_wind(REAL3 *fext, tri3f *faces, REAL3 *v, REAL3 *fn, REAL *fa, Wind *dw, int num)
{
	LEN_CHK(num);

	int id0 = faces[idx].id0();
	int id1 = faces[idx].id1();
	int id2 = faces[idx].id2();
	REAL3 n = fn[idx];
	REAL a = fa[idx];

	REAL3 fw = wind_force(id0, id1, id2, *dw, v, n, a) / 3.0;

	atomicAdd3(fext + id0, fw);
	atomicAdd3(fext + id1, fw);
	atomicAdd3(fext + id2, fw);
}

__global__ void
kernel_step_mesh(REAL3 *x, REAL3 *v, REAL dt, int num)
{
	LEN_CHK(num);
	x[idx] += v[idx] * dt;
}

__global__ void
kernel_update_velocity(REAL3 *x, REAL3 *xx, REAL3 *v, REAL dt, int num)
{
	LEN_CHK(num);
	v[idx] += (x[idx] - xx[idx]) / dt;
}


__global__ void
kernel_face_ws(g_cone *cones, g_box *bxs, bool ccd, REAL3 *nrm, tri3f *face, REAL3 *x, REAL3 *ox, REAL thickness, int num)
{
	LEN_CHK(num);

	int id0 = face[idx].id0();
	int id1 = face[idx].id1();
	int id2 = face[idx].id2();

	REAL3 ox0 = ox[id0];
	REAL3 ox1 = ox[id1];
	REAL3 ox2 = ox[id2];
	REAL3 x0 = x[id0];
	REAL3 x1 = x[id1];
	REAL3 x2 = x[id2];

	bxs[idx].set(ox0, ox1);
	bxs[idx].add(ox2);
	
	if (ccd) {
		bxs[idx].add(x0);
		bxs[idx].add(x1);
		bxs[idx].add(x2);
	} 

	bxs[idx].enlarge(thickness);

	nrm[idx] = normalize(cross(x1 - x0, x2 - x0));

#ifdef USE_NC
	cones[idx].set(nrm[idx]);
	if (ccd) {
		REAL3 n0 = normalize(cross(ox1 - ox0, ox2 - ox0));
		cones[idx] += n0;
	}

	/*
	if (!ccd)
	cones[idx].set(nrm[idx]);
	else {
	REAL3 va = x0 - ox0;
	REAL3 vb = x1 - ox1;
	REAL3 vc = x2 - ox2;

	REAL3 n0 = normalize(cross(ox1 - ox0, ox2 - ox0));
	REAL3 n1 = nrm[idx];
	REAL3 s = vb - va;
	REAL3 t = vc - va;
	REAL3 delta = cross(s, t);
	REAL3 n2 = normalize(n0 + n1 - delta);

	cones[idx].set(n0, n1);
	cones[idx] += n2;
	}
	*/
#endif
}

__global__ void
kernel_face_init_strain(REAL2 *str, REAL strMin, REAL strMax, int num)
{
	LEN_CHK(num);

	str[idx] = make_REAL2(strMin, strMax);
}

__global__ void
kernel_edge_ws(g_box *bxs, bool ccd, uint2 *en, uint2 *ef, REAL *er, REAL *et, REAL3 *nrm, REAL3 *x, REAL3 *ox, REAL thickness, int num)
{
	LEN_CHK(num);

	int id0 = en[idx].x;
	int id1 = en[idx].y;

	REAL3 ox0 = ox[id0];
	REAL3 ox1 = ox[id1];
	REAL3 x0 = x[id0];
	REAL3 x1 = x[id1];

	bxs[idx].set(x0, x1);
	if (ccd) {
		bxs[idx].add(ox0);
		bxs[idx].add(ox1);
	}

	bxs[idx].enlarge(thickness);

	et[idx] = dihedral_angle(make_uint4(id0, id1, 0, 0), ef[idx], x, nrm, er[idx]);
}

__global__ void
kernel_node_ws(g_box *bxs, bool ccd, REAL3 *x, REAL3 *ox, REAL3 *n,
int *n2vIdx, int *n2vData, int *adjIdx, int *adjData, REAL3 *fAreas,
REAL thickness, int num)
{
	LEN_CHK(num);

	REAL3 ox0 = ox[idx];
	REAL3 x0 = x[idx];

	bxs[idx].set(x0);
	if (ccd) {
		bxs[idx].add(ox0);
	}
	//else
		bxs[idx].enlarge(thickness);

	n[idx] = node_normal(idx, n2vIdx, n2vData, adjIdx, adjData, fAreas);
}

typedef struct {
	uint numNode, numFace, numEdge, numVert;
	int _n2vNum; // for n2v ...
	int _adjNum; // for vertex's adj faces ...
	int _mtrNum; // for materials

	// device memory
	// node attributes
	REAL3 *_dx, *_dx0, *_dv, *_dn;
	REAL *_da, *_dm;
	// from node to verts
	int *_dn2vIdx, *_dn2vData;

	// face attributes
	tri3f *_dfnod; //nodes
	tri3f *_dfvrt; //verts
	tri3f *dfedg;
	REAL3 *_dfn; // local normal, exact
	REAL *dfa, *dfm; // area, mass
	REAL2x2 *_dfdm, *_dfidm; // finite element matrix
	REAL2 *_dfstr; // strain_min/strain_max
	int *_dfmtr; // material indices

	// jaccobi strain limiting
	int *_dfstrWeights;
	REAL3 *_dfstrTemps;

	//edge attributes
	uint2 *den; //nodes
	uint2 *def; // faces
	REAL *detheta, *delen; // hihedra angle & length
	REAL *_deitheta; // rest dihedral angle, 
	REAL *_deref; //just to get sign of dihedral_angle() right

	// vertex attributes
	REAL2 *_dvu; // material space, usually you should should _du[nid]...
	int *_dvn; //vert->node->index
	int *_dvAdjIdx; // adjacent faces
	int *_dvAdjData;

	// host memory
	REAL3 *hx;          // use for save, dynamic
	tri3f *hfvrt;    // use for save, static

	// for bounding volumes
	g_box *_vtxBxs, *_triBxs, *_edgeBxs;

	// for normal cone culling
	g_cone *_triCones;
	int *_triParents;

	// for collision detection
	uint *_dfmask;

	// for debugging
	g_box *hvBxs, *htBxs, *heBxs;

	void getBxs()
	{
		if (hvBxs == NULL)
			hvBxs = new g_box[numNode];
		if (htBxs == NULL)
			htBxs = new g_box[numFace];
		if (heBxs == NULL)
			heBxs = new g_box[numEdge];

		cudaMemcpy(hvBxs, _vtxBxs, numNode*sizeof(g_box), cudaMemcpyDeviceToHost);
		cudaMemcpy(htBxs, _triBxs, numFace*sizeof(g_box), cudaMemcpyDeviceToHost);
		cudaMemcpy(heBxs, _edgeBxs, numEdge*sizeof(g_box), cudaMemcpyDeviceToHost);
	}

	// init function
	void init()
	{
		numNode = 0;
		numFace = 0;
		numEdge = 0;
		numVert = 0;

		_adjNum = 0;
		_n2vNum = 0;
		_mtrNum = 0;
		_dfmask = NULL;

		hvBxs = NULL;
		htBxs = NULL;
		heBxs = NULL;
	}

	// merge all into one ...
	void mergeObjs();
	void mergeClothes();

	// host function
	void destroy()
	{
		// for edges
		checkCudaErrors(cudaFree(den));
		checkCudaErrors(cudaFree(def));
		checkCudaErrors(cudaFree(detheta));
		checkCudaErrors(cudaFree(delen));
		checkCudaErrors(cudaFree(_deitheta));
		checkCudaErrors(cudaFree(_deref));

		// for nodes
		checkCudaErrors(cudaFree(_dx));
		checkCudaErrors(cudaFree(_dx0));
		checkCudaErrors(cudaFree(_dv));
		checkCudaErrors(cudaFree(_da));
		checkCudaErrors(cudaFree(_dm));
		checkCudaErrors(cudaFree(_dn));
		checkCudaErrors(cudaFree(_dn2vIdx));
		checkCudaErrors(cudaFree(_dn2vData));
		delete[] hx;

		// for faces
		checkCudaErrors(cudaFree(_dfnod));
		checkCudaErrors(cudaFree(_dfvrt));
		checkCudaErrors(cudaFree(dfedg));
		checkCudaErrors(cudaFree(_dfn));
		checkCudaErrors(cudaFree(dfa));
		checkCudaErrors(cudaFree(dfm));
		checkCudaErrors(cudaFree(_dfdm));
		checkCudaErrors(cudaFree(_dfidm));
		checkCudaErrors(cudaFree(_dfmtr));

		if (_dfstr)
			checkCudaErrors(cudaFree(_dfstr));
		if (_dfstrWeights)
			checkCudaErrors(cudaFree(_dfstrWeights));
		if (_dfstrTemps)
			checkCudaErrors(cudaFree(_dfstrTemps));

		delete[] hfvrt;

		// for vertex
		checkCudaErrors(cudaFree(_dvu));
		checkCudaErrors(cudaFree(_dvAdjIdx));
		checkCudaErrors(cudaFree(_dvAdjData));

		// for BVH
		checkCudaErrors(cudaFree(_vtxBxs));
		checkCudaErrors(cudaFree(_edgeBxs));
		checkCudaErrors(cudaFree(_triBxs));
		// _triBxs will be free by bvh (allocated there)
		checkCudaErrors(cudaFree(_triCones));
		checkCudaErrors(cudaFree(_triParents));

		if (_dfmask)
			checkCudaErrors(cudaFree(_dfmask));
	}

	void save(const string &fname) {
		checkCudaErrors(cudaMemcpy(hx, _dx, numNode*sizeof(REAL3), cudaMemcpyDeviceToHost));

		fstream file(fname.c_str(), ios::out);
		for (uint n = 0; n<numNode; n++) {
			file << "v " << hx[n].x << " " << hx[n].y << " " << hx[n].z << endl;
		}

		for (uint n = 0; n<numFace; n++) {
			file << hfvrt[n];
		}
	}

	void loadVtx(const string &fname) {
		input2(fname.c_str(), (REAL *)hx, numNode * 3);
		checkCudaErrors(cudaMemcpy(_dx, hx, numNode*sizeof(REAL3), cudaMemcpyHostToDevice));
	}

	void saveVtx(const string &fname) {
		checkCudaErrors(cudaMemcpy(hx, _dx, numNode*sizeof(REAL3), cudaMemcpyDeviceToHost));
		output2(fname.c_str(), (REAL *)hx, numNode * 3);
	}

	void saveObj(const string &fname) {
		save(fname);
	}

	void allocEdges()
	{
		checkCudaErrors(cudaMalloc((void **)&den, numEdge*sizeof(uint2)));
		checkCudaErrors(cudaMalloc((void **)&def, numEdge*sizeof(uint2)));
		checkCudaErrors(cudaMalloc((void **)&detheta, numEdge*sizeof(REAL)));
		checkCudaErrors(cudaMalloc((void **)&delen, numEdge*sizeof(REAL)));
		checkCudaErrors(cudaMalloc((void **)&_deitheta, numEdge*sizeof(REAL)));
		checkCudaErrors(cudaMalloc((void **)&_deref, numEdge*sizeof(REAL)));
		checkCudaErrors(cudaMalloc(&_edgeBxs, numEdge*sizeof(g_box)));
	}

	void pushEdges(int num, uint2 *n, uint2 *f, REAL *t, REAL *l, REAL *i, REAL *r,
		int offset, enum cudaMemcpyKind kind)
	{
		checkCudaErrors(cudaMemcpy(den + offset, n, num*sizeof(uint2), kind));
		checkCudaErrors(cudaMemcpy(def + offset, f, num*sizeof(uint2), kind));
		checkCudaErrors(cudaMemcpy(detheta + offset, t, num*sizeof(REAL), kind));
		checkCudaErrors(cudaMemcpy(delen + offset, l, num*sizeof(REAL), kind));
		checkCudaErrors(cudaMemcpy(_deitheta + offset, i, num*sizeof(REAL), kind));
		checkCudaErrors(cudaMemcpy(_deref + offset, r, num*sizeof(REAL), kind));
	}

	void pushEdges(int num, void *n, void *f, REAL *t, REAL *l, REAL *i, REAL *r)
	{
		numEdge = num;
		allocEdges();
		pushEdges(num, (uint2 *)n, (uint2 *)f, t, l, i, r, 0, cudaMemcpyHostToDevice);
	}

	void allocVertices()
	{
		checkCudaErrors(cudaMalloc((void **)&_dvu, numVert*sizeof(REAL2)));
		checkCudaErrors(cudaMalloc((void **)&_dvn, numVert*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&_dvAdjIdx, numVert*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&_dvAdjData, _adjNum*sizeof(int)));
	}

	void pushVertices(int num, int adjNum, REAL2 *vu, int *vn, int *adjIdx, int *adjData,
		int offset1, int offset2, int offset3, enum cudaMemcpyKind kind)
	{
		checkCudaErrors(cudaMemcpy(_dvu + offset1, vu, num*sizeof(REAL2), kind));
		checkCudaErrors(cudaMemcpy(_dvn + offset3, vn, num*sizeof(int), kind));
		checkCudaErrors(cudaMemcpy(_dvAdjIdx + offset1, adjIdx, num*sizeof(int), kind));
		checkCudaErrors(cudaMemcpy(_dvAdjData + offset2, adjData, adjNum*sizeof(int), kind));
	}

	void pushVertices(int num, REAL *vu, int *vn, int *adjIdx, int *adjData, int adjNum)
	{
		numVert = num;
		_adjNum = adjNum;

		allocVertices();
		pushVertices(num, adjNum, (REAL2 *)vu, vn, adjIdx, adjData, 0, 0, 0, cudaMemcpyHostToDevice);
	}

	void allocFaces()
	{
		checkCudaErrors(cudaMalloc((void **)&_dfnod, numFace*sizeof(tri3f)));
		checkCudaErrors(cudaMalloc((void **)&_dfvrt, numFace*sizeof(tri3f)));
		checkCudaErrors(cudaMalloc((void **)&dfedg, numFace*sizeof(tri3f)));
		checkCudaErrors(cudaMalloc((void **)&_dfn, numFace*sizeof(REAL3)));
		checkCudaErrors(cudaMalloc((void **)&dfa, numFace*sizeof(REAL)));
		checkCudaErrors(cudaMalloc((void **)&dfm, numFace*sizeof(REAL)));
		checkCudaErrors(cudaMalloc((void **)&_dfdm, numFace*sizeof(REAL2x2)));
		checkCudaErrors(cudaMalloc((void **)&_dfidm, numFace*sizeof(REAL2x2)));
		checkCudaErrors(cudaMalloc((void **)&_dfmtr, numFace*sizeof(int)));
		checkCudaErrors(cudaMalloc(&_triBxs, numFace*sizeof(g_box)));
		checkCudaErrors(cudaMalloc(&_triCones, numFace*sizeof(g_cone)));
		checkCudaErrors(cudaMalloc(&_triParents, numFace*sizeof(int)));

		hfvrt = new tri3f[numFace];
		_dfstr = NULL;
		_dfstrWeights = NULL;
		_dfstrTemps = NULL;
	}

	void pushFaces(int num, tri3f *nods, tri3f *vrts, tri3f *edgs, REAL3 *nrms,
		REAL *a, REAL *m, REAL2x2 *dm, REAL2x2 *idm, int *midx,
		int offset, enum cudaMemcpyKind kind)
	{
		checkCudaErrors(cudaMemcpy(_dfnod + offset, nods, num*sizeof(tri3f), kind));
		checkCudaErrors(cudaMemcpy(_dfvrt + offset, vrts, num*sizeof(tri3f), kind));
		checkCudaErrors(cudaMemcpy(dfedg + offset, edgs, num*sizeof(tri3f), kind));
		checkCudaErrors(cudaMemcpy(_dfn + offset, nrms, num*sizeof(REAL3), kind));
		checkCudaErrors(cudaMemcpy(dfa + offset, a, num*sizeof(REAL), kind));
		checkCudaErrors(cudaMemcpy(dfm + offset, m, num*sizeof(REAL), kind));
		checkCudaErrors(cudaMemcpy(_dfdm + offset, dm, num*sizeof(REAL2x2), kind));
		checkCudaErrors(cudaMemcpy(_dfidm + offset, idm, num*sizeof(REAL2x2), kind));
		checkCudaErrors(cudaMemcpy(_dfmtr + offset, midx, num*sizeof(int), kind));

		if (kind == cudaMemcpyHostToDevice)
			memcpy(hfvrt + offset, vrts, num*sizeof(tri3f));
		if (kind == cudaMemcpyDeviceToDevice)
			checkCudaErrors(cudaMemcpy(hfvrt + offset, vrts, num*sizeof(tri3f), cudaMemcpyDeviceToHost));
	}

	void pushFaces(int num, void *nods, void *vrts, void *edgs, REAL *nrms,
		REAL *a, REAL *m, REAL *dm, REAL*idm, int *midx, int mn)
	{
		numFace = num;
		_mtrNum = mn;

		allocFaces();
		pushFaces(num, (tri3f *)nods, (tri3f *)vrts, (tri3f *)edgs, (REAL3 *)nrms, a, m, (REAL2x2 *)dm, (REAL2x2 *)idm, midx,
			0, cudaMemcpyHostToDevice);
	}

	void pushNodes(int num, REAL *x, REAL3 *dxx, REAL dt)
	{
		checkCudaErrors(cudaMemcpy(dxx, _dx, num*sizeof(REAL3), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(_dx, x, num*sizeof(REAL3), cudaMemcpyHostToDevice));
		{
			BLK_PAR(numNode);
			kernel_update_velocity << <B, T >> > (_dx, dxx, _dv, dt, numNode);
			getLastCudaError("kernel_update_velocity");
		}
	}

	void dumpVtx()
	{
		REAL3 *hb = new REAL3[numNode];

		checkCudaErrors(cudaMemcpy(hb, _dx0,
			numNode*sizeof(REAL3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(hb, _dx,
			numNode*sizeof(REAL3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(hb, _dv,
			numNode*sizeof(REAL3), cudaMemcpyDeviceToHost));

		delete[] hb;
	}

	void allocNodes()
	{
		checkCudaErrors(cudaMalloc((void **)&_dx, numNode*sizeof(REAL3)));
		checkCudaErrors(cudaMalloc((void **)&_dx0, numNode*sizeof(REAL3)));
		checkCudaErrors(cudaMalloc((void **)&_dv, numNode*sizeof(REAL3)));
		checkCudaErrors(cudaMalloc((void **)&_da, numNode*sizeof(REAL)));
		checkCudaErrors(cudaMalloc((void **)&_dm, numNode*sizeof(REAL)));
		checkCudaErrors(cudaMalloc((void **)&_dn, numNode*sizeof(REAL3)));

		checkCudaErrors(cudaMalloc((void **)&_dn2vIdx, numNode*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&_dn2vData, _n2vNum*sizeof(int)));

		checkCudaErrors(cudaMalloc(&_vtxBxs, numNode*sizeof(g_box)));
		hx = new REAL3[numNode];
	}

	void pushNodes(int num, REAL3 *x, REAL3 *x0, REAL3 *v, REAL *a, REAL *m, REAL3 *n,
		int n2vNum, int *n2vIdx, int *n2vData,
		int offset1, int offset2, enum cudaMemcpyKind kind)
	{
		checkCudaErrors(cudaMemcpy(_dx + offset1, x, num*sizeof(REAL3), kind));
		checkCudaErrors(cudaMemcpy(_dx0 + offset1, x0, num*sizeof(REAL3), kind));
		checkCudaErrors(cudaMemcpy(_dv + offset1, v, num*sizeof(REAL3), kind));
		checkCudaErrors(cudaMemcpy(_da + offset1, a, num*sizeof(REAL), kind));
		checkCudaErrors(cudaMemcpy(_dm + offset1, m, num*sizeof(REAL), kind));
		checkCudaErrors(cudaMemcpy(_dn + offset1, n, num*sizeof(REAL3), kind));

		checkCudaErrors(cudaMemcpy(_dn2vIdx + offset1, n2vIdx, num*sizeof(int), kind));
		checkCudaErrors(cudaMemcpy(_dn2vData + offset2, n2vData, n2vNum*sizeof(int), kind));
	}

	void pushNodes(int num, REAL *x, REAL *x0, REAL *v, REAL *a, REAL *m, REAL *n,
		int n2vNum, int *n2vIdx, int *n2vData)
	{
		numNode = num;
		_n2vNum = n2vNum;

		allocNodes();
		pushNodes(num, (REAL3 *)x, (REAL3 *)x0, (REAL3 *)v, a, m, (REAL3 *)n, n2vNum, n2vIdx, n2vData, 0, 0, cudaMemcpyHostToDevice);
	}

	void popNodes(int num, REAL *x, REAL *x0)
	{
		checkCudaErrors(cudaMemcpy(x0, _dx0, num*sizeof(REAL3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(x, _dx, num*sizeof(REAL3), cudaMemcpyDeviceToHost));
	}

	void popNodes(int num, REAL *x)
	{
		checkCudaErrors(cudaMemcpy(x, _dx, num*sizeof(REAL3), cudaMemcpyDeviceToHost));
	}

	void incFaces(int num, REAL3 *n, int offset)
	{
		assert(numFace >= num);
		checkCudaErrors(cudaMemcpy(_dfn + offset, n, num*sizeof(REAL3), cudaMemcpyHostToDevice));
	}

	void incEdges(int num, REAL *t, int offset)
	{
		assert(numEdge >= num);
		checkCudaErrors(cudaMemcpy(detheta + offset, t, num*sizeof(REAL), cudaMemcpyHostToDevice));
	}

	void incNodes(int num, REAL3 *x, REAL3 *x0, REAL3 *v, REAL *a, REAL *m, REAL3 *n, int offset)
	{
		assert(numNode >= num);

		checkCudaErrors(cudaMemcpy(_dx + offset, x, num*sizeof(REAL3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(_dx0 + offset, x0, num*sizeof(REAL3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(_dv + offset, v, num*sizeof(REAL3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(_da + offset, a, num*sizeof(REAL), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(_dm + offset, m, num*sizeof(REAL), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(_dn + offset, n, num*sizeof(REAL3), cudaMemcpyHostToDevice));
	}

	void stepMesh(REAL dt)
	{
		if (numNode == 0)
			return;

		BLK_PAR(numNode);
		kernel_step_mesh << <B, T >> > (_dx, _dv, dt, numNode);
		getLastCudaError("kernel_step_mesh");
	}

	void resetVelocity()
	{
		checkCudaErrors(cudaMemset(_dv, 0, numNode*sizeof(REAL3)));
	}

	void updateNodes(REAL3 *ddv, REAL dt, bool update_positions)
	{
		int num = numNode;

		BLK_PAR(num);
		kernel_update_nodes << <B, T >> >(_dv, _dx, ddv, dt, update_positions, num);
		getLastCudaError("kernel_update_nodes");
	}

	void updateX0()
	{
		checkCudaErrors(cudaMemcpy(_dx0, _dx, numNode*sizeof(REAL3), cudaMemcpyDeviceToDevice));
	}

	void project_outside(REAL3 *ddx, REAL *w, REAL *om, REAL3 *ox, REAL mrt, REAL mpt)
	{
		if (getConstraintNum() == 0)
			return;

		int num = numNode;
		checkCudaErrors(cudaMemset(ddx, 0, num*sizeof(REAL3)));
		checkCudaErrors(cudaMemset(w, 0, num*sizeof(REAL)));

		{
			int num = getConstraintNum();
			BLK_PAR(num);
			kernel_project_outside1 << <B, T >> >(getConstraints(), ddx, w, num,
				_dm, om, _dx, ox, mrt, mpt);
			getLastCudaError("kernel_project_outside1");
		}

		{
			int num = numNode;
			BLK_PAR(num);
			kernel_project_outside2 << <B, T >> >(_dx, ddx, w, num);
			getLastCudaError("kernel_project_outside2");
		}
	}

	void computeWSdata(REAL thickness, bool ccd)
	{
		if (numFace == 0)
			return;

		{
			int num = numFace;
			BLK_PAR(num);
			kernel_face_ws << <B, T >> > (_triCones, _triBxs, ccd, _dfn, _dfnod, _dx, _dx0, thickness, num);
			getLastCudaError("kernel_face_ws");
		}
		{
			int num = numEdge;
			BLK_PAR(num);
			kernel_edge_ws << <B, T >> > (_edgeBxs, ccd, den, def, _deref, detheta, _dfn, _dx, _dx0, thickness, num);
			getLastCudaError("kernel_edge_ws");
		}
		{
			int num = numNode;

			BLK_PAR(num);
			kernel_node_ws << <B, T >> >(_vtxBxs, ccd, _dx, _dx0, _dn,
				_dn2vIdx, _dn2vData, _dvAdjIdx, _dvAdjData, _dfn,
				thickness, num);
			getLastCudaError("kernel_node_ws");
		}
	}

	// could be moved to GPU, but only once, so CPU is ok for now...
	void buildMask()
	{
		tri3f *tris = new tri3f[numFace];
		checkCudaErrors(cudaMemcpy(tris, _dfnod, numFace*sizeof(tri3f), cudaMemcpyDeviceToHost));
		tri3f *edgs = new tri3f[numFace];
		checkCudaErrors(cudaMemcpy(edgs, dfedg, numFace*sizeof(tri3f), cudaMemcpyDeviceToHost));

		unsigned int *fmask = new unsigned int[numFace];

		bool *vtx_marked = new bool[numNode];
		for (unsigned int i = 0; i<numNode; i++)
			vtx_marked[i] = false;

		bool *edge_marked = new bool[numEdge];
		for (unsigned int i = 0; i<numEdge; i++)
			edge_marked[i] = false;

		for (unsigned int i = 0; i<numFace; i++) {
			fmask[i] = 0;

			tri3f *vtx = tris + i;
			for (int j = 0; j<3; j++) {
				unsigned int vid = vtx->id(j);
				if (vtx_marked[vid] == false) {
					fmask[i] |= (0x1 << j);
					vtx_marked[vid] = true;
				}
			}

			tri3f *edge = edgs + i;
			for (int j = 0; j<3; j++) {
				unsigned int eid = edge->id(j);
				if (edge_marked[eid] == false) {
					fmask[i] |= (0x8 << j);
					edge_marked[eid] = true;
				}
			}
		}

		delete[] vtx_marked;
		delete[] edge_marked;
		delete[] tris;
		delete[] edgs;

		checkCudaErrors(cudaMalloc((void **)&_dfmask, numFace*sizeof(uint)));
		checkCudaErrors(cudaMemcpy(_dfmask, fmask, numFace*sizeof(uint), cudaMemcpyHostToDevice));

		delete[] fmask;
	}

	void initFaceStrains(REAL strMin, REAL strMax)
	{
		if (_dfstr != NULL) return;

		checkCudaErrors(cudaMalloc(&_dfstr, numFace*sizeof(REAL2)));
		{
			int num = numFace;
			BLK_PAR(num);
			kernel_face_init_strain << <B, T >> > (_dfstr, strMin, strMax, num);
			getLastCudaError("kernel_face_init_strain");
		}
	}
} g_mesh;

// another aux data for cloth, such as materials, fext, Jext used for time integration ...
typedef struct {
	// for time integration
	REAL3 *dFext;
	REAL3x3 *dJext;

	REAL3 *_db, *_dx; // for A*x = b
	REAL3 *_df; // for friction
	REAL *_dw; // _dx and _w will be used by project_outside

	REAL2 *_fdists, *_vdists, *_edists; // buffers for constraint filtering

	void init(int nn, int fn, int en) {
		checkCudaErrors(cudaMalloc((void **)&dFext, nn*sizeof(REAL3)));
		checkCudaErrors(cudaMalloc((void **)&dJext, nn*sizeof(REAL3x3)));
		checkCudaErrors(cudaMalloc((void **)&_db, nn*sizeof(REAL3)));
		checkCudaErrors(cudaMalloc((void **)&_dx, nn*sizeof(REAL3)));
		checkCudaErrors(cudaMalloc((void **)&_dw, nn*sizeof(REAL)));
		checkCudaErrors(cudaMalloc((void **)&_df, nn*sizeof(REAL3)));

		_vdists = new REAL2[nn];
		_fdists = new REAL2[fn];
		_edists = new REAL2[en];
	}

	void reset(int nn, int fn, int en) {
		checkCudaErrors(cudaMemset(dFext, 0, nn*sizeof(REAL3)));
		checkCudaErrors(cudaMemset(dJext, 0, nn*sizeof(REAL3x3)));
		checkCudaErrors(cudaMemset(_db, 0, nn*sizeof(REAL3)));
		checkCudaErrors(cudaMemset(_df, 0, nn*sizeof(REAL3)));
		checkCudaErrors(cudaMemset(_dx, 0, nn*sizeof(REAL3)));
		checkCudaErrors(cudaMemset(_dw, 0, nn*sizeof(REAL)));

		memset(_vdists, 0, nn*sizeof(REAL2));
		memset(_fdists, 0, fn*sizeof(REAL2));
		memset(_edists, 0, en*sizeof(REAL2));
	}

	void destroy() {
		checkCudaErrors(cudaFree(dFext));
		checkCudaErrors(cudaFree(dJext));
		checkCudaErrors(cudaFree(_db));
		checkCudaErrors(cudaFree(_df));
		checkCudaErrors(cudaFree(_dx));

		checkCudaErrors(cudaFree(_dw));

		delete[] _vdists;
		delete[] _fdists;
		delete[] _edists;
	}

} g_aux;

static int numCloth;
static g_mesh *_clothes;

static int numObstacles;
static g_mesh *obstacles;

static g_mesh *_currentMesh;
//static g_aux *currentAux;

static g_mesh currentObj;
static g_mesh currentCloth;
static g_aux totalAux;
static REAL m_weakening, m_damping, m_damage;

void push_material_gpu(const void *s, const void *b, int n, const void *w, const void *g, REAL damping, REAL weakening, REAL damage)
{
	checkCudaErrors(cudaMalloc(&dMaterialStretching, n*sizeof(StretchingSamples)));
	checkCudaErrors(cudaMemcpy(dMaterialStretching, s, n*sizeof(StretchingSamples), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&dMaterialBending, n*sizeof(BendingData)));
	checkCudaErrors(cudaMemcpy(dMaterialBending, b, n*sizeof(BendingData), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&dWind, sizeof(Wind)));
	checkCudaErrors(cudaMemcpy(dWind, w, sizeof(Wind), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&dGravity, sizeof(Gravity)));
	checkCudaErrors(cudaMemcpy(dGravity, g, sizeof(Gravity), cudaMemcpyHostToDevice));

	m_weakening = weakening;
	m_damping = damping;
	m_damage = damage;
}

void push_num_gpu(int nC, int nO)
{
	numCloth = nC;
	numObstacles = nO;

	_clothes = new g_mesh[nC];
	for (int i = 0; i<nC; i++)
		_clothes[i].init();

	obstacles = new g_mesh[nO];
	for (int i = 0; i<nO; i++)
		obstacles[i].init();
}

void merge_obstacles_gpu()
{
	currentObj.init();
	currentObj.mergeObjs();
}

void merge_clothes_gpu()
{
	currentCloth.init();
	currentCloth.mergeClothes();
}

__global__ void
kernel_add_offset(tri3f *x, int offset, int num)
{
	LEN_CHK(num);

	x[idx]._ids.x += offset;
	x[idx]._ids.y += offset;
	x[idx]._ids.z += offset;
}

__global__ void
kernel_add_offset(uint2 *x, int offset, int num, bool face)
{
	LEN_CHK(num);

	if (face && x[idx].x == -1)
		;
	else
		x[idx].x += offset;

	if (face && x[idx].y == -1)
		;
	else
		x[idx].y += offset;
}

__global__ void
kernel_add_offset(int *x, int offset, int num)
{
	LEN_CHK(num);

	x[idx] += offset;
}

__global__ void
kernel_add_offset(uint *x, int offset, int num)
{
	LEN_CHK(num);

	x[idx] += offset;
}

void offset_indices(tri3f *data, int offset, int num)
{
	BLK_PAR(num);
	kernel_add_offset << <B, T >> > (data, offset, num);
	getLastCudaError("kernel_add_offset");
}

void offset_indices(uint2*data, int offset, int num, bool face = false)
{
	BLK_PAR(num);
	kernel_add_offset << <B, T >> > (data, offset, num, face);
	getLastCudaError("kernel_add_offset");
}

void offset_indices(int*data, int offset, int num)
{
	BLK_PAR(num);
	kernel_add_offset << <B, T >> > (data, offset, num);
	getLastCudaError("kernel_add_offset");
}

void offset_indices(uint*data, int offset, int num)
{
	BLK_PAR(num);
	kernel_add_offset << <B, T >> > (data, offset, num);
	getLastCudaError("kernel_add_offset");
}

void g_mesh::mergeObjs()
{
	if (numObstacles == 0)
		return;

	int *nodeOffsets = new int[numObstacles];
	int *faceOffsets = new int[numObstacles];
	int *edgeOffsets = new int[numObstacles];
	int *vertOffsets = new int[numObstacles];
	int *adjIdxOffsets = new int[numObstacles];
	int *n2vIdxOffsets = new int[numObstacles];

	for (int i = 0; i<numObstacles; i++) {
		numNode += obstacles[i].numNode;
		numFace += obstacles[i].numFace;
		numEdge += obstacles[i].numEdge;
		numVert += obstacles[i].numVert;
		_adjNum += obstacles[i]._adjNum;
		_n2vNum += obstacles[i]._n2vNum;

		nodeOffsets[i] = (i == 0) ? 0 : nodeOffsets[i - 1] + obstacles[i - 1].numNode;
		faceOffsets[i] = (i == 0) ? 0 : faceOffsets[i - 1] + obstacles[i - 1].numFace;
		edgeOffsets[i] = (i == 0) ? 0 : edgeOffsets[i - 1] + obstacles[i - 1].numEdge;
		vertOffsets[i] = (i == 0) ? 0 : vertOffsets[i - 1] + obstacles[i - 1].numVert;
		adjIdxOffsets[i] = (i == 0) ? 0 : adjIdxOffsets[i - 1] + obstacles[i - 1]._adjNum;
		n2vIdxOffsets[i] = (i == 0) ? 0 : n2vIdxOffsets[i - 1] + obstacles[i - 1]._n2vNum;
	}

	{// merge nodes
		allocNodes();

		int offset1 = 0;
		int offset2 = 0;
		for (int i = 0; i<numObstacles; i++) {
			g_mesh &ob = obstacles[i];
			if (ob.numNode == 0)
				continue;

			pushNodes(ob.numNode, ob._dx, ob._dx0, ob._dv, ob._da, ob._dm, ob._dn,
				ob._n2vNum, ob._dn2vIdx, ob._dn2vData,
				offset1, offset2, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dn2vIdx + offset1, n2vIdxOffsets[i], ob.numNode);
				offset_indices(_dn2vData + offset2, vertOffsets[i], ob._n2vNum);
			}

			offset1 += ob.numNode;
			offset2 += ob._n2vNum;
		}
	}

	{// merge faces
		allocFaces();
		int offset = 0;
		for (int i = 0; i<numObstacles; i++) {
			g_mesh &ob = obstacles[i];
			if (ob.numFace == 0)
				continue;

			pushFaces(ob.numFace, ob._dfnod, ob._dfvrt, ob.dfedg, ob._dfn, ob.dfa, ob.dfm, ob._dfdm, ob._dfidm, ob._dfmtr,
				offset, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dfnod + offset, nodeOffsets[i], ob.numFace);
				offset_indices(_dfvrt + offset, vertOffsets[i], ob.numFace);
				offset_indices(dfedg + offset, edgeOffsets[i], ob.numFace);
			}

			offset += ob.numFace;
		}
		checkCudaErrors(cudaMemcpy(hfvrt, _dfnod, numFace*sizeof(tri3f), cudaMemcpyDeviceToHost));
	}

	{// merge edges
		allocEdges();

		int offset = 0;
		for (int i = 0; i<numObstacles; i++) {
			g_mesh &ob = obstacles[i];
			if (ob.numEdge == 0)
				continue;

			pushEdges(ob.numEdge, ob.den, ob.def, ob.detheta, ob.delen, ob._deitheta, ob._deref,
				offset, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(den + offset, nodeOffsets[i], ob.numEdge);
				offset_indices(def + offset, faceOffsets[i], ob.numEdge, true);
			}

			offset += ob.numEdge;
		}
	}

	{// merge vertices
		allocVertices();

		int offset1 = 0;
		int offset2 = 0;
		int offset3 = 0;
		for (int i = 0; i<numObstacles; i++) {
			g_mesh &ob = obstacles[i];
			if (ob.numVert == 0)
				continue;

			pushVertices(ob.numVert, ob._adjNum, ob._dvu, ob._dvn, ob._dvAdjIdx, ob._dvAdjData,
				offset1, offset2, offset3, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dvAdjIdx + offset1, adjIdxOffsets[i], ob.numVert);
				offset_indices(_dvAdjData + offset2, faceOffsets[i], ob._adjNum);
				offset_indices(_dvn + offset3, nodeOffsets[i], ob.numNode);
			}

			offset1 += ob.numVert;
			offset2 += ob._adjNum;
			offset3 += ob.numNode;
		}
	}

	delete[] nodeOffsets;
	delete[] faceOffsets;
	delete[] edgeOffsets;
	delete[] vertOffsets;
	delete[] adjIdxOffsets;
	delete[] n2vIdxOffsets;
}

void g_mesh::mergeClothes()
{
	assert(numCloth != 0);

	int *nodeOffsets = new int[numCloth];
	int *faceOffsets = new int[numCloth];
	int *edgeOffsets = new int[numCloth];
	int *vertOffsets = new int[numCloth];
	int *adjIdxOffsets = new int[numCloth];
	int *n2vIdxOffsets = new int[numCloth];
	int *midxOffsets = new int[numCloth];

	for (int i = 0; i<numCloth; i++) {
		numNode += _clothes[i].numNode;
		numFace += _clothes[i].numFace;
		numEdge += _clothes[i].numEdge;
		numVert += _clothes[i].numVert;
		_adjNum += _clothes[i]._adjNum;
		_n2vNum += _clothes[i]._n2vNum;
		_mtrNum += _clothes[i]._mtrNum;

		nodeOffsets[i] = (i == 0) ? 0 : nodeOffsets[i - 1] + _clothes[i - 1].numNode;
		faceOffsets[i] = (i == 0) ? 0 : faceOffsets[i - 1] + _clothes[i - 1].numFace;
		edgeOffsets[i] = (i == 0) ? 0 : edgeOffsets[i - 1] + _clothes[i - 1].numEdge;
		vertOffsets[i] = (i == 0) ? 0 : vertOffsets[i - 1] + _clothes[i - 1].numVert;
		adjIdxOffsets[i] = (i == 0) ? 0 : adjIdxOffsets[i - 1] + _clothes[i - 1]._adjNum;
		n2vIdxOffsets[i] = (i == 0) ? 0 : n2vIdxOffsets[i - 1] + _clothes[i - 1]._n2vNum;
		midxOffsets[i] = (i == 0) ? 0 : midxOffsets[i - 1] + _clothes[i - 1]._mtrNum;
	}

	{// merge nodes
		allocNodes();

		int offset1 = 0;
		int offset2 = 0;
		for (int i = 0; i<numCloth; i++) {
			g_mesh &ob = _clothes[i];
			pushNodes(ob.numNode, ob._dx, ob._dx0, ob._dv, ob._da, ob._dm, ob._dn,
				ob._n2vNum, ob._dn2vIdx, ob._dn2vData,
				offset1, offset2, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dn2vIdx + offset1, n2vIdxOffsets[i], ob.numNode);
				offset_indices(_dn2vData + offset2, vertOffsets[i], ob._n2vNum);
			}

			offset1 += ob.numNode;
			offset2 += ob._n2vNum;
		}
	}

	{// merge faces
		allocFaces();
		int offset = 0;
		for (int i = 0; i<numCloth; i++) {
			g_mesh &ob = _clothes[i];
			pushFaces(ob.numFace, ob._dfnod, ob._dfvrt, ob.dfedg, ob._dfn, ob.dfa, ob.dfm, ob._dfdm, ob._dfidm, ob._dfmtr,
				offset, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dfnod + offset, nodeOffsets[i], ob.numFace);
				offset_indices(_dfvrt + offset, vertOffsets[i], ob.numFace);
				offset_indices(dfedg + offset, edgeOffsets[i], ob.numFace);
				offset_indices(_dfmtr + offset, midxOffsets[i], ob.numFace);
			}

			offset += ob.numFace;
		}
		checkCudaErrors(cudaMemcpy(hfvrt, _dfnod, numFace*sizeof(tri3f), cudaMemcpyDeviceToHost));
	}

	{// merge edges
		allocEdges();

		int offset = 0;
		for (int i = 0; i<numCloth; i++) {
			g_mesh &ob = _clothes[i];
			pushEdges(ob.numEdge, ob.den, ob.def, ob.detheta, ob.delen, ob._deitheta, ob._deref,
				offset, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(den + offset, nodeOffsets[i], ob.numEdge);
				offset_indices(def + offset, faceOffsets[i], ob.numEdge, true);
			}

			offset += ob.numEdge;
		}
	}

	{// merge vertices
		allocVertices();

		int offset1 = 0;
		int offset2 = 0;
		int offset3 = 0;
		for (int i = 0; i<numCloth; i++) {
			g_mesh &ob = _clothes[i];
			pushVertices(ob.numVert, ob._adjNum, ob._dvu, ob._dvn, ob._dvAdjIdx, ob._dvAdjData,
				offset1, offset2, offset3, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dvAdjIdx + offset1, adjIdxOffsets[i], ob.numVert);
				offset_indices(_dvAdjData + offset2, faceOffsets[i], ob._adjNum);
				offset_indices(_dvn + offset3, nodeOffsets[i], ob.numNode);
			}

			offset1 += ob.numVert;
			offset2 += ob._adjNum;
			offset3 += ob.numNode;
		}
	}

	delete[] nodeOffsets;
	delete[] faceOffsets;
	delete[] edgeOffsets;
	delete[] vertOffsets;
	delete[] adjIdxOffsets;
	delete[] n2vIdxOffsets;
	delete[] midxOffsets;
}

void set_current_gpu(int idx, bool isCloth)
{
	_currentMesh = isCloth ? _clothes + idx : obstacles + idx;
}

void pop_cloth_gpu(int num, REAL *x, REAL *f)
{
	currentCloth.popNodes(num, x);

	if (f)
		checkCudaErrors(cudaMemcpy(f, totalAux._df, num*sizeof(REAL3), cudaMemcpyDeviceToHost));

}

void push_node_gpu(int num, REAL *x, REAL *x0, REAL *v, REAL *a, REAL *m, REAL *n, int n2vNum, int *n2vIdx, int *n2vData)
{
	_currentMesh->pushNodes(num, x, x0, v, a, m, n, n2vNum, n2vIdx, n2vData);
}

void push_node_gpu(int num, REAL *x, REAL dt)
{
	currentCloth.pushNodes(num, x, totalAux._dx, dt);
}

void build_mask_gpu()
{
	currentObj.buildMask();
	currentCloth.buildMask();
}

void inc_face_gpu(int idx, int num, REAL *fn)
{
	int offset = 0;
	for (int i = 0; i<idx; i++)
		offset += obstacles[i].numFace;

	currentObj.incFaces(num, (REAL3 *)fn, offset);
}

void inc_edge_gpu(int idx, int num, REAL *theta)
{
	int offset = 0;
	for (int i = 0; i<idx; i++)
		offset += obstacles[i].numEdge;

	currentObj.incEdges(num, theta, offset);
}

void inc_node_gpu(int idx, int num, REAL *x, REAL *x0, REAL *v, REAL *a, REAL *m, REAL *n)
{
	int offset = 0;
	for (int i = 0; i<idx; i++)
		offset += obstacles[i].numNode;

	currentObj.incNodes(num, (REAL3 *)x, (REAL3 *)x0, (REAL3 *)v, a, m, (REAL3 *)n, offset);
}

void push_face_gpu(int num, void *nods, void *vrts, void *edgs, REAL *nrms, REAL *a, REAL *m, REAL *dm, REAL *idm, int *midx, int mn)
{
	_currentMesh->pushFaces(num, nods, vrts, edgs, nrms, a, m, dm, idm, midx, mn);
}

void push_vert_gpu(int num, REAL *vu, int *vn, int *adjIdx, int *adjData, int adjNum)
{
	_currentMesh->pushVertices(num, vu, vn, adjIdx, adjData, adjNum);
}

void push_edge_gpu(int num, void *n, void *f, REAL *t, REAL *l, REAL *i, REAL *r)
{
	_currentMesh->pushEdges(num, n, f, t, l, i, r);
}

void check_gpu()
{
	return;
	reportMemory();
	currentObj.dumpVtx();
}

void step_mesh_gpu(REAL dt, REAL mrt)
{
	currentCloth.stepMesh(dt);
	currentObj.stepMesh(dt);

	currentCloth.computeWSdata(mrt, true);
	currentObj.computeWSdata(mrt, true);

}

void next_step_mesh_gpu(REAL mrt)
{
	currentCloth.computeWSdata(mrt, true);
	currentObj.computeWSdata(mrt, true);

	currentCloth.updateX0();
	currentObj.updateX0();
}

static void add_external_forces(REAL dt)
{
	{
		int num = currentCloth.numNode;
		BLK_PAR(num);
		kernel_add_gravity << <B, T >> > (totalAux.dFext, currentCloth._dm, dGravity, num);
		getLastCudaError("kernel_add_gravity");
	}

	{
		int num = currentCloth.numFace;
		BLK_PAR(num);
		kernel_add_wind << <B, T >> > (
			totalAux.dFext, currentCloth._dfnod,
			currentCloth._dv, currentCloth._dfn, currentCloth.dfa,
			dWind, num);
		getLastCudaError("kernel_add_wind");
	}

	if (true) {
		int num = getGlueNum();
		if (num) {
			REAL ratio = 0.2;

			BLK_PAR(num);
			kernel_fill_glue_forces << <B, T >> >
				(totalAux.dFext, getGlues(), dt,
				currentCloth._dx, currentCloth._dv, ratio, num);
			getLastCudaError("kernel_fill_glue_forces");

		}
	}

}

#include "matrix_builder.cuh"

static void implicit_update_dynamic(int max_iter, int nn, REAL dt, bool update, REAL mrt, REAL mpt, bool jacobi, REAL damping, REAL weakening, REAL damage)
{
	bool bsr = true;

	SparseMatrixBuilder builder;

	builder.init(nn);

	builder.getColSpace(dt);
	builder.getColIndex(dt, bsr);
	CooMatrix coo;

	coo.init(builder.length(), nn, bsr);

	builder.generateIdx(coo);

	builder.fillValues(dt, coo, mrt, damping, weakening, damage);

	builder.solveJacobi(max_iter, coo);

	builder.destroy();
	coo.destroy();

	currentCloth.updateNodes(totalAux._dx, dt, update);
	// all the information about the obstacles will be packed into currentObj
	currentCloth.project_outside(totalAux._dx, totalAux._dw, currentObj._dm, currentObj._dx, mrt, mpt);
}

void update_bvs_gpu(bool is_cloth, bool ccd, REAL mrt)
{
	if (is_cloth)
		currentCloth.computeWSdata(mrt, ccd);
	else
		currentObj.computeWSdata(mrt, ccd);
}

void init_aux_gpu()
{
	totalAux.init(currentCloth.numNode, currentCloth.numFace, currentCloth.numEdge);
	totalAux.reset(currentCloth.numNode, currentCloth.numFace, currentCloth.numEdge);
}

void physics_step_gpu(int max_iter, REAL dt, REAL mrt, REAL mpt, bool dynamic, bool jacobi)
{
	totalAux.reset(currentCloth.numNode, currentCloth.numFace, currentCloth.numVert);

	add_external_forces(dt);

	implicit_update_dynamic(max_iter, currentCloth.numNode, dt, false, mrt, mpt, jacobi, m_damping, m_weakening, m_damage);

	step_mesh_gpu(dt, mrt);
}

extern "C" void save_objs_gpu(const std::string &prefix)
{
	char buffer[512];
	sprintf(buffer, "%s_ob.obj", prefix.c_str());
	currentObj.saveObj(buffer);
}

//#######################################################
#include "pair.cuh"
g_pair pairs[2];
g_pair vf_pairs, ee_pairs;

__device__ uint2 get_e(int id, bool free, uint2 *ce, uint2 *oe)
{
	return free ? ce[id] : oe[id];
}

__device__ tri3f get_t(int id, bool free, tri3f *ct, tri3f *ot)
{
	return free ? ct[id] : ot[id];
}

__device__ REAL3 get_x(int id, bool free, REAL3 *cx, REAL3 *ox)
{
	return free ? cx[id] : ox[id];
}

__device__ REAL get_m(int id, bool free, REAL *cm, REAL *om)
{
	return free ? cm[id] : om[id];
}

inline __device__ void doProximityVF(
	int vid, int fid, bool freev, bool freef,
	REAL3 *cx, tri3f *ctris, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData, REAL3 *cnn, REAL3 *cfn, REAL2 *cvu,
	REAL3 *ox, tri3f *otris, int *oAdjIdx, int *oAdjData, int *on2vIdx, int *on2vData, REAL3 *onn, REAL3 *ofn, REAL2 *ovu,
	REAL *cfa, REAL *cna,
	REAL mu, REAL mu_obs, REAL mrt, REAL mcs,
	g_IneqCon *cstrs, uint *cstrIdx)
{
	tri3f t = get_t(fid, freef, ctris, otris);
	REAL3 x1 = get_x(t.id0(), freef, cx, ox);
	REAL3 x2 = get_x(t.id1(), freef, cx, ox);
	REAL3 x3 = get_x(t.id2(), freef, cx, ox);
	REAL3 x4 = get_x(vid, freev, cx, ox);

	REAL3 n;
	REAL w[4];
	REAL d = signed_vf_distance(x4, x1, x2, x3, &n, w);
	d = abs(d);

	const REAL dmin = 2 * mrt;
	if (d >= dmin)
		return;

	bool inside = (min(min(-w[1], -w[2]), -w[3]) >= -1e-6);
	if (!inside)
		return;

	uint sidev = -1, sidef = -1;
	if (freev)
		sidev = dot(n, get_x(vid, freev, cnn, onn)) >= 0 ? 0 : 1;
	if (freef)
		sidef = dot(-n, get_x(fid, freef, cfn, ofn)) >= 0 ? 0 : 1;

	g_IneqCon ineq;
	make_vf_constraint(vid, fid, freev, freef,
		cx, ctris, cAdjIdx, cAdjData, cn2vIdx, cn2vData,
		ox, otris, oAdjIdx, oAdjData, on2vIdx, on2vData,
		cfa, cna,
		mu, mu_obs, mcs, ineq);

	if (freev) {
		ineq._sides[0] = sidev;
		ineq._dist = d;
		ineq._ids.x = vid;
		ineq._ids.y = fid;
		ineq._which = 0;
		ineq._vf = true;
		ineq._valid = true;
		addConstraint(cstrs, cstrIdx, ineq);
	}

	if (freef) {
		ineq._sides[1] = sidef;
		ineq._dist = d;
		ineq._ids.x = vid;
		ineq._ids.y = fid;
		ineq._which = 1;
		ineq._vf = true;
		ineq._valid = true;
		addConstraint(cstrs, cstrIdx, ineq);
	}
}

inline __device__ REAL3 xvpos(REAL3 x, REAL3 v, REAL t)
{
	return x + v*t;
}

inline __device__ int sgn(REAL x) { return x<0 ? -1 : 1; }

inline __device__ int solve_quadratic(REAL a, REAL b, REAL c, REAL x[2]) {
	// http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
	REAL d = b*b - 4 * a*c;
	if (d < 0) {
		x[0] = -b / (2 * a);
		return 0;
	}
	REAL q = -(b + sgn(b)*sqrt(d)) / 2;
	int i = 0;
	if (abs(a) > 1e-12*abs(q))
		x[i++] = q / a;
	if (abs(q) > 1e-12*abs(c))
		x[i++] = c / q;
	if (i == 2 && x[0] > x[1])
		fswap(x[0], x[1]);
	return i;
}

inline __device__ REAL newtons_method(REAL a, REAL b, REAL c, REAL d, REAL x0,
	int init_dir) {
	if (init_dir != 0) {
		// quadratic approximation around x0, assuming y' = 0
		REAL y0 = d + x0*(c + x0*(b + x0*a)),
			ddy0 = 2 * b + x0*(6 * a);
		x0 += init_dir*sqrt(abs(2 * y0 / ddy0));
	}
	for (int iter = 0; iter < 100; iter++) {
		REAL y = d + x0*(c + x0*(b + x0*a));
		REAL dy = c + x0*(2 * b + x0 * 3 * a);
		if (dy == 0)
			return x0;
		REAL x1 = x0 - y / dy;
		if (abs(x0 - x1) < 1e-6)
			return x0;
		x0 = x1;
	}
	return x0;
}

// solves a x^3 + b x^2 + c x + d == 0
inline __device__ int solve_cubic(REAL a, REAL b, REAL c, REAL d, REAL x[3]) {
	REAL xc[2];
	int ncrit = solve_quadratic(3 * a, 2 * b, c, xc);
	if (ncrit == 0) {
		x[0] = newtons_method(a, b, c, d, xc[0], 0);
		return 1;
	}
	else if (ncrit == 1) {// cubic is actually quadratic
		return solve_quadratic(b, c, d, x);
	}
	else {
		REAL yc[2] = { d + xc[0] * (c + xc[0] * (b + xc[0] * a)),
			d + xc[1] * (c + xc[1] * (b + xc[1] * a)) };
		int i = 0;
		if (yc[0] * a >= 0)
			x[i++] = newtons_method(a, b, c, d, xc[0], -1);
		if (yc[0] * yc[1] <= 0) {
			int closer = abs(yc[0])<abs(yc[1]) ? 0 : 1;
			x[i++] = newtons_method(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
		}
		if (yc[1] * a <= 0)
			x[i++] = newtons_method(a, b, c, d, xc[1], 1);
		return i;
	}
}

inline __device__ bool collision_test(
	const REAL3 &x0, const REAL3 &x1, const REAL3 &x2, const REAL3 &x3,
	const REAL3 &v0, const REAL3 &v1, const REAL3 &v2, const REAL3 &v3,
	ImpactType type, g_impact &imp)
{
	REAL a0 = stp(x1, x2, x3),
		a1 = stp(v1, x2, x3) + stp(x1, v2, x3) + stp(x1, x2, v3),
		a2 = stp(x1, v2, v3) + stp(v1, x2, v3) + stp(v1, v2, x3),
		a3 = stp(v1, v2, v3);

	REAL t[4];
	int nsol = solve_cubic(a3, a2, a1, a0, t);
	t[nsol] = 1; // also check at end of timestep
	for (int i = 0; i < nsol; i++) {
		if (t[i] < 0 || t[i] > 1)
			continue;

		imp._t = t[i];
		REAL3 tx0 = xvpos(x0, v0, t[i]), tx1 = xvpos(x1 + x0, v1 + v0, t[i]),
			tx2 = xvpos(x2 + x0, v2 + v0, t[i]), tx3 = xvpos(x3 + x0, v3 + v0, t[i]);
		REAL3 &n = imp._n;
		REAL *w = imp._w;
		REAL d;
		bool inside;
		if (type == I_VF) {
			d = signed_vf_distance(tx0, tx1, tx2, tx3, &n, w);
			inside = (std::fmin(-w[1], std::fmin(-w[2], -w[3])) >= -1e-6);
		}
		else {// Impact::EE
			d = signed_ee_distance(tx0, tx1, tx2, tx3, &n, w);
			inside = (std::fmin(std::fmin(w[0], w[1]), std::fmin(-w[2], -w[3])) >= -1e-6);
		}
		if (dot(n, w[1] * v1 + w[2] * v2 + w[3] * v3) > 0)
			n = -n;
		if (fabs(d) < 1e-6 && inside)
			return true;
	}
	return false;
}

#ifdef USE_DNF_FILTER
#include "dnf-filter.cuh"
#endif

inline __device__ void doImpactSelfVF(int vid, int fid, int2 *pairs, uint *pairIdx,
	bool freev, bool freef,
	REAL3 *cx, REAL3 *cx0, tri3f *ctris, 
	REAL3 *ox, REAL3 *ox0, tri3f *otris)
{
	addPair(vid, fid, pairs, pairIdx);
}

__device__ void doImpactSelfEE(int edge0, int edge1, int2 *pairs, uint *pairIdx,
	bool free0, bool free1,
	REAL3 *cx, REAL3 *cx0, uint2 *cen, 
	REAL3 *ox, REAL3 *ox0, uint2 *oen)
{
	addPair(edge0, edge1, pairs, pairIdx);
}

inline __device__ void doImpactVF(
	int vid, int fid, bool freev, bool freef,
	REAL3 *cx, REAL3 *cx0, tri3f *ctris, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData, REAL *cm,
	REAL3 *ox, REAL3 *ox0, tri3f *otris, int *oAdjIdx, int *oAdjData, int *on2vIdx, int *on2vData, REAL *om,
	REAL mu, REAL mu_obs,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx, int iii)
{
	tri3f t = get_t(fid, freef, ctris, otris);
	REAL3 x00 = get_x(vid, freev, cx0, ox0);
	REAL3 x10 = get_x(t.id0(), freef, cx0, ox0);
	REAL3 x20 = get_x(t.id1(), freef, cx0, ox0);
	REAL3 x30 = get_x(t.id2(), freef, cx0, ox0);
	REAL3 x0 = get_x(vid, freev, cx, ox);
	REAL3 x1 = get_x(t.id0(), freef, cx, ox);
	REAL3 x2 = get_x(t.id1(), freef, cx, ox);
	REAL3 x3 = get_x(t.id2(), freef, cx, ox);

	g_impact imp;
	imp._type = I_VF;
	imp._nodes[0] = vid;
	imp._nodes[1] = t.id0();
	imp._nodes[2] = t.id1();
	imp._nodes[3] = t.id2();
	imp._frees[0] = freev;
	imp._frees[1] = freef;
	imp._frees[2] = freef;
	imp._frees[3] = freef;

	REAL3 p0 = x00;
	REAL3 p1 = x10 - x00;
	REAL3 p2 = x20 - x00;
	REAL3 p3 = x30 - x00;
	REAL3 v0 = x0 - x00;
	REAL3 v1 = x1 - x10 - v0;
	REAL3 v2 = x2 - x20 - v0;
	REAL3 v3 = x3 - x30 - v0;

#ifdef  USE_DNF_FILTER
	bool ret1 = dnf_filter(x00, x10, x20, x30, x0, x1, x2, x3);
	if (!ret1) return;
#endif

	bool ret = collision_test(p0, p1, p2, p3, v0, v1, v2, v3, I_VF, imp);
	if (ret) {
		addImpact(imps, impIdx, imp);
		addNodeInfo(inodes, inIdx, vid, freev, x00, x0, get_m(vid, freev, cm, om));
		addNodeInfo(inodes, inIdx, t.id0(), freef, x10, x1, get_m(t.id0(), freef, cm, om));
		addNodeInfo(inodes, inIdx, t.id1(), freef, x20, x2, get_m(t.id1(), freef, cm, om));
		addNodeInfo(inodes, inIdx, t.id2(), freef, x30, x3, get_m(t.id2(), freef, cm, om));
	}
}

__device__ bool in_wedge(REAL w,
	int edge0, int edge1, bool free0, bool free1,
	REAL3 *cx, tri3f *ctris, uint2 *cef, uint2 *cen, REAL3 *cn,
	REAL3 *ox, tri3f *otris, uint2 *oef, uint2 *oen, REAL3 *on)
{
	int e0[2], e1[2];
	int f0[2], f1[2];

	uint2 t = get_e(edge0, free0, cen, oen);
	e0[0] = t.x, e0[1] = t.y;
	t = get_e(edge1, free1, cen, oen);
	e1[0] = t.x, e1[1] = t.y;

	t = get_e(edge0, free0, cef, oef);
	f0[0] = t.x, f0[1] = t.y;
	t = get_e(edge1, free1, cef, oef);
	f1[0] = t.x, f1[1] = t.y;

	REAL3 x = (1 - w)*get_x(e0[0], free0, cx, ox) + w*get_x(e0[1], free0, cx, ox);

	bool in = true;
	for (int s = 0; s < 2; s++) {
		int fid = f1[s];
		if (fid == -1)
			continue;

		int n0 = e1[s], n1 = e1[1 - s];
		REAL3 e = get_x(n1, free1, cx, ox) - get_x(n0, free1, cx, ox);
		REAL3 n = get_x(fid, free1, cn, on);
		REAL3 r = x - get_x(n0, free1, cx, ox);
		in &= (stp(e, n, r) >= 0);
	}
	return in;
}

__device__ void doImpactEE(
	int edge0, int edge1, bool free0, bool free1,
	REAL3 *cx, REAL3 *cx0, tri3f *ctris, uint2 *cef, uint2 *cen, REAL3 *cn, REAL *cm,
	REAL3 *ox, REAL3 *ox0, tri3f *otris, uint2 *oef, uint2 *oen, REAL3 *on, REAL *om,
	REAL mu, REAL mu_obs,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx)
{
	uint2 e0 = get_e(edge0, free0, cen, oen);
	uint2 e1 = get_e(edge1, free1, cen, oen);
	REAL3 x10 = get_x(e0.x, free0, cx0, ox0);
	REAL3 x20 = get_x(e0.y, free0, cx0, ox0);
	REAL3 x30 = get_x(e1.x, free1, cx0, ox0);
	REAL3 x40 = get_x(e1.y, free1, cx0, ox0);
	REAL3 x1 = get_x(e0.x, free0, cx, ox);
	REAL3 x2 = get_x(e0.y, free0, cx, ox);
	REAL3 x3 = get_x(e1.x, free1, cx, ox);
	REAL3 x4 = get_x(e1.y, free1, cx, ox);

	g_impact imp;
	imp._type = I_EE;
	imp._nodes[0] = e0.x;
	imp._nodes[1] = e0.y;
	imp._nodes[2] = e1.x;
	imp._nodes[3] = e1.y;
	imp._frees[0] = free0;
	imp._frees[1] = free0;
	imp._frees[2] = free1;
	imp._frees[3] = free1;

	REAL3 p0 = x10;
	REAL3 p1 = x20 - x10;
	REAL3 p2 = x30 - x10;
	REAL3 p3 = x40 - x10;
	REAL3 v0 = x1 - x10;
	REAL3 v1 = x2 - x20 - v0;
	REAL3 v2 = x3 - x30 - v0;
	REAL3 v3 = x4 - x40 - v0;

#ifdef  USE_DNF_FILTER
	bool ret1 = dnf_filter(x10, x20, x30, x40, x1, x2, x3, x4);
	if (!ret1) return;
#endif

	bool ret = collision_test(p0, p1, p2, p3, v0, v1, v2, v3, I_EE, imp);
	if (ret) {
		addImpact(imps, impIdx, imp);
		addNodeInfo(inodes, inIdx, e0.x, free0, x10, x1, get_m(e0.x, free0, cm, om));
		addNodeInfo(inodes, inIdx, e0.y, free0, x20, x2, get_m(e0.y, free0, cm, om));
		addNodeInfo(inodes, inIdx, e1.x, free1, x30, x3, get_m(e1.x, free1, cm, om));
		addNodeInfo(inodes, inIdx, e1.y, free1, x40, x4, get_m(e1.y, free1, cm, om));
	}
}

__device__ void doProximityEE(
	int edge0, int edge1, bool free0, bool free1,
	REAL3 *cx, tri3f *ctris, uint2 *cef, uint2 *cen, REAL3 *cn, REAL3 *cnn,
	REAL3 *ox, tri3f *otris, uint2 *oef, uint2 *oen, REAL3 *on, REAL3 *onn,
	REAL *cfa,
	REAL mu, REAL mu_obs, REAL mrt, REAL mcs,
	g_IneqCon *cstrs, uint *cstrIdx)
{
	uint2 e0 = get_e(edge0, free0, cen, oen);
	uint2 e1 = get_e(edge1, free1, cen, oen);
	REAL3 e00 = get_x(e0.x, free0, cx, ox);
	REAL3 e01 = get_x(e0.y, free0, cx, ox);
	REAL3 e10 = get_x(e1.x, free1, cx, ox);
	REAL3 e11 = get_x(e1.y, free1, cx, ox);

	REAL3 n;
	REAL w[4];
	REAL d = signed_ee_distance(e00, e01, e10, e11, &n, w);
	d = abs(d);

	const REAL dmin = 2 * mrt;
	if (d >= dmin)
		return;

	bool inside = min(min(w[0], w[1]), min(-w[2], -w[3])) >= -1e-6;
	if (!inside) return;

	bool i1 = in_wedge(w[1], edge0, edge1, free0, free1,
		cx, ctris, cef, cen, cn, ox, otris, oef, oen, on);
	if (!i1) return;

	bool i2 = in_wedge(-w[3], edge1, edge0, free1, free0,
		cx, ctris, cef, cen, cn, ox, otris, oef, oen, on);
	if (!i2) return;

	uint side0 = -1, side1 = -1;
	if (free0) {
		REAL3 en = get_x(e0.x, free0, cnn, onn) + get_x(e0.y, free0, cnn, onn);
		side0 = dot(n, en) >= 0 ? 0 : 1;
	}
	if (free1) {
		REAL3 en = get_x(e1.x, free1, cnn, onn) + get_x(e1.y, free1, cnn, onn);
		side1 = dot(-n, en) >= 0 ? 0 : 1;
	}

	g_IneqCon ineq;
	make_ee_constraint(edge0, edge1, free0, free1,
		cx, ctris, cef, cen,
		ox, otris, oef, oen,
		cfa,
		mu, mu_obs, mcs, ineq);

	if (free0) {
		ineq._sides[0] = side0;
		ineq._dist = d;
		ineq._ids.x = edge0;
		ineq._ids.y = edge1;
		ineq._which = 0;
		ineq._vf = false;
		ineq._valid = true;
		addConstraint(cstrs, cstrIdx, ineq);
	}

	if (free1) {
		ineq._sides[0] = side1;
		ineq._dist = d;
		ineq._ids.x = edge0;
		ineq._ids.y = edge1;
		ineq._which = 1;
		ineq._vf = false;
		ineq._valid = true;
		addConstraint(cstrs, cstrIdx, ineq);
	}
}

#define DO_VF(vid, fid, free0, free1) \
	doProximityVF(vid, fid, free0, free1, \
		cx, ctris, cAdjIdx, cAdjData, cn2vIdx, cn2vData, cnn, cfn, cvu,\
		ox, otris, oAdjIdx, oAdjData, on2vIdx, on2vData, onn, ofn, ovu,\
		cfa, cna, mu, mu_obs, mrt, mcs, cstrs, cstrIdx);

#define DO_SELF_VF(vid, fid) \
	doProximityVF(vid, fid, true, true, \
		cx, ctris, cAdjIdx, cAdjData, cn2vIdx, cn2vData, cnn, cfn, cvu,\
		cx, ctris, cAdjIdx, cAdjData, cn2vIdx, cn2vData, cnn, cfn, cvu,\
		cfa, cna, mu, mu, mrt, mcs, cstrs, cstrIdx);

#define DO_EE(e0, e1, free0, free1) \
	doProximityEE(e0, e1, free0, free1, \
		cx, ctris, cef, cen, cfn, cnn, ox, otris, oef, oen, ofn, onn, cfa, mu, mu_obs, mrt, mcs, cstrs, cstrIdx);

#define DO_SELF_EE(e0, e1) \
	doProximityEE(e0, e1, true, true, \
		cx, ctris, cef, cen, cfn, cnn, cx, ctris, cef, cen, cfn, cnn, cfa, mu, mu, mrt, mcs, cstrs, cstrIdx);

#define DO_VF_IMPACT(vid, fid, free0, free1) \
	doImpactVF(vid, fid, free0, free1, \
		cx, cx0, ctris, cAdjIdx, cAdjData, cn2vIdx, cn2vData, cm,\
		ox, ox0, otris, oAdjIdx, oAdjData, on2vIdx, on2vData, om,\
		mu, mu_obs, imps, impIdx, inodes, inIdx, 0);

#define DO_SELF_VF_IMPACT(vid, fid) \
	doImpactVF(vid, fid, true, true, \
		cx, cx0, ctris, cAdjIdx, cAdjData, cn2vIdx, cn2vData, cm,\
		cx, cx0, ctris, cAdjIdx, cAdjData, cn2vIdx, cn2vData, cm,\
		mu, mu, imps, impIdx, inodes, inIdx, 0);

#define REC_SELF_VF_IMPACT(vid, fid, pairs, pairIdx) \
	doImpactSelfVF(vid, fid, pairs, pairIdx, true, true, cx, cx0, ctris, cx, cx0, ctris);

#define DO_EE_IMPACT(e0, e1, free0, free1) \
	doImpactEE(e0, e1, free0, free1, \
		cx, cx0, ctris, cef, cen, cn, cm, ox, ox0, otris, oef, oen, on, om, mu, mu_obs, imps, impIdx, inodes, inIdx);

#define DO_SELF_EE_IMPACT(e0, e1) \
	doImpactEE(e0, e1, true, true, \
		cx, cx0, ctris, cef, cen, cn, cm, cx, cx0, ctris, cef, cen, cn, cm, mu, mu, imps, impIdx, inodes, inIdx);

#define REC_SELF_EE_IMPACT(e0, e1, pairs, pairIdx) \
	doImpactSelfEE(e0, e1, pairs, pairIdx, true, true, cx, cx0, cen, cx, cx0, cen);

inline __device__ bool VtxMask(uint *maskes, uint tri_id, uint i)
{
	return maskes[tri_id] & (0x1 << i) ? true : false;
}

inline __device__	bool EdgeMask(uint *maskes, uint tri_id, uint i)
{
	return maskes[tri_id] & (0x8 << i) ? true : false;
}

__global__ void kernelGetImpacts(
	int2 *pairs, int num,
	REAL3 *cx, REAL3 *cx0, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, REAL3 *cn, REAL *cm,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, uint *cmask, int *cn2vIdx, int *cn2vData,
	REAL3 *ox, REAL3 *ox0, tri3f *otris, tri3f *oedgs, int *oAdjIdx, int *oAdjData, uint2 *oef, uint2 *oen, REAL3 *on, REAL *om,
	g_box *ovbxs, g_box *oebxs, g_box *ofbxs, uint *omask, int *on2vIdx, int *on2vData,
	REAL mu, REAL mu_obs,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx,
	int2 *vfPairs, uint *vfPairIdx,
	int2 *eePairs, uint *eePairIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {

		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;

		int2 pair = pairs[idx];
		int fid1 = pair.x;
		int fid2 = pair.y;

		tri3f t1 = ctris[fid1];
		for (int i = 0; i<3; i++) {
			int vid = t1.id(i);
			if (VtxMask(cmask, fid1, i) && cvbxs[vid].overlaps(ofbxs[fid2])) {
#ifdef REC_VF_EE_PAIRS
				addPair(vid+1, -(fid2+1), vfPairs, vfPairIdx);
#else
				DO_VF_IMPACT(vid, fid2, true, false);
#endif
			}
		}

		tri3f t2 = otris[fid2];
		for (int i = 0; i<3; i++) {
			int vid = t2.id(i);
			if (VtxMask(omask, fid2, i) && ovbxs[vid].overlaps(cfbxs[fid1])) {
#ifdef REC_VF_EE_PAIRS
				addPair(-(vid+1), fid1+1, vfPairs, vfPairIdx);
#else
				DO_VF_IMPACT(vid, fid1, false, true);
#endif
			}
		}

		tri3f e1 = cedgs[fid1];
		tri3f e2 = oedgs[fid2];
		for (int i = 0; i<3; i++)
			for (int j = 0; j<3; j++) {
				int ee1 = e1.id(i);
				int ee2 = e2.id(j);

				if (EdgeMask(cmask, fid1, i) && EdgeMask(omask, fid2, j) && cebxs[ee1].overlaps(oebxs[ee2]))
#ifdef REC_VF_EE_PAIRS
					addPair(ee1+1, -(ee2+1), eePairs, eePairIdx);
#else
					DO_EE_IMPACT(ee1, ee2, true, false);
#endif
			}
	}
}

__global__ void kernelDoInterVFImpacts(
	int2 *pairs, int num,
	REAL3 *cx, REAL3 *cx0, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, REAL3 *cn, REAL *cm,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, uint *cmask, int *cn2vIdx, int *cn2vData,
	REAL3 *ox, REAL3 *ox0, tri3f *otris, tri3f *oedgs, int *oAdjIdx, int *oAdjData, uint2 *oef, uint2 *oen, REAL3 *on, REAL *om,
	g_box *ovbxs, g_box *oebxs, g_box *ofbxs, uint *omask, int *on2vIdx, int *on2vData,
	REAL mu, REAL mu_obs,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i < stride; i++) {

		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;

		int2 pair = pairs[idx];
		int vid = pair.x;
		int fid = pair.y;
		bool free0 = vid > 0;
		bool free1 = fid > 0;
		if (!free0) vid = -vid;
		if (!free1) fid = -fid;

		DO_VF_IMPACT(vid-1, fid-1, free0, free1);
	}
}


__global__ void kernelDoInterEEImpacts(
	int2 *pairs, int num,
	REAL3 *cx, REAL3 *cx0, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, REAL3 *cn, REAL *cm,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, uint *cmask, int *cn2vIdx, int *cn2vData,
	REAL3 *ox, REAL3 *ox0, tri3f *otris, tri3f *oedgs, int *oAdjIdx, int *oAdjData, uint2 *oef, uint2 *oen, REAL3 *on, REAL *om,
	g_box *ovbxs, g_box *oebxs, g_box *ofbxs, uint *omask, int *on2vIdx, int *on2vData,
	REAL mu, REAL mu_obs,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {

		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;

		int2 pair = pairs[idx];
		int e0 = pair.x;
		int e1 = pair.y;
		bool free0 = e0 > 0;
		bool free1 = e1 > 0;
		if (!free0) e0 = -e0;
		if (!free1) e1 = -e1;

		DO_EE_IMPACT(e0 - 1, e1 - 1, free0, free1);
	}
}

__global__ void kernelGetProximities(
	int2 *pairs, int num,
	REAL3 *cx, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, REAL3 *cfn, REAL2 *cvu,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, uint *cmask, int *cn2vIdx, int *cn2vData, REAL3 *cnn,
	REAL3 *ox, tri3f *otris, tri3f *oedgs, int *oAdjIdx, int *oAdjData, uint2 *oef, uint2 *oen, REAL3 *ofn, REAL2 *ovu,
	g_box *ovbxs, g_box *oebxs, g_box *ofbxs, uint *omask, int *on2vIdx, int *on2vData, REAL3 *onn,
	REAL *cfa, REAL *cna,
	REAL mu, REAL mu_obs, REAL mrt, REAL mcs,
	g_IneqCon *cstrs, uint *cstrIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {

		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;

		int2 pair = pairs[idx];
		int fid1 = pair.x;
		int fid2 = pair.y;

		tri3f t1 = ctris[fid1];
		for (int i = 0; i<3; i++) {
			int vid = t1.id(i);
			if (VtxMask(cmask, fid1, i) && cvbxs[vid].overlaps(ofbxs[fid2])) {
				DO_VF(vid, fid2, true, false);
			}
		}

		tri3f t2 = otris[fid2];
		for (int i = 0; i<3; i++) {
			int vid = t2.id(i);
			if (VtxMask(omask, fid2, i) && ovbxs[vid].overlaps(cfbxs[fid1])) {
				DO_VF(vid, fid1, false, true);
			}
		}

		tri3f e1 = cedgs[fid1];
		tri3f e2 = oedgs[fid2];
		for (int i = 0; i<3; i++)
			for (int j = 0; j<3; j++) {
				int ee1 = e1.id(i);
				int ee2 = e2.id(j);

				if (EdgeMask(cmask, fid1, i) && EdgeMask(omask, fid2, j) && cebxs[ee1].overlaps(oebxs[ee2]))
					DO_EE(ee1, ee2, true, false);
			}
	}
}

__global__ void kernelGetVFProximities(
	int2 *pairs, int num,
	REAL3 *cx, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, REAL3 *cfn, REAL2 *cvu,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, uint *cmask, int *cn2vIdx, int *cn2vData, REAL3 *cnn,
	REAL3 *ox, tri3f *otris, tri3f *oedgs, int *oAdjIdx, int *oAdjData, uint2 *oef, uint2 *oen, REAL3 *ofn, REAL2 *ovu,
	g_box *ovbxs, g_box *oebxs, g_box *ofbxs, uint *omask, int *on2vIdx, int *on2vData, REAL3 *onn,
	REAL *cfa, REAL *cna,
	REAL mu, REAL mu_obs, REAL mrt, REAL mcs,
	g_IneqCon *cstrs, uint *cstrIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {

		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;

		int2 pair = pairs[idx];
		int vid = pair.x;
		int fid = pair.y;

		bool free0 = vid > 0;
		bool free1 = fid > 0;
		if (!free0) vid = -vid;
		if (!free1) fid = -fid;

		DO_VF(vid-1, fid-1, free0, free1);
	}
}

__global__ void kernelGetEEProximities(
	int2 *pairs, int num,
	REAL3 *cx, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, REAL3 *cfn, REAL2 *cvu,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, uint *cmask, int *cn2vIdx, int *cn2vData, REAL3 *cnn,
	REAL3 *ox, tri3f *otris, tri3f *oedgs, int *oAdjIdx, int *oAdjData, uint2 *oef, uint2 *oen, REAL3 *ofn, REAL2 *ovu,
	g_box *ovbxs, g_box *oebxs, g_box *ofbxs, uint *omask, int *on2vIdx, int *on2vData, REAL3 *onn,
	REAL *cfa, REAL *cna,
	REAL mu, REAL mu_obs, REAL mrt, REAL mcs,
	g_IneqCon *cstrs, uint *cstrIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {
		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;

		int2 pair = pairs[idx];
		int e0 = pair.x;
		int e1 = pair.y;

		bool free0 = e0 > 0;
		bool free1 = e1 > 0;
		if (!free0) e0 = -e0;
		if (!free1) e1 = -e1;

		DO_EE(e0-1, e1-1, free0, free1);
	}
}

__device__ void doSelfVF(
	int nid, int ffid, int fv,
	REAL3 *cx, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, REAL3 *cfn,
	int *cn2vIdx, int *cn2vData, REAL2 *cvu,
	REAL mu, REAL3 *cnn, REAL *cfa, REAL *cna, REAL mrt, REAL mcs,
	g_IneqCon *cstrs, uint *cstrIdx)
{
	VLST_BEGIN(cn2vIdx, cn2vData, nid)
		FLST_BEGIN(cAdjIdx, cAdjData, vid)

		if (!covertex(ffid, fid, ctris)) {
			if (fid == fv) {
				DO_SELF_VF(nid, ffid);
			}
			else
				return;
		}

	FLST_END
		VLST_END
}

__device__ void doSelfVFImpact(
	int nid, int ffid, int fv,
	REAL3 *cx, REAL3 *cx0, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, REAL3 *cn,
	int *cn2vIdx, int *cn2vData, REAL *cm,
	REAL mu,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx,
	int2 *pairs, uint *pairIdx)
{
	VLST_BEGIN(cn2vIdx, cn2vData, nid)
		FLST_BEGIN(cAdjIdx, cAdjData, vid)

		if (!covertex(ffid, fid, ctris)) {
			if (fid == fv) {
#ifdef REC_VF_EE_PAIRS
				REC_SELF_VF_IMPACT(nid, ffid, pairs, pairIdx);
#else
				DO_SELF_VF_IMPACT(nid, ffid)
#endif
			}
			else
				return;
		}

	FLST_END
		VLST_END
}

__device__ void doSelfEEImpact(
	int e1, int e2, int f1, int f2,
	REAL3 *cx, REAL3 *cx0, tri3f *ctris, uint2 *cef, uint2 *cen, REAL3 *cn, REAL *cm,
	REAL mu,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx,
	int2 *pairs, uint *pairIdx)
{
	unsigned int e[2];
	unsigned int f[2];

	if (e1 > e2) {
		e[0] = e1, e[1] = e2;
		f[0] = f1, f[1] = f2;
	}
	else {
		e[0] = e2, e[1] = e1;
		f[0] = f2, f[1] = f1;
	}


	for (int i = 0; i<2; i++)
		for (int j = 0; j<2; j++) {
			uint2 ef0 = cef[e[0]];
			uint2 ef1 = cef[e[1]];

			uint ff1 = (i == 0) ? ef0.x : ef0.y;
			uint ff2 = (j == 0) ? ef1.x : ef1.y;

			if (ff1 == -1 || ff2 == -1)
				continue;

			if (!covertex(ff1, ff2, ctris)) {
				if (ff1 == f[0] && ff2 == f[1]) {
#ifdef REC_VF_EE_PAIRS
					REC_SELF_EE_IMPACT(e1, e2, pairs, pairIdx)
#else
					DO_SELF_EE_IMPACT(e1, e2)
#endif
				}
				else
					return;
			}
		}
}

__device__ void doSelfEE(
	int e1, int e2, int f1, int f2,
	REAL3 *cx, tri3f *ctris, uint2 *cef, uint2 *cen, REAL3 *cfn, REAL mu, REAL3 *cnn, REAL2 *cvu, REAL *cfa,
	REAL mrt, REAL mcs,
	g_IneqCon *cstrs, uint *cstrIdx)
{
	unsigned int e[2];
	unsigned int f[2];

	if (e1 > e2) {
		e[0] = e1, e[1] = e2;
		f[0] = f1, f[1] = f2;
	}
	else {
		e[0] = e2, e[1] = e1;
		f[0] = f2, f[1] = f1;
	}

	for (int i = 0; i<2; i++)
		for (int j = 0; j<2; j++) {
			uint2 ef0 = cef[e[0]];
			uint2 ef1 = cef[e[1]];

			uint ff1 = (i == 0) ? ef0.x : ef0.y;
			uint ff2 = (j == 0) ? ef1.x : ef1.y;

 			if (ff1 == -1 || ff2 == -1)
				continue;

			if (!covertex(ff1, ff2, ctris)) {
				if (ff1 == f[0] && ff2 == f[1]) {
					DO_SELF_EE(e1, e2)
				}
				else
					return;
			}
		}
}

__global__ void kernelDoVFImpacts(
	int2 *pairs, int num,
	REAL3 *cx, REAL3 *cx0, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData,
	uint2 *cef, uint2 *cen, REAL3 *cn, REAL *cm,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, REAL mu,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {
		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;
		int2 pair = pairs[idx];

		DO_SELF_VF_IMPACT(pair.x, pair.y);
	}
}

__global__ void kernelDoEEImpacts(
	int2 *pairs, int num,
	REAL3 *cx, REAL3 *cx0, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData,
	uint2 *cef, uint2 *cen, REAL3 *cn, REAL *cm,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, REAL mu,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {
		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;
		int2 pair = pairs[idx];

		DO_SELF_EE_IMPACT(pair.x, pair.y);
	}
}

__global__ void kernelGetSelfImpacts(
	int2 *pairs, int num,
	REAL3 *cx, REAL3 *cx0, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData,
	uint2 *cef, uint2 *cen, REAL3 *cn, REAL *cm,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, REAL mu,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx,
	int2 *vfPairs, uint *vfPairIdx,
	int2 *eePairs, uint *eePairIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {

		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;

		int2 pair = pairs[idx];
		int fid1 = pair.x;
		int fid2 = pair.y;

		tri3f t1 = ctris[fid1];
		for (int i = 0; i<3; i++) {
			int vid = t1.id(i);
			if (cvbxs[vid].overlaps(cfbxs[fid2])) {
				doSelfVFImpact(vid, fid2, fid1,
					cx, cx0, ctris, cedgs, cAdjIdx, cAdjData, cef, cen, cn, cn2vIdx, cn2vData, cm, mu, imps, impIdx, inodes, inIdx, vfPairs, vfPairIdx);
			}
		}

		tri3f t2 = ctris[fid2];
		for (int i = 0; i<3; i++) {
			int vid = t2.id(i);
			if (cvbxs[vid].overlaps(cfbxs[fid1])) {
				doSelfVFImpact(vid, fid1, fid2,
					cx, cx0, ctris, cedgs, cAdjIdx, cAdjData, cef, cen, cn, cn2vIdx, cn2vData, cm, mu, imps, impIdx, inodes, inIdx, vfPairs, vfPairIdx);
			}
		}

		tri3f e1 = cedgs[fid1];
		tri3f e2 = cedgs[fid2];
		for (int i = 0; i<3; i++)
			for (int j = 0; j<3; j++) {
				int ee1 = e1.id(i);
				int ee2 = e2.id(j);

				if (cebxs[ee1].overlaps(cebxs[ee2]))
					doSelfEEImpact(ee1, ee2, fid1, fid2, cx, cx0, ctris, cef, cen, cn, cm, mu, imps, impIdx, inodes, inIdx, eePairs, eePairIdx);
			}
	}
}

__global__ void kernelGetSelfProximities(
	int2 *pairs, int num,
	REAL3 *cx, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData,
	uint2 *cef, uint2 *cen, REAL3 *cfn, REAL3 *cnn, REAL2 *cvu,
	REAL *cfa, REAL *cna,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, REAL mu, REAL mrt, REAL mcs,
	g_IneqCon *cstrs, uint *cstrIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {

		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;

		int2 pair = pairs[idx];
		int fid1 = pair.x;
		int fid2 = pair.y;

		tri3f t1 = ctris[fid1];
		for (int i = 0; i<3; i++) {
			int vid = t1.id(i);
			if (cvbxs[vid].overlaps(cfbxs[fid2])) {
				doSelfVF(vid, fid2, fid1,
					cx, ctris, cedgs, cAdjIdx, cAdjData, cef, cen, cfn, cn2vIdx, cn2vData, cvu, mu, cnn, cfa, cna, mrt, mcs, cstrs, cstrIdx);
			}
		}

		tri3f t2 = ctris[fid2];
		for (int i = 0; i<3; i++) {
			int vid = t2.id(i);
			if (cvbxs[vid].overlaps(cfbxs[fid1])) {
				doSelfVF(vid, fid1, fid2,
					cx, ctris, cedgs, cAdjIdx, cAdjData, cef, cen, cfn, cn2vIdx, cn2vData, cvu, mu, cnn, cfa, cna, mrt, mcs, cstrs, cstrIdx);
			}
		}

		tri3f e1 = cedgs[fid1];
		tri3f e2 = cedgs[fid2];
		for (int i = 0; i<3; i++)
			for (int j = 0; j<3; j++) {
				int ee1 = e1.id(i);
				int ee2 = e2.id(j);

				if (cebxs[ee1].overlaps(cebxs[ee2]))
					doSelfEE(ee1, ee2, fid1, fid2, cx, ctris, cef, cen, cfn, mu, cnn, cvu, cfa, mrt, mcs, cstrs, cstrIdx);
			}
	}
}

__global__ void kernelGetSelfVFProximities(
	int2 *pairs, int num,
	REAL3 *cx, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData,
	uint2 *cef, uint2 *cen, REAL3 *cfn, REAL3 *cnn, REAL2 *cvu,
	REAL *cfa, REAL *cna,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, REAL mu, REAL mrt, REAL mcs,
	g_IneqCon *cstrs, uint *cstrIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i < stride; i++) {

		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;

		int2 pair = pairs[idx];
		int vid = pair.x;
		int fid = pair.y;

		DO_SELF_VF(vid, fid);
	}
}

__global__ void kernelGetSelfEEProximities(
	int2 *pairs, int num,
	REAL3 *cx, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData,
	uint2 *cef, uint2 *cen, REAL3 *cfn, REAL3 *cnn, REAL2 *cvu,
	REAL *cfa, REAL *cna,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, REAL mu, REAL mrt, REAL mcs,
	g_IneqCon *cstrs, uint *cstrIdx,
	int stride)
{
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i < stride; i++) {

		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;

		int2 pair = pairs[idx];
		int e0 = pair.x;
		int e1 = pair.y;

		DO_SELF_EE(e0, e1);
	}
}

int g_pair::getProximityConstraints(bool self, REAL mu, REAL mu_obs, REAL mrt, REAL mcs)
{
	int num = length();

#ifdef OUTPUT_TXT
	if (self)
		printf("self pair = %d\n", num);
	else
		printf("inter-obj pair = %d\n", num);
#endif

	if (num == 0)
		return 0;

	if (self) {
		int stride = 4;
#ifdef FIX_BT_NUM
		BLK_PAR3(num, stride, 32);
#else
		BLK_PAR3(num, stride, getBlkSize((void *)kernelGetSelfProximities));
#endif

		kernelGetSelfProximities << < B, T >> >(_dPairs, num,
			currentCloth._dx, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData, currentCloth._dn2vIdx, currentCloth._dn2vData,
			currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dn, currentCloth._dvu, currentCloth.dfa, currentCloth._da,
			currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, mu, mrt, mcs,
			Cstrs._dIneqs, Cstrs._dIneqNum,
			stride);
		getLastCudaError("kernelGetSelfProximities");
	}
	else {
		int stride = 4;
#ifdef FIX_BT_NUM
		BLK_PAR3(num, stride, 32);
#else
		BLK_PAR3(num, stride, getBlkSize((void *)kernelGetProximities));
#endif

		kernelGetProximities << < B, T >> >(_dPairs, num,
			currentCloth._dx, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData,
			currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dvu,
			currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, currentCloth._dfmask, currentCloth._dn2vIdx, currentCloth._dn2vData, currentCloth._dn,
			currentObj._dx, currentObj._dfnod, currentObj.dfedg, currentObj._dvAdjIdx, currentObj._dvAdjData,
			currentObj.def, currentObj.den, currentObj._dfn, currentObj._dvu,
			currentObj._vtxBxs, currentObj._edgeBxs, currentObj._triBxs, currentObj._dfmask, currentObj._dn2vIdx, currentObj._dn2vData, currentObj._dn,
			currentCloth.dfa, currentCloth._da,
			mu, mu_obs, mrt, mcs,
			Cstrs._dIneqs, Cstrs._dIneqNum,
			stride);
		getLastCudaError("kernelGetProximities");
	}

	int len = Cstrs.updateLength();
#ifdef OUTPUT_TXT
	printf("constraint num = %d\n", len);
#endif

	return 0;
}

int g_pair::getImpacts(bool self, REAL mu, REAL mu_obs, g_pair &vfPairs, g_pair &eePairs, int &vfLen, int &eeLen)
{
	int num = length();

#ifdef OUTPUT_TXT
	if (self)
		printf("self pair = %d\n", num);
	else
		printf("inter-obj pair = %d\n", num);
#endif

	if (num == 0)
		return 0;

	if (self) {
		int stride = 4;
#ifdef FIX_BT_NUM
		//BLK_PAR2(num, stride);
		//BLK_PAR3(num, stride, 16);
		// optimized selection...
		BLK_PAR3(num, stride, 32);
#else
		BLK_PAR3(num, stride, 32);
		//BLK_PAR3(num, stride, getBlkSize((void *)kernelGetSelfImpacts));
#endif
	
		kernelGetSelfImpacts << < B, T >> >(_dPairs, num,
			currentCloth._dx, currentCloth._dx0, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData,
			currentCloth._dn2vIdx, currentCloth._dn2vData,
			currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dm,
			currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, mu,
			Impcts._dImps, Impcts._dImpNum,
			Impcts._dNodes, Impcts._dNodeNum,
			vfPairs._dPairs, vfPairs._dIdx,
			eePairs._dPairs, eePairs._dIdx,
			stride);
		getLastCudaError("kernelGetSelfImpacts");
		cudaThreadSynchronize();

#ifdef REC_VF_EE_PAIRS
		vfLen = vfPairs.length()-vfPairs._offset;
		eeLen = eePairs.length()-eePairs._offset;
#ifdef OUTPUT_TXT
		printf("self-cd: %d vf pairs, %d ee pairs ...\n", vfLen, eeLen);
#endif

		if (vfLen) {
			int stride = 4;
#ifdef FIX_BT_NUM
			BLK_PAR3(vfLen, stride, 32);
#else
			BLK_PAR3(vfLen, stride, 32);
			//BLK_PAR3(vfLen, stride, getBlkSize((void *)kernelDoVFImpacts));
#endif

			kernelDoVFImpacts << < B, T >> >(vfPairs._dPairs + vfPairs._offset, vfLen,
				currentCloth._dx, currentCloth._dx0, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData,
				currentCloth._dn2vIdx, currentCloth._dn2vData,
				currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dm,
				currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, mu,
				Impcts._dImps, Impcts._dImpNum,
				Impcts._dNodes, Impcts._dNodeNum,
				stride);
			getLastCudaError("kernelDoVFImpacts");
		}
		if (eeLen) {
			int stride = 4;
#ifdef FIX_BT_NUM
			BLK_PAR3(eeLen, stride, 32);
#else
			BLK_PAR3(eeLen, stride, 32);
			//BLK_PAR3(eeLen, stride, getBlkSize((void *)kernelDoEEImpacts));
#endif

			kernelDoEEImpacts << < B, T >> >(eePairs._dPairs + eePairs._offset, eeLen,
				currentCloth._dx, currentCloth._dx0, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData,
				currentCloth._dn2vIdx, currentCloth._dn2vData,
				currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dm,
				currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, mu,
				Impcts._dImps, Impcts._dImpNum,
				Impcts._dNodes, Impcts._dNodeNum,
				stride);
			getLastCudaError("kernelDoEEImpacts");
		}
#endif

	}
	else {
		int stride = 4;
#ifdef FIX_BT_NUM
		//BLK_PAR2(num, stride);
		//BLK_PAR3(num, stride, 16);
		// optimized selection...
		BLK_PAR3(num, stride, 32);
#else
		BLK_PAR3(num, stride, getBlkSize((void *)kernelGetImpacts));
#endif

		kernelGetImpacts << < B, T >> >(_dPairs, num,
			currentCloth._dx, currentCloth._dx0, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData,
			currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dm,
			currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, currentCloth._dfmask, currentCloth._dn2vIdx, currentCloth._dn2vData,
			currentObj._dx, currentObj._dx0, currentObj._dfnod, currentObj.dfedg, currentObj._dvAdjIdx, currentObj._dvAdjData,
			currentObj.def, currentObj.den, currentObj._dfn, currentObj._dm,
			currentObj._vtxBxs, currentObj._edgeBxs, currentObj._triBxs, currentObj._dfmask, currentObj._dn2vIdx, currentObj._dn2vData,
			mu, mu_obs,
			Impcts._dImps, Impcts._dImpNum,
			Impcts._dNodes, Impcts._dNodeNum,
			vfPairs._dPairs + vfPairs._offset, vfPairs._dIdx,
			eePairs._dPairs + eePairs._offset, eePairs._dIdx,
			stride);
		getLastCudaError("kernelGetImpacts");

#ifdef REC_VF_EE_PAIRS
		vfLen = vfPairs.length() - vfPairs._offset;
		eeLen = eePairs.length() - eePairs._offset;
#ifdef OUTPUT_TXT
		printf("inter-cd: %d vf pairs, %d ee pairs ...\n", vfLen, eeLen);
#endif

		if (vfLen) {
			int stride = 4;
			BLK_PAR3(vfLen, stride, getBlkSize((void *)kernelDoInterVFImpacts));

			kernelDoInterVFImpacts << < B, T >> >(vfPairs._dPairs + vfPairs._offset, vfLen,
				currentCloth._dx, currentCloth._dx0, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData,
				currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dm,
				currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, currentCloth._dfmask, currentCloth._dn2vIdx, currentCloth._dn2vData,
				currentObj._dx, currentObj._dx0, currentObj._dfnod, currentObj.dfedg, currentObj._dvAdjIdx, currentObj._dvAdjData,
				currentObj.def, currentObj.den, currentObj._dfn, currentObj._dm,
				currentObj._vtxBxs, currentObj._edgeBxs, currentObj._triBxs, currentObj._dfmask, currentObj._dn2vIdx, currentObj._dn2vData,
				mu, mu_obs,
				Impcts._dImps, Impcts._dImpNum,
				Impcts._dNodes, Impcts._dNodeNum,
				stride);
			getLastCudaError("kernelDoInterVFImpacts");
		}
		if (eeLen) {
			int stride = 4;
#ifdef FIX_BT_NUM
			BLK_PAR3(eeLen, stride, 32);
#else
			BLK_PAR3(eeLen, stride, getBlkSize((void *)kernelDoInterEEImpacts));
#endif

			kernelDoInterEEImpacts << < B, T >> >(eePairs._dPairs+eePairs._offset, eeLen,
				currentCloth._dx, currentCloth._dx0, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData,
				currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dm,
				currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, currentCloth._dfmask, currentCloth._dn2vIdx, currentCloth._dn2vData,
				currentObj._dx, currentObj._dx0, currentObj._dfnod, currentObj.dfedg, currentObj._dvAdjIdx, currentObj._dvAdjData,
				currentObj.def, currentObj.den, currentObj._dfn, currentObj._dm,
				currentObj._vtxBxs, currentObj._edgeBxs, currentObj._triBxs, currentObj._dfmask, currentObj._dn2vIdx, currentObj._dn2vData,
				mu, mu_obs,
				Impcts._dImps, Impcts._dImpNum,
				Impcts._dNodes, Impcts._dNodeNum,
				stride);
			getLastCudaError("kernelDoInterEEImpacts");
		}
#endif

	}

	int len = Impcts.updateLength();
#ifdef OUTPUT_TXT
	printf("impact num = %d\n", len);
#endif
	return 0;
}

void init_pairs_gpu()
{
	pairs[0].init();
	pairs[1].init();

#ifdef REC_VF_EE_PAIRS
	vf_pairs.init();
	ee_pairs.init();
#endif
}

#include "./spatial-hashing/SpatialHashCD.h"
SpatialHashCD sphcd;

void getProximityConstraints(int vfSt, int eeSt, int vfLen, int eeLen, bool self, REAL mu, REAL mu_obs, REAL mrt, REAL mcs)
{
	if (vfLen) {
		if (self) {
			int stride = 4;
#ifdef FIX_BT_NUM
			BLK_PAR3(vfLen, stride, 32);
#else
			BLK_PAR3(vfLen, stride, getBlkSize((void *)kernelGetSelfVFProximities));
#endif
			kernelGetSelfVFProximities << < B, T >> >(vf_pairs._dPairs + vfSt, vfLen,
				currentCloth._dx, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData, currentCloth._dn2vIdx, currentCloth._dn2vData,
				currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dn, currentCloth._dvu, currentCloth.dfa, currentCloth._da,
				currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, mu, mrt, mcs,
				Cstrs._dIneqs, Cstrs._dIneqNum,
				stride);

			getLastCudaError("kernelGetSelfVFProximities");
		}
		else {
			int stride = 4;
#ifdef FIX_BT_NUM
			BLK_PAR3(vfLen, stride, 32);
#else
			BLK_PAR3(vfLen, stride, getBlkSize((void *)kernelGetVFProximities));
#endif

			kernelGetVFProximities << < B, T >> >(vf_pairs._dPairs + vfSt, vfLen,
				currentCloth._dx, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData,
				currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dvu,
				currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, currentCloth._dfmask, currentCloth._dn2vIdx, currentCloth._dn2vData, currentCloth._dn,
				currentObj._dx, currentObj._dfnod, currentObj.dfedg, currentObj._dvAdjIdx, currentObj._dvAdjData,
				currentObj.def, currentObj.den, currentObj._dfn, currentObj._dvu,
				currentObj._vtxBxs, currentObj._edgeBxs, currentObj._triBxs, currentObj._dfmask, currentObj._dn2vIdx, currentObj._dn2vData, currentObj._dn,
				currentCloth.dfa, currentCloth._da,
				mu, mu_obs, mrt, mcs,
				Cstrs._dIneqs, Cstrs._dIneqNum,
				stride);

			getLastCudaError("kernelGetVFProximities");
		}
	}

	if (eeLen) {
		if (self) {
			int stride = 4;
#ifdef FIX_BT_NUM
			BLK_PAR3(eeLen, stride, 32);
#else
			BLK_PAR3(eeLen, stride, getBlkSize((void *)kernelGetSelfEEProximities));
#endif
			kernelGetSelfEEProximities << < B, T >> >(ee_pairs._dPairs + eeSt, eeLen,
				currentCloth._dx, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData, currentCloth._dn2vIdx, currentCloth._dn2vData,
				currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dn, currentCloth._dvu, currentCloth.dfa, currentCloth._da,
				currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, mu, mrt, mcs,
				Cstrs._dIneqs, Cstrs._dIneqNum,
				stride);

			getLastCudaError("kernelGetSelfEEProximities");
		}
		else {
			int stride = 4;
#ifdef FIX_BT_NUM
			BLK_PAR3(eeLen, stride, 32);
#else
			BLK_PAR3(eeLen, stride, getBlkSize((void *)kernelGetEEProximities));
#endif

			kernelGetEEProximities << < B, T >> >(ee_pairs._dPairs + eeSt, eeLen,
				currentCloth._dx, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData,
				currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dvu,
				currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, currentCloth._dfmask, currentCloth._dn2vIdx, currentCloth._dn2vData, currentCloth._dn,
				currentObj._dx, currentObj._dfnod, currentObj.dfedg, currentObj._dvAdjIdx, currentObj._dvAdjData,
				currentObj.def, currentObj.den, currentObj._dfn, currentObj._dvu,
				currentObj._vtxBxs, currentObj._edgeBxs, currentObj._triBxs, currentObj._dfmask, currentObj._dn2vIdx, currentObj._dn2vData, currentObj._dn,
				currentCloth.dfa, currentCloth._da,
				mu, mu_obs, mrt, mcs,
				Cstrs._dIneqs, Cstrs._dIneqNum,
				stride);

			getLastCudaError("kernelGetEEProximities");
		}
	}

	int len = Cstrs.updateLength();
#ifdef OUTPUT_TXT
	printf("constraint num = %d\n", len);
#endif
}

void get_collisions_after2(REAL dt, REAL mu, REAL mu_obs, REAL mrt, REAL mcs, bool self_cd)
{
	int vfLen = vf_pairs.length();
	int eeLen = ee_pairs.length();

	TIMING_BEGIN
		Cstrs.clear();

	TIMING_BEGIN
		getProximityConstraints(0, 0, vf_pairs._offset, ee_pairs._offset, 0, mu, mu_obs, mrt, mcs);
	TIMING_END("%%%get_inter_proximitiy")
	
	TIMING_BEGIN
	if (self_cd)
		getProximityConstraints(vf_pairs._offset, ee_pairs._offset, vfLen - vf_pairs._offset, eeLen - ee_pairs._offset, 1, mu, mu_obs, mrt, mcs);
	TIMING_END("%%%get_self_proximity")

	cudaThreadSynchronize();
	Cstrs.filtering(
		totalAux._vdists, totalAux._fdists, totalAux._edists,
		currentCloth.numNode, currentCloth.numFace, currentCloth.numEdge);
	TIMING_END("$$$get_collisions_after");
}

extern void refitBVH(bool isCloth);
extern void self_collision_culling(bool ccd);

void get_collisions_gpu_SpatialHashing(REAL dt, REAL mu, REAL mu_obs, REAL mrt, REAL mcs, bool self_cd)
{
#ifdef USE_NC
	TIMING_BEGIN
	refitBVH(true);
	self_collision_culling(false);
	TIMING_END("%%% Normal Cone Tests")
#endif

	TIMING_BEGIN
	pairs[0].clear();
	pairs[1].clear();
	Cstrs.clear();

	TIMING_BEGIN
	sphcd.initFaceBoxs(currentCloth.numFace, currentCloth._triBxs, currentObj.numFace, currentObj._triBxs);
#ifdef USE_NC
	sphcd.getCollisionPair(currentCloth._dfnod, pairs[1]._dPairs, pairs[1]._dIdx, pairs[0]._dPairs, pairs[0]._dIdx, currentCloth._triParents);
#else
	sphcd.getCollisionPair(currentCloth._dfnod, pairs[1]._dPairs, pairs[1]._dIdx, pairs[0]._dPairs, pairs[0]._dIdx, NULL);
#endif

	TIMING_END("%%%get_collisions_gpu_12")

	TIMING_BEGIN
	pairs[0].getProximityConstraints(0, mu, mu_obs, mrt, mcs);
	TIMING_END("%%%get_collisions_gpu_3")

	TIMING_BEGIN
	if (self_cd) {
		pairs[1].getProximityConstraints(1, mu, mu_obs, mrt, mcs);
		cudaThreadSynchronize();
	}
	TIMING_END("%%%get_collisions_gpu_4")

	Cstrs.filtering(
		totalAux._vdists, totalAux._fdists, totalAux._edists,
		currentCloth.numNode, currentCloth.numFace, currentCloth.numEdge);

	cudaThreadSynchronize();
	TIMING_END("$$$get_collisions_gpu")
}

void get_collisions_gpu(REAL dt, REAL mu, REAL mu_obs, REAL mrt, REAL mcs, bool self_cd)
{
	get_collisions_gpu_SpatialHashing(dt, mu, mu_obs, mrt, mcs, self_cd);
}

int get_impacts_gpu(REAL dt, REAL mu, REAL mu_obs, REAL mrt, REAL mcs, bool self_cd)
{
	TIMING_BEGIN
	pairs[0].clear();
	pairs[1].clear();
	Impcts.clear();

	if (true) {
#ifdef USE_NC
		TIMING_BEGIN
		refitBVH(true);
		self_collision_culling(true);
		TIMING_END("%%% Normal Cone Tests")
#endif

		TIMING_BEGIN
		sphcd.initFaceBoxs(currentCloth.numFace, currentCloth._triBxs, currentObj.numFace, currentObj._triBxs);
#ifdef USE_NC
		sphcd.getCollisionPair(currentCloth._dfnod, pairs[1]._dPairs, pairs[1]._dIdx, pairs[0]._dPairs, pairs[0]._dIdx, currentCloth._triParents);
#else
		sphcd.getCollisionPair(currentCloth._dfnod, pairs[1]._dPairs, pairs[1]._dIdx, pairs[0]._dPairs, pairs[0]._dIdx, NULL);
#endif
		TIMING_END("%%%get_impacts_gpu_12")

#ifdef REC_VF_EE_PAIRS
		vf_pairs.clear();
		ee_pairs.clear();
#endif

		int vfLen = 0, eeLen = 0;
		TIMING_BEGIN
		pairs[0].getImpacts(0, mu, mu_obs, vf_pairs, ee_pairs, vfLen, eeLen);
		TIMING_END("%%%get_impacts_gpu_3")

		vf_pairs._offset = vfLen;
		ee_pairs._offset = eeLen;

		TIMING_BEGIN
			if (self_cd) {
				pairs[1].getImpacts(1, mu, mu_obs, vf_pairs, ee_pairs, vfLen, eeLen);
				cudaThreadSynchronize();
			}
		TIMING_END("%%%get_impacts_gpu_4")
	}

	cudaThreadSynchronize();
	TIMING_END("$$$get_impacts_gpu")

	int len = Impcts.length();

#ifdef ONE_PASS_CD
	if (len == 0) {//preparing next time step
		get_collisions_after2(dt, mu, mu_obs, mrt, mcs, self_cd);
	}
#endif

	return len;
}

void get_impact_data_gpu(void *data, void *nodes, int num)
{
	assert(num == Impcts.length());

	checkCudaErrors(cudaMemcpy(data,
		Impcts.data(), sizeof(g_impact)*num, cudaMemcpyDeviceToHost));

	if (nodes)
	checkCudaErrors(cudaMemcpy(nodes,
		Impcts.nodes(), sizeof(g_impNode)*num * 4, cudaMemcpyDeviceToHost));
}

__global__ void
kernel_updatingX(g_impNode *nodes, REAL3 *x, int num)
{
	LEN_CHK(num);

	g_impNode n = nodes[idx];

	if (n._f) {
		x[n._n] = n._x;
	}
}

__global__ void
kernel_updatingV(g_impNode *nodes, REAL3 *v, REAL dt, int num)
{
	LEN_CHK(num);

	g_impNode n = nodes[idx];

	if (n._f) {
		v[n._n] += (n._x - n._ox) / dt;
	}
}

void put_impact_node_gpu(void *data, int num, REAL mrt)
{
	// using Impcts.nodes() as a temporary buffer ...
	checkCudaErrors(cudaMemcpy(Impcts.nodes(), data, sizeof(g_impNode)*num, cudaMemcpyHostToDevice));

	{ // now updating x
		BLK_PAR(num);
		kernel_updatingX << <B, T >> >(Impcts.nodes(), currentCloth._dx, num);
		getLastCudaError("kernel_updatingX");
	}

	// updating bouding volumes
	currentCloth.computeWSdata(mrt, true);
}

void put_impact_vel_gpu(void *data, int num, REAL dt)
{
	// using Impcts.nodes() as a temporary buffer ...
	checkCudaErrors(cudaMemcpy(Impcts.nodes(), data, sizeof(g_impNode)*num, cudaMemcpyHostToDevice));

	{ // now updating x
		BLK_PAR(num);
		kernel_updatingV << <B, T >> >(Impcts.nodes(), currentCloth._dv, dt, num);
		getLastCudaError("kernel_updatingV");
	}
}

// need more
void set_cache_perf_gpu()
{
	cudaFuncSetCacheConfig(kernelGetSelfImpacts, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelGetImpacts, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelGetSelfProximities, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelGetProximities, cudaFuncCachePreferL1);

#ifdef REC_VF_EE_PAIRS
	cudaFuncSetCacheConfig(kernelDoVFImpacts, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelDoEEImpacts, cudaFuncCachePreferL1);
#endif
}

// a[i] *= b[i]
__global__ void
kernel_vector_mul(REAL *a, REAL *b, int num)
{
	LEN_CHK(num);

	a[idx] *= b[idx];
}

__global__ void
kernel_vector_mul2(REAL *a, REAL *b, REAL *c, int num)
{
	LEN_CHK(num);

	c[idx] = a[idx] * b[idx];
}

__global__ void
kernel_vector_mul3(REAL3 *a, REAL3x3 *b, REAL3 *c, int num)
{
	LEN_CHK(num);

	c[idx] = b[idx] * a[idx];
}

void cublasDvmul(REAL *a, REAL *b, REAL *c, int num)
{
	num /= 3;

	BLK_PAR(num);
	kernel_vector_mul3 << <B, T >> >((REAL3 *)a, (REAL3x3 *)b, (REAL3 *)c, num);
	getLastCudaError("kernel_vector_mul3");
}

//////////////////////////////////////////////////////////////////////////
// restart a simulation
/////////////////////////////////////////////////////////////////////////
void cuda_solver_destory();

void clear_data_gpu()
{
	Cstrs.destroy();
	Hdls.destroy();
	Glus.destroy();
	Impcts.destroy();

	checkCudaErrors(cudaFree(dMaterialStretching));
	checkCudaErrors(cudaFree(dMaterialBending));
	checkCudaErrors(cudaFree(dGravity));
	checkCudaErrors(cudaFree(dWind));

	for (int i = 0; i < numCloth; i++)
	{
		_clothes[i].destroy();
	}
	delete[]_clothes;
	numCloth = 0;

	for (int i = 0; i < numObstacles; i++)
	{
		obstacles[i].destroy();
	}
	delete[]obstacles;
	numObstacles = 0;

	currentCloth.destroy();
	currentObj.destroy();
	//initialCloth.destroy();

	totalAux.destroy();

	pairs[0].destroy();
	pairs[1].destroy();
	vf_pairs.destroy();
	ee_pairs.destroy();
	sphcd.destroy();

	cuda_solver_destory();
}

///////////////////////////////////////////////////////////////////////////////////////
// Normal Cone Culling
///////////////////////////////////////////////////////////////////////////////////////

#if 1
#include "bvh.cuh"

g_bvh coBVH[2];
g_front fronts[3];

g_cone_front cone_front[2];
int currentCF = 0, nextCF = 1;

uint *cone_front_buffer = NULL;
uint cone_front_len = 0;

extern void filtering_cone_front(g_cone_front &fIn, g_cone_front &fout, int nodeNum);

///////////////////////////////////////////////////////////

void refitBVH_Serial(bool isCloth, int length)
{
	refit_serial_kernel << <1, 1, 0 >> >
		(coBVH[isCloth]._bvh, coBVH[isCloth]._bxs, coBVH[isCloth]._triBxs,
		coBVH[isCloth]._cones, coBVH[isCloth]._triCones,
		length == 0 ? coBVH[isCloth]._num : length);

	getLastCudaError("refit_serial_kernel");
	cudaThreadSynchronize();
}

void refitBVH_Parallel(bool isCloth, int st, int length)
{
	BLK_PAR(length);

	refit_kernel << < B, T >> >
		(coBVH[isCloth]._bvh, coBVH[isCloth]._bxs, coBVH[isCloth]._triBxs,
		coBVH[isCloth]._cones, coBVH[isCloth]._triCones,
		st, length);

	getLastCudaError("refit_kernel");
	cudaThreadSynchronize();
}

void refitBVH(bool isCloth)
{
	// before refit, need to get _tri_boxes !!!!
	// copying !!!
	for (int i = coBVH[isCloth]._max_level - 1; i >= 0; i--) {
		int st = coBVH[isCloth]._level_idx[i];
		int ed = (i != coBVH[isCloth]._max_level - 1) ?
			coBVH[isCloth]._level_idx[i + 1] - 1 : coBVH[isCloth]._num - 1;

		int length = ed - st + 1;
		if (i < 5) {
			refitBVH_Serial(isCloth, length + st);
			break;
		}
		else
		{
			refitBVH_Parallel(isCloth, st, length);
		}
		//coBVH[isCloth].getBxs();
	}
}

// for deubing...
static uint *dCounting = NULL; // for counting in kernels ...
static uint hCounting[10];
static bool bvh_front_init = false;

void pushBVHContour(bool isCloth, unsigned int *ctIdx, unsigned int *ctLst,
	int ctLen, int length, int triNum)
{
	if (!isCloth) {
		coBVH[isCloth]._ctNum = 0;
		coBVH[isCloth]._ctIdx = coBVH[isCloth]._ctLst = NULL;
		coBVH[isCloth]._cones = NULL;
		coBVH[isCloth]._triCones = NULL;
		coBVH[isCloth]._ctFlags = NULL;
		coBVH[isCloth]._ctPts = NULL;
		coBVH[isCloth]._ctVels = NULL;
		return;
	}

	coBVH[isCloth]._ctNum = ctLen;

	checkCudaErrors(
		cudaMalloc((void**)&coBVH[isCloth]._ctIdx, length*sizeof(uint)));
	checkCudaErrors(
		cudaMemcpy(coBVH[isCloth]._ctIdx, ctIdx, length*sizeof(uint), cudaMemcpyHostToDevice));

	checkCudaErrors(
		cudaMalloc((void**)&coBVH[isCloth]._ctFlags, length*sizeof(bool)));
	checkCudaErrors(
		cudaMalloc((void**)&coBVH[isCloth]._ctPts, ctLen*sizeof(REAL3)));
	checkCudaErrors(
		cudaMalloc((void**)&coBVH[isCloth]._ctVels, ctLen*sizeof(REAL3)));

	checkCudaErrors(
		cudaMalloc((void**)&coBVH[isCloth]._ctLst, ctLen*sizeof(uint)));
	checkCudaErrors(
		cudaMemcpy(coBVH[isCloth]._ctLst, ctLst, ctLen*sizeof(uint), cudaMemcpyHostToDevice));

	checkCudaErrors(
		cudaMalloc((void**)&coBVH[isCloth]._cones, length*sizeof(g_cone)));

	checkCudaErrors(
		cudaMalloc((void**)&dCounting, 10 * sizeof(uint)));
}

void pushBVHIdx(int max_level, unsigned int *level_idx, bool isCloth)
{
	coBVH[isCloth]._max_level = max_level;
	coBVH[isCloth]._level_idx = new uint[max_level];
	memcpy(coBVH[isCloth]._level_idx, level_idx, sizeof(uint)*max_level);
}

void pushBVH(unsigned int length, int *ids, bool isCloth)
{
	coBVH[isCloth]._num = length;
	checkCudaErrors(cudaMalloc((void**)&coBVH[isCloth]._bvh, length*sizeof(int) * 2));
	checkCudaErrors(cudaMemcpy(coBVH[isCloth]._bvh, ids, length*sizeof(int) * 2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&coBVH[isCloth]._bxs, length*sizeof(g_box)));
	checkCudaErrors(cudaMemset(coBVH[isCloth]._bxs, 0, length*sizeof(g_box)));
	coBVH[isCloth].hBxs = NULL;

	coBVH[1]._triBxs = currentCloth._triBxs;
	coBVH[1]._triCones = currentCloth._triCones;

	coBVH[0]._triBxs = currentObj._triBxs;
	coBVH[0]._triCones = NULL;
}

void pushBVHLeaf(unsigned int length, int *idf, bool isCloth)
{
	checkCudaErrors(cudaMalloc((void**)&coBVH[isCloth]._bvh_leaf, length*sizeof(int)));
	checkCudaErrors(cudaMemcpy(coBVH[isCloth]._bvh_leaf, idf, length*sizeof(int), cudaMemcpyHostToDevice));
}


//===================================================

void pushFront(bool self, int num, unsigned int *data)
{
	g_front *f = &fronts[self];

	f->init();
	f->push(num, (uint4 *)data);
}

#define STACK_SIZE 50
#define EMPTY (nIdx == 0)

#define PUSH_PAIR(nd1, nd2)  {\
	nStack[nIdx].x = nd1;\
	nStack[nIdx].y = nd2;\
	nIdx++;\
}

#define POP_PAIR(nd1, nd2) {\
	nIdx--;\
	nd1 = nStack[nIdx].x;\
	nd2 = nStack[nIdx].y;\
}

#define NEXT(n1, n2) 	POP_PAIR(n1, n2)


inline __device__ void pushToFront(int a, int b, uint4 *front, uint *idx, uint ptr)
{
	//	(*idx)++;
	if (*idx < MAX_FRONT_NUM)
	{
		uint offset = atomicAdd(idx, 1);
		front[offset] = make_uint4(a, b, 0, ptr);
	}
}

inline __device__ void sproutingAdaptive(int left, int right,
	int *bvhA, g_box *bxsA, int *bvhB, g_box *bxsB,
	uint4 *front, uint *frontIdx,
	uint2 *pairs, uint *pairIdx, bool update, uint ptr)
{
	uint2 nStack[STACK_SIZE];
	uint nIdx = 0;

	for (int i = 0; i<4; i++)
	{
		if (isLeaf(left, bvhA) && isLeaf(right, bvhB)) {
			pushToFront(left, right, front, frontIdx, ptr);
		}
		else {
			if (!overlaps(left, right, bxsA, bxsB)) {
				pushToFront(left, right, front, frontIdx, ptr);
			}
			else {
				if (isLeaf(left, bvhA)) {
					PUSH_PAIR(left, getLeftChild(right, bvhB));
					PUSH_PAIR(left, getRightChild(right, bvhB));
				}
				else {
					PUSH_PAIR(getLeftChild(left, bvhA), right);
					PUSH_PAIR(getRightChild(left, bvhA), right);
				}
			}
		}

		if (EMPTY)
			return;

		NEXT(left, right);
	}

	while (!EMPTY) {
		NEXT(left, right);
		pushToFront(left, right, front, frontIdx, ptr);
	}
}

inline __device__ void sprouting(int left, int right,
	int *bvhA, g_box *bxsA, int *bvhB, g_box *bxsB,
	uint4 *front, uint *frontIdx,
	int2 *pairs, uint *pairIdx, bool update, uint ptr)
{
	uint2 nStack[STACK_SIZE];
	uint nIdx = 0;

	while (1)
	{
		if (isLeaf(left, bvhA) && isLeaf(right, bvhB)) {
			if (update)
				pushToFront(left, right, front, frontIdx, ptr);

			if (overlaps(left, right, bxsA, bxsB))
				addPair(getTriID(left, bvhA), getTriID(right, bvhB), pairs, pairIdx);
		}
		else {
			if (!overlaps(left, right, bxsA, bxsB)) {
				if (update)
					pushToFront(left, right, front, frontIdx, ptr);

			}
			else {
				if (isLeaf(left, bvhA)) {
					PUSH_PAIR(left, getLeftChild(right, bvhB));
					PUSH_PAIR(left, getRightChild(right, bvhB));
				}
				else {
					PUSH_PAIR(getLeftChild(left, bvhA), right);
					PUSH_PAIR(getRightChild(left, bvhA), right);
				}
			}
		}

		if (EMPTY)
			return;

		NEXT(left, right);
	}
}

inline __device__ void doPropogate(
	uint4 *front, uint *frontIdx, int num,
	int *bvhA, g_box *bxsA, int bvhAnum,
	int *bvhB, g_box *bxsB, int bvhBnum,
	int2 *pairs, uint *pairIdx, bool update, tri3f *Atris, int idx, bool *flags)
{
	uint4 node = front[idx];
	if (node.z != 0) {
#if defined(_DEBUG) || defined(OUTPUT_TXT)
		atomicAdd(frontIdx + 1, 1);
#endif
		return;
	}

#ifdef USE_NC
	if (flags != NULL && flags[node.w] == 0) {
#if defined(_DEBUG) || defined(OUTPUT_TXT)
		atomicAdd(frontIdx + 2, 1);
#endif
		return;
	}
#endif

	uint left = node.x;
	uint right = node.y;

	if (isLeaf(left, bvhA) && isLeaf(right, bvhB)) {
		if (overlaps(left, right, bxsA, bxsB))
			if (bvhA != bvhB)
				addPair(getTriID(left, bvhA), getTriID(right, bvhB), pairs, pairIdx);
			else { // for self ccd, we need to remove adjacent triangles, they will be processed seperatedly with orphan set
				if (!covertex(getTriID(left, bvhA), getTriID(right, bvhB), Atris))
					addPair(getTriID(left, bvhA), getTriID(right, bvhB), pairs, pairIdx);
			}
			return;
	}

	if (!overlaps(left, right, bxsA, bxsB))
		return;

	if (update)
		front[idx].z = 1;

	int ptr = node.w;
	if (isLeaf(left, bvhA)) {
		sprouting(left, getLeftChild(right, bvhB), bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr);
		sprouting(left, getRightChild(right, bvhB), bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr);
	}
	else {
		sprouting(getLeftChild(left, bvhA), right, bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr);
		sprouting(getRightChild(left, bvhA), right, bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr);
	}
}

__global__ void kernelPropogate(uint4 *front, uint *frontIdx, int num,
	int *bvhA, g_box *bxsA, int bvhAnum,
	int *bvhB, g_box *bxsB, int bvhBnum,
	int2 *pairs, uint *pairIdx, bool update, tri3f *Atris, int stride, bool *flags)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {
		int j = idx*stride + i;
		if (j >= num)
			return;

		doPropogate(front, frontIdx, num,
			bvhA, bxsA, bvhAnum, bvhB, bxsB, bvhBnum, pairs, pairIdx, update, Atris, j, flags);
	}
}

int g_front::propogate(bool &update, bool self, bool ccd)
{
	uint dummy[1];
	cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
#ifdef OUTPUT_TXT
	printf("Before propogate, length = %d\n", dummy[0]);
#endif

#if defined(_DEBUG) || defined(OUTPUT_TXT)
	uint dummy2[5] = { 0, 0, 0, 0, 0 };
	cutilSafeCall(cudaMemcpy(_dIdx + 1, dummy2, 5 * sizeof(int), cudaMemcpyHostToDevice));
#endif

	if (dummy[0] != 0) {
		g_bvh *pb1 = &coBVH[1];
		g_bvh *pb2 = (self) ? &coBVH[1] : &coBVH[0];
		tri3f *faces = (self ? currentCloth._dfnod : NULL);

		int stride = 4;
#ifdef FIX_BT_NUM
		BLK_PAR2(dummy[0], stride);
#else
		BLK_PAR3(dummy[0], stride, getBlkSize((void *)kernelPropogate));
#endif

		kernelPropogate << < B, T >> >
			(_dFront, _dIdx, dummy[0],
			pb1->_bvh, pb1->_bxs, pb1->_num,
			pb2->_bvh, pb2->_bxs, pb2->_num,
			pairs[self]._dPairs, pairs[self]._dIdx, update, faces, stride, self ? coBVH[1]._ctFlags : NULL);
		//pairs[self]._dPairs, pairs[self]._dIdx, update, faces, stride, (self && !ccd) ? coBVH[1]._ctFlags : NULL);

		cudaThreadSynchronize();
		getLastCudaError("kernelPropogate");
	}

	cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
#ifdef OUTPUT_TXT
	printf("After propogate, length = %d\n", dummy[0]);
#endif

#if defined(_DEBUG) || defined(OUTPUT_TXT)
	cutilSafeCall(cudaMemcpy(dummy2, _dIdx + 1, 5 * sizeof(int), cudaMemcpyDeviceToHost));
	printf("Invalid = %d, NC culled = %d\n", dummy2[0], dummy2[1]);
#endif

	if (update && dummy[0] > SAFE_FRONT_NUM) {
		printf("Too long front, stop updating ...\n");
		update = false;
	}

	if (dummy[0] > MAX_FRONT_NUM) {
		printf("Too long front, exiting ...\n");
		exit(0);
	}
	return dummy[0];
}

#define CONE_STACK_SIZE 50

//#define PUSH_NODE(nd) { nStack[nIdx++] = nd;}
#define PUSH_NODE(nd) {\
	if (nIdx >= CONE_STACK_SIZE)\
		printf("cone stack full!!!!\n");\
		else\
		nStack[nIdx++] = nd;\
}

#define POP_NODE(nd) {nd = nStack[--nIdx];}
#define NEXT_NODE(nd) POP_NODE(nd)

inline __device__ void pushToFront(uint nd, uint3 *front, uint *idx, uint prt)
{
	uint offset = atomicAdd(idx, 1);
	front[offset] = make_uint3(nd, prt, 0);
}

inline __device__ void sproutingCone(uint node,
	int *bvhA, uint3 *front, uint *frontIdx, uint ptr,
	g_cone *bvh_cones, uint *ctIdx, uint *ctLst, REAL3 *ctPts, REAL3 *ctVels, int ctNum,
	REAL3 *x, REAL3 *ox, bool ccd)
{
	uint nStack[CONE_STACK_SIZE];
	uint nIdx = 0;

	while (1)
	{
		if (isLeaf(node, bvhA)) {
			pushToFront(node, front, frontIdx, ptr);
		}
		else {
			if (false == cone_test(node, bvh_cones, ctIdx, ctLst,
				ctPts, ctVels, ctNum, x, ox, ccd))
				pushToFront(node, front, frontIdx, ptr);
			else {
				PUSH_NODE(getLeftChild(node, bvhA));
				PUSH_NODE(getRightChild(node, bvhA));
			}
		}

		if (EMPTY)
			return;

		NEXT_NODE(node);
	}
}

inline __device__ void doConePropogate(
	uint3 *front, uint *frontIdx, int *bvhA, int idx,
	g_cone *bvh_cones, uint *ctIdx, uint *ctLst, REAL3 *ctPts, REAL3 *ctVels, int ctNum,
	REAL3 *x, REAL3 *ox, bool ccd)

{
	uint3 node = front[idx];
	if (node.z != 0)
		return;

	uint current = node.x;
	uint parent = node.y; // useless ...

	if (isLeaf(current, bvhA))
	{
		return;
	}

	if (false == cone_test(current, bvh_cones, ctIdx, ctLst,
		ctPts, ctVels, ctNum, x, ox, ccd))
	{
		//perfect, just skip this node.
		return;
	}

	//need to test its childern
	front[idx].z = 1;

	sproutingCone(getLeftChild(current, bvhA),
		bvhA, front, frontIdx, current,
		bvh_cones, ctIdx, ctLst, ctPts, ctVels, ctNum, x, ox, ccd);
	sproutingCone(getRightChild(current, bvhA),
		bvhA, front, frontIdx, current,
		bvh_cones, ctIdx, ctLst, ctPts, ctVels, ctNum, x, ox, ccd);
}

inline __device__ void doConeFlags(
	uint3 *front, int *bvhA, int bvhAnum, int idx, bool *flags)
{
	uint3 node = front[idx];
	if (node.z != 0)
		return;

	if (node.x == 10)
		printf("hello!\n");

	uint parent = getParent(node.x, bvhA, bvhAnum);
	while (parent != -1) {
		flags[parent] = true;
		parent = getParent(parent, bvhA, bvhAnum);
	}
}

inline __device__ void doConeSHFlags(uint3 *front, int idx, bool *flags)
{
	uint3 node = front[idx];
	if (node.z != 0)
		return;

	flags[node.x] = true;
}

inline __device__ void doConeFrontFiltering(uint3 *frontA, uint *refs, int idx, uint3 *frontB, uint *frontBIdx, int *bvhA, int bvhAnum)
{
	uint3 node = frontA[idx];

	if (node.z != 0)
		return;

	uint parent = getParent(node.x, bvhA, bvhAnum);
	if (parent == -1)
		return;

	if (refs[parent] == 2) {
		if (getLeftChild(parent, bvhA) == node.x) {
			pushToFront(parent, frontB, frontBIdx, 0);
		}
	}
	else {
		pushToFront(node.x, frontB, frontBIdx, 0);
	}
}

__global__ void kernelConeFrontFiltering(uint3 *frontA, uint *refs, uint3 *frontB, uint *frontBIdx, int num, int stride, int *bvhA, int bvhAnum)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {
		int j = idx*stride + i;
		if (j >= num)
			return;

		doConeFrontFiltering(frontA, refs, j, frontB, frontBIdx, bvhA, bvhAnum);
	}
}

inline __device__ void doConeFrontMark(uint3 *front, uint *refs, int idx, int *bvhA, int bvhAnum)
{
	uint3 node = front[idx];

	if (node.z != 0)
		return;

	uint parent = getParent(node.x, bvhA, bvhAnum);
	if (parent != -1) {
		atomicAdd(refs + parent, 1);
	}
}

__global__ void kernelConeFrontMark(uint3 *front, uint *refs, int num, int stride, int *bvhA, int bvhAnum)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {
		int j = idx*stride + i;
		if (j >= num)
			return;

		doConeFrontMark(front, refs, j, bvhA, bvhAnum);
	}
}

__global__ void kernelConePropogate(uint3 *front, uint *frontIdx, int num,
	int *bvhA, int stride,
	g_cone *bvh_cones, uint *ctIdx, uint *ctLst, REAL3 *ctPts, REAL3 *ctVels, int ctNum,
	REAL3 *x, REAL3 *ox, bool ccd)

{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//if (num == 7)
	//	printf("hereG!");

	for (int i = 0; i<stride; i++) {
		int j = idx*stride + i;
		if (j >= num)
			return;

		doConePropogate(front, frontIdx, bvhA, j,
			bvh_cones, ctIdx, ctLst, ctPts, ctVels, ctNum, x, ox, ccd);
	}
}

__global__ void kernelConeFlags(uint3 *front, int num,
	int *bvhA, int bvhAnum, int stride, bool *flags)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {
		int j = idx*stride + i;
		if (j >= num)
			return;

		doConeFlags(front, bvhA, bvhAnum, j, flags);
	}
}

__global__ void kernelConeSHFlags(uint3 *front, int num, int stride, bool *flags)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {
		int j = idx*stride + i;
		if (j >= num)
			return;

		doConeSHFlags(front, j, flags);
	}
}

inline __device__ void doSetParent(int *leaf, bool *flags, int *bvhA, int bvhAnum, int idx, int *parent, int frontLen)
{
	int node = leaf[idx];
	parent[idx] = frontLen + idx + 100;

	uint p = getParent(node, bvhA, bvhAnum);
	while (p != -1) {
		if (flags[p] == true) {
			parent[idx] = p + 1;
			break;
		}
		p = getParent(p, bvhA, bvhAnum);
	}
}

__global__ void kernelSetParents(int *leaf, bool *flags, int *parent, int *bvhA, int bvhAnum, int num, int stride, int frontLen)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {
		int j = idx*stride + i;
		if (j >= num)
			return;

		doSetParent(leaf, flags, bvhA, bvhAnum, j, parent, frontLen);
	}
}

int g_cone_front::propogate(bool ccd)
{
	uint dummy[1];
	cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));

	if (dummy[0] != 0) {
		int stride = 4;
		BLK_PAR2(dummy[0], stride);

		g_bvh *pb1 = &coBVH[1];
		REAL3 *x = currentCloth._dx;
		REAL3 *ox = currentCloth._dx0;

		kernelConePropogate << < B, T >> >
			(_dFront, _dIdx, dummy[0],
			pb1->_bvh, stride,
			pb1->_cones, pb1->_ctIdx, pb1->_ctLst,
			pb1->_ctPts, pb1->_ctVels, pb1->_ctNum, x, ox, ccd);

		cudaThreadSynchronize();
		getLastCudaError("kernelConePropogate");
	}

	cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
	//set flags ...
	if (dummy[0] != 0) {
		int stride = 4;
		BLK_PAR2(dummy[0], stride);

		g_bvh *pb1 = &coBVH[1];
		bool *flags = pb1->_ctFlags;

#ifdef USE_NC
		kernelConeSHFlags << < B, T >> >(_dFront, dummy[0], stride, flags);
#else
		kernelConeFlags << < B, T >> >
			(_dFront, dummy[0],
			pb1->_bvh, pb1->_num, stride, flags);
#endif

		cudaThreadSynchronize();
		getLastCudaError("kernelConeFlags");
	}

#ifdef USE_NC
	//set leaf parents ...
	{
		int stride = 4;
		int length = currentCloth.numFace;
		g_bvh *pb1 = &coBVH[1];

		BLK_PAR2(length, stride);
		kernelSetParents << <B, T >> >(pb1->_bvh_leaf, pb1->_ctFlags, currentCloth._triParents, pb1->_bvh, pb1->_num, length, stride, dummy[0]);
		cudaThreadSynchronize();
		getLastCudaError("kernelSetParents");
	}
#endif

	return dummy[0];
}



//static thrust::device_ptr<bool> g_flags;
void self_collision_culling(bool ccd)
{
	bool isCloth = true;

	checkCudaErrors(cudaMemset(dCounting, 0, 10 * sizeof(uint)));
	coBVH[isCloth].selfCollisionCulling(currentCloth._dx, currentCloth._dx0, ccd, dCounting);
	checkCudaErrors(cudaMemcpy(hCounting, dCounting, sizeof(uint) * 10, cudaMemcpyDeviceToHost));
#ifdef OUTPUT_TXT
	printf("Marked %d nodes ...\n", hCounting[0]);
#endif

#define FILTER_CF
#ifdef FILTER_CF
	filtering_cone_front(cone_front[currentCF], cone_front[nextCF], coBVH[isCloth]._num);
#endif

#ifdef USE_NC
	if (ccd) {
		if (cone_front_buffer == NULL)
			cone_front_buffer = new uint[102400 * 3];

		uint dummy[1];
		cutilSafeCall(cudaMemcpy(dummy, cone_front[currentCF]._dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
		cone_front_len = min(dummy[0], 102400);
		checkCudaErrors(cudaMemcpy(cone_front_buffer, cone_front[currentCF]._dFront, sizeof(uint3) * cone_front_len, cudaMemcpyDeviceToHost));
	}
#endif

}

// Predicate functor
struct valid_front_node
{
	inline __host__ __device__
		bool operator() (const uint4 x)
	{
		return x.z == 0;
		//return x.z == 0 && g_flags[x.w] != 0;
	}
};

void filtering_cone_front(g_cone_front &fIn, g_cone_front &fout, int nodeNum)
{
	static uint *refs = NULL;

	uint dummy[1];
	cutilSafeCall(cudaMemcpy(dummy, fIn._dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
	int len = dummy[0];

	if (len == 1)
		return;

	if (refs == NULL) {
		checkCudaErrors(cudaMalloc(&refs, nodeNum*sizeof(uint)));
	}
	checkCudaErrors(cudaMemset(refs, 0, nodeNum*sizeof(uint)));

	int stride = 4;

	g_bvh *pb1 = &coBVH[1];

	{
		BLK_PAR3(len, stride, getBlkSize((void *)kernelConeFrontMark));

		kernelConeFrontMark << < B, T >> >
			(fIn._dFront, refs, len, stride, pb1->_bvh, pb1->_num);

		cudaThreadSynchronize();
		getLastCudaError("kernelConeFrontMark");
	}

	{
		uint dummy = 0;
		checkCudaErrors(cudaMemcpy(fout._dIdx, &dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));

		BLK_PAR3(len, stride, getBlkSize((void *)kernelConeFrontFiltering));

		kernelConeFrontFiltering << <B, T >> >
			(fIn._dFront, refs, fout._dFront, fout._dIdx, len, stride, pb1->_bvh, pb1->_num);
		cudaThreadSynchronize();
		getLastCudaError("kernelConeFrontFiltering");
	}

	::swap(currentCF, nextCF);
}

void
g_bvh::selfCollisionCulling(REAL3 *x, REAL3 *ox, bool ccd, uint *counting)
{
	checkCudaErrors(cudaMemset(_ctFlags, 0, _num*sizeof(bool)));

	if (false == bvh_front_init) {
		bvh_front_init = true;
		cone_front[currentCF].init(_num / 2 + 1);
		cone_front[nextCF].init(_num / 2 + 1);
	}

	int len = cone_front[currentCF].propogate(ccd);
}

#endif
