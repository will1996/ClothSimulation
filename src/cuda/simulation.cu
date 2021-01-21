//atomicAdd(XX, 1) -> atomicInc !!!!!!!

// CUDA Runtime
#include <cuda_runtime.h>

#include <cuda_profiler_api.h>
#include <assert.h>

// Utilities and system includes
#include "helper_functions.h"  // helper for shared functions common to CUDA SDK samples
#include "helper_cuda.h"       // helper for CUDA error checking

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

bool enable_collision = true;

// for deubing...
// removed later ...
static double3 *hf;
static double *hm;
static int *hr;
static double3 *hb;
static double9 *hF;
static double9x9 *hJ;
static double12 *hhF;
static double12x12 *hhJ;

int iii=0;

extern void input2(const char *fname, double *data, int len);

extern void output2(const char *fname, double *data, int len);
extern void output1(const char *fname, double *data, int len);
extern void output(char *fname, double *data, int len);
extern void output9(char *fname, double *data, int len);
extern void output9x9(char *fname, double *data, int len);
extern void output12(char *fname, double *data, int len);
extern void output12x12(char *fname, double *data, int len);

extern void CheckI(int *p, int N);

typedef unsigned int uint;

typedef struct {
	uint3 _ids;
	
	inline __device__ __host__ uint id0() const {return _ids.x;}
	inline __device__ __host__ uint id1() const {return _ids.y;}
	inline __device__ __host__ uint id2() const {return _ids.z;}
	inline __device__ __host__ uint id(int i) const { return (i==0 ? id0() : ((i == 1) ? id1() : id2()));  }
} tri3f;

inline __device__ bool covertex(int tA, int tB, tri3f *Atris)
{
		for (int i=0; i<3; i++)
			for (int j=0; j<3; j++) {
				if (Atris[tA].id(i) == Atris[tB].id(j))
					return true;
			}

		return false;
}

typedef struct {
	uint4 _ids;
	
	inline __device__ __host__ uint id0() const {return _ids.x;}
	inline __device__ __host__ uint id1() const {return _ids.y;}
	inline __device__ __host__ uint id2() const {return _ids.z;}
	inline __device__ __host__ uint id3() const {return _ids.w;}
	inline __device__ __host__ uint id(int i) const {
		return (i==3 ? id3() : (i==0 ? id0() : ((i == 1) ? id1() : id2())));  }
} tri4f;

#define MaterialDumping 0.0
//(*::materials)[face->label]->weakening
#define MaterialWeakening 0.0
//face->damage
#define FaceDamage 0.0

// should get from magic::
//#define magic_repulsion_thickness 0.001
//#define magic_projection_thickness 0.0001
//#define magic_collision_stiffness 1.0e9
#define double_infinity 1.0e30

#include "constraint.cuh"
g_constraints Cstrs;
g_handles Hdls;

#include "impact.cuh"
g_impacts Impcts;

__global__ void
kernel_project_outside1(g_IneqCon *cons, double3 *dx, double *w, int num,
						double *cm, double *om, double3 *cx, double3 *ox, double mrt, double mpt)
{
	LEN_CHK(num);

	g_IneqCon *c = cons+idx;
	if (false == c->_valid)
		return;

	MeshGrad dxc;
	c->project(dxc, cm, om, cx, ox, mrt, mpt);
	for (int i=0; i<4; i++) {
		if (c->free[i]) {
			int id = c->nodes[i];

			double wn = norm2(dxc[i]);
			//w[id] += wn;
			//dx[id] += wn*dxc[i];

			atomicAdd(w+id, wn);
			atomicAdd(dx+id, wn*dxc[i]);
		}
	}
}

__global__ void
kernel_project_outside2(double3 *x, double3 *dx, double *w, int num)
{
	LEN_CHK(num);

	if (w[idx] == 0)
		return;

	x[idx] += dx[idx]/w[idx];
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

void push_eq_cstr_gpu(int nid, double *x, double *n, double stiff, int id)
{
	EqCon data(nid, (double3 *)x, (double3 *)n, stiff);
	//data.print();

	checkCudaErrors(cudaMemcpy(Hdls.data()+id, &data, sizeof(EqCon), cudaMemcpyHostToDevice));
}

inline int getConstraintNum()
{
	return Cstrs.length();
}

inline g_IneqCon *getConstraints()
{
	return Cstrs.data();
}

struct StretchingSamples {double2x2 s[40][40][40];};
struct BendingData {double d[3][5];};

struct Wind {
    double density;
    double3 velocity;
    double drag;
};

struct Gravity {
	double3 _g;
};

//(*::materials)[face->label]->stretching

StretchingSamples *dMaterialStretching;
BendingData *dMaterialBending;
Wind *dWind;
Gravity *dGravity;

extern void gpuSolver(int cooN, int *cooRowInd, int *cooColInd, double *cooData, bool bsr,
	double *dSolverB, double *dSolverX, int vtxN);


static const int nsamples = 30;

inline __host__  ostream &operator<< (ostream &out, const tri3f &tri)
{
//	out << "f " << tri.id0() << " " << tri.id1() << " " << tri.id2() << endl;
	out << "f " << tri._ids.x+1 << " " << tri._ids.y+1 << " " << tri._ids.z+1 << endl;
	return out;
}

// return locations for bsr matrix
inline __device__
int get_indices(int i, int j, int *inc)
{
	int beginIdx = (i == 0) ? 0 : inc[i - 1];
	return (beginIdx + j) * 9;
}

// return locations for each sub matrix at (i, j)
inline __device__
void get_indices(int i, int j, int loc[], int *inc)
{
	int beginIdx = (i == 0) ? 0 : inc[i-1]*9;
	int curLoc = j*3;
	int lineOffset = (i == 0) ? inc[0] : inc[i]-inc[i-1];
	lineOffset *= 3;

	loc[0] = beginIdx+curLoc;
	loc[1] = loc[0]+1;
	loc[2] = loc[0]+2;

	loc[3] = loc[0]+lineOffset;
	loc[4] = loc[3]+1;
	loc[5] = loc[3]+2;

	loc[6] = loc[3]+lineOffset;
	loc[7] = loc[6]+1;
	loc[8] = loc[6]+2;
}

__global__ void
kernel_generate_idx(int **matIdx, int *rowInc, int *rows, int *cols, int num)
{
	LEN_CHK(num);

	int *here = matIdx[idx];
	int len = (idx == 0) ? rowInc[0] : rowInc[idx] - rowInc[idx-1];

	for (int l=0; l<len; l++) {
		int locs[9];
		get_indices(idx, l, locs, rowInc);

		int s = 0, r=idx*3, c=here[l]*3;
		rows[locs[s]] = r;
		cols[locs[s++]] = c;
		rows[locs[s]] = r;
		cols[locs[s++]] = c+1;
		rows[locs[s]] = r;
		cols[locs[s++]] = c+2;

		rows[locs[s]] = r+1;
		cols[locs[s++]] = c;
		rows[locs[s]] = r+1;
		cols[locs[s++]] = c+1;
		rows[locs[s]] = r+1;
		cols[locs[s++]] = c+2;

		rows[locs[s]] = r+2;
		cols[locs[s++]] = c;
		rows[locs[s]] = r+2;
		cols[locs[s++]] = c+1;
		rows[locs[s]] = r+2;
		cols[locs[s++]] = c+2;
	}
}

// moved sorted & unique indices into citmes ...
__global__ void
kernel_compress_idx(int **matIdx, int *cItems, int *rowInc, int num)
{
	LEN_CHK(num);

	int loc = rowInc[idx];
	int *src = matIdx[idx];
	int len = (idx == 0) ? loc : loc - rowInc[idx-1];
	int *dst = cItems+ ((idx==0) ? 0 : rowInc[idx-1]);

	for (int i=0; i<len; i++)
		dst[i] = src[i];
}

__global__ void
kernel_update_nodes(double3 *v, double3 *x, double3 *dv, double dt, bool update, int num)
{
	LEN_CHK(num);

	v[idx] += dv[idx];

	if (update)
		x[idx] += v[idx]*dt;

	//acc[idx] = dv[idx]/dt;
}


inline __device__ void
_sort(int *data, int num)
{
	// bubble sort
	for (int i=0; i<num-1; i++) {
		for (int j=num-1; j>i; j--) {
			if (data[j] < data[j-1]) {
				int tmp = data[j];
				data[j] = data[j-1];
				data[j-1] = tmp;
			}
		}
	}
}

inline __device__ int
_unique(int *data, int num)
{
	int loc=0;
	for (int i=1; i<num; i++) {
		if (data[i] == data[loc])
			continue;
		else
			data[++loc] = data[i];
	}

	return loc+1;
}

// input: unsorted array (data, num)
// ouput: in place sorted & unique array (data), return its really length
inline __device__ int
gpu_sort(int *data, int num)
{
	_sort(data, num);
	return _unique(data, num);
}

__global__ void
kernel_sort_idx(int **mIdx, int *col, int *inc, int num)
{
	LEN_CHK(num);

	inc[idx] = gpu_sort(mIdx[idx], (idx==0 ? col[0] : col[idx]-col[idx-1]));
}

__global__ void
kernel_set_matIdx(int **mIdx, int *col, int *items, int num)
{
	LEN_CHK(num);

	 mIdx[idx] = items + ((idx == 0) ? 0 : col[idx-1]);
}

__device__ void mat_add(int i, int j, int *rowIdx, int **matIdx)
{
	int index=atomicAdd(&rowIdx[i],1);
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
kernel_add_face_forces (tri3f *faces, int *colLen, int *rowIdx, int **matIdx, int num, bool counting)
{
	LEN_CHK(num);

	tri3f *t = faces+idx;
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
kernel_add_constraint_forces (g_IneqCon *cstrs, int *colLen, int *rowIdx, int **matIdx, int num, bool counting)
{
	LEN_CHK(num);

	g_IneqCon *cstr = cstrs+idx;
	if (false == cstr->_valid)
		return;

	for (int i=0; i<4; i++) {
		if (!cstr->free[i]) continue;
		int ni = cstr->nodes[i];

		for (int j=0; j<4; j++) {
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

/*
__global__ void
kernel_add_friction_forces (g_IneqCon *cstrs, int *colLen, int *rowIdx, int **matIdx, int num, bool counting)
{
	LEN_CHK(num);

	g_IneqCon *cstr = cstrs+idx;
	MeshHess jac;
    MeshGrad force;
	cstr->friction(dt, jac, force);
	...
}
*/

/*
Vert *edge_opp_vert (const Edge *edge, int side) {
    Face *face = (Face*)edge->adjf[side];
    if (!face)
        return NULL;
    for (int j = 0; j < 3; j++)
        if (face->v[j]->node == edge->n[side])
            return face->v[PREV(j)];
    return NULL;
}


Vert *edge_vert (const Edge *edge, int side, int i) {
    Face *face = (Face*)edge->adjf[side];
    if (!face)
        return NULL;
    for (int j = 0; j < 3; j++)
        if (face->v[j]->node == edge->n[i])
            return face->v[j];
    return NULL;
}
*/

inline __device__ int edge_opp_node(tri3f *faces, int ef, int en) {
	tri3f *f =faces+ef;

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
kernel_add_edge_forces (uint2 *ens, uint2 *efs, tri3f *faces, int *colLen, int *rowIdx, int **matIdx, int num, bool counting)
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
	int len = (i == 0) ? rowInc[0] : rowInc[i]-rowInc[i-1];

	for (int l=0; l<len; l++)
		if (matIdx[i][l] == v)
			return l;

	assert(0);
	return -1;
}

inline __device__
void add_val(int loc, double *vals, const double3x3 &v)
{
	atomicAdd(vals + loc + 0, getIJ(v, 0, 0));
	atomicAdd(vals + loc + 1, getIJ(v, 0, 1));
	atomicAdd(vals + loc + 2, getIJ(v, 0, 2));
	atomicAdd(vals + loc + 3, getIJ(v, 1, 0));
	atomicAdd(vals + loc + 4, getIJ(v, 1, 1));
	atomicAdd(vals + loc + 5, getIJ(v, 1, 2));
	atomicAdd(vals + loc + 6, getIJ(v, 2, 0));
	atomicAdd(vals + loc + 7, getIJ(v, 2, 1));
	atomicAdd(vals + loc + 8, getIJ(v, 2, 2));
}

inline __device__
void add_val(int *locs, double *vals, const double3x3 &v)
{
	atomicAdd(vals+locs[0], getIJ(v, 0, 0));
	atomicAdd(vals+locs[1], getIJ(v, 0, 1));
	atomicAdd(vals+locs[2], getIJ(v, 0, 2));
	atomicAdd(vals+locs[3], getIJ(v, 1, 0));
	atomicAdd(vals+locs[4], getIJ(v, 1, 1));
	atomicAdd(vals+locs[5], getIJ(v, 1, 2));
	atomicAdd(vals+locs[6], getIJ(v, 2, 0));
	atomicAdd(vals+locs[7], getIJ(v, 2, 1));
	atomicAdd(vals+locs[8], getIJ(v, 2, 2));
}

inline __device__ double3
subvec3(const double3x3 &b, int i)
{
	if (i == 0)
		return b.column0;
	if (i == 1)
		return b.column1;
	if (i == 2)
		return b.column2;

	assert(0);
	return make_double3(0.0);
}

inline __device__ double3
subvec4(const double12 &b, int i)
{
	return b.m[i];
}

inline __device__ double3x3
submat3(const double9x9 &A, int i, int j)
{
	return A.m[i][j];
}

inline __device__ double3x3
submat4(const double12x12 &A, int i, int j)
{
	return A.m[i][j];
}

inline __device__ void
add_submat3(const double9x9 &asub, tri3f *f,  double *vals, int *rowInc, int **matIdx, bool bsr)
{
	int locs[9];

	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++) {
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
	if (i  == 0) return f.x;
	else if (i == 1) return f.y;
	else if (i == 2) return f.z;
	else return f.w;
}

inline __device__ void
add_submat4(double12x12 asub, uint4 &f,  double *vals, int *rowInc, int **matIdx, bool bsr)
{
	int locs[9];

	for (int i=0; i<4; i++)
		for (int j=0; j<4; j++) {
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
add_subvec3(const double3x3 &bsub, tri3f *f, double3 *b)
{
	for (int i=0; i<3; i++)
		//b[f->id(i)] += subvec3(bsub, i);
		atomicAdd(b+f->id(i), subvec3(bsub, i));
}

inline __device__ void
add_subvec4(const double12 &bsub, uint4 &f, double3 *b)
{
	for (int i=0; i<4; i++) {
		double3 d = subvec4(bsub, i);
		int id = get(f, i);

		//b[get(f, i)] += subvec4(bsub, i);
		atomicAdd(b+id, d);
	}
}

inline __device__ double2
derivative (double a0, double a1, double a2, const double2x2 &idm) {
    return getTrans(idm) * make_double2(a1-a0, a2-a0);
}

inline __device__ double3x2
derivative (double3 &w0, double3 &w1, double3 &w2, const double2x2 &idm) {
    return make_double3x2(w1-w0, w2-w0) * idm;
}

inline __device__ double2x3
derivative (const double2x2 &idm) {
//    return face->invDm.t()*Mat2x3::rows(Vec3(-1,1,0), Vec3(-1,0,1));
    return getTrans(idm)*make_double2x3(
		make_double2(-1, -1),
		make_double2(1, 0),
		make_double2(0, 1));
}

inline __device__ double2x2
stretching_stiffness (const double2x2 &G, const StretchingSamples &samples) {
    double a=(getIJ(G, 0,0)+0.25)*nsamples;
    double b=(getIJ(G, 1,1)+0.25)*nsamples;
    double c=fabsf(getIJ(G, 0,1))*nsamples;
    a=clamp(a, 0.0, nsamples-1-1e-5);
    b=clamp(b, 0.0, nsamples-1-1e-5);
    c=clamp(c, 0.0, nsamples-1-1e-5);
    int ai=(int)floor(a);
    int bi=(int)floor(b);
    int ci=(int)floor(c);
    if(ai<0)        ai=0;
    if(bi<0)        bi=0;
    if(ci<0)        ci=0;
    if(ai>nsamples-2)        ai=nsamples-2;
    if(bi>nsamples-2)        bi=nsamples-2;
    if(ci>nsamples-2)        ci=nsamples-2;
    a=a-ai;
    b=b-bi;
    c=c-ci;
    double weight[2][2][2];
    weight[0][0][0]=(1-a)*(1-b)*(1-c);
    weight[0][0][1]=(1-a)*(1-b)*(  c);
    weight[0][1][0]=(1-a)*(  b)*(1-c);
    weight[0][1][1]=(1-a)*(  b)*(  c);
    weight[1][0][0]=(  a)*(1-b)*(1-c);
    weight[1][0][1]=(  a)*(1-b)*(  c);
    weight[1][1][0]=(  a)*(  b)*(1-c);
    weight[1][1][1]=(  a)*(  b)*(  c);
    double2x2 stiffness = zero2x2();
    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
            for(int k=0; k<2; k++)
                for(int l=0; l<4; l++)
                    {
                        //stiffness[l]+=samples.s[ai+i][bi+j][ci+k][l]*weight[i][j][k];
						getI(stiffness, l) += getI(samples.s[ai+i][bi+j][ci+k], l)*weight[i][j][k];
                    }
    return stiffness;
}

inline __device__ double3x9
kronecker(const double3 &A, const double3x3 &B)
{
	double3x9 t;

	t.m[0] = A.x * B;
	t.m[1] = A.y * B;
	t.m[2] = A.z * B;
	return t;
}

inline __device__ double
unwrap_angle (double theta, double theta_ref) {
    if (theta - theta_ref > M_PI)
        theta -= 2*M_PI;
    if (theta - theta_ref < -M_PI)
        theta += 2*M_PI;
    return theta;
}

inline __device__ double
dihedral_angle (uint4 en, uint2 ef, double3 *x, double3 *n, double ref)
{
    // if (!hinge.edge[0] || !hinge.edge[1]) return 0;
    // const Edge *edge0 = hinge.edge[0], *edge1 = hinge.edge[1];
    // int s0 = hinge.s[0], s1 = hinge.s[1];

	//if (!edge->adjf[0] || !edge->adjf[1])
    //    return 0;
	if (ef.x == -1 || ef.y == -1)
		return 0.0;

    //Vec3 e = normalize(pos<s>(edge->n[0]) - pos<s>(edge->n[1]));
	double3 e = normalize(x[en.x] - x[en.y]);
    if (norm2(e)==0) return 0.0;

    //Vec3 n0 = nor<s>(edge->adjf[0]), n1 = nor<s>(edge->adjf[1]);
	double3 n0 = n[ef.x], n1 = n[ef.y];
    if (norm2(n0)==0 || norm2(n1)==0) return 0.0;

    double cosine = dot(n0, n1), sine = dot(e, cross(n0, n1));
    double theta = atan2(sine, cosine);
    //return unwrap_angle(theta, edge->reference_angle);
	return unwrap_angle(theta, ref);
}

inline __device__ double
bending_stiffness (int side, const BendingData &data, 
				   uint2 ef, uint4 en, tri3f *vrts,
				   double eTheta, double eLen, double A12,
				   double2 *vU, tri3f *nods,
                   double initial_angle)
{
    //double curv = edge->theta*edge->l/(edge->adjf[0]->a + edge->adjf[1]->a);
	double curv = eTheta * eLen / A12;
    double alpha = curv/2;
    double value = alpha*0.2; // because samples are per 0.05 cm^-1 = 5 m^-1
    if(value>4) value=4;
    int  value_i=(int)value;
    if(value_i<0)   value_i=0;
    if(value_i>3)   value_i=3;
    value-=value_i;

	int vid1 = edge_vert(vrts+(side == 0 ? ef.x : ef.y), nods+(side == 0 ? ef.x : ef.y), en.y);
	int vid2 = edge_vert(vrts+(side == 0 ? ef.x : ef.y), nods+(side == 0 ? ef.x : ef.y), en.x);
    //Vec2 du = edge_vert(edge, side, 1)->u - edge_vert(edge, side, 0)->u;
	double2 du =	vU[vid1] - vU[vid2];

    double    bias_angle=(atan2f(du.y, du.x)+initial_angle)*4/M_PI;
    if(bias_angle<0)        bias_angle= -bias_angle;
    if(bias_angle>4)        bias_angle=8-bias_angle;
    if(bias_angle>2)        bias_angle=4-bias_angle;
    int             bias_id=(int)bias_angle;
    if(bias_id<0)   bias_id=0;
    if(bias_id>1)   bias_id=1;
    bias_angle-=bias_id;
    double actual_ke = data.d[bias_id]  [value_i]  *(1-bias_angle)*(1-value)
                     + data.d[bias_id+1][value_i]  *(  bias_angle)*(1-value)
                     + data.d[bias_id]  [value_i+1]*(1-bias_angle)*(  value)
                     + data.d[bias_id+1][value_i+1]*(  bias_angle)*(  value);
    if(actual_ke<0) actual_ke=0;
    return actual_ke;
}

inline __device__ void
bending_force (uint4 en, uint2 ef, double el, double et, double eti, double eref,
	tri3f *vrts, double *fa, double3 *x, double3 *fn, double2x2 *idms, double2 *vu, tri3f *nods,
	double12x12 &oJ, double12 &oF, BendingData *bd)
{
	uint f0 = ef.x;
	uint f1 = ef.y;

	if (f0 == 42 && f1 == 3152)
		f0 = 42;

    double theta = dihedral_angle(en, ef, x, fn, eref);
    double a = fa[f0] + fa[f1]; //face0->a + face1->a;
    //Vec3 x0 = pos<s>(edge->n[0]),
    //     x1 = pos<s>(edge->n[1]),
    //     x2 = pos<s>(edge_opp_vert(edge, 0)->node),
    //     x3 = pos<s>(edge_opp_vert(edge, 1)->node);
	double3 x0 = x[en.x], x1=x[en.y], x2=x[en.z], x3=x[en.w];

    double h0 = distance(x2, x0, x1), h1 = distance(x3, x0, x1);

    //Vec3 n0 = nor<s>(face0), n1 = nor<s>(face1);
	double3 n0 = fn[ef.x], n1 = fn[ef.y];

	double2	w_f0 = barycentric_weights(x2, x0, x1),
					w_f1 = barycentric_weights(x3, x0, x1);

    double12 dtheta = make_double3x4(
								-(w_f0.x*n0/h0 + w_f1.x*n1/h1),
								-(w_f0.y*n0/h0 + w_f1.y*n1/h1),
								n0/h0,
								n1/h1);

    //const BendingData &bend0 = (*::materials)[face0->label]->bending,
    //                  &bend1 = (*::materials)[face1->label]->bending;
    //double ke = min(bending_stiffness(edge, 0, bend0),
    //                bending_stiffness(edge, 1, bend1));

	double d1 = bending_stiffness(0, *bd, ef, en, vrts, et, el, a, vu, nods, 0.0);
	double d2 = bending_stiffness(1, *bd, ef, en, vrts, et, el, a, vu, nods, 0.0);
	double ke = fminf(d1, d2);

    //double weakening = max((*::materials)[face0->label]->weakening,
    //                       (*::materials)[face1->label]->weakening);
    //ke *= 1/(1 + weakening*edge->damage);
	ke *= 1.0;

    //double shape = sq(edge->l)/(2*a);
	double shape = (el*el)/(2*a);
    
	//return make_pair(-ke*shape*outer(dtheta, dtheta)/2.,
    //                 -ke*shape*(theta - edge->theta_ideal)*dtheta/2.);
	oJ = -ke*shape*outer(dtheta, dtheta)*0.5;
	//oF = -ke*shape*(theta - edge->theta_ideal)*dtheta*0.5;
	oF = -ke*shape*(theta - eti)*dtheta*0.5;
}

inline __device__ void
stretching_force (
	const tri3f &t, double fa, double3 *x, const double2x2 &idm,
	double9x9 &oJ, double3x3 &oF, StretchingSamples *ss)
{
    double3x2 F = derivative(x[t.id0()], x[t.id1()], x[t.id2()], idm);

    double2x2 G = (getTrans(F)*F - make_double2x2(1.0))*0.5;
    double2x2 k = stretching_stiffness(G, *ss);
    double weakening = MaterialWeakening;
    k *= 1/(1 + weakening*FaceDamage);

    // eps = 1/2(F'F - I) = 1/2([x_u^2 & x_u x_v \\ x_u x_v & x_v^2] - I)
    // e = 1/2 k0 eps00^2 + k1 eps00 eps11 + 1/2 k2 eps11^2 + k3 eps01^2
    // grad e = k0 eps00 grad eps00 + ...
    //        = k0 eps00 Du' x_u + ...
    double2x3 D = derivative(idm);
    //double3 du = D.row(0), dv = D.row(1);
	double3 du = getRow(D, 0), dv = getRow(D, 1);

	double3x3 I = identity3x3();
	double3x9 Du = kronecker(du, I);
	double3x9 Dv = kronecker(dv, I);
//	double3x9 Du = kronecker(rowmat(du), Mat3x3(1)),
  //           Dv = kronecker(rowmat(dv), Mat3x3(1));

    //const Vec3 &xu = F.col(0), &xv = F.col(1); // should equal Du*mat_to_vec(X)
	const double3 &xu = F.column0;
	const double3 &xv = F.column1;

	//Vec9 fuu = Du.t()*xu, fvv = Dv.t()*xv, fuv = (Du.t()*xv + Dv.t()*xu)/2.;
	double9 fuu = getTrans(Du)*xu;
	double9 fvv = getTrans(Dv)*xv;
	double9 fuv = (getTrans(Du)*xv + getTrans(Dv)*xu)*0.5;

	//Vec9 grad_e = k[0]*G(0,0)*fuu + k[2]*G(1,1)*fvv
	//		              + k[1]*(G(0,0)*fvv + G(1,1)*fuu) + 2*k[3]*G(0,1)*fuv;
	double9 grad_e =
		getIJ(k, 0, 0)*getIJ(G, 0, 0)*fuu +
		getIJ(k, 0, 1)*getIJ(G, 1, 1)*fvv +
		getIJ(k, 1, 0)*(getIJ(G, 0, 0)*fvv + getIJ(G, 1, 1)*fuu) +
		2.0*getIJ(k, 1, 1)*getIJ(G, 0, 1)*fuv;


    //Mat9x9 hess_e = k[0]*(outer(fuu,fuu) + max(G(0,0),0.)*Du.t()*Du)
    //              + k[2]*(outer(fvv,fvv) + max(G(1,1),0.)*Dv.t()*Dv)
    //              + k[1]*(outer(fuu,fvv) + max(G(0,0),0.)*Dv.t()*Dv
    //                      + outer(fvv,fuu) + max(G(1,1),0.)*Du.t()*Du)
    //              + 2.*k[3]*(outer(fuv,fuv));
/*	double9x9 hess_e = getIJ(k, 0, 0)*outer(fuu,fuu) + max(getIJ(G, 0,0),0.)*(getTrans(Du)*Du)
                  + getIJ(k, 0, 1)*(outer(fvv,fvv) + max(getIJ(G, 1,1),0.)*(getTrans(Dv)*Dv))
                  + getIJ(k, 1, 0)*(outer(fuu,fvv) + max(getIJ(G, 0,0),0.)*(getTrans(Dv)*Dv)
										+ outer(fvv,fuu) + max(getIJ(G, 1,1),0.)*(getTrans(Du)*Du))
                  + 2.*getIJ(k, 1, 1)*outer(fuv,fuv);*/

	double9x9 hess_e = 
		getIJ(k, 0, 0)*(outer(fuu,fuu) + max(getIJ(G, 0,0),0.)*(getTrans(Du)*Du)) +
		getIJ(k, 0, 1)*(outer(fvv,fvv) + max(getIJ(G, 1,1),0.)*(getTrans(Dv)*Dv)) +
		getIJ(k, 1, 0)*(outer(fuu,fvv) + max(getIJ(G, 0,0),0.)*(getTrans(Dv)*Dv)
							+outer(fvv,fuu) + max(getIJ(G, 1,1),0.)*(getTrans(Du)*Du)) +
		2.0*getIJ(k, 1, 1)*outer(fuv,fuv);
/*
	hess_e += max(getIJ(G, 0,0),0.)*(getTrans(Du)*Du);
       hess_e += getIJ(k, 0, 1)*(outer(fvv,fvv) + max(getIJ(G, 1,1),0.)*(getTrans(Dv)*Dv));
        hess_e += getIJ(k, 1, 0)*(outer(fuu,fvv) + max(getIJ(G, 0,0),0.)*(getTrans(Dv)*Dv)
										+ outer(fvv,fuu) + max(getIJ(G, 1,1),0.)*(getTrans(Du)*Du));
        hess_e += 2.*getIJ(k, 1, 1)*outer(fuv,fuv);
*/

    // ignoring G(0,1)*(Du.t()*Dv+Dv.t()*Du)/2. term
    // because may not be positive definite
    
	//return make_pair(-face->a*hess_e, -face->a*grad_e);
	oJ = -fa*hess_e;
	oF = -fa*grad_e;
}

__global__ void
kernel_internal_face_forces(
				tri3f *faces, double3 *v, double *fa, double3 *x, double2x2 *idms,
				double dt, double *vals, bool bsr, int num,
				int **matIdx, int *rowInc, double3 *b, double *m,
				double3 *Fext, double3x3 *Jext, StretchingSamples *ss,
				double9 *oF, double9x9 *oJ)
{
	LEN_CHK(num);

	tri3f *f = faces+idx;
	double9 vs = make_double3x3(v[f->id0()], v[f->id1()], v[f->id2()]);

	double9x9 J;
	double3x3 F;
	stretching_force(faces[idx], fa[idx], x, idms[idx], J, F, ss);

	// for debug
	if (oF != NULL) oF[idx] = F;
	if (oJ != NULL) oJ[idx] = J;

	if (dt == 0) {
		add_submat3(-J, f, vals, rowInc, matIdx, bsr);
		add_subvec3(F, f, b);
	} else {
		double damping = MaterialDumping;
		add_submat3(-dt*(dt+damping)*J, f, vals, rowInc, matIdx, bsr);
		add_subvec3(dt*(F + (dt+damping)*J*vs), f, b);
	}
}

__global__ void
kernel_internal_edge_forces(
				uint2 *ens, uint2 *efs, tri3f *nods, double3 *v,
				double *els, double *ets, double *etis, double *eref,
				double *fas, double3 *fns, double2x2 *idms, double2 *vus, double3 *x, tri3f *vrts,
				double dt, double *vals, bool bsr, int num,
				int **matIdx, int *rowInc, double3 *b, double *m, double3 *Fext, double3x3 *Jext, BendingData *bd,
				double12 *oF, double12x12 *oJ)
{
	LEN_CHK(num);

	uint2 en = ens[idx];
	uint2 ef = efs[idx];

	if (en.x == 697 && en.y == 1544)
		en.x = 697;

	if (ef.x == -1 || ef.y == -1)
		return;

//	if (idx == 410)
//		idx = 410;

	int aa = en.x;
	int bb = en.y;
	int cc = edge_opp_node(nods, ef.x, aa);
	int dd = edge_opp_node(nods, ef.y, bb);
	uint4 em = make_uint4(aa, bb, cc, dd);
	double12 vs = make_double3x4(v[aa], v[bb], v[cc], v[dd]);

	double12x12 J;
	double12 F;
	bending_force(em, ef, els[idx], ets[idx], etis[idx], eref[idx],
		vrts, fas, x, fns, idms, vus, nods,
		J, F, bd);

	// for debug
	if (oF != NULL) oF[idx] = F;
	if (oJ != NULL) oJ[idx] = J;

	if (dt == 0) {
		add_submat4(-J, em, vals, rowInc, matIdx, bsr);
		add_subvec4(F, em, b);
	} else {
		double damping = MaterialDumping;
						//((*::materials)[edge->adjf[0]->label]->damping +
						//(*::materials)[edge->adjf[1]->label]->damping)/2.;
		add_submat4(-dt*(dt+damping)*J, em, vals, rowInc, matIdx, bsr);

		double12 t = dt*(F + (dt+damping)*J*vs);
		add_subvec4(t, em, b);
	}
}

__global__ void
kernel_fill_constraint_forces (g_IneqCon *cstrs, 
		double dt, double *vals, bool bsr, int num,
		int **matIdx, int *rowInc, double3 *b,
		double3 *cx, double3 *ox, double3 *cv, double3 *ov, double mrt, int iii)
{
	LEN_CHK(num);
	g_IneqCon &cstr = cstrs[idx];

/*
	if (iii == 8 && idx == 1) {
		iii = 8;
	}

	if (iii == 8 &&
		cstr.nodes[0] == 25 &&
		cstr.nodes[1] == 4 &&
		cstr.nodes[2] == 866 &&
		cstr.nodes[3] == 447)
		cstr.nodes[0] = 25;
*/

	if (false == cstr._valid)
		return;

/*	if (cstr.nodes[0] == 47 &&
		cstr.nodes[1] == 953 &&
		cstr.nodes[2] == 278 &&
		cstr.nodes[3] == 447)
		cstr.nodes[0] = 47;
*/
	double v = cstr.value(cx, ox, mrt);
	double g = cstr.energy_grad(v, mrt);
	double h = cstr.energy_hess(v, mrt);

	MeshGrad grad;
	cstr.gradient(grad);

	double v_dot_grad = 0;
	for (int i=0; i<4; i++) {
		v_dot_grad += dot(grad[i], cstr.get_x(i, cv, ov));
	}
	
	for (int i=0; i<4; i++) {
		if (!cstr.free[i]) continue;
		int ni = cstr.nodes[i];

		for (int j=0; j<4; j++) {
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
					double3x3 m = dt*dt*h*outer(grad[i], grad[j]);
					add_val(loc, vals, m);
				}
			}
			else {
				get_indices(ni, k, locs, rowInc);

				if (dt == 0) {
					add_val(locs, vals, h*outer(grad[i], grad[j]));
				}
				else {
					double3x3 m = dt*dt*h*outer(grad[i], grad[j]);
					add_val(locs, vals, m);
				}
			}
		}

		if (dt == 0)
			atomicAdd(b+ni, -g*grad[i]);
		else {
			double3 dd = -dt*(g+dt*h*v_dot_grad)*grad[i];
			atomicAdd(b+ni, dd);
		}
	}
}


__global__ void
kernel_fill_handle_forces (EqCon *hdls, 
		double dt, double *vals, bool bsr, int num,
		int **matIdx, int *rowInc, double3 *b,
		double3 *cx, double3 *cv)
{
	LEN_CHK(num);
	EqCon &cstr = hdls[idx];

	double v = cstr.value(cx);
	double g = cstr.energy_grad(v);
	double h = cstr.energy_hess();

	double3 grad;
	cstr.gradient(grad);

	int ni = cstr.node;
	double v_dot_grad = dot(grad, cv[ni]);
	
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
		atomicAdd(b + ni, -g*grad);
	else
		atomicAdd(b + ni, -dt*(g + dt*h*v_dot_grad)*grad);
}

__global__ void
kernel_fill_friction_forces (g_IneqCon *cstrs, 
		double dt, double *vals, bool bsr, int num,
		int **matIdx, int *rowInc, double3 *b,
		double3 *cx, double3 *ox, double3 *cv, double3 *ov, double *cm, double *om, double mrt)
{
	LEN_CHK(num);
	g_IneqCon &cstr = cstrs[idx];
	if (false == cstr._valid)
		return;

	MeshHess jac;
	MeshGrad force;
	cstr.friction(dt, jac, force, cm, om, cx, ox, cv, ov, mrt);

	for (int i=0; i<4; i++) {
		if (!cstr.free[i]) continue;
		int id = cstr.nodes[i];
		atomicAdd(b+id, dt*force[i]);
	}

	for (int i=0; i<4; i++)
		for (int j=0; j<4; j++) {
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
kernel_mat_fill (double dt, double *vals, int num,
				 int **matIdx, int *rowInc, double3 *b, double *m, double3 *Fext, double3x3 *Jext)
{
	LEN_CHK(num);

	int locs[9];
	int j = find_location(matIdx, rowInc, idx, idx);
	get_indices(idx, j, locs, rowInc);
	add_val(locs, vals, make_double3x3(m[idx])-dt*dt*Jext[idx]);
	b[idx] += dt*Fext[idx];
}

__global__ void
kernel_mat_fill_bsr(double dt, double *vals, int num,
int **matIdx, int *rowInc, double3 *b, double *m, double3 *Fext, double3x3 *Jext)
{
	LEN_CHK(num);

	int j = find_location(matIdx, rowInc, idx, idx);
	int loc = get_indices(idx, j, rowInc);
	add_val(loc, vals, make_double3x3(m[idx]) - dt*dt*Jext[idx]);
	b[idx] += dt*Fext[idx];
}

__global__ void
kernel_mat_add (int *colLen, int *rowIdx, int **matIdx, int num, bool counting)
{
	LEN_CHK(num);

	if (counting)
		colLen[idx] += 1;
	else
		mat_add(idx, idx, rowIdx, matIdx);
}

__global__ void
kernel_add_gravity(double3 *fext, double *m, Gravity *g, int num)
{
	LEN_CHK(num);
	fext[idx] += m[idx]*g->_g;
}


inline __device__ double3 wind_force (
	int id0, int id1, int id2, const Wind &wind, double3 *v, double3 &fn, double fa)
{
    //Vec3 vface = (face->v[0]->node->v + face->v[1]->node->v
    //              + face->v[2]->node->v)/3.;
	double3 vface = (v[id0]+v[id1]+v[id2])/3.0;
    double3 vrel = wind.velocity - vface;
    double vn = dot(fn, vrel);
    double3 vt = vrel - vn*fn;
    return wind.density*fa*abs(vn)*vn*fn + wind.drag*fa*vt;
}

__global__ void 
kernel_add_wind(double3 *fext, tri3f *faces, double3 *v, double3 *fn, double *fa, Wind *dw, int num)
{
	LEN_CHK(num);

	int id0 = faces[idx].id0();
	int id1 = faces[idx].id1();
	int id2 = faces[idx].id2();
	double3 n = fn[idx];
	double a = fa[idx];

	double3 fw = wind_force(id0, id1, id2, *dw, v, n, a)/3.0;

	atomicAdd(fext+id0, fw);
	atomicAdd(fext+id1, fw);
	atomicAdd(fext+id2, fw);
}

__global__ void 
kernel_step_mesh (double3 *x, double3 *v, double dt, int num)
{
	LEN_CHK(num);
	x[idx] += v[idx]*dt;
}

__global__ void 
kernel_update_velocity(double3 *x, double3 *xx, double3 *v, double dt, int num)
{
	LEN_CHK(num);
	v[idx] += (x[idx] - xx[idx])/dt;
}


__global__ void
kernel_face_ws (g_box *bxs, bool ccd, double3 *nrm, tri3f *face, double3 *x, double3 *ox, double thickness, int num)
{
	LEN_CHK(num);

	int id0 = face[idx].id0();
	int id1 = face[idx].id1();
	int id2 = face[idx].id2();

	double3 ox0 = ox[id0];
	double3 ox1 = ox[id1];
	double3 ox2 = ox[id2];
	double3 x0 = x[id0];
	double3 x1 = x[id1];
	double3 x2 = x[id2];

	bxs[idx].set(ox0, ox1);
	bxs[idx].add(ox2);
	if (ccd) {
		bxs[idx].add(x0);
		bxs[idx].add(x1);
		bxs[idx].add(x2);
	}

	bxs[idx].enlarge(thickness);
	nrm[idx] = normalize(cross(x1-x0, x2-x0));
}

__global__ void
kernel_edge_ws (g_box *bxs, bool ccd, uint2 *en, uint2 *ef, double *er, double *et, double3 *nrm, double3 *x, double3 *ox, double thickness, int num)
{
	LEN_CHK(num);

	int id0 = en[idx].x;
	int id1 = en[idx].y;

	double3 ox0 = ox[id0];
	double3 ox1 = ox[id1];
	double3 x0 = x[id0];
	double3 x1 = x[id1];

	bxs[idx].set(ox0, ox1);
	if (ccd) {
		bxs[idx].add(x0);
		bxs[idx].add(x1);
	}
	bxs[idx].enlarge(thickness);

	et[idx] = dihedral_angle(make_uint4(id0, id1, 0, 0), ef[idx], x, nrm, er[idx]);
}

__global__ void
kernel_node_ws (g_box *bxs, bool ccd, double3 *x, double3 *ox, double3 *n,
				int *n2vIdx, int *n2vData, int *adjIdx, int *adjData, double3 *fAreas,
				double thickness, int num)
{
	LEN_CHK(num);

	double3 ox0 = ox[idx];
	double3 x0 = x[idx];

	bxs[idx].set(ox0);
	if (ccd) {
		bxs[idx].add(x0);
	}
	bxs[idx].enlarge(thickness);

	n[idx] = node_normal(idx, n2vIdx, n2vData, adjIdx, adjData, fAreas);
}

typedef struct {
	uint numNode, numFace, numEdge, numVert;
	int _n2vNum; // for n2v ...
	int _adjNum; // for vertex's adj faces ...

	// device memory
	// node attributes
	double3 *_dx, *_dx0, *_dv, *_dn;
	double *_da, *_dm;
	// from node to verts
	int *_dn2vIdx, *_dn2vData;

	//uint *_dn2v; // node to vert, should be verts ...
	//double2 *_du; // should belong to vert, but need to access from face (only have node id)

	// face attributes
	tri3f *_dfnod; //nodes
	tri3f *_dfvrt; //verts
	tri3f *dfedg;
	double3 *_dfn; // local normal, exact
	double *dfa, *dfm; // area, mass
	double2x2 *_dfdm, *_dfidm; // finite element matrix

	//edge attributes
	uint2 *den; //nodes
	uint2 *def; // faces
	double *detheta, *delen; // hihedra angle & length
	double *_deitheta; // rest dihedral angle, 
	double *_deref; //just to get sign of dihedral_angle() right

	// vertex attributes
	double2 *_dvu; // material space, usually you should should _du[nid]...
	int *_dvAdjIdx; // adjacent faces
	int *_dvAdjData;

	// host memory
	double3 *hx;          // use for save, dynamic
	tri3f *hfvrt;    // use for save, static

	// for bounding volumes
	g_box *_vtxBxs, *_triBxs, *_edgeBxs;

	// for collision detection
	uint *_dfmask;

	// for debugging
	g_box *hvBxs, *htBxs, *heBxs;

	void getBxs()
	{
		if (hvBxs == NULL)
			hvBxs = new g_box [numNode];
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
		delete [] hx;

		// for faces
		checkCudaErrors(cudaFree(_dfnod));
		checkCudaErrors(cudaFree(_dfvrt));
		checkCudaErrors(cudaFree(dfedg));
		checkCudaErrors(cudaFree(_dfn));
		checkCudaErrors(cudaFree(dfa));
		checkCudaErrors(cudaFree(dfm));
		checkCudaErrors(cudaFree(_dfdm));
		checkCudaErrors(cudaFree(_dfidm));
		delete [] hfvrt;

		// for vertex
		checkCudaErrors(cudaFree(_dvu));
		checkCudaErrors(cudaFree(_dvAdjIdx));
		checkCudaErrors(cudaFree(_dvAdjData));

		// for BVH
		checkCudaErrors(cudaFree(_vtxBxs));
		checkCudaErrors(cudaFree(_edgeBxs));
		checkCudaErrors(cudaFree(_triBxs));
		// _triBxs will be free by bvh (allocated there)

		if (_dfmask)
		checkCudaErrors(cudaFree(_dfmask));
	}

	void save(const string &fname) {
		checkCudaErrors(cudaMemcpy(hx, _dx, numNode*sizeof(double3), cudaMemcpyDeviceToHost));

		fstream file(fname.c_str(), ios::out);
		for (uint n=0; n<numNode; n++) {
			file << "v " << hx[n].x << " " << hx[n].y << " " << hx[n].z << endl;
		}

		for (uint n=0; n<numFace; n++) {
			file << hfvrt[n];
		}
	}

	void loadVtx(const string &fname) {
		input2(fname.c_str(), (double *)hx, numNode*3);
		checkCudaErrors(cudaMemcpy(_dx, hx, numNode*sizeof(double3), cudaMemcpyHostToDevice));
	}
	
	void saveVtx (const string &fname) {
		checkCudaErrors(cudaMemcpy(hx, _dx, numNode*sizeof(double3), cudaMemcpyDeviceToHost));
		//output1(fname.c_str(), (double *)hx, numNode*3);
		output2(fname.c_str(), (double *)hx, numNode*3);
	}
	
	void saveObj(const string &fname) {
		save(fname);
	}

	void allocEdges()
	{
		checkCudaErrors(cudaMalloc((void **)&den, numEdge*sizeof(uint2)));
		checkCudaErrors(cudaMalloc((void **)&def, numEdge*sizeof(uint2)));
		checkCudaErrors(cudaMalloc((void **)&detheta, numEdge*sizeof(double)));
		checkCudaErrors(cudaMalloc((void **)&delen, numEdge*sizeof(double)));
		checkCudaErrors(cudaMalloc((void **)&_deitheta, numEdge*sizeof(double)));
		checkCudaErrors(cudaMalloc((void **)&_deref, numEdge*sizeof(double)));
		checkCudaErrors(cudaMalloc(&_edgeBxs, numEdge*sizeof(g_box)));
	}

	void pushEdges(int num, uint2 *n, uint2 *f, double *t, double *l, double *i, double *r,
		int offset, enum cudaMemcpyKind kind)
	{
		checkCudaErrors(cudaMemcpy(den+offset, n, num*sizeof(uint2), kind));
		checkCudaErrors(cudaMemcpy(def+offset, f, num*sizeof(uint2), kind));
		checkCudaErrors(cudaMemcpy(detheta+offset, t, num*sizeof(double), kind));
		checkCudaErrors(cudaMemcpy(delen+offset, l, num*sizeof(double), kind));
		checkCudaErrors(cudaMemcpy(_deitheta+offset, i, num*sizeof(double), kind));
		checkCudaErrors(cudaMemcpy(_deref+offset, r, num*sizeof(double), kind));
	}

	void pushEdges(int num, void *n, void *f, double *t, double *l, double *i, double *r)
	{
		numEdge = num;
		allocEdges();
		pushEdges(num, (uint2 *)n, (uint2 *)f, t, l, i, r, 0, cudaMemcpyHostToDevice);
	}

	void allocVertices()
	{
		checkCudaErrors(cudaMalloc((void **)&_dvu, numVert*sizeof(double2)));
		checkCudaErrors(cudaMalloc((void **)&_dvAdjIdx, numVert*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&_dvAdjData, _adjNum*sizeof(int)));
	}

	void pushVertices(int num, int adjNum, double2 *vu, int *adjIdx, int *adjData,
		int offset1, int offset2, enum cudaMemcpyKind kind)
	{
		checkCudaErrors(cudaMemcpy(_dvu+offset1, vu, num*sizeof(double2), kind));
		checkCudaErrors(cudaMemcpy(_dvAdjIdx+offset1, adjIdx, num*sizeof(int), kind));
		checkCudaErrors(cudaMemcpy(_dvAdjData+offset2, adjData, adjNum*sizeof(int), kind));
	}

	void pushVertices(int num, double *vu, int *adjIdx, int *adjData, int adjNum)
	{
		numVert = num;
		// not need to be equal, but numNode >= numVert
		//assert(numVert == numNode);
		
		// not hold for dress-blue.json
		//assert(numNode >= numVert);

		_adjNum = adjNum;

		allocVertices();
		pushVertices(num, adjNum, (double2 *)vu, adjIdx, adjData, 0, 0, cudaMemcpyHostToDevice);
	}

	void allocFaces()
	{
		checkCudaErrors(cudaMalloc((void **)&_dfnod, numFace*sizeof(tri3f)));
		checkCudaErrors(cudaMalloc((void **)&_dfvrt, numFace*sizeof(tri3f)));
		checkCudaErrors(cudaMalloc((void **)&dfedg, numFace*sizeof(tri3f)));
		checkCudaErrors(cudaMalloc((void **)&_dfn, numFace*sizeof(double3)));
		checkCudaErrors(cudaMalloc((void **)&dfa, numFace*sizeof(double)));
		checkCudaErrors(cudaMalloc((void **)&dfm, numFace*sizeof(double)));
		checkCudaErrors(cudaMalloc((void **)&_dfdm, numFace*sizeof(double2x2)));
		checkCudaErrors(cudaMalloc((void **)&_dfidm, numFace*sizeof(double2x2)));
		checkCudaErrors(cudaMalloc(&_triBxs, numFace*sizeof(g_box)));
		hfvrt = new tri3f[numFace];
	}

	void pushFaces(int num, tri3f *nods, tri3f *vrts, tri3f *edgs, double3 *nrms,
		double *a, double *m, double2x2 *dm, double2x2 *idm,
		int offset, enum cudaMemcpyKind kind)
	{
		checkCudaErrors(cudaMemcpy(_dfnod+offset, nods, num*sizeof(tri3f), kind));
		checkCudaErrors(cudaMemcpy(_dfvrt+offset, vrts, num*sizeof(tri3f), kind));
		checkCudaErrors(cudaMemcpy(dfedg+offset, edgs, num*sizeof(tri3f), kind));
		checkCudaErrors(cudaMemcpy(_dfn+offset, nrms, num*sizeof(double3), kind));
		checkCudaErrors(cudaMemcpy(dfa+offset, a, num*sizeof(double), kind));
		checkCudaErrors(cudaMemcpy(dfm+offset, m, num*sizeof(double), kind));
		checkCudaErrors(cudaMemcpy(_dfdm+offset, dm, num*sizeof(double2x2), kind));
		checkCudaErrors(cudaMemcpy(_dfidm+offset, idm, num*sizeof(double2x2), kind));

		if (kind == cudaMemcpyHostToDevice)
			memcpy(hfvrt+offset, vrts, num*sizeof(tri3f));
		if (kind == cudaMemcpyDeviceToDevice)
			checkCudaErrors(cudaMemcpy(hfvrt+offset, vrts, num*sizeof(tri3f), cudaMemcpyDeviceToHost));
	}

	void pushFaces(int num, void *nods, void *vrts, void *edgs, double *nrms,
		double *a, double *m, double *dm, double*idm)
	{
		numFace = num;

		allocFaces();
		pushFaces(num, (tri3f *)nods, (tri3f *)vrts, (tri3f *)edgs, (double3 *)nrms, a, m, (double2x2 *)dm, (double2x2 *)idm,
			0, cudaMemcpyHostToDevice);
	}

	void pushNodes(int num, double *x, double3 *dxx, double dt)
	{
		checkCudaErrors(cudaMemcpy(dxx, _dx, num*sizeof(double3), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(_dx, x, num*sizeof(double3), cudaMemcpyHostToDevice));
		{
		BLK_PAR(numNode);
		kernel_update_velocity<<<B, T>>> (_dx, dxx, _dv, dt, numNode);
	    getLastCudaError("kernel_update_velocity");
		}
	}

	void dumpVtx()
	{
		double3 *hb = new double3[numNode];

		checkCudaErrors(cudaMemcpy(hb, _dx0,
			numNode*sizeof(double3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(hb, _dx,
			numNode*sizeof(double3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(hb, _dv,
			numNode*sizeof(double3), cudaMemcpyDeviceToHost));

//		output1("e:\\temp2\\oxxxx.txt", (double *)hb, numNode*3);

		delete [] hb;
	}

	void allocNodes()
	{
		checkCudaErrors(cudaMalloc((void **)&_dx, numNode*sizeof(double3)));
		checkCudaErrors(cudaMalloc((void **)&_dx0, numNode*sizeof(double3)));
		checkCudaErrors(cudaMalloc((void **)&_dv, numNode*sizeof(double3)));
		checkCudaErrors(cudaMalloc((void **)&_da, numNode*sizeof(double)));
		checkCudaErrors(cudaMalloc((void **)&_dm, numNode*sizeof(double)));
		checkCudaErrors(cudaMalloc((void **)&_dn, numNode*sizeof(double3)));

		checkCudaErrors(cudaMalloc((void **)&_dn2vIdx, numNode*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&_dn2vData, _n2vNum*sizeof(int)));

		checkCudaErrors(cudaMalloc(&_vtxBxs, numNode*sizeof(g_box)));
		hx = new double3[numNode];
	}

	void pushNodes(int num, double3 *x, double3 *x0, double3 *v, double *a, double *m, double3 *n,
		int n2vNum, int *n2vIdx, int *n2vData,
		int offset1, int offset2, enum cudaMemcpyKind kind)
	{
		checkCudaErrors(cudaMemcpy(_dx+offset1, x, num*sizeof(double3), kind));
		checkCudaErrors(cudaMemcpy(_dx0+offset1, x0, num*sizeof(double3), kind));
		checkCudaErrors(cudaMemcpy(_dv+offset1, v, num*sizeof(double3), kind));
		checkCudaErrors(cudaMemcpy(_da+offset1, a, num*sizeof(double), kind));
		checkCudaErrors(cudaMemcpy(_dm+offset1, m, num*sizeof(double), kind));
		checkCudaErrors(cudaMemcpy(_dn+offset1, n, num*sizeof(double3), kind));

		checkCudaErrors(cudaMemcpy(_dn2vIdx+offset1, n2vIdx, num*sizeof(int), kind));
		checkCudaErrors(cudaMemcpy(_dn2vData+offset2, n2vData, n2vNum*sizeof(int), kind));
	}

	void pushNodes(int num, double *x, double *x0, double *v, double *a, double *m, double *n,
		int n2vNum, int *n2vIdx, int *n2vData)
	{
		numNode = num;
		_n2vNum = n2vNum;

		allocNodes();
		pushNodes(num, (double3 *)x, (double3 *)x0, (double3 *)v, a, m, (double3 *)n, n2vNum, n2vIdx, n2vData, 0, 0, cudaMemcpyHostToDevice);
	}

	void popNodes(int num, double *x, double *x0)
	{
		checkCudaErrors(cudaMemcpy(x0, _dx0, num*sizeof(double3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(x, _dx, num*sizeof(double3), cudaMemcpyDeviceToHost));
	}

	void popNodes(int num, double *x)
	{
		checkCudaErrors(cudaMemcpy(x, _dx, num*sizeof(double3), cudaMemcpyDeviceToHost));
	}

	void incFaces(int num, double3 *n, int offset)
	{
		assert(numFace >= num);
		checkCudaErrors(cudaMemcpy(_dfn+offset, n, num*sizeof(double3), cudaMemcpyHostToDevice));
	}

	void incEdges(int num, double *t, int offset)
	{
		assert(numEdge >= num);
		checkCudaErrors(cudaMemcpy(detheta+offset, t, num*sizeof(double), cudaMemcpyHostToDevice));
	}

	void incNodes(int num, double3 *x, double3 *x0, double3 *v, double *a, double *m, double3 *n, int offset)
	{
		//now, only update part of it...
		assert(numNode >= num);

		checkCudaErrors(cudaMemcpy(_dx+offset, x, num*sizeof(double3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(_dx0+offset, x0, num*sizeof(double3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(_dv+offset, v, num*sizeof(double3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(_da+offset, a, num*sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(_dm+offset, m, num*sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(_dn+offset, n, num*sizeof(double3), cudaMemcpyHostToDevice));
	}

	void stepMesh(double dt)
	{
		if (numNode == 0)
			return;

		//mesh.nodes[n]->x += mesh.nodes[n]->v*dt;
		BLK_PAR(numNode);
		kernel_step_mesh<<<B, T>>> (_dx, _dv, dt, numNode);
	    getLastCudaError("kernel_step_mesh");
	}

	void updateNodes(double3 *ddv, double dt, bool update_positions)
	{
		int num = numNode;

		BLK_PAR(num);
		kernel_update_nodes<<<B, T>>>(_dv, _dx, ddv, dt, update_positions, num);
		getLastCudaError("kernel_update_nodes");
	}

	void updateX0()
	{
		checkCudaErrors(cudaMemcpy(_dx0, _dx, numNode*sizeof(double3), cudaMemcpyDeviceToDevice));
	}

	void project_outside(double3 *ddx, double *w, double *om, double3 *ox, double mrt, double mpt)
	{
		if (getConstraintNum() == 0)
			return;

		int num = numNode;
		checkCudaErrors(cudaMemset(ddx, 0, num*sizeof(double3)));
		checkCudaErrors(cudaMemset(w, 0, num*sizeof(double)));

/*		for (int c = 0; c < cons.size(); c++) {
			MeshGrad dxc = cons[c]->project();
			for (MeshGrad::iterator it = dxc.begin(); it != dxc.end(); it++) {
				const Node *node = it->first;
				double wn = norm2(it->second);
				int n = node->index;
				if (n >= mesh.nodes.size() || mesh.nodes[n] != node)
					continue;
				w[n] += wn;
				dx[n] += wn*it->second;
			}
		}
*/
		{
			int num = getConstraintNum();
			BLK_PAR(num);
			kernel_project_outside1<<<B, T>>>(getConstraints(), ddx, w, num,
				_dm, om, _dx, ox, mrt, mpt);
			getLastCudaError("kernel_project_outside1");
		}

	/*    for (int n = 0; n < nn; n++) {
			if (w[n] == 0)
				continue;
			mesh.nodes[n]->x += dx[n]/w[n];
		}
	*/
		{
			int num = numNode;
			BLK_PAR(num);
			kernel_project_outside2<<<B, T>>>(_dx, ddx, w, num);
			getLastCudaError("kernel_project_outside2");
		}
	}

	void computeWSdata(double thickness, bool ccd)
	{
		if (numFace == 0)
			return;

		{
		int num = numFace;
		BLK_PAR(num);
		kernel_face_ws<<<B, T>>> (_triBxs, ccd, _dfn, _dfnod, _dx, _dx0, thickness, num);
		getLastCudaError("kernel_face_ws");
		}
		{
		int num = numEdge;
		//    edge->theta = dihedral_angle<WS>(edge);

		BLK_PAR(num);
		kernel_edge_ws<<<B, T>>> (_edgeBxs, ccd, den, def, _deref, detheta, _dfn, _dx, _dx0, thickness, num);
		getLastCudaError("kernel_edge_ws");
		}
		{
		int num = numNode;

		BLK_PAR(num);
		kernel_node_ws<<<B, T>>>(_vtxBxs, ccd, _dx, _dx0, _dn, 
			_dn2vIdx, _dn2vData, _dvAdjIdx, _dvAdjData, _dfn,
			thickness, num);
		getLastCudaError("kernel_node_ws");
		
		/*
		node->n = Vec3(0);
		for (int v = 0; v < node->verts.size(); v++) {
			const Vert *vert = node->verts[v];
			const vector<Face*> &adjfs = vert->adjf;
			for (int i = 0; i < adjfs.size(); i++) {
				Face const* face = adjfs[i];
				int j = find(vert, face->v), j1 = (j+1)%3, j2 = (j+2)%3;
				Vec3 e1 = face->v[j1]->node->x - node->x,
					 e2 = face->v[j2]->node->x - node->x;
				node->n += cross(e1,e2)/(2*norm2(e1)*norm2(e2));
			}
		}
		node->n = normalize(node->n);
		*/
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
		for (unsigned int i=0; i<numNode; i++)
			vtx_marked[i] = false;

		bool *edge_marked = new bool[numEdge];
		for (unsigned int i=0; i<numEdge; i++)
			edge_marked[i] = false;

		for (unsigned int i=0; i<numFace; i++) {
			fmask[i] = 0;

			tri3f *vtx = tris+i;
			for (int j=0; j<3; j++) {
				unsigned int vid = vtx->id(j);
				if (vtx_marked[vid] == false) {
					fmask[i] |= (0x1 << j);
					vtx_marked[vid] = true;
				}
			}

			tri3f *edge = edgs+i;
			for (int j=0; j<3; j++) {
				unsigned int eid = edge->id(j);
				if (edge_marked[eid] == false) {
					fmask[i] |= (0x8 << j);
					edge_marked[eid] = true;
				}
			}
		}

		delete [] vtx_marked;
		delete [] edge_marked;	
		delete [] tris;
		delete [] edgs;

		checkCudaErrors(cudaMalloc((void **)&_dfmask, numFace*sizeof(uint)));
		checkCudaErrors(cudaMemcpy(_dfmask, fmask, numFace*sizeof(uint), cudaMemcpyHostToDevice));

		delete [] fmask;
	}

} g_mesh;

// another aux data for cloth, such as materials, fext, Jext used for time integration ...
typedef struct {
	// for time integration
	double3 *dFext;
	double3x3 *dJext;

	double3 *_db, *_dx; // for A*x = b
	double *_dw; // _dx and _w will be used by project_outside

	double2 *_fdists, *_vdists, *_edists; // buffers for constraint filtering

	void init(int nn, int fn, int en) {
		checkCudaErrors(cudaMalloc((void **)&dFext, nn*sizeof(double3)));
		checkCudaErrors(cudaMalloc((void **)&dJext, nn*sizeof(double3x3)));
		checkCudaErrors(cudaMalloc((void **)&_db, nn*sizeof(double3)));
		checkCudaErrors(cudaMalloc((void **)&_dx, nn*sizeof(double3)));
		checkCudaErrors(cudaMalloc((void **)&_dw, nn*sizeof(double)));

		//checkCudaErrors(cudaMalloc(&_vdists, nn*sizeof(double2)));
		//checkCudaErrors(cudaMalloc(&_fdists, fn*sizeof(double2)));
		//checkCudaErrors(cudaMalloc(&_edists, en*sizeof(double2)));
		_vdists = new double2[nn];
		_fdists = new double2[fn];
		_edists = new double2[en];
	}

	void reset(int nn, int fn, int en) {
		checkCudaErrors(cudaMemset(dFext, 0, nn*sizeof(double3)));
		checkCudaErrors(cudaMemset(dJext, 0, nn*sizeof(double3x3)));
		checkCudaErrors(cudaMemset(_db, 0, nn*sizeof(double3)));
		checkCudaErrors(cudaMemset(_dx, 0, nn*sizeof(double3)));
		checkCudaErrors(cudaMemset(_dw, 0, nn*sizeof(double)));

		//checkCudaErrors(cudaMemset(_vdists, 0, nn*sizeof(double2)));
		//checkCudaErrors(cudaMemset(_fdists, 0, fn*sizeof(double2)));
		//checkCudaErrors(cudaMemset(_edists, 0, en*sizeof(double2)));
		memset(_vdists, 0, nn*sizeof(double2));
		memset(_fdists, 0, fn*sizeof(double2));
		memset(_edists, 0, en*sizeof(double2));
	}

	void destroy() {
		checkCudaErrors(cudaFree(dFext));
		checkCudaErrors(cudaFree(dJext));
		checkCudaErrors(cudaFree(_db));
		checkCudaErrors(cudaFree(_dx));
		checkCudaErrors(cudaFree(_dw));

		//checkCudaErrors(cudaFree(_vdists));
		//checkCudaErrors(cudaFree(_fdists));
		//checkCudaErrors(cudaFree(_edists));
		delete [] _vdists;
		delete [] _fdists;
		delete [] _edists;
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

void push_material_gpu(const void *s, const void *b, const void *w, const void *g)
{
	checkCudaErrors(cudaMalloc(&dMaterialStretching, sizeof(StretchingSamples)));
	checkCudaErrors(cudaMemcpy(dMaterialStretching, s, sizeof(StretchingSamples), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&dMaterialBending, sizeof(BendingData)));
	checkCudaErrors(cudaMemcpy(dMaterialBending, b, sizeof(BendingData), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&dWind, sizeof(Wind)));
	checkCudaErrors(cudaMemcpy(dWind, w, sizeof(Wind), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&dGravity, sizeof(Gravity)));
	checkCudaErrors(cudaMemcpy(dGravity, g, sizeof(Gravity), cudaMemcpyHostToDevice));
}

void push_num_gpu(int nC, int nO)
{
	numCloth = nC;
	numObstacles = nO;

	_clothes = new g_mesh[nC];
	for (int i=0; i<nC; i++)
		_clothes[i].init();

	obstacles = new g_mesh[nO];
	for (int i=0; i<nO; i++)
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
	kernel_add_offset <<<B, T>>> (data, offset, num);
	getLastCudaError("kernel_add_offset");
}

void offset_indices(uint2*data, int offset, int num, bool face = false)
{
	BLK_PAR(num);
	kernel_add_offset <<<B, T>>> (data, offset, num, face);
	getLastCudaError("kernel_add_offset");
}

void offset_indices(int*data, int offset, int num)
{
	BLK_PAR(num);
	kernel_add_offset <<<B, T>>> (data, offset, num);
	getLastCudaError("kernel_add_offset");
}

void offset_indices(uint*data, int offset, int num)
{
	BLK_PAR(num);
	kernel_add_offset <<<B, T>>> (data, offset, num);
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

	for (int i=0; i<numObstacles; i++) {
		numNode += obstacles[i].numNode;
		numFace += obstacles[i].numFace;
		numEdge += obstacles[i].numEdge;
		numVert += obstacles[i].numVert;
		_adjNum += obstacles[i]._adjNum;
		_n2vNum += obstacles[i]._n2vNum;

		nodeOffsets[i] = (i == 0) ? 0 : nodeOffsets[i-1] + obstacles[i-1].numNode;
		faceOffsets[i] = (i == 0) ? 0 : faceOffsets[i-1] + obstacles[i-1].numFace;
		edgeOffsets[i] = (i == 0) ? 0 : edgeOffsets[i-1] + obstacles[i-1].numEdge;
		vertOffsets[i] = (i == 0) ? 0 : vertOffsets[i-1] + obstacles[i-1].numVert;
		adjIdxOffsets[i] = (i == 0) ? 0 : adjIdxOffsets[i-1] + obstacles[i-1]._adjNum;
		n2vIdxOffsets[i] = (i == 0) ? 0 : n2vIdxOffsets[i-1] + obstacles[i-1]._n2vNum;
	}

	{// merge nodes
		allocNodes();

		int offset1 = 0;
		int offset2 = 0;
		for (int i=0; i<numObstacles; i++) {
			g_mesh &ob = obstacles[i];
			if (ob.numNode == 0)
				continue;

			pushNodes(ob.numNode, ob._dx, ob._dx0, ob._dv, ob._da, ob._dm, ob._dn,
				ob._n2vNum, ob._dn2vIdx, ob._dn2vData,
				offset1, offset2, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dn2vIdx+offset1, n2vIdxOffsets[i], ob.numNode);
				offset_indices(_dn2vData+offset2, vertOffsets[i], ob._n2vNum);
			}

			offset1 += ob.numNode;
			offset2 += ob._n2vNum;
		}
	}

	{// merge faces
		allocFaces();
		int offset = 0;
		for (int i=0; i<numObstacles; i++) {
			g_mesh &ob = obstacles[i];
			if (ob.numFace == 0)
				continue;

			pushFaces(ob.numFace, ob._dfnod, ob._dfvrt, ob.dfedg, ob._dfn, ob.dfa, ob.dfm, ob._dfdm, ob._dfidm,
				offset, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dfnod+offset, nodeOffsets[i], ob.numFace);
				offset_indices(_dfvrt+offset, vertOffsets[i], ob.numFace);
				offset_indices(dfedg+offset, edgeOffsets[i], ob.numFace);
			}

			offset += ob.numFace;
		}
		checkCudaErrors(cudaMemcpy(hfvrt, _dfnod, numFace*sizeof(tri3f), cudaMemcpyDeviceToHost));
	}

	{// merge edges
		allocEdges();

		int offset = 0;
		for (int i=0; i<numObstacles; i++) {
			g_mesh &ob = obstacles[i];
			if (ob.numEdge == 0)
				continue;

			pushEdges(ob.numEdge, ob.den, ob.def, ob.detheta, ob.delen, ob._deitheta, ob._deref,
				offset, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(den+offset, nodeOffsets[i], ob.numEdge);
				offset_indices(def+offset, faceOffsets[i], ob.numEdge, true);
			}

			offset += ob.numEdge;
		}
	}

	{// merge vertices
		allocVertices();

		int offset1 = 0;
		int offset2 = 0;
		for (int i=0; i<numObstacles; i++) {
			g_mesh &ob = obstacles[i];
			if (ob.numVert == 0)
				continue;

			pushVertices(ob.numVert, ob._adjNum, ob._dvu, ob._dvAdjIdx, ob._dvAdjData,
				offset1, offset2, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dvAdjIdx+offset1, adjIdxOffsets[i], ob.numVert);
				offset_indices(_dvAdjData+offset2, faceOffsets[i], ob._adjNum);
			}

			offset1 += ob.numVert;
			offset2 += ob._adjNum;
		}
	}

	delete [] nodeOffsets;
	delete [] faceOffsets;
	delete [] edgeOffsets;
	delete [] vertOffsets;
	delete [] adjIdxOffsets;
	delete [] n2vIdxOffsets;
}

void g_mesh::mergeClothes()
{
	assert (numCloth != 0);

	int *nodeOffsets = new int[numCloth];
	int *faceOffsets = new int[numCloth];
	int *edgeOffsets = new int[numCloth];
	int *vertOffsets = new int[numCloth];
	int *adjIdxOffsets = new int[numCloth];
	int *n2vIdxOffsets = new int[numCloth];

	for (int i=0; i<numCloth; i++) {
		numNode += _clothes[i].numNode;
		numFace += _clothes[i].numFace;
		numEdge += _clothes[i].numEdge;
		numVert += _clothes[i].numVert;
		_adjNum += _clothes[i]._adjNum;
		_n2vNum += _clothes[i]._n2vNum;

		nodeOffsets[i] = (i == 0) ? 0 : nodeOffsets[i-1] + _clothes[i-1].numNode;
		faceOffsets[i] = (i == 0) ? 0 : faceOffsets[i-1] + _clothes[i-1].numFace;
		edgeOffsets[i] = (i == 0) ? 0 : edgeOffsets[i-1] + _clothes[i-1].numEdge;
		vertOffsets[i] = (i == 0) ? 0 : vertOffsets[i-1] + _clothes[i-1].numVert;
		adjIdxOffsets[i] = (i == 0) ? 0 : adjIdxOffsets[i-1] + _clothes[i-1]._adjNum;
		n2vIdxOffsets[i] = (i == 0) ? 0 : n2vIdxOffsets[i-1] + _clothes[i-1]._n2vNum;
	}

	{// merge nodes
		allocNodes();

		int offset1 = 0;
		int offset2 = 0;
		for (int i=0; i<numCloth; i++) {
			g_mesh &ob = _clothes[i];
			pushNodes(ob.numNode, ob._dx, ob._dx0, ob._dv, ob._da, ob._dm, ob._dn,
				ob._n2vNum, ob._dn2vIdx, ob._dn2vData,
				offset1, offset2, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dn2vIdx+offset1, n2vIdxOffsets[i], ob.numNode);
				offset_indices(_dn2vData+offset2, vertOffsets[i], ob._n2vNum);
			}

			offset1 += ob.numNode;
			offset2 += ob._n2vNum;
		}
	}

	{// merge faces
		allocFaces();
		int offset = 0;
		for (int i=0; i<numCloth; i++) {
			g_mesh &ob = _clothes[i];
			pushFaces(ob.numFace, ob._dfnod, ob._dfvrt, ob.dfedg, ob._dfn, ob.dfa, ob.dfm, ob._dfdm, ob._dfidm,
				offset, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dfnod+offset, nodeOffsets[i], ob.numFace);
				offset_indices(_dfvrt+offset, vertOffsets[i], ob.numFace);
				offset_indices(dfedg+offset, edgeOffsets[i], ob.numFace);
			}

			offset += ob.numFace;
		}
		checkCudaErrors(cudaMemcpy(hfvrt, _dfnod, numFace*sizeof(tri3f), cudaMemcpyDeviceToHost));
	}

	{// merge edges
		allocEdges();

		int offset = 0;
		for (int i=0; i<numCloth; i++) {
			g_mesh &ob = _clothes[i];
			pushEdges(ob.numEdge, ob.den, ob.def, ob.detheta, ob.delen, ob._deitheta, ob._deref,
				offset, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(den+offset, nodeOffsets[i], ob.numEdge);
				offset_indices(def+offset, faceOffsets[i], ob.numEdge, true);
			}

			offset += ob.numEdge;
		}
	}

	{// merge vertices
		allocVertices();

		int offset1 = 0;
		int offset2 = 0;
		for (int i=0; i<numCloth; i++) {
			g_mesh &ob = _clothes[i];
			pushVertices(ob.numVert, ob._adjNum, ob._dvu, ob._dvAdjIdx, ob._dvAdjData,
				offset1, offset2, cudaMemcpyDeviceToDevice);

			if (i != 0) {
				offset_indices(_dvAdjIdx+offset1, adjIdxOffsets[i], ob.numVert);
				offset_indices(_dvAdjData+offset2, faceOffsets[i], ob._adjNum);
			}

			offset1 += ob.numVert;
			offset2 += ob._adjNum;
		}
	}

	delete [] nodeOffsets;
	delete [] faceOffsets;
	delete [] edgeOffsets;
	delete [] vertOffsets;
	delete [] adjIdxOffsets;
	delete [] n2vIdxOffsets;
}

void set_current_gpu(int idx, bool isCloth)
{
	_currentMesh = isCloth ? _clothes+idx : obstacles+idx;
}

void pop_cloth_gpu(int num, double *x)
{
	currentCloth.popNodes(num, x);
}

void push_node_gpu(int num, double *x, double *x0, double *v, double *a, double *m, double *n, int n2vNum, int *n2vIdx, int *n2vData)
{
	_currentMesh->pushNodes(num, x, x0, v, a, m, n, n2vNum, n2vIdx, n2vData);
}

void push_node_gpu(int num, double *x, double dt)
{
	currentCloth.pushNodes(num, x, totalAux._dx, dt);
}

void backup_node_gpu()
{
	checkCudaErrors(
		cudaMemcpy(totalAux._dx, currentCloth._dx,
			currentCloth.numNode*sizeof(double3), cudaMemcpyDeviceToDevice));
}

void build_mask_gpu()
{
	currentObj.buildMask();
	currentCloth.buildMask();
}

void inc_face_gpu(int idx, int num,  double *fn)
{
	int offset=0;
	for (int i=0; i<idx; i++)
		offset += obstacles[i].numFace;

	currentObj.incFaces(num, (double3 *)fn, offset);
}

void inc_edge_gpu(int idx, int num, double *theta)
{
	int offset=0;
	for (int i=0; i<idx; i++)
		offset += obstacles[i].numEdge;

	currentObj.incEdges(num, theta, offset);
}

void inc_node_gpu(int idx, int num, double *x, double *x0, double *v, double *a, double *m, double *n)
{
	int offset=0;
	for (int i=0; i<idx; i++)
		offset += obstacles[i].numNode;

	currentObj.incNodes(num, (double3 *)x, (double3 *)x0, (double3 *)v, a, m, (double3 *)n, offset);
}

void push_face_gpu(int num, void *nods, void *vrts, void *edgs, double *nrms, double *a, double *m, double *dm, double *idm)
{
	_currentMesh->pushFaces(num, nods, vrts, edgs, nrms, a, m, dm, idm);
}

void push_vert_gpu(int num, double *vu, int *adjIdx, int *adjData, int adjNum)
{
	_currentMesh->pushVertices(num, vu, adjIdx, adjData, adjNum);
}

void push_edge_gpu(int num, void *n, void *f, double *t, double *l, double *i, double *r)
{
	_currentMesh->pushEdges(num, n, f, t, l, i, r);
}

void check_gpu()
{
	return;
	currentObj.dumpVtx();
}

void step_mesh_gpu(double dt, double mrt)
{
/*	for (int i=0; i<numCloth; i++) {
		clothes[i].stepMesh(dt);
	}
*/
	currentCloth.stepMesh(dt);
	//currentCloth.saveVtx("e:\\temp4\\1.vtx");
	//currentCloth.loadVtx("e:\\temp4\\1.vtx");
	
	currentObj.stepMesh(dt);

	currentCloth.computeWSdata(mrt, true);
	currentObj.computeWSdata(mrt, true);

}

void next_step_mesh_gpu(double mrt)
{
	currentCloth.computeWSdata(mrt, true);
	currentObj.computeWSdata(mrt, true);

	currentCloth.updateX0();
	currentObj.updateX0();
}

static void add_external_forces()
{
	{
	// fext[n] += mesh.nodes[n]->m*gravity;
	int num = currentCloth.numNode;
	BLK_PAR(num);
	kernel_add_gravity <<<B, T>>> (totalAux.dFext, currentCloth._dm, dGravity, num);
	getLastCudaError("kernel_add_gravity");
	}

	{
	// const Face *face = mesh.faces[f];
    // Vec3 fw = wind_force(face, wind);
    // for (int v = 0; v < 3; v++)
    //	fext[face->v[v]->node->index] += fw/3.;
	int num = currentCloth.numFace;
	BLK_PAR(num);
	kernel_add_wind <<<B, T>>> (
		totalAux.dFext, currentCloth._dfnod, 
		currentCloth._dv, currentCloth._dfn, currentCloth.dfa,
		dWind, num);
	getLastCudaError("kernel_add_wind");
	}
}

struct CooMatrix {
	int _num;
	int *_rows;
	int *_cols;
	double *_vals;
	bool _bsr;

	void init(int nn, int nvtx, bool bsr)
	{
		_bsr = bsr;

		if (!_bsr) {
			nn *= 9;
			_num = nn;

			checkCudaErrors(cudaMalloc((void **)&_rows, nn*sizeof(int)));
			checkCudaErrors(cudaMalloc((void **)&_cols, nn*sizeof(int)));
			checkCudaErrors(cudaMalloc((void **)&_vals, nn*sizeof(double)));

			checkCudaErrors(cudaMemset(_rows, 0, nn*sizeof(int)));
			checkCudaErrors(cudaMemset(_cols, 0, nn*sizeof(int)));
			checkCudaErrors(cudaMemset(_vals, 0, nn*sizeof(double)));
		}
		else {
			_num = nn;

			checkCudaErrors(cudaMalloc((void **)&_rows, (nvtx+1)*sizeof(int)));
			checkCudaErrors(cudaMemset(_rows, 0, (nvtx + 1)*sizeof(int)));

			checkCudaErrors(cudaMalloc((void **)&_cols, nn*sizeof(int)));
			checkCudaErrors(cudaMemset(_cols, 0, nn*sizeof(int)));

			checkCudaErrors(cudaMalloc((void **)&_vals, nn * 9 * sizeof(double)));
			checkCudaErrors(cudaMemset(_vals, 0, nn * 9 * sizeof(double)));
		}
	}

	void destroy() {
		checkCudaErrors(cudaFree(_rows));
		checkCudaErrors(cudaFree(_cols));
		checkCudaErrors(cudaFree(_vals));
	}

};

typedef struct {
	int *_colLen; // for counting, used for alloc matIdx
	int *_rowIdx; // for adding, inc with atomicAdd
	int *_rowInc; // for fast locating
	int **_matIdx;
	int _matDim;

	int *_tItems, *_cItems; // all the nodes & compressed nodes
	int _cNum; // compressed length

	int *_hBuffer; // read back buffer on host, used for total only, can be removed later

	void init(int nn) {
		_matDim = nn;
		checkCudaErrors(cudaMalloc((void **)&_colLen, nn*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&_rowInc, nn*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&_rowIdx, nn*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&_matIdx, nn*sizeof(int *)));

		checkCudaErrors(cudaMemset(_colLen, 0, nn*sizeof(int)));
		checkCudaErrors(cudaMemset(_rowInc, 0, nn*sizeof(int)));
		checkCudaErrors(cudaMemset(_rowIdx, 0, nn*sizeof(int)));
		checkCudaErrors(cudaMemset(_matIdx, 0, nn*sizeof(int *)));

		_hBuffer = new int[nn];

		_tItems = _cItems = NULL;
		_cNum = 0;
	}

	int length() {
		return _cNum;
	}

	void destroy() {
		checkCudaErrors(cudaFree(_colLen));
		checkCudaErrors(cudaFree(_rowInc));
		checkCudaErrors(cudaFree(_rowIdx));
		checkCudaErrors(cudaFree(_matIdx));
		delete [] _hBuffer;

		if (_tItems)
			checkCudaErrors(cudaFree(_tItems));
		if (_cItems)
			checkCudaErrors(cudaFree(_cItems));

	}

	void	mat_fill_constraint_forces(double dt, CooMatrix &A, double mrt, int  iii)
	{
		{// for NodeHandle, it only put data at A(ni, ni), so this is only place for modifications
		int num = getHandleNum();
		if (num) {
		BLK_PAR(num);
		kernel_fill_handle_forces<<<B, T>>>
			(getHandles(), dt, A._vals, A._bsr, num, _matIdx, _rowInc, totalAux._db,
			currentCloth._dx, currentCloth._dv);
		getLastCudaError("kernel_fill_handle_forces");
		}
		}

		{
		int num = getConstraintNum();
		if (num) {
		BLK_PAR(num);
		kernel_fill_constraint_forces<<<B, T>>>
			(getConstraints(), dt, A._vals, A._bsr, num, _matIdx, _rowInc, totalAux._db,
			currentCloth._dx, currentObj._dx, currentCloth._dv, currentObj._dv, mrt, iii);
		getLastCudaError("kernel_fill_constraint_forces");
		}
		}
	}
	
	void mat_fill_friction_forces(double dt, CooMatrix &A, double mrt)
	{
		int num = getConstraintNum();
		if (num) {
		BLK_PAR(num);
		kernel_fill_friction_forces<<<B, T>>>
			(getConstraints(), dt, A._vals, A._bsr, num, _matIdx, _rowInc, totalAux._db,
			currentCloth._dx, currentObj._dx, currentCloth._dv, currentObj._dv,
			currentCloth._dm, currentObj._dm, mrt);
		getLastCudaError("kernel_fill_friction_forces");
		}
	}

	void mat_fill_diagonals(double dt, CooMatrix &A)
	{
		int num = _matDim;
		BLK_PAR(num);

		if (A._bsr) {
			kernel_mat_fill_bsr << <B, T >> >(dt, A._vals, num, _matIdx, _rowInc,
				totalAux._db, currentCloth._dm, totalAux.dFext, totalAux.dJext);
			getLastCudaError("kernel_mat_fill_bsr");
		}
		else {
			kernel_mat_fill <<<B, T >>>(dt, A._vals, num, _matIdx, _rowInc,
				totalAux._db, currentCloth._dm, totalAux.dFext, totalAux.dJext);
			getLastCudaError("kernel_mat_fill");
		}

//#define DEBUG_7
#ifdef DEBUG_7
		hb = new double3[_matDim];
		checkCudaErrors(cudaMemcpy(hb, totalAux._db,
			_matDim*sizeof(double3), cudaMemcpyDeviceToHost));

		hm = new double[A._num];
		checkCudaErrors(cudaMemcpy(hm, A._vals, 
			A._num*sizeof(double), cudaMemcpyDeviceToHost));

		delete [] hb;
		delete [] hm;
#endif
	}

	void mat_fill_internal_forces(double dt, CooMatrix &A)
	{
		{
		int num = currentCloth.numFace;

		double9 *dF = NULL;
		double9x9 *dJ = NULL;

//#define DEBUG_8
#ifdef DEBUG_8
		
		cudaMalloc(&dF, sizeof(double9)*num);
		cudaMalloc(&dJ, sizeof(double9x9)*num);

		hF = new double9[num];
		hJ = new double9x9[num];
#endif

		BLK_PAR(num);
		kernel_internal_face_forces<<<B, T>>>(
			currentCloth._dfnod, currentCloth._dv,
			currentCloth.dfa, currentCloth._dx,
			currentCloth._dfidm,
			dt, A._vals, A._bsr, num, _matIdx, _rowInc,
			totalAux._db, currentCloth._dm, totalAux.dFext, totalAux.dJext, dMaterialStretching, dF, dJ);
		getLastCudaError("kernel_internal_face_forces");

#ifdef DEBUG_8
		cudaMemcpy(hF, dF, sizeof(double9)*num, cudaMemcpyDeviceToHost);
		cudaMemcpy(hJ, dJ, sizeof(double9x9)*num, cudaMemcpyDeviceToHost);

#ifdef _DEBUG
		output9("e:\\temp2\\2\\aa.txt", (double *)hF, num);
		output9x9("e:\\temp2\\2\\bb.txt", (double *)hJ, num);
#else
		output9("e:\\temp2\\1\\aa.txt", (double *)hF, num);
		output9x9("e:\\temp2\\1\\bb.txt", (double *)hJ, num);
#endif

		cudaFree(dF);
		cudaFree(dJ);
		delete [] hF;
		delete [] hJ;

		exit(0);
#endif
		}

		{
//#define DEBUG_7
#ifdef DEBUG_7
		hb = new double3[_matDim];

		checkCudaErrors(cudaMemcpy(hb, totalAux._db,
			_matDim*sizeof(double3), cudaMemcpyDeviceToHost));

		hm = new double[A._num];
		checkCudaErrors(cudaMemcpy(hm, A._vals, 
			A._num*sizeof(double), cudaMemcpyDeviceToHost));

		output("c:\\temp\\00.txt", (double *)hb, _matDim*3);
		output("c:\\temp\\11.txt", hm, A._num);
		exit(0);

		delete [] hb;
		delete [] hm;
#endif
		}

		{
		int num = currentCloth.numEdge;
		double12 *dF = NULL;
		double12x12 *dJ = NULL;

//#define DEBUG_9
#ifdef DEBUG_9
		
		cudaMalloc(&dF, sizeof(double12)*num);
		cudaMalloc(&dJ, sizeof(double12x12)*num);
		cudaMemset(dF, 0, sizeof(double12)*num);
		cudaMemset(dJ, 0, sizeof(double12x12)*num);

		hhF = new double12[num];
		hhJ = new double12x12[num];
#endif

		BLK_PAR(num);
		kernel_internal_edge_forces<<<B, T>>>(
			currentCloth.den, currentCloth.def,
			currentCloth._dfnod, currentCloth._dv,
			currentCloth.delen, currentCloth.detheta,
			currentCloth._deitheta, currentCloth._deref,
			currentCloth.dfa, currentCloth._dfn, currentCloth._dfidm,
			currentCloth._dvu, currentCloth._dx, currentCloth._dfvrt,
			dt, A._vals, A._bsr, num, _matIdx, _rowInc,
			totalAux._db, currentCloth._dm, totalAux.dFext, totalAux.dJext, dMaterialBending, dF, dJ);
		getLastCudaError("kernel_internal_edge_forces");

#ifdef DEBUG_9
		cudaMemcpy(hhF, dF, sizeof(double12)*num, cudaMemcpyDeviceToHost);
		cudaMemcpy(hhJ, dJ, sizeof(double12x12)*num, cudaMemcpyDeviceToHost);

		output12("c:\\temp\\a.txt", (double *)hhF, num);
		output12x12("c:\\temp\\b.txt", (double *)hhJ, num);

		cudaFree(dF);
		cudaFree(dJ);
		delete [] hhF;
		delete [] hhJ;

		exit(0);
#endif
		}

		{
//#define DEBUG_A
#ifdef DEBUG_A
		hb = new double3[_matDim];

		checkCudaErrors(cudaMemcpy(hb, totalAux._db,
			_matDim*sizeof(double3), cudaMemcpyDeviceToHost));

		hm = new double[A._num];
		checkCudaErrors(cudaMemcpy(hm, A._vals, 
			A._num*sizeof(double), cudaMemcpyDeviceToHost));

		output("c:\\temp\\00.txt", (double *)hb, _matDim*3);
		output("c:\\temp\\11.txt", hm, A._num);
		exit(0);

		delete [] hb;
		delete [] hm;
#endif
		}
	}

	void mat_add_diagonals(bool counting)
	{
		//for (int i=0; i<matDim; i++)
		//	mat_add(i, i);

		BLK_PAR(_matDim);
		kernel_mat_add<<<B, T>>> (_colLen, _rowIdx, _matIdx, _matDim, counting);
	    getLastCudaError("kernel_mat_add");
	}

	void mat_add_constraint_forces(bool counting)
	{
		int num = getConstraintNum();
		if (num == 0)
			return;

		BLK_PAR(num);
		kernel_add_constraint_forces<<<B, T>>>(getConstraints(), _colLen, _rowIdx, _matIdx, num, counting);
		getLastCudaError("kernel_add_constraint_forces");
	}

/*	void mat_add_friction_forces(bool counting)
	{
		int num = getConstriantNum();
		BLK_PAR(num);
		kernel_add_friction_forces<<<B, T>>>(getConstraints(), _colLen, _rowIdx, _matIdx, _matDim, counting);
		getLastCudaError("kernel_add_friction_forces");
	}
*/

	void mat_add_internal_forces(bool counting)
	{
		{
			int num = currentCloth.numFace;
			BLK_PAR(num);
			kernel_add_face_forces<<<B, T>>>(currentCloth._dfnod, _colLen, _rowIdx, _matIdx, num, counting);
			getLastCudaError("kernel_add_face_forces");
		}
		{
			int num = currentCloth.numEdge;
			BLK_PAR(num);
			kernel_add_edge_forces<<<B, T>>>(
				currentCloth.den, currentCloth.def, currentCloth._dfnod,
				_colLen, _rowIdx, _matIdx, num, counting);
			getLastCudaError("kernel_add_edge_forces");
		}

/*
		const Mesh &mesh = cloth.mesh;
		for (int f = 0; f < mesh.faces.size(); f++) {
			const Face* face = mesh.faces[f];
			const Node *n0 = face->v[0]->node, *n1 = face->v[1]->node, *n2 = face->v[2]->node;
			mat_add_submat(n0->index, n1->index, n2->index);
		}
		for (int e = 0; e < mesh.edges.size(); e++) {
			const Edge *edge = mesh.edges[e];
			if (!edge->adjf[0] || !edge->adjf[1])
				continue;

			const Node *n0 = edge->n[0], *n1 = edge->n[1],
				*n2 = edge_opp_vert(edge, 0)->node,
				*n3 = edge_opp_vert(edge, 1)->node;

			mat_add_submat(n0->index, n1->index, n2->index, n3->index);
		}
*/
	}

	void mat_build_inc(bool bsr)
	{
		{
		/*int LEN = 100;
		int *buffer = new int[LEN];
		checkCudaErrors(cudaMemcpy(buffer, _tItems, 100*sizeof(int), cudaMemcpyDeviceToHost));
		int total = _hBuffer[N-1];

		for (int i=0; i<20; i++)
			printf("i = %d\n", _hBuffer[i]);
		*/
		}

		// make each matIdx[i] sorted and unique
		// return its new length in colLen
		{
		BLK_PAR(_matDim);
		kernel_sort_idx<<<B, T>>> (_matIdx, _colLen, _rowInc, _matDim);
	    getLastCudaError("kernel_idx_sort");
		}

//#define DEBUG_4
#ifdef DEBUG_4
		{
		// should move to gpu reduction, prefix sum calculation ...
		checkCudaErrors(cudaMemcpy(_hBuffer, _rowInc, _matDim*sizeof(int), cudaMemcpyDeviceToHost));

		int total = 0;
		for (int i=0; i<_matDim; i++)
			total += _hBuffer[i];

		for (int i=0; i<20; i++)
			printf("ci = %d\n", _hBuffer[i]);

		printf("compressed total = %d\n", total);
		}
#endif

		thrust::device_ptr<int> dev_data(_rowInc);
		thrust::inclusive_scan(dev_data, dev_data+_matDim, dev_data);

		checkCudaErrors(cudaMemcpy(&_cNum, _rowInc+_matDim-1, sizeof(int), cudaMemcpyDeviceToHost));
		printf("compressed total = %d\n", _cNum);

#ifdef DEBUG_4
		{
		// should move to gpu reduction, prefix sum calculation ...
		checkCudaErrors(cudaMemcpy(_hBuffer, _rowInc, _matDim*sizeof(int), cudaMemcpyDeviceToHost));
		for (int i=0; i<20; i++)
			printf("di = %d\n", _hBuffer[i]);
		}
#endif

		checkCudaErrors(cudaMalloc((void **)&_cItems, _cNum*sizeof(int)));

		{
		int num = _matDim;

		BLK_PAR(num);
		kernel_compress_idx<<<B, T>>> (_matIdx, _cItems, _rowInc, num);
	    getLastCudaError("kernel_compress_idx");

//#define DEBUG_5
#ifdef DEBUG_5
		int *data = new int[_cNum];
		checkCudaErrors(cudaMemcpy(data, _cItems, _cNum*sizeof(int), cudaMemcpyDeviceToHost));
		for (int i=0; i<20; i++)
			printf("xi = %d\n", data[i]);
		delete [] data;
#endif

		// for bsr, _cItems is the colIdx, and _rowInc is the rowIdx, for CSR
		// this kernel can be ignored for bsr
		if (!bsr) {
			kernel_set_matIdx << <B, T >> >(_matIdx, _rowInc, _cItems, num);
			getLastCudaError("kernel_set_matIdx");
		}
		}
	}

	void mat_build_space()
	{
#ifdef DEBUG_2
		// should move to gpu reduction, prefix sum calculation ...
		checkCudaErrors(cudaMemcpy(_hBuffer, _colLen, _matDim*sizeof(int), cudaMemcpyDeviceToHost));

		int total = 0;
		for (int i=0; i<_matDim; i++)
			total += _hBuffer[i];

		for (int i=0; i<20; i++)
			printf("i = %d\n", _hBuffer[i]);

		printf("total = %d\n", total);
#else
		int N = _matDim;

		thrust::device_ptr<int> dev_data(_colLen);
		thrust::inclusive_scan(dev_data, dev_data+N, dev_data);
/*
		checkCudaErrors(cudaMemcpy(_hBuffer, _colLen, N*sizeof(int), cudaMemcpyDeviceToHost));
		int total = _hBuffer[N-1];

		for (int i=0; i<20; i++)
			printf("i = %d\n", _hBuffer[i]);
*/
		int total;
		checkCudaErrors(cudaMemcpy(&total, _colLen+N-1, sizeof(int), cudaMemcpyDeviceToHost));
		printf("total = %d\n", total);
#endif

		checkCudaErrors(cudaMalloc((void **)&_tItems, total*sizeof(int)));

/*		// need to be done in kernel, can be done in parallel ...
		// Now, it is wrong !!!
		for (int i=0; i<matDim; i++) {
			if (i == 0)
				matIdx[0] = tItems;
			else
				matIdx[i] = tItems+colLen[i-1];
		}
*/
		{
			int num = _matDim;

			BLK_PAR(num);
			kernel_set_matIdx<<<B, T>>>(_matIdx, _colLen, _tItems, num);
			getLastCudaError("kernel_set_matIdx");
		}

	}

	void getColSpace(double dt)
	{
		mat_add_diagonals(true);
		mat_add_internal_forces(true);
		mat_add_constraint_forces(true);

		// for proximities, mat_add_friction_forces just fill at the positions, so no need to call it...
		//mat_add_friction_forces(true, dt);
		mat_build_space();
	}

	void getColIndex(double dt, bool bsr) {
		mat_add_diagonals(false);
		mat_add_internal_forces(false);
		mat_add_constraint_forces(false);

		// for proximities, mat_add_friction_forces just fill at the positions, so no need to call it...
		//mat_add_friction_forces(false, dt);
		mat_build_inc(bsr);
	}

	void generateIdx(CooMatrix &A)
	{
		//int *rows, int *cols, int len, bool bsr) {
		int num = _matDim;

		BLK_PAR(num);
		if (A._bsr) {
			//kernel_generate_idx_bsr <<<B, T >>>(_matIdx, _rowInc, A._rows, A._cols, num);
			cudaMemcpy(A._cols, _cItems, _cNum*sizeof(int), cudaMemcpyDeviceToDevice);
			// offset one from the _rowInc
			cudaMemcpy(A._rows+1, _rowInc, num*sizeof(int), cudaMemcpyDeviceToDevice);

			CheckI(A._cols, _cNum);
			CheckI(A._rows, num + 1);
		}
		else {
			kernel_generate_idx <<<B, T >>>(_matIdx, _rowInc, A._rows, A._cols, num);
			getLastCudaError("kernel_generate_idx");
		}

//#define DEBUG_6
#ifdef DEBUG_6
		int len = A._num;
		hr = new int[len];
		checkCudaErrors(cudaMemcpy(hr, rows, len*sizeof(int), cudaMemcpyDeviceToHost));
		printf("len = %d\n", len);
		checkCudaErrors(cudaMemcpy(hr, cols, len*sizeof(int), cudaMemcpyDeviceToHost));
		delete [] hr;
#endif
	}

	void fill(double dt, CooMatrix &A, double mrt) {
		iii++;

//#define DEBUG_MATRIX_DATA
#ifdef DEBUG_MATRIX_DATA
		if (iii == 8)
			printf("here!");

		hb = new double3[_matDim];
		mat_fill_diagonals(dt, A);
		checkCudaErrors(cudaMemcpy(hb, totalAux._db,
			_matDim*sizeof(double3), cudaMemcpyDeviceToHost));
		mat_fill_internal_forces(dt, A);
		checkCudaErrors(cudaMemcpy(hb, totalAux._db,
			_matDim*sizeof(double3), cudaMemcpyDeviceToHost));
		mat_fill_constraint_forces(dt, A, mrt, iii);
		checkCudaErrors(cudaMemcpy(hb, totalAux._db,
			_matDim*sizeof(double3), cudaMemcpyDeviceToHost));
		mat_fill_friction_forces(dt, A, mrt);
		checkCudaErrors(cudaMemcpy(hb, totalAux._db,
			_matDim*sizeof(double3), cudaMemcpyDeviceToHost));
		delete [] hb;
#else
		mat_fill_diagonals(dt, A);
		mat_fill_internal_forces(dt, A);
		mat_fill_constraint_forces(dt, A, mrt, iii);
		mat_fill_friction_forces(dt, A, mrt);
#endif
	}

	void solve(CooMatrix &A) {
		extern void CheckIF(int *p, int N, char *fname);

		//CheckIF(A._rows, _matDim);
		//CheckIF(A._cols, A._num);
		//CheckIF(A._rows, _matDim, "e:\\temp2\\aa.txt");
		//CheckIF(A._cols, A._num, "e:\\temp2\\bb.txt");

		gpuSolver(A._num, A._rows, A._cols, A._vals, A._bsr,
			(double *)totalAux._db, (double *)totalAux._dx, _matDim*3);
		getLastCudaError("solve");


//#define DEBUG_B
#ifdef DEBUG_B
		//if (false && iii == 8)
		{
		{
		hb = new double3[_matDim];

		checkCudaErrors(cudaMemcpy(hb, totalAux._db,
			_matDim*sizeof(double3), cudaMemcpyDeviceToHost));

		output1("e:\\temp2\\ob2.txt", (double *)hb, _matDim*3);

		delete [] hb;
		}
		{
			double *hb = new double [A._num];
			cudaMemcpy(hb, A._vals, A._num*sizeof(double), cudaMemcpyDeviceToHost);

			output1("e:\\temp2\\oa2.txt", (double *)hb, A._num);
			delete [] hb;
		}
		{
		hb = new double3[_matDim];

		checkCudaErrors(cudaMemcpy(hb, totalAux._dx,
			_matDim*sizeof(double3), cudaMemcpyDeviceToHost));

		output1("e:\\temp2\\ox2.txt", (double *)hb, _matDim*3);

		delete [] hb;
		}
		//exit(0);
		}
#endif
	}

} SparseMatrixBuilder;

static void implicit_update(int nn, double dt, bool update, double mrt, double mpt)
{
	bool bsr = true;

	SparseMatrixBuilder builder;

	builder.init(nn);

	builder.getColSpace(dt);
	builder.getColIndex(dt, bsr);

	CooMatrix coo;

	coo.init(builder.length(), nn, bsr);

	builder.generateIdx(coo);

	builder.fill(dt, coo, mrt);

	builder.solve(coo);

	builder.destroy();
	coo.destroy();

	currentCloth.updateNodes(totalAux._dx, dt, update);
	// all the information about the obstacles will be packed into currentObj
	currentCloth.project_outside(totalAux._dx, totalAux._dw, currentObj._dm, currentObj._dx, mrt, mpt);
}

void update_bvs_gpu(bool is_cloth, bool ccd, double mrt)
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

void physics_step_gpu (double dt, double mrt, double mpt) 
{
	totalAux.reset(currentCloth.numNode, currentCloth.numFace, currentCloth.numVert);

//#define DEBUG_1
#ifdef DEBUG_1
	hf = new double3[num];
	checkCudaErrors(cudaMemcpy(hf, totalAux.dFext, num*sizeof(double3), cudaMemcpyDeviceToHost));
	hm = new double[num];
	checkCudaErrors(cudaMemcpy(hm, currentCloth.dm, num*sizeof(double), cudaMemcpyDeviceToHost));
#endif

	add_external_forces();

#ifdef DEBUG_1
	checkCudaErrors(cudaMemcpy(hf, totalAux.dFext, num*sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hm, currentCloth.dm, num*sizeof(double), cudaMemcpyDeviceToHost));
	delete [] hf;
	delete [] hm;
#endif


	implicit_update(currentCloth.numNode, dt, false, mrt, mpt);

/*	for (int c = 0; c < sim.cloths.size(); c++) {
		int nn = sim.cloths[c].mesh.nodes.size();
		vector<Vec3> fext(nn, Vec3(0));
		vector<Mat3x3> Jext(nn, Mat3x3(0));
		add_external_forces(sim.cloths[c], sim.gravity, sim.wind, fext, Jext);
		TIMING_BEGIN
			implicit_update(sim.cloths[c], fext, Jext, cons, sim.step_time, false);
		TIMING_END("implicit_update")
	}
*/
	step_mesh_gpu(dt, mrt);
}

//========================================
inline string stringf (const string &format, ...) {
    char buf[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buf, 256, format.c_str(), args);
    va_end(args);
    return std::string(buf);
}

extern "C" void save_objs_gpu(const std::string &prefix)
{
	//currentCloth.saveVtx(stringf("%s_00.vtx", prefix.c_str()));
	//currentObj.saveVtx(stringf("%s_00.vto", prefix.c_str()));
	currentObj.saveObj(stringf("%s_ob.obj", prefix.c_str()));
}

//#######################################################
#include "bvh.cuh"

g_bvh coBVH[2];
g_front fronts[2];
g_pair pairs[2];

///////////////////////////////////////////////////////////

void refitBVH_Serial(bool isCloth, int length)
{
	refit_serial_kernel<<<1, 1, 0>>>
		(coBVH[isCloth]._bvh, coBVH[isCloth]._bxs, coBVH[isCloth]._triBxs,
		length==0 ? coBVH[isCloth]._num  : length);

	getLastCudaError("refit_serial_kernel");
    cudaThreadSynchronize();
}

void refitBVH_Parallel(bool isCloth, int st, int length)
{
    BLK_PAR(length);

	refit_kernel<<< B, T>>>(coBVH[isCloth]._bvh, coBVH[isCloth]._bxs, coBVH[isCloth]._triBxs, st, length);

	getLastCudaError("refit_kernel");
    cudaThreadSynchronize();
}

void refitBVH(bool isCloth)
{
	if (false && !isCloth) {
		currentObj.getBxs();
		refitBVH_Serial(isCloth, coBVH[isCloth]._num);
		coBVH[isCloth].getBxs();
		coBVH[isCloth].printBxs("e:\\temp2\\1.txt");
		exit(0);
		getLastCudaError("refitBVH");
	}


	// before refit, need to get _tri_boxes !!!!
	// copying !!!
	for (int i= coBVH[isCloth]._max_level-1; i>=0; i--) {
		int st = coBVH[isCloth]._level_idx[i];
		int ed = (i != coBVH[isCloth]._max_level-1) ? 
			coBVH[isCloth]._level_idx[i+1]-1 : coBVH[isCloth]._num-1;

		int length = ed-st+1;
		if ( i < 5 ) {
			refitBVH_Serial(isCloth, length+st);
			break;
		} else
		{
			refitBVH_Parallel(isCloth, st, length);
		}
		//coBVH[isCloth].getBxs();
	}

	if (false) {
	coBVH[isCloth].getBxs();

	if (!isCloth) {
		coBVH[isCloth].printBxs("e:\\temp2\\2.txt");
		exit(0);
	}
	}
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
	checkCudaErrors(cudaMalloc((void**)&coBVH[isCloth]._bvh, length*sizeof(int)*2));
	checkCudaErrors(cudaMemcpy(coBVH[isCloth]._bvh, ids, length*sizeof(int)*2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&coBVH[isCloth]._bxs, length*sizeof(g_box)));
	checkCudaErrors(cudaMemset(coBVH[isCloth]._bxs, 0, length*sizeof(g_box)));
	coBVH[isCloth].hBxs = NULL;

	coBVH[1]._triBxs = currentCloth._triBxs;
	coBVH[0]._triBxs = currentObj._triBxs;
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


inline __device__ void pushToFront(int a, int b, uint4 *front, uint *idx)
{
//	(*idx)++;
	if (*idx < MAX_FRONT_NUM) 
	{
		uint offset = atomicAdd(idx, 1);
		front[offset] = make_uint4(a, b, 0, 0);
	}
}

inline __device__ void sproutingAdaptive(int left, int right,
		int *bvhA, g_box *bxsA, int *bvhB, g_box *bxsB, 
		uint4 *front, uint *frontIdx,
		uint2 *pairs, uint *pairIdx, bool update)
{
	uint2 nStack[STACK_SIZE];
	uint nIdx=0;

	for (int i=0; i<4; i++)
	{
		if (isLeaf(left, bvhA) && isLeaf(right, bvhB)) {
				pushToFront(left, right, front, frontIdx);
		} else {
			if (!overlaps(left, right, bxsA, bxsB)) {
					pushToFront(left, right, front, frontIdx);
			} else {
				if (isLeaf(left, bvhA)) {
					PUSH_PAIR(left, getLeftChild(right, bvhB));
					PUSH_PAIR(left, getRightChild(right, bvhB));
				} else {
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
		pushToFront(left, right, front, frontIdx);
	}
}

inline __device__ void addPair(uint a, uint b, uint2 *pairs, uint *idx)
{
	if (*idx < MAX_PAIR_NUM) 
	{
		uint offset = atomicAdd(idx, 1);
		pairs[offset].x = a;
		pairs[offset].y = b;
	}
}

inline __device__ void sprouting(int left, int right,
		int *bvhA, g_box *bxsA, int *bvhB, g_box *bxsB, 
		uint4 *front, uint *frontIdx,
		uint2 *pairs, uint *pairIdx, bool update)
{
	uint2 nStack[STACK_SIZE];
	uint nIdx=0;

	while (1)
	{
		if (isLeaf(left, bvhA) && isLeaf(right, bvhB)) {
			if (update)
				pushToFront(left, right, front, frontIdx);

			if (overlaps(left, right, bxsA, bxsB))
				addPair(getTriID(left, bvhA), getTriID(right, bvhB), pairs, pairIdx);
		} else {
			if (!overlaps(left, right, bxsA, bxsB)) {
				if (update)
					pushToFront(left, right, front, frontIdx);
					
			} else {
				if (isLeaf(left, bvhA)) {
					PUSH_PAIR(left, getLeftChild(right, bvhB));
					PUSH_PAIR(left, getRightChild(right, bvhB));
				} else {
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

__device__ void doPropogate(
	uint4 *front,  uint *frontIdx, int num, 
	int *bvhA, g_box *bxsA, int bvhAnum,
	int *bvhB, g_box *bxsB, int bvhBnum,
	uint2 *pairs, uint *pairIdx, bool update, tri3f *Atris, int idx)
{
	uint4 node = front[idx];
	if (node.z != 0)
		return;

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
		
	if (isLeaf(left, bvhA)) {
		sprouting(left, getLeftChild(right, bvhB), bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update);
		sprouting(left, getRightChild(right, bvhB), bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update);
	} else {
		sprouting(getLeftChild(left, bvhA), right, bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update);
		sprouting(getRightChild(left, bvhA), right, bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update);
	}
}

__global__ void kernelPropogate(uint4 *front,  uint *frontIdx, int num, 
			int *bvhA, g_box *bxsA, int bvhAnum, 
			int *bvhB, g_box *bxsB, int bvhBnum,
			uint2 *pairs, uint *pairIdx, bool update, tri3f *Atris, int stride)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i=0; i<stride; i++) {
		int j = idx*stride+i;
		if (j >= num)
			return;
		
		doPropogate(front,  frontIdx, num, 
			bvhA, bxsA, bvhAnum, bvhB, bxsB, bvhBnum, pairs, pairIdx, update, Atris, j);
	}
}

int g_front::propogate(bool &update, bool self)
{
	uint dummy[1];
	cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1*sizeof(uint), cudaMemcpyDeviceToHost));
	printf("Before propogate, length = %d\n", dummy[0]);
	
	if (dummy[0] != 0) {
	int stride=4;
	BLK_PAR2(dummy[0], stride);

	g_bvh *pb1 = &coBVH[1];
	g_bvh *pb2 = (self) ? &coBVH[1] : &coBVH[0];
	tri3f *faces = (self ? currentCloth._dfnod : NULL);

	if (false) {
	coBVH[0].getBxs();
	coBVH[1].getBxs();
	g_box bx1 = coBVH[0].hBxs[0];
	g_box bx2 = coBVH[1].hBxs[0];
	if (bx1.overlaps(bx2))
		printf("here!\n");

	g_box bx3, bx4;
	cudaMemcpy(&bx3, pb1->_bxs, sizeof(g_box), cudaMemcpyDeviceToHost);
	cudaMemcpy(&bx4, pb2->_bxs, sizeof(g_box), cudaMemcpyDeviceToHost);

	if (bx3.overlaps(bx4))
		printf("here!\n");
	}

    kernelPropogate<<< B, T>>>
		(_dFront, _dIdx, dummy[0],
		pb1->_bvh, pb1->_bxs, pb1->_num,
		pb2->_bvh, pb2->_bxs, pb2->_num,
		pairs[self]._dPairs, pairs[self]._dIdx, update, faces, stride);

	cudaThreadSynchronize();
    getLastCudaError("kernelPropogate");
	}

	cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1*sizeof(uint), cudaMemcpyDeviceToHost));
	printf("After propogate, length = %d\n", dummy[0]);
	
	if (dummy[0] > SAFE_FRONT_NUM) {
		printf("Too long front, stop updating ...\n");
		update = false;
	}
	
	if (dummy[0] > MAX_FRONT_NUM) {
		printf("Too long front, exiting ...\n");
		exit(0);
	}
	return dummy[0];
}

__device__ uint2 get_e(int id, bool free, uint2 *ce, uint2 *oe)
{
	return free ? ce[id] : oe[id];
}

__device__ tri3f get_t(int id, bool free, tri3f *ct, tri3f *ot)
{
	return free ? ct[id] : ot[id];
}

__device__ double3 get_x(int id, bool free, double3 *cx, double3 *ox)
{
	return free ? cx[id] : ox[id];
}

__device__ double get_m(int id, bool free, double *cm, double *om)
{
	return free ? cm[id] : om[id];
}

inline __device__ void doProximityVF(
			int vid, int fid, bool freev, bool freef,
			double3 *cx, tri3f *ctris, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData, double3 *cnn, double3 *cfn, double2 *cvu,
			double3 *ox, tri3f *otris, int *oAdjIdx, int *oAdjData, int *on2vIdx, int *on2vData, double3 *onn, double3 *ofn, double2 *ovu,
			double *cfa, double *cna,
			double mu, double mu_obs, double mrt, double mcs,
			g_IneqCon *cstrs, uint *cstrIdx)
{
	tri3f t = get_t(fid, freef, ctris, otris);
	double3 x1 = get_x(t.id0(), freef, cx, ox);
	double3 x2 = get_x(t.id1(), freef, cx, ox);
	double3 x3 = get_x(t.id2(), freef, cx, ox);
	double3 x4 = get_x(vid, freev, cx, ox);

	double3 n;
	double w[4];
    double d = signed_vf_distance(x4, x1, x2, x3, &n, w);
    d = abs(d);
	    
	const double dmin = 2*mrt;
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

inline __device__ double3 xvpos(double3 x, double3 v, double t)
{
	return x+v*t;
}

inline __device__ int sgn (double x) {return x<0 ? -1 : 1;}

inline __device__ int solve_quadratic (double a, double b, double c, double x[2]) {
    // http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
    double d = b*b - 4*a*c;
    if (d < 0) {
        x[0] = -b/(2*a);
        return 0;
    }
    double q = -(b + sgn(b)*sqrt(d))/2;
    int i = 0;
    if (abs(a) > 1e-12*abs(q))
        x[i++] = q/a;
    if (abs(q) > 1e-12*abs(c))
        x[i++] = c/q;
    if (i==2 && x[0] > x[1])
        fswap(x[0], x[1]);
    return i;
}

inline __device__ double newtons_method (double a, double b, double c, double d, double x0,
                       int init_dir) {
    if (init_dir != 0) {
        // quadratic approximation around x0, assuming y' = 0
        double y0 = d + x0*(c + x0*(b + x0*a)),
               ddy0 = 2*b + x0*(6*a);
        x0 += init_dir*sqrt(abs(2*y0/ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        double y = d + x0*(c + x0*(b + x0*a));
        double dy = c + x0*(2*b + x0*3*a);
        if (dy == 0)
            return x0;
        double x1 = x0 - y/dy;
        if (abs(x0 - x1) < 1e-6)
            return x0;
        x0 = x1;
    }
    return x0;
}

// solves a x^3 + b x^2 + c x + d == 0
inline __device__ int solve_cubic (double a, double b, double c, double d, double x[3]) {
    double xc[2];
    int ncrit = solve_quadratic(3*a, 2*b, c, xc);
    if (ncrit == 0) {
        x[0] = newtons_method(a, b, c, d, xc[0], 0);
        return 1;
    } else if (ncrit == 1) {// cubic is actually quadratic
        return solve_quadratic(b, c, d, x);
    } else {
        double yc[2] = {d + xc[0]*(c + xc[0]*(b + xc[0]*a)),
                        d + xc[1]*(c + xc[1]*(b + xc[1]*a))};
        int i = 0;
        if (yc[0]*a >= 0)
            x[i++] = newtons_method(a, b, c, d, xc[0], -1);
        if (yc[0]*yc[1] <= 0) {
            int closer = abs(yc[0])<abs(yc[1]) ? 0 : 1;
            x[i++] = newtons_method(a, b, c, d, xc[closer], closer==0?1:-1);
        }
        if (yc[1]*a <= 0)
            x[i++] = newtons_method(a, b, c, d, xc[1], 1);
        return i;
    }
}

inline __device__ bool collision_test(
	const double3 &x0, const double3 &x1, const double3 &x2, const double3 &x3,
	const double3 &v0, const double3 &v1, const double3 &v2, const double3 &v3,
	ImpactType type, g_impact &imp)
{
	double a0 = stp(x1, x2, x3),
           a1 = stp(v1, x2, x3) + stp(x1, v2, x3) + stp(x1, x2, v3),
           a2 = stp(x1, v2, v3) + stp(v1, x2, v3) + stp(v1, v2, x3),
           a3 = stp(v1, v2, v3);

	double t[4];
    int nsol = solve_cubic(a3, a2, a1, a0, t);
    t[nsol] = 1; // also check at end of timestep
    for (int i = 0; i < nsol; i++) {
        if (t[i] < 0 || t[i] > 1)
            continue;

        imp._t = t[i];
        double3 tx0 = xvpos(x0, v0, t[i]), tx1 = xvpos(x1+x0, v1+v0, t[i]),
             tx2 = xvpos(x2+x0, v2+v0, t[i]), tx3 = xvpos(x3+x0, v3+v0, t[i]);
        double3 &n = imp._n;
        double *w = imp._w;
        double d;
        bool inside;
        if (type == I_VF) {
            d = signed_vf_distance(tx0, tx1, tx2, tx3, &n, w);
            inside = (fmin(-w[1], fmin(-w[2], -w[3])) >= -1e-6);
        } else {// Impact::EE
            d = signed_ee_distance(tx0, tx1, tx2, tx3, &n, w);
            inside = (fmin(fmin(w[0], w[1]), fmin(-w[2], -w[3])) >= -1e-6);
        }
        if (dot(n, w[1]*v1 + w[2]*v2 + w[3]*v3) > 0)
            n = -n;
        if (fabs(d) < 1e-6 && inside)
            return true;
    }
    return false;
}

inline __device__ void doImpactVF(
			int vid, int fid, bool freev, bool freef,
			double3 *cx, double3 *cx0, tri3f *ctris, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData, double *cm,
			double3 *ox, double3 *ox0, tri3f *otris, int *oAdjIdx, int *oAdjData, int *on2vIdx, int *on2vData, double *om,
			double mu, double mu_obs, 
			g_impact *imps, uint *impIdx,
			g_impNode *inodes, uint *inIdx, int iii)
{
	tri3f t = get_t(fid, freef, ctris, otris);
	double3 x00 = get_x(vid, freev, cx0, ox0);
	double3 x10 = get_x(t.id0(), freef, cx0, ox0);
	double3 x20 = get_x(t.id1(), freef, cx0, ox0);
	double3 x30 = get_x(t.id2(), freef, cx0, ox0);
	double3 x0 = get_x(vid, freev, cx, ox);
	double3 x1 = get_x(t.id0(), freef, cx, ox);
	double3 x2 = get_x(t.id1(), freef, cx, ox);
	double3 x3 = get_x(t.id2(), freef, cx, ox);

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

	double3 p0 = x00;
	double3 p1 = x10-x00;
	double3 p2 = x20-x00;
	double3 p3 = x30-x00;
	double3 v0 = x0-x00;
	double3 v1 = x1-x10-v0;
	double3 v2 = x2-x20-v0;
	double3 v3 = x3-x30-v0;

/*	if (iii == 69 &&
		vid == 162 && t.id0() == 4 &&
		t.id1() == 44 && t.id2() == 32)
		vid = 162;
*/

	bool ret = collision_test(p0, p1, p2, p3, v0, v1,  v2, v3, I_VF, imp);
	if (ret) {
		addImpact(imps, impIdx, imp);
		addNodeInfo(inodes, inIdx, vid, freev, x00, x0, get_m(vid, freev, cm, om));
		addNodeInfo(inodes, inIdx, t.id0(), freef, x10, x1, get_m(t.id0(), freef, cm, om));
		addNodeInfo(inodes, inIdx, t.id1(), freef, x20, x2, get_m(t.id1(), freef, cm, om));
		addNodeInfo(inodes, inIdx, t.id2(), freef, x30, x3, get_m(t.id2(), freef, cm, om));
	}
}

__device__ bool in_wedge (double w, 
	int edge0, int edge1, bool free0, bool free1,
	double3 *cx, tri3f *ctris, uint2 *cef, uint2 *cen, double3 *cn,
	double3 *ox, tri3f *otris, uint2 *oef, uint2 *oen, double3 *on)
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

	//double3 x = (1-w)*edge0->n[0]->x + w*edge0->n[1]->x;
	double3 x = (1-w)*get_x(e0[0], free0, cx, ox) + w*get_x(e0[1], free0, cx, ox);

    bool in = true;
    for (int s = 0; s < 2; s++) {
        //const Face *face = edge1->adjf[s];
        //if (!face)
        //    continue;
		int fid = f1[s];
		if (fid == -1)
			continue;

        //const Node *node0 = edge1->n[s], *node1 = edge1->n[1-s];
        //double3 e = node1->x - node0->x, n = face->n, r = x - node0->x;
		int n0 = e1[s], n1 = e1[1-s];
		double3 e = get_x(n1, free1, cx, ox) - get_x(n0, free1, cx, ox);
		double3 n = get_x(fid, free1, cn, on);
		double3 r = x - get_x(n0, free1, cx, ox);
        in &= (stp(e, n, r) >= 0);
    }
    return in;
}

__device__ void doImpactEE (
	int edge0, int edge1, bool free0, bool free1,
	double3 *cx, double3 *cx0, tri3f *ctris, uint2 *cef, uint2 *cen, double3 *cn, double *cm,
	double3 *ox, double3 *ox0, tri3f *otris, uint2 *oef, uint2 *oen, double3 *on, double *om,
	double mu, double mu_obs,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx)
{
	uint2 e0 = get_e(edge0, free0, cen, oen);
	uint2 e1 = get_e(edge1, free1, cen, oen);
	double3 x10 = get_x(e0.x, free0, cx0, ox0);
	double3 x20 = get_x(e0.y, free0, cx0, ox0);
	double3 x30 = get_x(e1.x, free1, cx0, ox0);
	double3 x40 = get_x(e1.y, free1, cx0, ox0);
	double3 x1 = get_x(e0.x, free0, cx, ox);
	double3 x2 = get_x(e0.y, free0, cx, ox);
	double3 x3 = get_x(e1.x, free1, cx, ox);
	double3 x4 = get_x(e1.y, free1, cx, ox);

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

	double3 p0 = x10;
	double3 p1 = x20-x10;
	double3 p2 = x30-x10;
	double3 p3 = x40-x10;
	double3 v0 = x1-x10;
	double3 v1 = x2-x20-v0;
	double3 v2 = x3-x30-v0;
	double3 v3 = x4-x40-v0;

/*	if (e0.x == 41 && e0.y == 624 &&
		e1.x == 599 && e1.y == 383)
		e0.x = 41;
*/
	bool ret = collision_test(p0, p1, p2, p3, v0, v1,  v2, v3, I_EE, imp);
	if (ret) {
		addImpact(imps, impIdx, imp);
		addNodeInfo(inodes, inIdx, e0.x, free0, x10, x1, get_m(e0.x, free0, cm, om));
		addNodeInfo(inodes, inIdx, e0.y, free0, x20, x2, get_m(e0.y, free0, cm, om));
		addNodeInfo(inodes, inIdx, e1.x, free1, x30, x3, get_m(e1.x, free1, cm, om));
		addNodeInfo(inodes, inIdx, e1.y, free1, x40, x4, get_m(e1.y, free1, cm, om));
	}
}

__device__ void doProximityEE (
	int edge0, int edge1, bool free0, bool free1,
	double3 *cx, tri3f *ctris, uint2 *cef, uint2 *cen, double3 *cn, double3 *cnn,
	double3 *ox, tri3f *otris, uint2 *oef, uint2 *oen, double3 *on, double3 *onn,
	double *cfa,
	double mu, double mu_obs, double mrt, double mcs,
	g_IneqCon *cstrs, uint *cstrIdx)
{
	uint2 e0 = get_e(edge0, free0, cen, oen);
	uint2 e1 = get_e(edge1, free1, cen, oen);
	double3 e00 = get_x(e0.x, free0, cx, ox);
	double3 e01 = get_x(e0.y, free0, cx, ox);
	double3 e10 = get_x(e1.x, free1, cx, ox);
	double3 e11 = get_x(e1.y, free1, cx, ox);


/*	if (e0.x == 891 && e0.y == 446 &&
		e1.x == 18 && e1.y == 188)
		e1.x = 18;
*/
    double3 n;
    double w[4];
    double d = signed_ee_distance(e00, e01, e10, e11, &n, w);
    d = abs(d);

	const double dmin = 2*mrt;
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

	uint side0=-1, side1=-1;
	if (free0) {
		double3 en = get_x(e0.x, free0, cnn, onn) + get_x(e0.y, free0, cnn, onn);
		side0 = dot(n, en) >= 0 ? 0 : 1;
	}
	if (free1) {
		double3 en = get_x(e1.x, free1, cnn, onn) + get_x(e1.y, free1, cnn, onn);
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
		mu, mu_obs, imps, impIdx, inodes, inIdx, iii);

#define DO_SELF_VF_IMPACT(vid, fid) \
	doImpactVF(vid, fid, true, true, \
		cx, cx0, ctris, cAdjIdx, cAdjData, cn2vIdx, cn2vData, cm,\
		cx, cx0, ctris, cAdjIdx, cAdjData, cn2vIdx, cn2vData, cm,\
		mu, mu, imps, impIdx, inodes, inIdx, 0);

#define DO_EE_IMPACT(e0, e1, free0, free1) \
	doImpactEE(e0, e1, free0, free1, \
		cx, cx0, ctris, cef, cen, cn, cm, ox, ox0, otris, oef, oen, on, om, mu, mu_obs, imps, impIdx, inodes, inIdx);

#define DO_SELF_EE_IMPACT(e0, e1) \
	doImpactEE(e0, e1, true, true, \
		cx, cx0, ctris, cef, cen, cn, cm, cx, cx0, ctris, cef, cen, cn, cm, mu, mu, imps, impIdx, inodes, inIdx);

inline __device__ bool VtxMask(uint *maskes, uint tri_id, uint i) 
{
	return maskes[tri_id] & (0x1 << i) ? true : false;
}

inline __device__	bool EdgeMask(uint *maskes, uint tri_id, uint i)
{
	return maskes[tri_id] & (0x8 << i) ? true : false;
}

__global__ void kernelGetImpacts(
	uint2 *pairs, int num,
	double3 *cx, double3 *cx0, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, double3 *cn, double *cm,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, uint *cmask, int *cn2vIdx, int *cn2vData,
	double3 *ox, double3 *ox0, tri3f *otris, tri3f *oedgs, int *oAdjIdx, int *oAdjData, uint2 *oef, uint2 *oen, double3 *on, double *om,
	g_box *ovbxs, g_box *oebxs, g_box *ofbxs, uint *omask, int *on2vIdx, int *on2vData,
	double mu, double mu_obs, 
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx,
	int stride, int iii)
{
    int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i=0; i<stride; i++) {

	int j=idxx*stride+i;
	if (j>=num)
		return;
		
	int idx = j;

	uint2 pair = pairs[idx];
	int fid1 = pair.x;
	int fid2 = pair.y;

	tri3f t1 = ctris[fid1];
	for (int i=0; i<3; i++) {
		int vid = t1.id(i);
		if (VtxMask(cmask, fid1, i) && cvbxs[vid].overlaps(ofbxs[fid2])) {
			DO_VF_IMPACT(vid, fid2, true, false);
		}
	}

	tri3f t2 = otris[fid2];
	for (int i=0; i<3; i++) {
		int vid = t2.id(i);
		if (VtxMask(omask, fid2, i) && ovbxs[vid].overlaps(cfbxs[fid1])) {
			DO_VF_IMPACT(vid, fid1, false, true);
		}
	}

	tri3f e1 = cedgs[fid1];
	tri3f e2 = oedgs[fid2];
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++) {
			int ee1 = e1.id(i);
			int ee2 = e2.id(j);

			if (EdgeMask(cmask, fid1, i) && EdgeMask(omask, fid2, j) && cebxs[ee1].overlaps(oebxs[ee2]))
				DO_EE_IMPACT(ee1, ee2, true, false);
		}
	}
}

__global__ void kernelGetProximities(
	uint2 *pairs, int num,
	double3 *cx, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, double3 *cfn, double2 *cvu,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, uint *cmask, int *cn2vIdx, int *cn2vData, double3 *cnn,
	double3 *ox, tri3f *otris, tri3f *oedgs, int *oAdjIdx, int *oAdjData, uint2 *oef, uint2 *oen, double3 *ofn, double2 *ovu,
	g_box *ovbxs, g_box *oebxs, g_box *ofbxs, uint *omask, int *on2vIdx, int *on2vData, double3 *onn,
	double *cfa, double *cna,
	double mu, double mu_obs, double mrt, double mcs,
	g_IneqCon *cstrs, uint *cstrIdx,
	int stride)
{
    int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i=0; i<stride; i++) {

	int j=idxx*stride+i;
	if (j>=num)
		return;
		
	int idx = j;

	uint2 pair = pairs[idx];
	int fid1 = pair.x;
	int fid2 = pair.y;

	tri3f t1 = ctris[fid1];
	for (int i=0; i<3; i++) {
		int vid = t1.id(i);
		if (VtxMask(cmask, fid1, i) && cvbxs[vid].overlaps(ofbxs[fid2])) {
			DO_VF(vid, fid2, true, false);
		}
	}

	tri3f t2 = otris[fid2];
	for (int i=0; i<3; i++) {
		int vid = t2.id(i);
		if (VtxMask(omask, fid2, i) && ovbxs[vid].overlaps(cfbxs[fid1])) {
			DO_VF(vid, fid1, false, true);
		}
	}

	tri3f e1 = cedgs[fid1];
	tri3f e2 = oedgs[fid2];
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++) {
			int ee1 = e1.id(i);
			int ee2 = e2.id(j);

			if (EdgeMask(cmask, fid1, i) && EdgeMask(omask, fid2, j) && cebxs[ee1].overlaps(oebxs[ee2]))
				DO_EE(ee1, ee2, true, false);
		}
	}
}

__device__ void doSelfVF(
	int nid, int ffid, int fv,
	double3 *cx, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, double3 *cfn,
	int *cn2vIdx, int *cn2vData, double2 *cvu,
	double mu, double3 *cnn, double *cfa, double *cna, double mrt, double mcs,
	g_IneqCon *cstrs, uint *cstrIdx)
{
/*	int id = vid;
	int adjStart =  (id == 0) ? 0 : cAdjIdx[id-1];
	int adjNum = cAdjIdx[id]-adjStart;
	for (int i=0; i<adjNum; i++) {
		int fid = cAdjData[i+adjStart];

		if (!covertex(ffid, fid, ctris)) {
			if (fid == fv) {
				DO_SELF_VF(vid, ffid);
			} else
				return;					
		}
	}
*/
	VLST_BEGIN(cn2vIdx, cn2vData, nid)
	FLST_BEGIN(cAdjIdx, cAdjData, vid)

		if (!covertex(ffid, fid, ctris)) {
			if (fid == fv) {
				DO_SELF_VF(nid, ffid);
			} else
				return;					
		}

	FLST_END
	VLST_END
}

__device__ void doSelfVFImpact(
	int nid, int ffid, int fv,
	double3 *cx, double3 *cx0, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, uint2 *cef, uint2 *cen, double3 *cn,
	int *cn2vIdx, int *cn2vData, double *cm,
	double mu,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx)
{
/*	int id = vid;
	int adjStart =  (id == 0) ? 0 : cAdjIdx[id-1];
	int adjNum = cAdjIdx[id]-adjStart;
	for (int i=0; i<adjNum; i++) {
		int fid = cAdjData[i+adjStart];

		if (!covertex(ffid, fid, ctris)) {
			if (fid == fv) {
				DO_SELF_VF_IMPACT(vid, ffid);
			} else
				return;					
		}
	}
*/
	VLST_BEGIN(cn2vIdx, cn2vData, nid)
	FLST_BEGIN(cAdjIdx, cAdjData, vid)

		if (!covertex(ffid, fid, ctris)) {
			if (fid == fv) {
				DO_SELF_VF_IMPACT(nid, ffid);
			} else
				return;					
		}

	FLST_END
	VLST_END
}

__device__ void doSelfEEImpact(
	int e1, int e2, int f1, int f2,
	double3 *cx, double3 *cx0, tri3f *ctris, uint2 *cef, uint2 *cen, double3 *cn, double *cm,
	double mu,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx)

{
	unsigned int e[2];
	unsigned int f[2];

	if (e1 > e2) {
		e[0] = e1, e[1] = e2;
		f[0] = f1, f[1] = f2;
	} else {
		e[0] = e2, e[1] = e1;
		f[0] = f2, f[1] = f1;
	}


	for (int i=0; i<2; i++)
		for (int j=0; j<2; j++) {
			uint2 ef0 = cef[e[0]];
			uint2 ef1 = cef[e[1]];

			uint ff1 = (i == 0) ? ef0.x : ef0.y; 
			uint ff2 = (j == 0) ? ef1.x : ef1.y;  

			if (ff1 == -1 || ff2 == -1)
				continue;

			if (!covertex(ff1, ff2, ctris)) {
				if (ff1 == f[0] && ff2 == f[1]) {
					DO_SELF_EE_IMPACT(e1, e2)
				} else
					return;
			}
		}
}

__device__ void doSelfEE(
	int e1, int e2, int f1, int f2,
	double3 *cx, tri3f *ctris, uint2 *cef, uint2 *cen, double3 *cfn, double mu, double3 *cnn, double2 *cvu, double *cfa,
	double mrt, double mcs,
	g_IneqCon *cstrs, uint *cstrIdx)

{
	unsigned int e[2];
	unsigned int f[2];

	if (e1 > e2) {
		e[0] = e1, e[1] = e2;
		f[0] = f1, f[1] = f2;
	} else {
		e[0] = e2, e[1] = e1;
		f[0] = f2, f[1] = f1;
	}


	for (int i=0; i<2; i++)
		for (int j=0; j<2; j++) {
			uint2 ef0 = cef[e[0]];
			uint2 ef1 = cef[e[1]];

			uint ff1 = (i == 0) ? ef0.x : ef0.y; 
			uint ff2 = (j == 0) ? ef1.x : ef1.y;  

			if (ff1 == -1 || ff2 == -1)
				continue;

			if (!covertex(ff1, ff2, ctris)) {
				if (ff1 == f[0] && ff2 == f[1]) {
					DO_SELF_EE(e1, e2)
				} else
					return;
			}
		}
}

__global__ void kernelGetSelfImpacts(
	uint2 *pairs, int num,
	double3 *cx, double3 *cx0, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData,
	uint2 *cef, uint2 *cen, double3 *cn, double *cm,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, double mu,
	g_impact *imps, uint *impIdx,
	g_impNode *inodes, uint *inIdx,
	int stride)
{
    int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i=0; i<stride; i++) {

	int j=idxx*stride+i;
	if (j>=num)
		return;
		
	int idx = j;

	uint2 pair = pairs[idx];
	int fid1 = pair.x;
	int fid2 = pair.y;

	tri3f t1 = ctris[fid1];
	for (int i=0; i<3; i++) {
		int vid = t1.id(i);
		if (cvbxs[vid].overlaps(cfbxs[fid2])) {
			doSelfVFImpact(vid, fid2, fid1,
					cx, cx0, ctris, cedgs, cAdjIdx, cAdjData, cef, cen, cn, cn2vIdx, cn2vData, cm, mu, imps, impIdx, inodes, inIdx);
		}
	}

	tri3f t2 = ctris[fid2];
	for (int i=0; i<3; i++) {
		int vid = t2.id(i);
		if (cvbxs[vid].overlaps(cfbxs[fid1])) {
			doSelfVFImpact(vid, fid1, fid2,
					cx, cx0, ctris, cedgs, cAdjIdx, cAdjData, cef, cen, cn, cn2vIdx, cn2vData, cm, mu, imps, impIdx, inodes, inIdx);
		}
	}

	tri3f e1 = cedgs[fid1];
	tri3f e2 = cedgs[fid2];
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++) {
			int ee1 = e1.id(i);
			int ee2 = e2.id(j);

			if (cebxs[ee1].overlaps(cebxs[ee2]))
				doSelfEEImpact(ee1, ee2, fid1, fid2, cx, cx0, ctris, cef, cen, cn, cm, mu, imps, impIdx, inodes, inIdx);
		}
	}
}

__global__ void kernelGetSelfProximities(
	uint2 *pairs, int num,
	double3 *cx, tri3f *ctris, tri3f *cedgs, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData,
	uint2 *cef, uint2 *cen, double3 *cfn, double3 *cnn, double2 *cvu,
	double *cfa, double *cna,
	g_box *cvbxs, g_box *cebxs, g_box *cfbxs, double mu, double mrt, double mcs,
	g_IneqCon *cstrs, uint *cstrIdx,
	int stride)
{
    int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i=0; i<stride; i++) {

	int j=idxx*stride+i;
	if (j>=num)
		return;
		
	int idx = j;

	uint2 pair = pairs[idx];
	int fid1 = pair.x;
	int fid2 = pair.y;

	tri3f t1 = ctris[fid1];
	for (int i=0; i<3; i++) {
		int vid = t1.id(i);
		if (cvbxs[vid].overlaps(cfbxs[fid2])) {
			doSelfVF(vid, fid2, fid1,
					cx, ctris, cedgs, cAdjIdx, cAdjData, cef, cen, cfn, cn2vIdx, cn2vData, cvu, mu, cnn, cfa, cna, mrt, mcs, cstrs, cstrIdx);
		}
	}

	tri3f t2 = ctris[fid2];
	for (int i=0; i<3; i++) {
		int vid = t2.id(i);
		if (cvbxs[vid].overlaps(cfbxs[fid1])) {
			doSelfVF(vid, fid1, fid2,
					cx, ctris, cedgs, cAdjIdx, cAdjData, cef, cen, cfn, cn2vIdx, cn2vData, cvu, mu, cnn, cfa, cna, mrt, mcs, cstrs, cstrIdx);
		}
	}

	tri3f e1 = cedgs[fid1];
	tri3f e2 = cedgs[fid2];
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++) {
			int ee1 = e1.id(i);
			int ee2 = e2.id(j);

			if (cebxs[ee1].overlaps(cebxs[ee2]))
				doSelfEE(ee1, ee2, fid1, fid2, cx, ctris, cef, cen, cfn, mu, cnn, cvu, cfa, mrt, mcs, cstrs, cstrIdx);
		}
	}
}

int g_pair::getProximityConstraints(bool self, double mu, double mu_obs, double mrt, double mcs)
{
	int num = length();
	if (self)
		printf("self pair = %d\n", num);
	else
		printf("inter-obj pair = %d\n", num);

	if (num == 0)
		return 0;

	int stride=4;
	//BLK_PAR2(num, stride);
	BLK_PAR3(num, stride, 32);

	if (self) {
		kernelGetSelfProximities<<< B, T>>>(_dPairs, num,
		currentCloth._dx, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData, currentCloth._dn2vIdx, currentCloth._dn2vData,
		currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dn, currentCloth._dvu, currentCloth.dfa, currentCloth._da,
		currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, mu, mrt, mcs,
		Cstrs._dIneqs, Cstrs._dIneqNum,
		stride);
		getLastCudaError("kernelGetSelfProximities");
	}
	else {
		kernelGetProximities<<< B, T>>>(_dPairs, num,
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
	printf("constraint num = %d\n", len);

	return 0;
}

	static int vvv=0;

int g_pair::getImpacts(bool self, double mu, double mu_obs)
{
	vvv++;

	int num = length();
	if (self)
		printf("self pair = %d\n", num);
	else
		printf("inter-obj pair = %d\n", num);

	if (num == 0)
		return 0;

	int stride=4;
	//BLK_PAR2(num, stride);
	//BLK_PAR3(num, stride, 16);
	// optimized selection...
	BLK_PAR3(num, stride, 32);

	if (self) {
		kernelGetSelfImpacts<<< B, T>>>(_dPairs, num,
		currentCloth._dx, currentCloth._dx0, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData,
		currentCloth._dn2vIdx, currentCloth._dn2vData,
		currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dm,
		currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, mu,
		Impcts._dImps, Impcts._dImpNum,
		Impcts._dNodes, Impcts._dNodeNum,
		stride);
		getLastCudaError("kernelGetSelfImpacts");
	}
	else {
		kernelGetImpacts<<< B, T>>>(_dPairs, num,
		currentCloth._dx, currentCloth._dx0, currentCloth._dfnod, currentCloth.dfedg, currentCloth._dvAdjIdx, currentCloth._dvAdjData,
		currentCloth.def, currentCloth.den, currentCloth._dfn, currentCloth._dm,
		currentCloth._vtxBxs, currentCloth._edgeBxs, currentCloth._triBxs, currentCloth._dfmask, currentCloth._dn2vIdx, currentCloth._dn2vData,
		currentObj._dx, currentObj._dx0, currentObj._dfnod, currentObj.dfedg, currentObj._dvAdjIdx, currentObj._dvAdjData,
		currentObj.def, currentObj.den, currentObj._dfn, currentObj._dm,
		currentObj._vtxBxs, currentObj._edgeBxs, currentObj._triBxs, currentObj._dfmask, currentObj._dn2vIdx, currentObj._dn2vData,
		mu, mu_obs,
		Impcts._dImps, Impcts._dImpNum,
		Impcts._dNodes, Impcts._dNodeNum,
		stride, vvv);
		getLastCudaError("kernelGetImpacts");
	}

	int len = Impcts.updateLength();
	printf("impact num = %d\n", len);

	return 0;
}

void init_pairs_gpu()
{
	pairs[0].init();
	pairs[1].init();
}

void get_collisions_gpu (double dt, double mu, double mu_obs, double mrt, double mcs)
{
	static bool update = true;
	
	TIMING_BEGIN
	pairs[0].clear();
	pairs[1].clear();
	Cstrs.clear();

	if (enable_collision) {
		refitBVH(true);
		refitBVH(false);
		
	TIMING_BEGIN
		//inter-object CD
		fronts[0].propogate(update, 0);
		//intra-object CD
		fronts[1].propogate(update, 1);
    cudaThreadSynchronize();
	TIMING_END("%%%get_collisions_gpu_1")

	TIMING_BEGIN
		pairs[0].getProximityConstraints(0, mu, mu_obs, mrt, mcs);
		pairs[1].getProximityConstraints(1, mu, mu_obs, mrt, mcs);
    cudaThreadSynchronize();
	TIMING_END("%%%get_collisions_gpu_2")

		Cstrs.filtering(
			totalAux._vdists, totalAux._fdists, totalAux._edists,
			currentCloth.numNode, currentCloth.numFace, currentCloth.numEdge);
	}

    cudaThreadSynchronize();
	TIMING_END("$$$get_collisions_gpu")
}

int get_impacts_gpu (double dt, double mu, double mu_obs)
{
	static bool update = true;

	TIMING_BEGIN
	pairs[0].clear();
	pairs[1].clear();
	Impcts.clear();
	
	if (enable_collision) {
		refitBVH(true);
		refitBVH(false);

	TIMING_BEGIN
		//inter-object CCD
		fronts[0].propogate(update, 0);
		//intra-object CCD
		fronts[1].propogate(update, 1);
    cudaThreadSynchronize();
	TIMING_END("%%%get_impacts_gpu_1")

	TIMING_BEGIN
		pairs[0].getImpacts(0, mu, mu_obs);
		pairs[1].getImpacts(1, mu, mu_obs);
    cudaThreadSynchronize();
	TIMING_END("%%%get_impacts_gpu_2")
	}

    cudaThreadSynchronize();
	TIMING_END("$$$get_impacts_gpu")
	
	return Impcts.length();
}

void get_impact_data_gpu(void *data, void *nodes, int num)
{
	assert(num == Impcts.length());

	checkCudaErrors(cudaMemcpy(data,
		Impcts.data(), sizeof(g_impact)*num, cudaMemcpyDeviceToHost));
		
	if (nodes)
	checkCudaErrors(cudaMemcpy(nodes,
		Impcts.nodes(), sizeof(g_impNode)*num*4, cudaMemcpyDeviceToHost));
}

__global__ void
kernel_updatingX(g_impNode *nodes, double3 *x, int num)
{
	LEN_CHK(num);

	g_impNode n = nodes[idx];

	if (n._f) {
		x[n._n] = n._x;
	}
}

__global__ void
kernel_updatingV(g_impNode *nodes, double3 *v, double dt, int num)
{
	LEN_CHK(num);

	g_impNode n = nodes[idx];

	if (n._f) {
		v[n._n] += (n._x - n._ox)/dt;
	}
}

void put_impact_node_gpu(void *data, int num, double mrt)
{
	// using Impcts.nodes() as a temporary buffer ...
	checkCudaErrors(cudaMemcpy(Impcts.nodes(), data, sizeof(g_impNode)*num, cudaMemcpyHostToDevice));

	{ // now updating x
		BLK_PAR(num);
		kernel_updatingX<<<B, T>>>(Impcts.nodes(), currentCloth._dx, num);
		getLastCudaError("kernel_updatingX");
	}

	// updating bouding volumes
	currentCloth.computeWSdata(mrt, true);
}

void put_impact_vel_gpu(void *data, int num, double dt)
{
	// using Impcts.nodes() as a temporary buffer ...
	checkCudaErrors(cudaMemcpy(Impcts.nodes(), data, sizeof(g_impNode)*num, cudaMemcpyHostToDevice));

	{ // now updating x
		BLK_PAR(num);
		kernel_updatingV<<<B, T>>>(Impcts.nodes(), currentCloth._dv, dt, num);
		getLastCudaError("kernel_updatingV");
	}
}

void load_x_gpu()
{
	FILE *fp = fopen("e:\\temp2\\x.dat", "rb");
	int num;
	fread(&num, sizeof(int), 1, fp);
	if (num != currentCloth.numNode) {
		printf("x file donot match!\n");
		exit(0);
	}

	fread(currentCloth.hx, sizeof(double3), num, fp);
	cudaMemcpy(currentCloth._dx, currentCloth.hx, sizeof(double3)*num, cudaMemcpyHostToDevice);
	fclose(fp);
}

// need more
void set_cache_perf_gpu()
{
	cudaFuncSetCacheConfig(kernelGetSelfImpacts, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelGetImpacts, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelGetSelfProximities, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelGetProximities, cudaFuncCachePreferL1);
}	


// for strain limiting

void backup_nodes_gpu()
{
	checkCudaErrors(
		cudaMemcpy(totalAux._dx, currentCloth._dx,
		currentCloth.numNode*sizeof(double3),
		cudaMemcpyDeviceToDevice));
}

//===============================================================
#define EPSILON_GD 10e-12
#define MAX_ITERATIONS 1000

struct tmImpactInfo {
	uint _nodes[4];
	bool _frees[4];
	double _w[4];

    ImpactType _type;
    double _t;
	double3 _n;
};

inline __device__ void
bake_impact(tmImpactInfo &imp, double *w, double3 &n,
double3 *cx, double3 *ox, double3 *cx0, double3 *ox0)
{
	double3 x[4], x0[4];

	for (int i = 0; i<4; i++) {
		x[i] = get_x(imp._nodes[i], imp._frees[i], cx, ox);
		x0[i] = get_x(imp._nodes[i], imp._frees[i], cx0, ox0);
	}

	double3 p0 = x0[0];
	double3 p1 = x0[1] - p0;
	double3 p2 = x0[2] - p0;
	double3 p3 = x0[3] - p0;
	double3 v0 = x[0] - x0[0];
	double3 v1 = x[1] - x0[1] - v0;
	double3 v2 = x[2] - x0[2] - v0;
	double3 v3 = x[3] - x0[3] - v0;

	g_impact dummy;
	bool ret = collision_test(p0, p1, p2, p3, v0, v1, v2, v3, imp._type, dummy);
	if (ret == false) {
		double d = 0;
		if (imp._type == I_VF)
			d = signed_vf_distance(x[0], x[1], x[2], x[3], &n, w);
		else
			d = signed_ee_distance(x[0], x[1], x[2], x[3], &n, w);
			
		n = -n;
	}
	else {
		n = dummy._n;
		w[0] = dummy._w[0];
		w[1] = dummy._w[1];
		w[2] = dummy._w[2];
		w[3] = dummy._w[3];
	}
}

inline __device__ void set_subvec(double *x, int i, const double3 &xi)
{
	x[i * 3 + 0] = xi.x;
	x[i * 3 + 1] = xi.y;
	x[i * 3 + 2] = xi.z;
}

inline __device__ double3 get_subvec(const double *x, int i)
{
	return make_double3(x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2]);
}

inline __device__ void add_subvec(double *x, int i, const double3 &xi)
{
	x[i * 3 + 0] += xi.x;
	x[i * 3 + 1] += xi.y;
	x[i * 3 + 2] += xi.z;
}

inline __device__ void
preProcessing(int ncon, tmImpactInfo *imps, uint *nodes, int &nvar,
double3 *cx, double3 *ox, double3 *cx0, double3 *ox0, double *m, double &invM)
{
	// get nodes
	nvar = 0;
	for (int i = 0; i<ncon; i++) {
		tmImpactInfo *imp = imps + i;

		for (int k = 0; k<4; k++) {
			if (imp->_frees[k] == false)
				continue;

			int n = imp->_nodes[k];
			bool find = false;
			for (int j = 0; j<nvar; j++) {
				if (nodes[j] == n) {
					find = true;
					break;
				}
			}

			if (!find) {
				nodes[nvar++] = n;
			}
		}
	}

	// baking impacts
	/*
	for (int i = 0; i<ncon; i++) {
		bake_impact(imps[i], w + i * 4, n[i], cx, ox, cx0, ox0);
	}
	*/

	// get invM
	invM = 0;
	for (int i = 0; i<nvar; i++)
		invM += 1 / m[nodes[i]];
	invM /= nvar;

}

inline __device__ void
initializeSX(double *sX, uint *nodes, int nv, double3 *x)
{
	for (int i = 0; i<nv; i++)
		set_subvec(sX, i, x[nodes[i]]);
}

inline __device__ void
finalizeSX(double *sX, uint *nodes, int nv, double3 *x)
{
	for (int i = 0; i<nv; i++) {
		x[nodes[i]] = get_subvec(sX, i);
	}
}

inline __device__ void
precomputeSX(double *sX, uint *nodes, int nv, double3 *x)
{
	finalizeSX(sX, nodes, nv, x);
}

inline __device__ double
squaredNormal(double *x, int nv)
{
	double sum = 0;
	for (int i = 0; i < nv; i++)
		sum += x[i] * x[i];
	return sum;
}

inline __device__ double clampViolation(double x, int sign)
{
	return (sign<0) ? fmax(x, 0.) : (sign>0) ? fmin(x, 0.) : x;
}

inline __device__ double objective(uint *nodes, int nv, double3 *outX, double3 *outXold, double *outM, double invM)
{
	double e = 0;
	for (int n = 0; n <nv; n++) {
		uint nd = nodes[n];
		double3 dx = outX[nd] - outXold[nd];
		e += invM * outM[nd] * dot(dx, dx) * 0.5;
	}
	return e;
}

inline __device__ void obj_grad(double *grad, uint *nodes, int nv, double3 *outX, double3 *outXold, double *outM, double invM)
{
	for (int n = 0; n <nv; n++) {
		uint nd = nodes[n];
		double3 dx = outX[nd] - outXold[nd];
		set_subvec(grad, n, invM*outM[nd] * dx);
	}
}

inline __device__ double
constraintSX(tmImpactInfo *imp, int &sign, double thickness, double3 *cx, double3 *ox)
{
	sign = 1;
	double c = -thickness;
	for (int i = 0; i<4; i++) {
		uint nd = imp->_nodes[i];
		bool f = imp->_frees[i];

		double3 x = get_x(nd, f, cx, ox);
		c += imp->_w[i] * dot(imp->_n, x);
	}
	return c;
}

inline __device__ int
find(uint id, uint *ids, int num)
{
	for (int i = 0; i < num; i++)
		if (id == ids[i])
			return i;

	return -1;
}

inline __device__ void
con_grad(tmImpactInfo *imp, double factor, double *grad, uint *nodes, int nv)
{
	for (int i = 0; i<4; i++) {
		if (imp->_frees[i]) {
			int k = find(imp->_nodes[i], nodes, nv);
			assert(k != -1);

			add_subvec(grad, k, factor*imp->_w[i] * imp->_n);
		}
	}
}

inline __device__ void
valueGrad(double *sx, double &val, double *grad, double mu, double *lamb, int nc,
uint *nodes, int nv, double3 *outX, double3 *outXold, double *outM, double invM,
tmImpactInfo *imps, double3 *outObjX, double thickness)
{
	precomputeSX(sx, nodes, nv, outX);
	val = objective(nodes, nv, outX, outXold, outM, invM);
	obj_grad(grad, nodes, nv, outX, outXold, outM, invM);

	for (int j = 0; j < nc; j++) {
		tmImpactInfo *imp = imps + j;

		int sign;
		double gj = constraintSX(imp, sign, thickness, outX, outObjX);
		double cj = clampViolation(gj + lamb[j] / mu, sign);
		if (cj != 0) {
			val += mu * 0.5 * cj * cj;
			con_grad(imp, mu*cj, grad, nodes, nv);
		}
	}
}

inline __device__ double
Value(double *sx, double mu, double *lamb, int nc,
uint *nodes, int nv, double3 *outX, double3 *outXold, double *outM, double invM,
tmImpactInfo *imps, double3 *outObjX, double thickness)
{
	precomputeSX(sx, nodes, nv, outX);
	double val = objective(nodes, nv, outX, outXold, outM, invM);

	for (int j = 0; j < nc; j++) {
		tmImpactInfo *imp = imps + j;

		int sign;
		double gj = constraintSX(imp, sign, thickness, outX, outObjX);
		double cj = clampViolation(gj + lamb[j] / mu, sign);
		if (cj != 0) {
			val += mu * 0.5 * cj * cj;
		}
	}
	return val;
}

inline __device__ void
interplateX(double *ret, double *x, double *dir, double step, int nv)
{
	for (int i = 0; i < nv; i++)
		ret[i] = x[i] + dir[i] * step;
}

inline __device__ double
lineSearch(double *x, double *g, int nv, double sgn, double val, double *x_plus_tdx, double mu, double *lamb,
int ncon,
uint *nodes, int nvar, double3 *outX, double3 *outXold, double *outM, double invM,
tmImpactInfo *imps, double3 *outObjX, double thickness)
{
	double currentVal = val;
	double m_ls_beta = 0.1;
	double m_ls_alpha = 0.25;
	double t = 1.0 / m_ls_beta;
	double lhs, rhs;

	do {
		t *= m_ls_beta;
		//x_plus_tdx = x + t*descent_dir;
		interplateX(x_plus_tdx, x, g, -t, nv);

		lhs = Value(x_plus_tdx, mu, lamb, ncon, nodes, nvar, outX, outXold, outM, invM, imps, outObjX, thickness);
		rhs = currentVal - m_ls_alpha * t * sgn;
	} while (lhs >= rhs && t > EPSILON_GD);

	return t;
}

inline __device__ double
lineSearch2(double *x, double *s, int nv, double sgn, double val, double *x_plus_tdx, double mu, double *lamb,
int ncon,
uint *nodes, int nvar, double3 *outX, double3 *outXold, double *outM, double invM,
tmImpactInfo *imps, double3 *outObjX, double thickness)
{
	double currentVal = val;
	double m_ls_beta = 0.1;
	double m_ls_alpha = 0.25;
	double t = 1.0 / m_ls_beta;
	double lhs, rhs;

	do {
		t *= m_ls_beta;
		//x_plus_tdx = x + t*descent_dir;
		interplateX(x_plus_tdx, x, s, t, nv);

		lhs = Value(x_plus_tdx, mu, lamb, ncon, nodes, nvar, outX, outXold, outM, invM, imps, outObjX, thickness);
		rhs = currentVal - m_ls_alpha * t * sgn;
	} while (lhs >= rhs && t > EPSILON_GD);

	return t;
}

inline __device__ void
getS(double *s, double *g, int nv)
{
	for (int i = 0; i<nv; i++)
		s[i] = -g[i];
}

inline __device__ void
updateS(double *s, double *g, double beta, int nv)
{
	for (int i = 0; i<nv; i++)
		s[i] = -g[i] + beta*s[i];
}

inline __device__ void
updateX(double *x, double *g, double step, int nv)
{
	for (int i = 0; i < nv; i++)
		x[i] -= g[i] * step;
}


inline __device__ void
multiplierUpdate(double *sx, double mu, double *lamb, int nc,
uint *nodes, int nv, double3 *outX, double3 *outXold, double *outM, double invM,
tmImpactInfo *imps, double3 *outObjX, double thickness)
{
	precomputeSX(sx, nodes, nv, outX);
	for (int j = 0; j < nc; j++) {
		tmImpactInfo *imp = imps + j;

		int sign;
		double gj = constraintSX(imp, sign, thickness, outX, outObjX);
		lamb[j] = clampViolation(lamb[j] + mu*gj, sign);
	}
}

inline __device__ void
augumented_lagrangian_gpu(
int ncon, tmImpactInfo *imps, uint *nodes,
double *sLambda, double *sX, double *sG,
double *xPlusTdx, double *sS,
double3 *cx, double3 *ox, double3 *cx0, double3 *ox0, double3 *cxOld,
double thickness, double *cm, int *ret)
{
	int nvar = 0;
	double invM = 0;
	for (int i = 0; i < ncon; i++)
		sLambda[i] = 0;

/*
	if (imps->_nodes[0] == 3800 &&
		imps->_nodes[1] == 805 &&
		imps->_nodes[2] == 434 &&
		imps->_nodes[3] == 1073)
		nvar = 0;
*/

	*ret = 0;
	preProcessing(ncon, imps, nodes, nvar, cx, ox, cx0, ox0, cm, invM);
	initializeSX(sX, nodes, nvar, cx);

	// x0, g0
	int nv = nvar * 3;
	double val;

	double sMu = 1e3;
	valueGrad(sX, val, sG, sMu, sLambda, ncon, nodes, nvar, cx, cxOld, cm, invM, imps, ox, thickness);

	double sgn = squaredNormal(sG, nv);
	if (sgn < EPSILON_GD) {
		printf("already converge at x0 ...\n");
		return;
	}

	double step = lineSearch(sX, sG, nv, sgn, val, xPlusTdx, sMu, sLambda, ncon, nodes, nvar, cx, cxOld, cm, invM, imps, ox, thickness);

	updateX(sX, sG, step, nv);
	multiplierUpdate(sX, sMu, sLambda, ncon, nodes, nvar, cx, cxOld, cm, invM, imps, ox, thickness);

	double sgn0 = sgn;
	double sgn1 = 0;

	//iterations
	int iter = 1;
	while (iter < MAX_ITERATIONS) {
		//sS = -sG
		getS(sS, sG, nv);
		valueGrad(sX, val, sG, sMu, sLambda, ncon, nodes, nvar, cx, cxOld, cm, invM, imps, ox, thickness);

		//Fletcher-Reeves Beta
		double sgn1 = squaredNormal(sG, nv);
		if (sgn1 < EPSILON_GD)
			break;

		double beta = fmax(0.0, sgn1 / sgn0);
		sgn0 = sgn1;

		//sS = -sG + beta * s;
		updateS(sS, sG, beta, nv);

		double sgn = squaredNormal(sS, nv);
		double alpha = lineSearch2(sX, sS, nv, sgn, val, xPlusTdx, sMu, sLambda, ncon, nodes, nvar, cx, cxOld, cm, invM, imps, ox, thickness);

		updateX(sX, sS, -alpha, nv);

		if (alpha < EPSILON_GD)
			break;

		multiplierUpdate(sX, sMu, sLambda, ncon, nodes, nvar, cx, cxOld, cm, invM, imps, ox, thickness);
		iter++;
	}

	*ret = iter;
	finalizeSX(sX, nodes, nvar, cx);
}


__global__ void
kernel_impact_zone(
int *count, int *offset, tmImpactInfo *impAll, uint *nodeAll, double *lambdaAll, double *sxAll,
double *sgAll, double *xptdAll, double *ssAll,
double3 *cx, double3 *ox, double3 *cx0, double3 *ox0, double3 *cxOld,
double thickness, double *cm, int num, int *retAll)
{
	LEN_CHK(num);

	int ncon = count[idx];
	tmImpactInfo *imps = impAll + offset[idx];
	uint *nodes = nodeAll + offset[idx] * 4;
	double *lambda = lambdaAll + offset[idx];
	double *sX = sxAll + offset[idx] * 12;
	double *sG = sgAll + offset[idx] * 12;
	double *xptd = xptdAll + offset[idx] * 12;
	double *sS = ssAll + offset[idx] * 12;
	int *ret = retAll+idx;

	augumented_lagrangian_gpu(
		ncon, imps, nodes,
		lambda, sX, sG,
		xptd, sS,
		cx, ox, cx0, ox0, cxOld,
		thickness, cm, ret);
}


void ImpactZoneGPU(tmImpactInfo *hImpAll, int *hCount, int *hOffset, int cnum, int inum, double thickness)
{
	static bool init = false;
	static int *dCount, *dOffset;
	static tmImpactInfo *dImpAll;
	static uint *dNodeAll;
	static double *dLambdaAll, *dSxAll, *dSgAll, *dXptdAll, *dSsAll, *dwAll;
	static double3 *dnAll;
	static int *dRet;

	if (!init) {
		init = true;

		cudaMalloc(&dCount, sizeof(int) * 2000);
		cudaMalloc(&dOffset, sizeof(int) * 2000);
		cudaMalloc(&dImpAll, sizeof(tmImpactInfo) * 20000);
		cudaMalloc(&dNodeAll, sizeof(uint) * 8000);
		cudaMalloc(&dLambdaAll, sizeof(double) * 2000);
		cudaMalloc(&dSxAll, sizeof(double) * 2000);
		cudaMalloc(&dSgAll, sizeof(double) * 2000);
		cudaMalloc(&dXptdAll, sizeof(double) * 2000);
		cudaMalloc(&dSsAll, sizeof(double) * 2000);
		cudaMalloc(&dRet, sizeof(int) * 2000);
		reportMemory();
	}

	cudaMemcpy(dCount, hCount, sizeof(int)*cnum, cudaMemcpyHostToDevice);
	cudaMemcpy(dOffset, hOffset, sizeof(int)*cnum, cudaMemcpyHostToDevice);
	cudaMemcpy(dImpAll, hImpAll, sizeof(tmImpactInfo)*inum, cudaMemcpyHostToDevice);
	//cudaMemset(dRet, 0, sizeof(int)*cnum);

	{
		int num = cnum;
		BLK_PAR(num);
		kernel_impact_zone << <B, T >> > (
			dCount, dOffset, dImpAll,
			dNodeAll, dLambdaAll, dSxAll,
			dSgAll, dXptdAll, dSsAll,
			currentCloth._dx, currentObj._dx, currentCloth._dx0, currentObj._dx0,
			totalAux._dx, thickness, currentCloth._dm, num, dRet);

		getLastCudaError("kernel_impact_zone");
	}
	
	if (false) {
		int *hRet = new int[cnum];
		cudaMemcpy(hRet, dRet, sizeof(int)*cnum, cudaMemcpyDeviceToHost);
		printf("Iterations:\n");
		for (int i=0; i<cnum; i++) {
			printf("%d:%d\n", i, hRet[i]);
		}
	}
}
