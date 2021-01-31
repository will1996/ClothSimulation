typedef REAL3 MeshGrad[4];
typedef REAL3x3 MeshHess[4][4];

inline __device__ void zeroMG(MeshGrad &g) {
	for (int i=0; i<4; i++)
		g[i] = zero3f();
}

inline __device__ void zeroMH(MeshHess &h) {
	for (int i=0; i<4; i++)
	for (int j=0; j<4; j++)
		h[i][j] = zero3x3();
}

inline __device__ REAL sq (REAL x) {return x*x;}

struct EqCon {
	int node;
    REAL3 x, n;
    REAL stiff;

	__host__ EqCon() {
		node = -1;
		stiff = 0;
	}

	__host__ EqCon(int nid, REAL3 *x, REAL3 *n, REAL stiff) {
		this->node = nid;
		this->x = *x;
		this->n = *n;
		this->stiff = stiff;
	}

	__host__ void print()
	{
		printf("n = %d, x=(%lf, %lf, %lf), n=(%lf, %lf, %lf), stiff = %d\n", node, x.x, x.y, x.z, n.x, n.y, n.z, stiff);
	}

	__device__ REAL value (REAL3 *cx)
	{
	    return dot(n, cx[node] - x);
	}

    __device__ void gradient (REAL3 &grad)
	{
		grad = n;
	}

    __device__ REAL energy (REAL value)
	{
		return stiff*sq(value)*0.5;
	}

    __device__ REAL energy_grad (REAL value)
	{
		return stiff*value;
	}

    __device__ REAL energy_hess ()
	{
		return stiff;
	}
};

typedef struct {
	uint nodes[4];

    REAL w[4];
    bool free[4];
    REAL3 n;
    REAL a; // area
    REAL mu; // friction
    REAL stiff;

	uint _sides[2];
	REAL _dist;
	uint2 _ids; //vf / ee
	uint _which; // 0,1 for v or f, e0 or e1, -1 N/A
	bool _valid;
	bool _vf;

	__device__ __host__ bool set_valid(REAL2 *vDists, REAL2 *fDists, REAL2 *eDists)
	{
		if (_vf == true) {
			if (_which == 0) {
				int vid = _ids.x;

				if (_sides[0])
					_valid = is_equal2(vDists[vid].x, _dist - REAL(1.0));
				else
					_valid = is_equal2(vDists[vid].y, _dist - REAL(1.0));

			} else {
				int fid = _ids.y;

				if (_sides[1])
					_valid = is_equal2(fDists[fid].x, _dist - REAL(1.0));
				else
					_valid = is_equal2(fDists[fid].y, _dist - REAL(1.0));
			}
		} else {
			if (_which == 0) {
				int eid = _ids.x;
				if (_sides[0])
					_valid = is_equal2(eDists[eid].x, _dist - REAL(1.0));
				else
					_valid = is_equal2(eDists[eid].y, _dist - REAL(1.0));

			} else {
				int eid = _ids.y;

				if (_sides[1])
					_valid = is_equal2(eDists[eid].x, _dist - REAL(1.0));
				else
					_valid = is_equal2(eDists[eid].y, _dist - REAL(1.0));
			}
		}

		return _valid;
	}

	__device__ __host__ void set_dist(REAL2 *vDists, REAL2 *fDists, REAL2 *eDists)
	{
		if (_vf == true) {
			if (_which == 0) {
				int vid = _ids.x;
				if (_sides[0])
					vDists[vid].x = fminf(vDists[vid].x, _dist-REAL(1.0));
				else
					vDists[vid].y = fminf(vDists[vid].y, _dist-REAL(1.0));

			} else {
				int fid = _ids.y;

				if (_sides[1])
					fDists[fid].x = fminf(fDists[fid].x, _dist-REAL(1.0));
				else
					fDists[fid].y = fminf(fDists[fid].y, _dist-REAL(1.0));
			}
		} else {
			if (_which == 0) {
				int eid = _ids.x;
				if (_sides[0])
					eDists[eid].x = fminf(eDists[eid].x, _dist-REAL(1.0));
				else
					eDists[eid].y = fminf(eDists[eid].y, _dist-REAL(1.0));

			} else {
				int eid = _ids.y;

				if (_sides[1])
					eDists[eid].x = fminf(eDists[eid].x, _dist-REAL(1.0));
				else
					eDists[eid].y = fminf(eDists[eid].y, _dist-REAL(1.0));
			}
		}
	}

	__device__ REAL get_m(int i, REAL *cm, REAL *om)
	{
		if (free[i])
			return cm[nodes[i]];
		else
			return om[nodes[i]];
	}

	__device__ REAL3 get_x(int i, REAL3 *cx, REAL3 *ox)
	{
		if (free[i])
			return cx[nodes[i]];
		else
			return ox[nodes[i]];
	}

	__device__ REAL value (REAL3 *cx, REAL3 *ox, REAL mrt)
	{
		REAL d = 0;
		for (int i = 0; i < 4; i++)
			d += w[i]*dot(n, get_x(i, cx, ox));
		d -= mrt;
		return d;
	}

    __device__ void gradient (MeshGrad &grad)
	{
		for (int i = 0; i < 4; i++)
			grad[i] = w[i]*n;
	}

	__device__ void project (MeshGrad &dx, REAL *cm, REAL *om, REAL3 *cx, REAL3 *ox, REAL mrt, REAL mpt)
	{
		REAL d = value(cx, ox, mrt) + mrt - mpt;
		if (d >= 0) {
			zeroMG(dx);
			return;
		}

		REAL inv_mass = 0;
		for (int i = 0; i < 4; i++)
			if (free[i])
				inv_mass += sq(w[i])/get_m(i, cm, om);

		for (int i = 0; i < 4; i++)
			if (free[i])
				dx[i] = -(w[i]/get_m(i, cm, om))/inv_mass*n*d;
	}

	__device__ REAL violation (REAL value) 
	{
		return max(-value, 0.);
	}

    __device__ REAL energy (REAL value, REAL mrt)
	{
		REAL v = violation(value);
		return stiff*v*v*v/mrt/6;
	}

    __device__ REAL energy_grad (REAL value, REAL mrt)
	{
	    return -stiff*sq(violation(value))/mrt/2;
	}

    __device__ REAL energy_hess (REAL value, REAL mrt)
	{
		return stiff*violation(value)/mrt;
	}
    
	__device__ void friction (REAL dt, MeshHess &jac, MeshGrad &force, REAL *cm, REAL *om, REAL3 *cx, REAL3 *ox, REAL3 *cv, REAL3 *ov, REAL mrt)
	{
		zeroMG(force);
		zeroMH(jac);

		if (mu == 0)
			return;

		REAL fn = abs(energy_grad(value(cx, ox, mrt), mrt));
		if (fn == 0)
	        return;

		REAL3 v = zero3f();
	    REAL inv_mass = 0;
		for (int i = 0; i < 4; i++) {
			v += w[i]*get_x(i, cv, ov);
			if (free[i])
				inv_mass += sq(w[i])/get_m(i, cm, om);
		}
    
		REAL3x3 T = identity3x3() - outer(n,n);
		REAL vt = length(T*v);
		//REAL f_by_v = min(mu*fn/vt, 1/(dt*inv_mass));
		REAL f_by_v = mu*fn / fmaxf(vt, REAL(1e-2));

		for (int i = 0; i < 4; i++) {
			if (free[i]) {
				force[i] = -w[i]*f_by_v*T*v;
				for (int j = 0; j < 4; j++) {
					if (free[j]) {
						jac[i][j] = -w[i]*w[j]*f_by_v*T;
					}
				}
			}
	    }
	}
} g_IneqCon;


typedef struct _g_GlueCon {
	uint n0, n1;
	REAL3 x0, x1;
	REAL stiff;

	__host__ _g_GlueCon() {
		this->n0 = -1;
		this->n1 = -1;
		this->x0 = zero3f();
		this->x1 = zero3f();
		this->stiff = 0;
	}

	__host__ _g_GlueCon(int n0, int n1, REAL3 *x0, REAL3 *x1, REAL stiff) {
		this->n0 = n0;
		this->n1 = n1;
		this->x0 = *x0;
		this->x1 = *x1;
		this->stiff = stiff;
	}

	__host__ void print()
	{
		printf("n0,n1 = %d,%d x0=(%lf, %lf, %lf),x1=(%lf, %lf, %lf), stiff = %d\n",
			n0, n1,
			x0.x, x0.y, x0.z,
			x1.x, x1.y, x1.z, 
			stiff);
	}
} g_GlueCon;

#define MAX_CSTRT_NUM 500000

struct g_glues {
	g_GlueCon *_dGlus;
	uint _nGlus;

	g_glues() {
		_dGlus = NULL;
		_nGlus = 0;
	}

	void reset() {
		if (_nGlus != 0)
			cutilSafeCall(cudaMemset(_dGlus, 0, _nGlus*sizeof(g_GlueCon)));
	}

	void init(int num) {
		destroy();

		_nGlus = num;
		if (_nGlus != 0)
			cutilSafeCall(cudaMalloc((void**)&_dGlus, _nGlus*sizeof(g_GlueCon)));
		else
			_dGlus = NULL;

		reset();
	}

	void destroy() {
		if (_nGlus != 0) {
			cudaFree(_dGlus);
			_nGlus = 0;
		}
	}

	int length() {
		return _nGlus;
	}

	g_GlueCon *data() {
		return _dGlus;
	}

	void checkData() {
		if (_nGlus == 0)
			return;

		g_GlueCon *hglus = new g_GlueCon[_nGlus];
		cutilSafeCall(cudaMemcpy(hglus, _dGlus, _nGlus*sizeof(g_GlueCon), cudaMemcpyDeviceToHost));
		for (int i = 0; i<_nGlus; i++)
			hglus[i].print();
		delete[] hglus;
	}
};

struct g_handles{
	EqCon *_dEqs;
	uint _nEqs;

	g_handles() {
		_dEqs = NULL;
		_nEqs = 0;
	}

	void reset() {
		if (_nEqs != 0)
			cutilSafeCall(cudaMemset(_dEqs, 0, _nEqs*sizeof(EqCon)) );
	}

	void init(int num) {
		destroy();

		_nEqs = num;
		if (_nEqs != 0)
			cutilSafeCall(cudaMalloc((void**)&_dEqs, _nEqs*sizeof(EqCon)) );
		else
			_dEqs = NULL;

		reset();
	}

	void destroy() {
		if (_nEqs != 0) {
			cudaFree(_dEqs);
			_nEqs = 0;
		}
	}

	int length() {
		return _nEqs;
	}

	EqCon *data() {
		return _dEqs;
	}

	void checkData() {
		if (_nEqs == 0)
			return;

		EqCon *heqs = new EqCon[_nEqs];
		cutilSafeCall(cudaMemcpy(heqs, _dEqs, _nEqs*sizeof(EqCon), cudaMemcpyDeviceToHost));
		for (int i=0; i<_nEqs; i++)
			heqs[i].print();
		delete [] heqs;
	}

};


typedef struct {
	g_IneqCon *_dIneqs;
	uint *_dIneqNum;
	uint _hLength;

	g_IneqCon *_hIneqs;


	void init() {
		uint dummy[] = {0};
		cutilSafeCall(cudaMalloc((void**)&_dIneqNum, 1*sizeof(uint)) );
		cutilSafeCall(cudaMemcpy(_dIneqNum, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		reportMemory();

		cutilSafeCall(cudaMalloc((void**)&_dIneqs, MAX_CSTRT_NUM*sizeof(g_IneqCon)) );
		cutilSafeCall(cudaMemset(_dIneqs, 0, MAX_CSTRT_NUM*sizeof(g_IneqCon)) );
		reportMemory();

		_hLength = 0;

		// buffer for cpu processing
		_hIneqs = new g_IneqCon[MAX_CSTRT_NUM];
	}

	void filtering(	REAL2 *vDists, REAL2 *fDists, REAL2 *eDists, int vNum, int fNum, int eNum) {

		if (_hLength == 0)
			return;

		cudaMemcpy(_hIneqs, _dIneqs, _hLength*sizeof(g_IneqCon), cudaMemcpyDeviceToHost);

		memset(fDists, 0, sizeof(REAL2)*fNum);
		memset(vDists, 0, sizeof(REAL2)*vNum);
		memset(eDists, 0, sizeof(REAL2)*eNum);

		for (int i=0; i<_hLength; i++) {
			_hIneqs[i].set_dist(vDists, fDists, eDists);
		}

		int count=0;
		for (int i=0; i<_hLength; i++) {
			if (_hIneqs[i].set_valid(vDists, fDists, eDists))
				count++;
		}
//		printf("#const before/after filtering = %d/%d\n", _hLength, count);
//#define DEBUG_OUTPUT

#ifdef DEBUG_OUTPUT
		{
		FILE *fp = fopen("e:\\temp\\a11.txt", "wt");
		for (int i=0; i<_hLength; i++) {
			_hIneqs[i].print(fp, false);
		}
		fclose(fp);
		exit(0);
		}
#endif

		// valid items ahead
		::sort(_hIneqs, _hIneqs+_hLength);

		// copyback to gpu
		setLength(count);
		cudaMemcpy(_dIneqs, _hIneqs, _hLength*sizeof(g_IneqCon), cudaMemcpyHostToDevice);

#ifdef DEBUG_OUTPUT
		::sort(_hIneqs, _hIneqs+_hLength);
		{
		FILE *fp = fopen("e:\\temp2\\a12.txt", "wt");
		for (int i=0; i<_hLength; i++) {
			_hIneqs[i].print(fp, true);
		}
		fclose(fp);
		}

		//exit(0);
#endif
	}

	void clear() {
		uint dummy[] = {0};
		cutilSafeCall(cudaMemcpy(_dIneqNum, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		_hLength = 0;
	}

	void destroy() {
		cudaFree(_dIneqs);
		cudaFree(_dIneqNum);
		delete [] _hIneqs;
	}

	int length() {
		return _hLength;
	}

	g_IneqCon *data() {
		return _dIneqs;
	}

	void setLength(uint num) {
		_hLength = num;
		cutilSafeCall(cudaMemcpy(_dIneqNum, &_hLength, 1*sizeof(uint), cudaMemcpyHostToDevice));
	}

	int updateLength() {
		cutilSafeCall(cudaMemcpy(&_hLength, _dIneqNum, 1*sizeof(uint), cudaMemcpyDeviceToHost));
		return _hLength;
	}
} g_constraints;

inline __device__ REAL3 get_x(int i, REAL3 *cx, REAL3 *ox, bool free)
{
	if (free) return cx[i];
	else return ox[i];
}

inline __device__ REAL face_area (int id, bool free, REAL *fa,
									tri3f *faces, REAL3 *x)
{
	if (free)
		return fa[id];

	tri3f *t = faces+id;
	int a = t->id0();
	int b = t->id1();
	int c = t->id2();

	REAL3 x0 = x[a];
	REAL3 x1 = x[b];
	REAL3 x2 = x[c];

    return length(cross(x1-x0, x2-x0))*0.5;
}

inline __device__ REAL node_area (int nid, bool free, REAL *na,
									int *n2vIdx, int *n2vData, int *adjIdx, int *adjData, tri3f *faces, REAL3 *x)
{
	if (free)
		return na[nid];

	REAL a = 0;

	VLST_BEGIN(n2vIdx, n2vData, nid)
	FLST_BEGIN(adjIdx, adjData, vid)
		REAL a1 = face_area(fid, false, NULL, faces, x)/3.0;
		a+=a1;
	FLST_END
	VLST_END

/*	int adjStart = (id == 0) ? 0 : adjIdx[id-1];
	int adjNum = adjIdx[id]-adjStart;
	for (int i=0; i<adjNum; i++) {
		int fid = adjData[i+adjStart];
		REAL a1 = face_area(fid, false, NULL, faces, x)/3.0;
		a += a1;
	}
*/
	return a;
}

inline __device__ REAL3 node_normal (int nid, int *n2vIdx, int *n2vData, int *adjIdx, int *adjData, REAL3 *fAreas)
{
	REAL3 a = zero3f();

	VLST_BEGIN(n2vIdx, n2vData, nid)
	FLST_BEGIN(adjIdx, adjData, vid)

		a += fAreas[fid];

	FLST_END
	VLST_END

	return normalize(a);
}

inline __device__ REAL edge_area (int id, bool free, REAL *fa, uint2 *adjf, tri3f *faces, REAL3 *x)
{
    REAL a = 0;

	int id0 = adjf[id].x;
	int id1 = adjf[id].y;

	if (id0 != -1)
		a += face_area(id0, free, fa, faces, x)/3.0;
	if (id1 != -1)
		a += face_area(id1, free, fa, faces, x)/3.0;

	return a;
}


__device__ REAL signed_vf_distance 
		(const REAL3 &x,
        const REAL3 &y0, const REAL3 &y1, const REAL3 &y2,
        REAL3 *n, REAL *w)
{
    REAL3 _n; if (!n) n = &_n;
    REAL _w[4]; if (!w) w = _w;
    *n = cross(normalize(y1-y0), normalize(y2-y0));
    if (norm2(*n) < 1e-6)
        return REAL_infinity;
    *n = normalize(*n);
    REAL h = dot(x-y0, *n);
    REAL b0 = stp(y1-x, y2-x, *n),
           b1 = stp(y2-x, y0-x, *n),
           b2 = stp(y0-x, y1-x, *n);
    w[0] = 1;
    w[1] = -b0/(b0 + b1 + b2);
    w[2] = -b1/(b0 + b1 + b2);
    w[3] = -b2/(b0 + b1 + b2);
    return h;
}

__device__ REAL signed_ee_distance (const REAL3 &x0, const REAL3 &x1,
                           const REAL3 &y0, const REAL3 &y1,
                           REAL3 *n, REAL *w) {
    REAL3 _n; if (!n) n = &_n;
    REAL _w[4]; if (!w) w = _w;
    *n = cross(normalize(x1-x0), normalize(y1-y0));
    if (norm2(*n) < 1e-6)
        return REAL_infinity;
    *n = normalize(*n);
    REAL h = dot(x0-y0, *n);
    REAL a0 = stp(y1-x1, y0-x1, *n), a1 = stp(y0-x0, y1-x0, *n),
           b0 = stp(x0-y1, x1-y1, *n), b1 = stp(x1-y0, x0-y0, *n);
    w[0] = a0/(a0 + a1);
    w[1] = a1/(a0 + a1);
    w[2] = -b0/(b0 + b1);
    w[3] = -b1/(b0 + b1);
    return h;
}

__device__ void make_vf_constraint (
	int vtx, int face, bool freev, bool freef,
	REAL3 *cx, tri3f *ctris, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData,
	REAL3 *ox, tri3f *otris, int *oAdjIdx, int *oAdjData, int *on2vIdx, int *on2vData,
	REAL *cfa, REAL *cna,
	REAL mu, REAL mu_obs, REAL mcs, g_IneqCon &ineq)
{
    g_IneqCon *con = &ineq;
	tri3f &t = (freef) ? ctris[face] : otris[face];

//	if (vtx == 753)
//		vtx = 753;

    con->nodes[0] = vtx;
	con->free[0] = freev;

    con->nodes[1] = t.id0();
	con->free[1] = freef;
    con->nodes[2] = t.id1();
	con->free[2] = freef;
    con->nodes[3] = t.id2();
	con->free[3] = freef;

	REAL a1 = node_area(vtx, freev, cna,
			freev ? cn2vIdx : on2vIdx,
			freev ? cn2vData : on2vData,
			freev ? cAdjIdx : oAdjIdx,
			freev ? cAdjData : oAdjData,
			freev ? ctris : otris,
			freev ? cx : ox);
	REAL a2 = face_area(face, freef, cfa,
			freef ? ctris : otris,
			freef ? cx : ox);

    REAL a = min(a1, a2);

    con->stiff = mcs*a;
    REAL d = signed_vf_distance(
		get_x(con->nodes[0], cx, ox, freev),
		get_x(con->nodes[1], cx, ox, freef),
		get_x(con->nodes[2], cx, ox, freef),
		get_x(con->nodes[3], cx, ox, freef),
		&con->n, con->w);

    if (d < 0)
        con->n = -con->n;

    con->mu = (!freev || !freef) ? mu_obs : mu;
}

__device__ void make_ee_constraint (
	int edge0, int edge1, bool free0, bool free1,
	REAL3 *cx, tri3f *ctris, uint2 *cef, uint2 *cen,
	REAL3 *ox, tri3f *otris, uint2 *oef, uint2 *oen,
	REAL *cfa,
	REAL mu, REAL mu_obs, REAL mcs, g_IneqCon &ineq)
{
    g_IneqCon *con = &ineq;
	con->nodes[0] = free0 ? cen[edge0].x : oen[edge0].x;
	con->free[0] = free0;
    con->nodes[1] = free0 ? cen[edge0].y : oen[edge0].y;
	con->free[1] = free0;
    con->nodes[2] = free1 ? cen[edge1].x : oen[edge1].x;
	con->free[2] = free1;
    con->nodes[3] = free1 ? cen[edge1].y : oen[edge1].y;
	con->free[3] = free1;

    REAL a = min(
		edge_area(edge0, free0, cfa,
			free0 ? cef : oef,
			free0 ? ctris : otris,
			free0 ? cx : ox),
		edge_area(edge1, free1, cfa,
			free1 ? cef : oef,
			free1 ? ctris : otris,
			free1 ? cx : ox));

    con->stiff = mcs*a;
    REAL d = signed_ee_distance(
		get_x(con->nodes[0], cx, ox, free0),
		get_x(con->nodes[1], cx, ox, free0),
		get_x(con->nodes[2], cx, ox, free1),
		get_x(con->nodes[3], cx, ox, free1),
		&con->n, con->w);

    if (d < 0)
        con->n = -con->n;
    con->mu = (!free0 || !free1) ? mu_obs : mu;
}

inline __device__ void addConstraint(g_IneqCon *cstrs, uint *idx, g_IneqCon &ic)
{
	if (*idx < MAX_CSTRT_NUM) 
	{
		uint offset = atomicAdd(idx, 1);
		cstrs[offset] = ic;
	}
}

#define LESS(a, b) {if (a != b) return a < b;}
#define BIGG(a, b) {if (a != b) return b < a;}

inline __host__ bool operator< (const g_IneqCon &ci0, const g_IneqCon &ci1) {
	BIGG(ci0._valid, ci1._valid);
	LESS(ci0.free[0], ci1.free[0]);
	LESS(ci0.free[1], ci1.free[1]);
	LESS(ci0.free[2], ci1.free[2]);
	LESS(ci0.free[3], ci1.free[3]);
	LESS(ci0.nodes[0], ci1.nodes[0]);
	LESS(ci0.nodes[1], ci1.nodes[1]);
	LESS(ci0.nodes[2], ci1.nodes[2]);
	LESS(ci0.nodes[3], ci1.nodes[3]);
	LESS(ci0._which, ci1._which);

	return true;
}
