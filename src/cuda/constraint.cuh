typedef double3 MeshGrad[4];
typedef double3x3 MeshHess[4][4];

inline __device__ void zeroMG(MeshGrad &g) {
	for (int i=0; i<4; i++)
		g[i] = zero3f();
}

inline __device__ void zeroMH(MeshHess &h) {
	for (int i=0; i<4; i++)
	for (int j=0; j<4; j++)
		h[i][j] = zero3x3();
}

inline __device__ double sq (double x) {return x*x;}

struct EqCon {
    // n . (node->x - x) = 0
    //Node *node;
	int node;
    double3 x, n;
    double stiff;

	__host__ EqCon() {
		node = -1;
		stiff = 0;
	}

	__host__ EqCon(int nid, double3 *x, double3 *n, double stiff) {
		this->node = nid;
		this->x = *x;
		this->n = *n;
		this->stiff = stiff;
	}

	__host__ void print()
	{
		printf("n = %d, x=(%lf, %lf, %lf), n=(%lf, %lf, %lf), stiff = %d\n", node, x.x, x.y, x.z, n.x, n.y, n.z, stiff);
	}

	__device__ double value (double3 *cx)
	{
	    return dot(n, cx[node] - x);
	}

    __device__ void gradient (double3 &grad)
	{
		grad = n;
	}

    __device__ double energy (double value)
	{
		return stiff*sq(value)*0.5;
	}

    __device__ double energy_grad (double value)
	{
		return stiff*value;
	}

    __device__ double energy_hess ()
	{
		return stiff;
	}
};

typedef struct {
    // n . sum(w[i] verts[i]->x) >= 0
    //Node *nodes[4];
	uint nodes[4];

    double w[4];
    bool free[4];
    double3 n;
    double a; // area
    double mu; // friction
    double stiff;

	uint _sides[2];
	double _dist;
	uint2 _ids; //vf / ee
	uint _which; // 0,1 for v or f, e0 or e1, -1 N/A
	bool _valid;
	bool _vf;

	__device__ __host__ bool set_valid(double2 *vDists, double2 *fDists, double2 *eDists)
	{
		if (_vf == true) {
			if (_which == 0) {
				int vid = _ids.x;

				if (_sides[0])
					_valid = (vDists[vid].x == _dist-1.0);
				else
					_valid = (vDists[vid].y == _dist-1.0);

			} else {
				int fid = _ids.y;

				if (_sides[1])
					_valid = (fDists[fid].x == _dist-1.0);
				else
					_valid = (fDists[fid].y == _dist-1.0);
			}
		} else {
			if (_which == 0) {
				int eid = _ids.x;
				if (_sides[0])
					_valid = (eDists[eid].x == _dist-1.0);
				else
					_valid = (eDists[eid].y == _dist-1.0);

			} else {
				int eid = _ids.y;

				if (_sides[1])
					_valid = (eDists[eid].x == _dist-1.0);
				else
					_valid = (eDists[eid].y == _dist-1.0);
			}
		}

		return _valid;
	}

	__device__ __host__ void set_dist(double2 *vDists, double2 *fDists, double2 *eDists)
	{
		if (_vf == true) {
			if (_which == 0) {
				int vid = _ids.x;
				if (_sides[0])
					vDists[vid].x = fmin(vDists[vid].x, _dist-1.0);
				else
					vDists[vid].y = fmin(vDists[vid].y, _dist-1.0);

			} else {
				int fid = _ids.y;

				if (_sides[1])
					fDists[fid].x = fmin(fDists[fid].x, _dist-1.0);
				else
					fDists[fid].y = fmin(fDists[fid].y, _dist-1.0);
			}
		} else {
			if (_which == 0) {
				int eid = _ids.x;
				if (_sides[0])
					eDists[eid].x = fmin(eDists[eid].x, _dist-1.0);
				else
					eDists[eid].y = fmin(eDists[eid].y, _dist-1.0);

			} else {
				int eid = _ids.y;

				if (_sides[1])
					eDists[eid].x = fmin(eDists[eid].x, _dist-1.0);
				else
					eDists[eid].y = fmin(eDists[eid].y, _dist-1.0);
			}
		}
	}

	__device__ double get_m(int i, double *cm, double *om)
	{
		if (free[i])
			return cm[nodes[i]];
		else
			return om[nodes[i]];
	}

	__device__ double3 get_x(int i, double3 *cx, double3 *ox)
	{
		if (free[i])
			return cx[nodes[i]];
		else
			return ox[nodes[i]];
	}

	__device__ double value (double3 *cx, double3 *ox, double mrt)
	{
		double d = 0;
		for (int i = 0; i < 4; i++)
			d += w[i]*dot(n, get_x(i, cx, ox)); //nodes[i]->x);
		d -= mrt;
		return d;
	}

    __device__ void gradient (MeshGrad &grad)
	{
		for (int i = 0; i < 4; i++)
			//grad[nodes[i]] = w[i]*n;
			grad[i] = w[i]*n;
	}

	__device__ void project (MeshGrad &dx, double *cm, double *om, double3 *cx, double3 *ox, double mrt, double mpt)
	{
		double d = value(cx, ox, mrt) + mrt - mpt;
		if (d >= 0) {
			zeroMG(dx);
			return;
		}

		double inv_mass = 0;
		for (int i = 0; i < 4; i++)
			if (free[i])
				inv_mass += sq(w[i])/get_m(i, cm, om); //nodes[i]->m;

		for (int i = 0; i < 4; i++)
			if (free[i])
				//dx[nodes[i]] = -(w[i]/nodes[i]->m)/inv_mass*n*d;
				//dx[i] = -(w[i]/nodes[i]->m)/inv_mass*n*d;
				dx[i] = -(w[i]/get_m(i, cm, om))/inv_mass*n*d;
	}

	__device__ double violation (double value) 
	{
		return max(-value, 0.);
	}

    __device__ double energy (double value, double mrt)
	{
		double v = violation(value);
		return stiff*v*v*v/mrt/6;
	}

    __device__ double energy_grad (double value, double mrt)
	{
	    return -stiff*sq(violation(value))/mrt/2;
	}

    __device__ double energy_hess (double value, double mrt)
	{
		return stiff*violation(value)/mrt;
	}
    
	__device__ void friction (double dt, MeshHess &jac, MeshGrad &force, double *cm, double *om, double3 *cx, double3 *ox, double3 *cv, double3 *ov, double mrt)
	{
		zeroMG(force);
		zeroMH(jac);

		if (mu == 0)
			return;

		double fn = abs(energy_grad(value(cx, ox, mrt), mrt));
		if (fn == 0)
	        return;

		double3 v = zero3f();
	    double inv_mass = 0;
		for (int i = 0; i < 4; i++) {
			v += w[i]*get_x(i, cv, ov); //nodes[i]->v;
			if (free[i])
				inv_mass += sq(w[i])/get_m(i, cm, om); //nodes[i]->m;
		}
    
		double3x3 T = identity3x3() - outer(n,n);
		double vt = length(T*v);
		double f_by_v = min(mu*fn/vt, 1/(dt*inv_mass));
		// double f_by_v = mu*fn/max(vt, 1e-1);

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
	
	__host__ void print(FILE *fp, bool valid) {
		if ((valid && _valid) || !valid)
			fprintf(fp, "n(%d, %d, %d, %d), f(%d, %d, %d, %d), s(%d, %d), vf-ee(%d, %d), vf(%d), dist(%lf)\n",
				nodes[0], nodes[1], nodes[2], nodes[3],
				free[0], free[1], free[2], free[3],
				_sides[0], _sides[1], _ids.x, _ids.y, _vf, _dist);
	}
} g_IneqCon;

#define MAX_CSTRT_NUM 500000

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
		if (_nEqs != 0)
			cudaFree(_dEqs);
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

	void filtering(	double2 *vDists, double2 *fDists, double2 *eDists, int vNum, int fNum, int eNum) {

		if (_hLength == 0)
			return;

		cudaMemcpy(_hIneqs, _dIneqs, _hLength*sizeof(g_IneqCon), cudaMemcpyDeviceToHost);

		memset(fDists, 0, sizeof(double2)*fNum);
		memset(vDists, 0, sizeof(double2)*vNum);
		memset(eDists, 0, sizeof(double2)*eNum);

		for (int i=0; i<_hLength; i++) {
			_hIneqs[i].set_dist(vDists, fDists, eDists);
		}

		int count=0;
		for (int i=0; i<_hLength; i++) {
			if (_hIneqs[i].set_valid(vDists, fDists, eDists))
				count++;
		}
		printf("#const before/after filtering = %d/%d\n", _hLength, count);
//#define DEBUG_OUTPUT

#ifdef DEBUG_OUTPUT
		{
		FILE *fp = fopen("e:\\temp2\\a11.txt", "wt");
		for (int i=0; i<_hLength; i++) {
			_hIneqs[i].print(fp, false);
		}
		fclose(fp);
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

inline __device__ double3 get_x(int i, double3 *cx, double3 *ox, bool free)
{
	if (free) return cx[i];
	else return ox[i];
}

inline __device__ double face_area (int id, bool free, double *fa,
									tri3f *faces, double3 *x)
{
	if (free)
		return fa[id];

	tri3f *t = faces+id;
	int a = t->id0();
	int b = t->id1();
	int c = t->id2();

	double3 x0 = x[a];
	double3 x1 = x[b];
	double3 x2 = x[c];

    return length(cross(x1-x0, x2-x0))*0.5;
}

inline __device__ double node_area (int nid, bool free, double *na,
									int *n2vIdx, int *n2vData, int *adjIdx, int *adjData, tri3f *faces, double3 *x)
{
	if (free)
		return na[nid];

	double a = 0;

	VLST_BEGIN(n2vIdx, n2vData, nid)
	FLST_BEGIN(adjIdx, adjData, vid)
		double a1 = face_area(fid, false, NULL, faces, x)/3.0;
		a+=a1;
	FLST_END
	VLST_END

/*	int adjStart = (id == 0) ? 0 : adjIdx[id-1];
	int adjNum = adjIdx[id]-adjStart;
	for (int i=0; i<adjNum; i++) {
		int fid = adjData[i+adjStart];
		double a1 = face_area(fid, false, NULL, faces, x)/3.0;
		a += a1;
	}
*/
	return a;
}

inline __device__ double3 node_normal (int nid, int *n2vIdx, int *n2vData, int *adjIdx, int *adjData, double3 *fAreas)
{
	double3 a = zero3f();

	VLST_BEGIN(n2vIdx, n2vData, nid)
	FLST_BEGIN(adjIdx, adjData, vid)

		a += fAreas[fid];

	FLST_END
	VLST_END

	return normalize(a);
}

inline __device__ double edge_area (int id, bool free, double *fa, uint2 *adjf, tri3f *faces, double3 *x)
{
    double a = 0;

	int id0 = adjf[id].x;
	int id1 = adjf[id].y;

	if (id0 != -1)
		a += face_area(id0, free, fa, faces, x)/3.0;
	if (id1 != -1)
		a += face_area(id1, free, fa, faces, x)/3.0;

	return a;
}


__device__ double signed_vf_distance 
		(const double3 &x,
        const double3 &y0, const double3 &y1, const double3 &y2,
        double3 *n, double *w)
{
    double3 _n; if (!n) n = &_n;
    double _w[4]; if (!w) w = _w;
    *n = cross(normalize(y1-y0), normalize(y2-y0));
    if (norm2(*n) < 1e-6)
        return double_infinity;
    *n = normalize(*n);
    double h = dot(x-y0, *n);
    double b0 = stp(y1-x, y2-x, *n),
           b1 = stp(y2-x, y0-x, *n),
           b2 = stp(y0-x, y1-x, *n);
    w[0] = 1;
    w[1] = -b0/(b0 + b1 + b2);
    w[2] = -b1/(b0 + b1 + b2);
    w[3] = -b2/(b0 + b1 + b2);
    return h;
}

__device__ double signed_ee_distance (const double3 &x0, const double3 &x1,
                           const double3 &y0, const double3 &y1,
                           double3 *n, double *w) {
    double3 _n; if (!n) n = &_n;
    double _w[4]; if (!w) w = _w;
    *n = cross(normalize(x1-x0), normalize(y1-y0));
    if (norm2(*n) < 1e-6)
        return double_infinity;
    *n = normalize(*n);
    double h = dot(x0-y0, *n);
    double a0 = stp(y1-x1, y0-x1, *n), a1 = stp(y0-x0, y1-x0, *n),
           b0 = stp(x0-y1, x1-y1, *n), b1 = stp(x1-y0, x0-y0, *n);
    w[0] = a0/(a0 + a1);
    w[1] = a1/(a0 + a1);
    w[2] = -b0/(b0 + b1);
    w[3] = -b1/(b0 + b1);
    return h;
}

__device__ void make_vf_constraint (
	int vtx, int face, bool freev, bool freef,
	double3 *cx, tri3f *ctris, int *cAdjIdx, int *cAdjData, int *cn2vIdx, int *cn2vData,
	double3 *ox, tri3f *otris, int *oAdjIdx, int *oAdjData, int *on2vIdx, int *on2vData,
	double *cfa, double *cna,
	double mu, double mu_obs, double mcs, g_IneqCon &ineq)
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

	double a1 = node_area(vtx, freev, cna,
			freev ? cn2vIdx : on2vIdx,
			freev ? cn2vData : on2vData,
			freev ? cAdjIdx : oAdjIdx,
			freev ? cAdjData : oAdjData,
			freev ? ctris : otris,
			freev ? cx : ox);
	double a2 = face_area(face, freef, cfa,
			freef ? ctris : otris,
			freef ? cx : ox);

    double a = min(a1, a2);

    con->stiff = mcs*a;
    double d = signed_vf_distance(
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
	double3 *cx, tri3f *ctris, uint2 *cef, uint2 *cen,
	double3 *ox, tri3f *otris, uint2 *oef, uint2 *oen,
	double *cfa,
	double mu, double mu_obs, double mcs, g_IneqCon &ineq)
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

    double a = min(
		edge_area(edge0, free0, cfa,
			free0 ? cef : oef,
			free0 ? ctris : otris,
			free0 ? cx : ox),
		edge_area(edge1, free1, cfa,
			free1 ? cef : oef,
			free1 ? ctris : otris,
			free1 ? cx : ox));

    con->stiff = mcs*a;
    double d = signed_ee_distance(
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
