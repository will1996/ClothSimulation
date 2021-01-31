
typedef struct {
	int _num;
	int *_bvh;
	int *_bvh_leaf;
	g_box *_bxs;
	g_cone *_cones;
	
	int _max_level;
	uint *_level_idx;

	// for contour test
	int _ctNum;
	uint *_ctIdx, *_ctLst;
	g_cone *_triCones; //directly borrow from g_mesh ...
	bool *_ctFlags;
	REAL3 *_ctPts, *_ctVels;

	uint2 *_fids;
	g_box *_triBxs; //directly borrow form g_mesh ...

	g_box *hBxs; // for debug;

	void getBxs() {
		if (hBxs == NULL)
			hBxs = new g_box [_num];

		cudaMemcpy(hBxs, _bxs, _num*sizeof(g_box), cudaMemcpyDeviceToHost);
	}

	void selfCollisionCulling(REAL3 *x, REAL3 *ox, bool ccd, uint *counting);
} g_bvh;

//the length of the front <= triangle num
typedef struct {
	uint3 *_dFront; // node_id, parent_id, valid 0/1
	uint *_dIdx;
	uint _iMax;

	void init(int max_num) {
		//_iMax = max_num;
		_iMax = max_num * 2; //allow invalid nodes ...

		//start from the root, so at lest one ...
		uint dummy[] = {1};
		cutilSafeCall(cudaMalloc((void**)&_dIdx, 1*sizeof(uint)) );
		cutilSafeCall(cudaMemcpy(_dIdx, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));

		cutilSafeCall(cudaMalloc((void**)&_dFront, _iMax*sizeof(uint3)) );
		cutilSafeCall(cudaMemset(_dFront, 0, _iMax*sizeof(uint3)) );
	}

	void destroy() {
		cudaFree(_dFront);
		cudaFree(_dIdx);
	}

	void reset() {
		//start from the root
		uint dummy[] = {1};
		cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));

		uint3 dummy2;
		dummy2.x = dummy2.y = dummy2.z = 0;
		cutilSafeCall(cudaMemcpy(_dFront, &dummy2, 1 * sizeof(uint3), cudaMemcpyHostToDevice));
	}

	void push(int length, uint3 *data) {
		uint dummy[] = {length};
		cutilSafeCall(cudaMemcpy(_dIdx, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		if (length)
			cutilSafeCall(cudaMemcpy(_dFront, data,length*sizeof(uint3), cudaMemcpyHostToDevice));
	}

	int propogate(bool ccd);
} g_cone_front;

#define SAFE_FRONT_NUM  170000000
#define MAX_FRONT_NUM   180000000

typedef struct {
	uint4 *_dFront; // left, right, valid 0/1, dummy
	uint *_dIdx;

	void init() {
		uint dummy[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		cutilSafeCall(cudaMalloc((void**)&_dIdx, 10*sizeof(uint)) );
		cutilSafeCall(cudaMemcpy(_dIdx, dummy,10*sizeof(uint), cudaMemcpyHostToDevice));

		cutilSafeCall(cudaMalloc((void**)&_dFront, MAX_FRONT_NUM*sizeof(uint4)) );
		cutilSafeCall(cudaMemset(_dFront, 0, MAX_FRONT_NUM*sizeof(uint4)) );
	}

	void destroy() {
		cudaFree(_dFront);
		cudaFree(_dIdx);
	}

	void push(int length, uint4 *data) {
		uint dummy[] = {length};
		cutilSafeCall(cudaMemcpy(_dIdx, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		if (length)
			cutilSafeCall(cudaMemcpy(_dFront, data,length*sizeof(uint4), cudaMemcpyHostToDevice));
	}

	int propogate(bool &update, bool self, bool ccd);

	uint length() {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
		return dummy[0];
	}

	void setLength(uint len) {
		cutilSafeCall(cudaMemcpy(_dIdx, &len, 1 * sizeof(uint), cudaMemcpyHostToDevice));
	}
} g_front;

inline __device__ int getParent(uint i, int *bvh_ids, int num)
{
	//return i-bvh_ids[num+i];
	return bvh_ids[num + i];
}

inline __device__ int getTriID(uint i, int *bvh_ids)
{
	return bvh_ids[i];
}

inline __device__ int getLeftChild(uint i, int *bvh_ids)
{
	return i-bvh_ids[i];
}

inline __device__ int getRightChild(uint i, int *bvh_ids)
{
	return i-bvh_ids[i]+1;
}

inline __device__ bool isLeaf(uint i, int *bvh_ids)
{
	return bvh_ids[i] >= 0;
}

inline __device__ bool overlaps(uint i, uint j, g_box *Abxs, g_box *Bbxs)
{
	return Abxs[i].overlaps(Bbxs[j]);
}

inline __device__ void refit(int i, int *bvh_ids, g_box *bvh_boxes, g_box *tri_boxes, g_cone *bvh_cones, g_cone *tri_cones)
{
	if (isLeaf(i, bvh_ids)) // isLeaf
	{
		int fid = getTriID(i, bvh_ids);
		bvh_boxes[i] = tri_boxes[fid];

		if (bvh_cones)
			bvh_cones[i].set(tri_cones[fid]);
	}
	else
	{
		int left = getLeftChild(i, bvh_ids);
		int right = getRightChild(i, bvh_ids);

		bvh_boxes[i].set(bvh_boxes[left], bvh_boxes[right]);

		if (bvh_cones) {
			bvh_cones[i].set(bvh_cones[left]);
			bvh_cones[i] += bvh_cones[right];
		}
	}
}

__global__ void refit_serial_kernel(int *bvh_ids, g_box *bvh_boxes, g_box *tri_boxes,
	g_cone *bvh_cones, g_cone *tri_cones,
	int num)
{
	for (int i=num-1; i>=0; i--) {
		refit(i, bvh_ids, bvh_boxes, tri_boxes, bvh_cones, tri_cones);
	}
}

__global__ void refit_kernel(int *bvh_ids, g_box *bvh_boxes, g_box *tri_boxes,
	g_cone *bvh_cones, g_cone *tri_cones,
	int st, int num)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= num)
		return;

	refit(idx + st, bvh_ids, bvh_boxes, tri_boxes, bvh_cones, tri_cones);
}

#define CLST_BEGIN(lstIdx, lstData, idd) \
	{int fst = (idd == 0) ? 0 : lstIdx[idd-1];\
	int fnum = lstIdx[idd] - fst;\
	if (fnum == 0) return true;\
	for (int fi=0; fi<fnum; fi++) {\
		uint cid = lstData[fi+fst];\

#define CLST_END }}

inline __device__ bool multiple_loops(uint *lstIdx, int idd)
{
	int fst = (idd == 0) ? 0 : lstIdx[idd - 1];
	int fnum = lstIdx[idd] - fst; 
	return fnum == 0;
}

inline __device__ bool self_intersect(
	int idx, uint *ctIdx, uint *ctLst,
	REAL3 *x, REAL3 *ox, REAL3 axis, REAL3 *pnts)
{
	int num = 0;

	//headache for one week! tangmin 08182017
	int offset = (idx == 0) ? 0 : ctIdx[idx - 1];
	pnts += offset;

	// check for self-collision on the bound
	CLST_BEGIN(ctIdx, ctLst, idx)
		pnts[num++] = x[cid];
	CLST_END

	REAL3 org = pnts[0];
	REAL3 center = make_REAL3(0, 0, 0);
	REAL3 R = (pnts[0]+pnts[1])*REAL(0.5f);

/*
	if (is_equal2(fabs(dot(axis, make_REAL3(0, 1, 0))), 1))
		R = make_REAL3(1, 0, 0);
	else
		R = cross(axis, make_REAL3(0, 1, 0));
*/

	for (int i = 0; i < num; i++) {
		REAL3 tmp = pnts[i];
		center += tmp;
	}
	center = center*(1.0 / num);

	int length = num;
//	if (length < 5)
//		printf("here!");

	RealSign s0 = rs_sign(center, pnts[length - 1], pnts[0], axis);

	if (s0 == RS_Unknow) {
		return true;
	}

	int intersection = 0;
	for (int i = 0; i < length - 1; i++) {
		RealSign s1 = rs_sign(center, pnts[i], pnts[i + 1], axis);

		if (s1 != s0) {
			return true;
		}

		//if (dot(R, pnts[i + 1] - pnts[i]) < 0) 
		{
			RealSign sa = rs_sign(pnts[i], center, R, axis);
			RealSign sb = rs_sign(pnts[i + 1], center, R, axis);

			if (sa == RS_Unknow || sb == RS_Unknow)
				return true;

			if (sa != sb && sb == s0) {
				intersection++;
				if (intersection > 1) {
					return true;
				}
			}
		}
	}

	if (intersection == 0)
		return true;

	return false;
}


inline __device__ bool self_intersect_ccd(
	int idx, uint *ctIdx, uint *ctLst,
	REAL3 *x, REAL3 *ox, REAL3 axis, REAL3 *pnts, REAL3 *vels)
{
	int num = 0;

	//headache for one week! tangmin 08182017
	pnts += (idx == 0) ? 0 : ctIdx[idx - 1];
	vels += (idx == 0) ? 0 : ctIdx[idx - 1];

	// check for self-collision on the bound
	CLST_BEGIN(ctIdx, ctLst, idx)
		pnts[num] = ox[cid];
		vels[num++] = x[cid] - ox[cid];
	CLST_END

	REAL3 org = pnts[0];
	REAL3 center = make_REAL3(0, 0, 0);
	REAL3 R = (pnts[0]+pnts[1])*REAL(0.5);
	
/*	if (is_equal2(fabs(dot(axis, make_REAL3(0, 1, 0))), 1))
		R = make_REAL3(1, 0, 0);
	else
		R = cross(axis, make_REAL3(0, 1, 0));
*/
	for (int i = 0; i < num; i++) {
		REAL3 tmp = pnts[i];
		center += tmp;
	}
	center = center*(1.0 / num);

	int length = num;
	RealSign s0 = rs_sign(center, pnts[length - 1], vels[length-1], pnts[0], vels[0], axis);

	if (s0 == RS_Unknow) {
		return true;
	}

	int intersection = 0;
	for (int i = 0; i < length - 1; i++) {
		RealSign s1 = rs_sign(center, pnts[i], vels[i], pnts[i+1], vels[i+1], axis);

		if (s1 != s0) {
			return true;
		}

		//if (dot(R, pnts[i + 1] - pnts[i]) < 0) 
		{
			RealSign sa = rs_sign(pnts[i], vels[i], center, R, axis);
			RealSign sb = rs_sign(pnts[i+1], vels[i+1], center, R, axis);
			if (sa == RS_Unknow || sb == RS_Unknow) {
				return true;
			}

			if (sa != sb && sb == s0) {
				intersection++;
				if (intersection > 1) {
					return true;
				}
			}
		}
	}

	if (intersection == 0)
		return true;

	return false;
}

#define SIMPLE_CONE_TEST

inline __device__ bool cone_test(
	int i, g_cone *bvh_cones, uint *ctIdx, uint *ctLst, REAL3 *ctPts, REAL3 *ctVels, int ctNum,
	REAL3 *x, REAL3 *ox, bool ccd)
{
#ifdef SIMPLE_CONE_TEST	
	return bvh_cones[i].full() || multiple_loops(ctIdx, i);
#else
	if (bvh_cones[i].full())
		return true;

	REAL3 axis = bvh_cones[i]._axis;

	if (ccd)
		return self_intersect_ccd(i, ctIdx, ctLst, x, ox, axis, ctPts, ctVels);
	else
		return self_intersect(i, ctIdx, ctLst, x, ox, axis, ctPts);
#endif
}

__global__ void cone_test_serial_kernel(
	int *bvh_ids, g_cone *bvh_cones,
	uint *ctIdx, uint *ctLst, REAL3 *ctPts, REAL3 *ctVels, int ctNum, bool *ctFlag,
	REAL3 *x, REAL3 *ox, bool ccd, uint *counting,
	int num)
{
	ctFlag[0] = true; //must test from the root ...
	for (int i = 0; i< num; i++)
	{
		if (ctFlag[i] == true) {
			if (isLeaf(i, bvh_ids))
				ctFlag[i] = false;
			else
				ctFlag[i] = cone_test(
					i, bvh_cones, ctIdx, ctLst, ctPts, ctVels, ctNum, x, ox, ccd);

			if (ctFlag[i] == true) {// need to test its children
				int left = getLeftChild(i, bvh_ids);
				int right = getRightChild(i, bvh_ids);
				ctFlag[left] = ctFlag[right] = true;
				atomicAdd(counting, 1);
			}
		}
	}
}

