typedef struct {
	int _num;
	int *_bvh;
	g_box *_bxs;
	
	int _max_level;
	uint *_level_idx;

	uint2 *_fids;
	g_box *_triBxs; //directly borrow form g_mesh ...

	g_box *hBxs; // for debug;

	void getBxs() {
		if (hBxs == NULL)
			hBxs = new g_box [_num];

		cudaMemcpy(hBxs, _bxs, _num*sizeof(g_box), cudaMemcpyDeviceToHost);
	}

	void printBxs(char *path) {
		FILE *fp = fopen(path, "wt");
		for (int i=0;i<_num;i++)
			hBxs[i].print(fp);
		fclose(fp);
	}
} g_bvh;

#define MAX_PAIR_NUM 20000000
typedef struct {
	uint2 *_dPairs;
	uint *_dIdx;

	void init() {
		uint dummy[] = {0};
		cutilSafeCall(cudaMalloc((void**)&_dIdx, 1*sizeof(uint)) );
		cutilSafeCall(cudaMemcpy(_dIdx, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		reportMemory();

		cutilSafeCall(cudaMalloc((void**)&_dPairs, MAX_PAIR_NUM*sizeof(uint2)) );
		cutilSafeCall(cudaMemset(_dPairs, 0, MAX_PAIR_NUM*sizeof(uint2)) );
		reportMemory();
	}

	void clear() {
		uint dummy[] = {0};
		cutilSafeCall(cudaMemcpy(_dIdx, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
	}

	int getProximityConstraints(bool self, double mu, double mu_obs, double mrt, double mcs);
	int getImpacts(bool self, double mu, double mu_obs);

	void destroy() {
		cudaFree(_dPairs);
		cudaFree(_dIdx);
	}

	int length() {
		uint dummy[] = {0};
		cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1*sizeof(uint), cudaMemcpyDeviceToHost));
		return dummy[0];
	}
} g_pair;

#define SAFE_FRONT_NUM  170000000
#define MAX_FRONT_NUM   180000000

typedef struct {
	uint4 *_dFront; // left, right, valid 0/1, dummy
	uint *_dIdx;

	void init() {
		uint dummy[] = {0};
		cutilSafeCall(cudaMalloc((void**)&_dIdx, 1*sizeof(uint)) );
		cutilSafeCall(cudaMemcpy(_dIdx, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		reportMemory();

		cutilSafeCall(cudaMalloc((void**)&_dFront, MAX_FRONT_NUM*sizeof(uint4)) );
		cutilSafeCall(cudaMemset(_dFront, 0, MAX_FRONT_NUM*sizeof(uint4)) );
		reportMemory();
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

	int propogate(bool &update, bool self);
} g_front;

inline __device__ int getParent(uint i, int *bvh_ids, int num)
{
	return i-bvh_ids[num+i];
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

inline __device__ void refit(int i, int *bvh_ids, g_box *bvh_boxes, g_box *tri_boxes)
{
	if (isLeaf(i, bvh_ids)) // isLeaf
	{
		bvh_boxes[i] = tri_boxes[getTriID(i, bvh_ids)];
	}
	else 
	{
		int left = getLeftChild(i, bvh_ids);
		int right = getRightChild(i, bvh_ids);

		bvh_boxes[i].set(bvh_boxes[left], bvh_boxes[right]);
	}
}

__global__ void refit_serial_kernel(int *bvh_ids, g_box *bvh_boxes, g_box *tri_boxes, int num)
{
	for (int i=num-1; i>=0; i--) {
		refit(i, bvh_ids, bvh_boxes, tri_boxes);
	}
}

__global__ void refit_kernel(int *bvh_ids, g_box *bvh_boxes, g_box *tri_boxes, int st, int num)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= num)
		return;

	refit(idx+st, bvh_ids, bvh_boxes, tri_boxes);
}

