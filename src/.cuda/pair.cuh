
typedef struct _g_pair {
	int2 *_dPairs;
	uint *_dIdx;
	int _offset;

	void init() {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMalloc((void**)&_dIdx, 1 * sizeof(uint)));
		cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));
		reportMemory();

		cutilSafeCall(cudaMalloc((void**)&_dPairs, MAX_PAIR_NUM*sizeof(int2)));
		cutilSafeCall(cudaMemset(_dPairs, 0, MAX_PAIR_NUM*sizeof(uint2)));
		reportMemory();

		_offset = 0;
	}

	void clear() {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));
		_offset = 0;
	}

	int getProximityConstraints(bool self, REAL mu, REAL mu_obs, REAL mrt, REAL mcs);
	int getImpacts(bool self, REAL mu, REAL mu_obs, _g_pair &vfPairs, _g_pair &eePairs, int &vfLen, int &eeLen);

	void destroy() {
		cudaFree(_dPairs);
		cudaFree(_dIdx);
	}

	uint length() {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
		return dummy[0];
	}

	void setLength(uint len) {
		cutilSafeCall(cudaMemcpy(_dIdx, &len, 1 * sizeof(uint), cudaMemcpyHostToDevice));
	}
} g_pair;
