#pragma once

typedef unsigned int uint;
#define MAX_PAIR_NUM 20000000

typedef struct {
	uint3 _ids;

	inline __device__ __host__ uint id0() const { return _ids.x; }
	inline __device__ __host__ uint id1() const { return _ids.y; }
	inline __device__ __host__ uint id2() const { return _ids.z; }
	inline __device__ __host__ uint id(int i) const { return (i == 0 ? id0() : ((i == 1) ? id1() : id2())); }
} tri3f;


typedef struct {
	uint4 _ids;

	inline __device__ __host__ uint id0() const { return _ids.x; }
	inline __device__ __host__ uint id1() const { return _ids.y; }
	inline __device__ __host__ uint id2() const { return _ids.z; }
	inline __device__ __host__ uint id3() const { return _ids.w; }
	inline __device__ __host__ uint id(int i) const {
		return (i == 3 ? id3() : (i == 0 ? id0() : ((i == 1) ? id1() : id2())));
	}
} tri4f;

inline __device__ bool covertex(int tA, int tB, tri3f *Atris)
{
	for (int i = 0; i<3; i++)
		for (int j = 0; j<3; j++) {
			if (Atris[tA].id(i) == Atris[tB].id(j))
				return true;
		}

	return false;
}

inline __device__ void addPair(uint a, uint b, int2 *pairs, uint *idx)
{
	if (*idx < MAX_PAIR_NUM)
	{
		uint offset = atomicAdd(idx, 1);
		pairs[offset].x = a;
		pairs[offset].y = b;
	}
}