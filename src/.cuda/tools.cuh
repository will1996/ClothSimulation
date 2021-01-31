#pragma once

#include <assert.h>

#define VLST_BEGIN(lstIdx, lstData, idd) \
	{int vst = (idd == 0) ? 0 : lstIdx[idd-1];\
	int vnum = lstIdx[idd] - vst;\
	for (int vi=0; vi<vnum; vi++) {\
		int vid = lstData[vi+vst];\

#define VLST_END }}

#define FLST_BEGIN(lstIdx, lstData, idd) \
	{int fst = (idd == 0) ? 0 : lstIdx[idd-1];\
	int fnum = lstIdx[idd] - fst;\
	for (int fi=0; fi<fnum; fi++) {\
		int fid = lstData[fi+fst];\

#define FLST_END }}

///////////////////////////////////////////////////////
// show memory usage of GPU
inline void  reportMemory()
{
#ifdef OUTPUT_TXT
	size_t free_byte;
	size_t total_byte;
	cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
	
	if ( cudaSuccess != cuda_status ) {
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
		exit(1);
	}
	
	REAL free_db = (REAL)free_byte; 
	REAL total_db = (REAL)total_byte;
	REAL used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
		used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
#endif
}


///////////////////////////////////////////////////////
// show memory usage of GPU

#define BLOCK_DIM 64

inline int BPG(int N, int TPB)
{
	int blocksPerGrid = (N + TPB - 1) / (TPB);
	//printf("(N=%d, TPB=%d, stride=1, BPG=%d)\n", N, TPB, blocksPerGrid);
	
	if (blocksPerGrid > 65536) {
		printf("blocksPerGrid is larger than 65536, aborting ... (N=%d, TPB=%d, BPG=%d)\n", N, TPB, blocksPerGrid);
		exit(0);
	}

	return blocksPerGrid;
}

inline int BPG(int N, int TPB, int &stride)
{
	int blocksPerGrid = 0;
	
	do {
		blocksPerGrid = (N + TPB*stride - 1) / (TPB*stride);
		if (blocksPerGrid <= 65536)
			return blocksPerGrid;

		stride *= 2;
	} while (1);

	assert(0);
	return 0;
}

#include "cuda_occupancy.h"
extern cudaDeviceProp deviceProp;

inline int evalOptimalBlockSize(cudaFuncAttributes attribs, cudaFuncCache cachePreference, size_t smemBytes) {
	cudaOccDeviceProp prop = deviceProp;
	cudaOccFuncAttributes occAttribs = attribs;
	cudaOccDeviceState occCache;

	switch (cachePreference) {
	case cudaFuncCachePreferNone:
		occCache.cacheConfig = CACHE_PREFER_NONE;
		break;
	case cudaFuncCachePreferShared:
		occCache.cacheConfig = CACHE_PREFER_SHARED;
		break;
	case cudaFuncCachePreferL1:
		occCache.cacheConfig = CACHE_PREFER_L1;
		break;
	case cudaFuncCachePreferEqual:
		occCache.cacheConfig = CACHE_PREFER_EQUAL;
		break;
	default:
		;	///< should throw error
	}

	int minGridSize, blockSize;
	cudaOccMaxPotentialOccupancyBlockSize(
		&minGridSize, &blockSize, &prop, &occAttribs, &occCache, nullptr, smemBytes);
	return blockSize;
}

#define LEN_CHK(l) \
    int idx = blockDim.x * blockIdx.x + threadIdx.x;\
	if (idx >= l) return;

#define BLK_PAR(l) \
   int T = BLOCK_DIM; \
    int B = BPG(l, T);

#define BLK_PAR2(l, s) \
   int T = BLOCK_DIM; \
    int B = BPG(l, T, s);

#define BLK_PAR3(l, s, n) \
   int T = n; \
    int B = BPG(l, T, s);

#define cutilSafeCall checkCudaErrors

#define M_PI       3.14159265358979323846
#define M_SQRT2    1.41421356237309504880

#include <map>
using namespace std;

typedef map<void *, int> FUNC_INT_MAP;
static  FUNC_INT_MAP blkSizeTable;

inline int getBlkSize(void *func)
{
	FUNC_INT_MAP::iterator it;

	it = blkSizeTable.find(func);
	if (it == blkSizeTable.end()) {
		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, func);
		int num = evalOptimalBlockSize(attr, cudaFuncCachePreferL1, 0);
		blkSizeTable[func] = num;
		return num;
	}
	else {
		return it->second;
	}
}