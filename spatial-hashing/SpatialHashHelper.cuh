#ifndef __SPHHELPER__
#define __SPHHELPER__

#include "CudaBasic.cuh"
#include "CollisionPair.h"
#include "./cuda/box.cuh"

#define HASHX 73856093
#define HASHY 19349663
#define HASHZ 83492991

__global__ void computeAABBBox(int numface, const int *facetable, const REAL *vertex, g_box * aabbs);
__global__ void computeWorldBoxSerial(int numface, g_box * aabbs, g_box *wolrdBox, REAL *averageSide);
__global__ void computeWorldBox(int numface, g_box * aabbs, g_box *wolrdBox, REAL *averageSide);

__global__ void countFaceHashTable(int numface, int faceHashTableSize, int *faceHashTableCnt, g_box *aabbs, g_box *worldBox, REAL *averageSide, int DIMX, int DIMY, int DIMZ);
__global__ void fillFaceHashTable(int numface, int faceHashTableSize, int *tmptr, int *faceHashTableVal, g_box *aabbs, g_box *worldBox, REAL *averageSide, CellIdType *faceCellIdVal, int DIMX, int DIMY, int DIMZ);
__global__ void countFaceHashTableP(int numface, int faceHashTableSize, int *faceHashTableCnt, int *faceHashTableCntP, g_box *aabbs, g_box *worldBox, REAL *averageSide, int DIMX, int DIMY, int DIMZ);
__global__ void fillFaceHashTableP(int numface, int faceHashTableSize, int *tmptr, int *faceHashTable, int *faceHashTableValP, g_box *aabbs, g_box *worldBox, REAL *averageSide, CellIdType *faceCellIdValP, int DIMX, int DIMY, int DIMZ);

__global__ void countLenTable(int tableSize, int * lenTable, int *hashTable, int *hashTableP);
__global__ void countCompressNum(int oldSize, int *cntArray, int *oldArray);
__global__ void compressArray(int oldSize, int * newArray, int *cntArray, int * oldArray, int *additionOldArray, int *additionNewArray);
__global__ void compressArray(int oldSize, int * newArray, int *cntArray, int * oldArray, int *additionOldArray, int *additionNewArray, int *additionOldArray1, int *additionNewArray1);
__global__ void compressArray(int oldSize, int * newArray, int *cntArray, int * oldArray, int *additionOldArray, int *additionNewArray, int *additionOldArray1, int *additionNewArray1, int *len, int *lenCompress);

__global__ void
countFaceFaceHashTable(int numface, int faceFaceHashTableSize, int *faceFaceHashTableCnt, int totalCompare, int *lenTable, int lenCompressed, int *lenTableCompressed, int *faceHashTableCompressed, int *faceHashTablePCompressed, int *faceHlenCompressed,
int faceHashTableSize, const int *faceHashTable, const int *faceHashTableVal, const CellIdType *faceCellIdVal, int DIMX, int DIMY, int DIMZ, const int *faceHashTableP, const int *faceHashTablePVal, const CellIdType * faceCellIdValP, const int *facetable, const REAL *vertex, g_box *worldBox, g_box *faceBox, REAL *averageSide, int *complist, bool outputlist, int *pNum, FFPair *ps);
__global__ void
fillFaceFaceHashTable(int numface, int faceFaceHashTableSize, int *faceFaceHashTable, int *faceFaceHashTableVal, int totalCompare, int *lenTable, int lenCompressed, int *lenTableCompressed, int *faceHashTableCompressed, int *faceHashTablePCompressed, int *faceHlenCompressed,
int faceHashTableSize, const int *faceHashTable, const int *faceHashTableVal, const int *faceHashTableP, const int *faceHashTableValP, const int *facetable, const REAL *vertex, g_box *worldBox, g_box *faceBox, REAL *averageSide);

__global__ void fillterFFPair(int numface, int faceFaceHashTableSize, int *faceFaceHashTable, int *pairNumGpu, FFPair * ffpair, const REAL *vertex, g_box *aabbs, int * faceFaceHashTableVal);



inline __host__ __device__ int
find(int * a, int size, int target){
	int start = 0, end = size - 1;
	while (start < end){
		int mid = start + (end - start + 1) / 2;
		if (target > a[mid]) start = mid;
		else if (target < a[mid]) end = mid - 1;
		else return mid;
	}
	return start;
}

inline __host__ __device__ void
decode(int n, int &i, int &j){
	i = sqrtf(2 * n);
	while (1){
		if (i * (i + 1) / 2 > n) break;
		++i;
	}
	j = n - i * (i - 1) / 2;
	if (n < 0){
		i = -99999999;
		j = -99999999;
	}
}

inline __device__ REAL3
lowCross(const g_box &b0, const g_box &b1)
{
	REAL3 ans;
	REAL3 m1 = b0.minV();
	REAL3 m2 = b1.minV();

	ans.x = fmaxf(m1.x, m2.x);
	ans.y = fmaxf(m1.y, m2.y);
	ans.z = fmaxf(m1.z, m2.z);

	return ans;
}

#endif