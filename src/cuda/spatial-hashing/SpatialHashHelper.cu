#include "SpatialHashHelper.cuh"
//#include  <device_atomic_functions.hpp>
#include "../box.cuh"


inline __device__ void
_sort_device(int *data, int num)
{
	// bubble sort
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
_unique_device(int *data, int num)
{
	int loc = 0;
	for (int i = 1; i < num; i++) {
		if (data[i] == data[loc])
			continue;
		else
			data[++loc] = data[i];
	}
	return loc + 1;
}

// ouput: in place sorted & unique array (data), return its really length
inline __device__ int
gpu_sort_device(int *data, int num)
{
	if (num == 0) return 0;
	_sort_device(data, num);
	return _unique_device(data, num);
}


__global__ void 
computeAABBBox(int numface, const int *facetable, const REAL *vertex, g_box * aabbs){
	LEN_CHK(numface);
	REAL3 v0 = make_REAL3(vertex[3 * facetable[idx * 3 + 0] + 0],
		vertex[3 * facetable[idx * 3 + 0] + 1], 
		vertex[3 * facetable[idx * 3 + 0] + 2]);
	REAL3 v1 = make_REAL3(vertex[3 * facetable[idx * 3 + 1] + 0],
		vertex[3 * facetable[idx * 3 + 1] + 1],
		vertex[3 * facetable[idx * 3 + 1] + 2]);
	REAL3 v2 = make_REAL3(vertex[3 * facetable[idx * 3 + 2] + 0],
		vertex[3 * facetable[idx * 3 + 2] + 1],
		vertex[3 * facetable[idx * 3 + 2] + 2]);

	g_box tmp;
	tmp.set(v0, v1);
	tmp.add(v2);

	aabbs[idx] = tmp;
}

#define THREADNUM 128
__global__ void
computeWorldBox(int numface, g_box * aabbs, g_box *worldBox, REAL *averageSide){
	__shared__ g_box tmpBox[THREADNUM];
	__shared__ REAL tmpSide[THREADNUM];
	LEN_CHK(THREADNUM);
	int k = THREADNUM;
	if (idx < numface){
		tmpBox[idx] = aabbs[idx];
		REAL3 d = aabbs[idx].maxV() - aabbs[idx].minV();
		tmpSide[idx] = d.x;
		tmpSide[idx] += d.y;
		tmpSide[idx] += d.z;
	}
	while (idx + k < numface){
		tmpBox[idx].add(aabbs[idx + k]);
		REAL3 d = aabbs[idx + k].maxV() - aabbs[idx + k].minV();
		tmpSide[idx] += d.x;
		tmpSide[idx] += d.y;
		tmpSide[idx] += d.z;
		k += THREADNUM;
	}
	__syncthreads();
	
	int i = 1;
	int strip;
	while ((idx &((1 << i) - 1)) == 0 && (strip = (1 << (i - 1)), idx + strip < (THREADNUM < numface ? THREADNUM : numface))){
		tmpBox[idx].add(tmpBox[idx + strip]);
		tmpSide[idx] += tmpSide[idx + strip];
		++i;
		__syncthreads();
	}
	if (idx == 0){
		worldBox[0] = tmpBox[0];
		averageSide[0] = tmpSide[0] / 3.0 / numface;
		averageSide[0] *= 1.5;
	}
}

__global__ void 
countFaceHashTable(int numface,int faceHashTableSize, int *faceHashTableCnt, g_box *aabbs, g_box *worldBox, REAL *averageSide, int DIMX, int DIMY, int DIMZ){
	LEN_CHK(numface);
#if 1 
	REAL3 d = aabbs[idx].minV() - worldBox[0].minV();
	unsigned int i = d.x / averageSide[0];
	unsigned int j = d.y / averageSide[0];
	unsigned int k = d.z / averageSide[0];

	d = aabbs[idx].maxV() - worldBox[0].minV();
	unsigned int maxi = d.x / averageSide[0];
	unsigned int maxj = d.y / averageSide[0];
	unsigned int maxk = d.z / averageSide[0];

	unsigned int hashkey = ((CellIdType)k + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY) % (unsigned int)faceHashTableSize;
	atomicAdd(faceHashTableCnt + hashkey, 1);
	for (unsigned int ii = i + 1; ii <= maxi; ++ii){
		hashkey = ((CellIdType)k + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)ii * (CellIdType)DIMZ * (CellIdType)DIMY) % (unsigned int)faceHashTableSize;
		atomicAdd(faceHashTableCnt + hashkey, 1);
	}
	for (unsigned int jj = j + 1; jj <= maxj; ++jj){
		hashkey = ((CellIdType)k + (CellIdType)jj * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY) % (unsigned int)faceHashTableSize;
		atomicAdd(faceHashTableCnt + hashkey, 1);
	}
	for (unsigned int kk = k + 1; kk <= maxk; ++kk){
		hashkey = ((CellIdType)kk + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY) % (unsigned int)faceHashTableSize;
		atomicAdd(faceHashTableCnt + hashkey, 1);
	}
#endif
}

__global__ void 
fillFaceHashTable(int numface, int faceHashTableSize, int *tmptr, int *faceHashTableVal, g_box *aabbs, g_box *worldBox, REAL *averageSide, CellIdType *faceCellIdVal, int DIMX, int DIMY, int DIMZ){
	LEN_CHK(numface);
#if 1 
	REAL3 d = aabbs[idx].minV() - worldBox[0].minV();
	unsigned int i = d.x / averageSide[0];
	unsigned int j = d.y / averageSide[0];
	unsigned int k = d.z / averageSide[0];

	d = aabbs[idx].maxV() - worldBox[0].minV();
	unsigned int maxi = d.x / averageSide[0];
	unsigned int maxj = d.y / averageSide[0];
	unsigned int maxk = d.z / averageSide[0];

	unsigned int hashkey = ((CellIdType)k + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY) % (unsigned int)faceHashTableSize;
	int index = atomicAdd(tmptr + hashkey, 1);
	faceHashTableVal[index] = idx;
	faceCellIdVal[index] = (CellIdType)k + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY;
	for (unsigned int ii = i + 1; ii <= maxi; ++ii){
		hashkey = ((CellIdType)k + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)ii * (CellIdType)DIMZ * (CellIdType)DIMY) % (unsigned int)faceHashTableSize;
		index = atomicAdd(tmptr + hashkey, 1);
		faceHashTableVal[index] = idx;
		faceCellIdVal[index] = (CellIdType)k + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)ii * (CellIdType)DIMZ * (CellIdType)DIMY;
	}
	for (unsigned int jj = j + 1; jj <= maxj; ++jj){
		hashkey = ((CellIdType)k + (CellIdType)jj * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY) % (unsigned int)faceHashTableSize;
		index = atomicAdd(tmptr + hashkey, 1);
		faceHashTableVal[index] = idx;
		faceCellIdVal[index] = (CellIdType)k + (CellIdType)jj * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY;
	}
	for (unsigned int kk = k + 1; kk <= maxk; ++kk){
		hashkey = ((CellIdType)kk + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY) % (unsigned int)faceHashTableSize;
		index = atomicAdd(tmptr + hashkey, 1);
		faceHashTableVal[index] = idx;
		faceCellIdVal[index] = (CellIdType)kk + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY;
	}
#endif
}


__global__ void
countFaceHashTableP(int numface, int faceHashTableSize, int *faceHashTableCnt, int *faceHashTableCntP, g_box *aabbs, g_box *worldBox, REAL *averageSide, int DIMX, int DIMY, int DIMZ){
	LEN_CHK(numface);

	REAL3 d = aabbs[idx].minV() - worldBox[0].minV();
	unsigned int mini = d.x / averageSide[0];
	unsigned int minj = d.y / averageSide[0];
	unsigned int mink = d.z / averageSide[0];

	d = aabbs[idx].maxV() - worldBox[0].minV();
	unsigned int maxi = d.x / averageSide[0];
	unsigned int maxj = d.y / averageSide[0];
	unsigned int maxk = d.z / averageSide[0];

#if 1 
	for (unsigned int i = mini; i <= maxi; ++i)
	for (unsigned int j = minj; j <= maxj; ++j)
	for (unsigned int k = mink; k <= maxk; ++k)
	{
		if ((!(i == mini && j == minj)) &&
			(!(i == mini && k == mink)) &&
			(!(j == minj && k == mink))){
			//unsigned int hashkey = ((i * HASHX) ^ (j * HASHY) ^ (k * HASHZ)) % (unsigned int)faceHashTableSize;
			unsigned int hashkey = ((CellIdType)k + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY) % (unsigned int)faceHashTableSize;
			if (faceHashTableCnt[hashkey] == 0) continue;
			atomicAdd(faceHashTableCntP + hashkey, 1);
		}
	}
#endif
}


__global__ void
fillFaceHashTableP(int numface, int faceHashTableSize, int *tmptr, int *faceHashTable, int *faceHashTableValP, g_box *aabbs, g_box *worldBox, REAL *averageSide, CellIdType *faceCellIdValP, int DIMX, int DIMY, int DIMZ){
	LEN_CHK(numface);
	REAL3 d = aabbs[idx].minV() - worldBox[0].minV();
	unsigned int mini = d.x / averageSide[0];
	unsigned int minj = d.y / averageSide[0];
	unsigned int mink = d.z / averageSide[0];

	d = aabbs[idx].maxV() - worldBox[0].minV();
	unsigned int maxi = d.x / averageSide[0];
	unsigned int maxj = d.y / averageSide[0];
	unsigned int maxk = d.z / averageSide[0];

#if 1 
	for (unsigned int i = mini; i <= maxi; ++i)
	for (unsigned int j = minj; j <= maxj; ++j)
	for (unsigned int k = mink; k <= maxk; ++k)
	{
		if ((!(i == mini && j == minj)) &&
			(!(i == mini && k == mink)) &&
			(!(j == minj && k == mink))){
			unsigned int hashkey = ((CellIdType)k + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY) % (unsigned int)faceHashTableSize;
			if (faceHashTable[hashkey + 1] - faceHashTable[hashkey] == 0) continue;
			int index = atomicAdd(tmptr + hashkey, 1);
			faceHashTableValP[index] = idx;
			faceCellIdValP[index] = (CellIdType)k + (CellIdType)j * (CellIdType)DIMZ + (CellIdType)i * (CellIdType)DIMZ * (CellIdType)DIMY;
		}
	}
#endif
}


//ff test
inline __device__ bool
commonVertex(int fid0, int fid1, const int *faceTable){
	int a = faceTable[3 * fid0 + 0];
	int b = faceTable[3 * fid0 + 1];
	int c = faceTable[3 * fid0 + 2];
	int d = faceTable[3 * fid1 + 0];
	int e = faceTable[3 * fid1 + 1];
	int f = faceTable[3 * fid1 + 2];
	if (a != d && a != e && a != f &&
		b != d && b != e && b != f &&
		c != d && c != e && c != f) return false;
	return true;
}

__global__ void
countLenTable(int tableSize, int * lenTable, int *hashTable, int *hashTableP){
	LEN_CHK(tableSize);
	if (idx != 0){
		int len = hashTable[idx] - hashTable[idx - 1];
		int len1 = hashTableP[idx] - hashTableP[idx - 1];
		lenTable[idx] = len * (len - 1) / 2 + len * len1;
	}
	else lenTable[idx] = 0;
}

__global__ void
countCompressNum(int oldSize, int *cntArray, int *oldArray){
	LEN_CHK(oldSize);
	if (idx != 0 && oldArray[idx] == oldArray[idx - 1]) cntArray[idx] = 1;
	else cntArray[idx] = 0;
}

__global__ void
compressArray(int oldSize, int * newArray, int *cntArray, int * oldArray, int *additionOldArray, int *additionNewArray){
	LEN_CHK(oldSize);
	if (idx == oldSize - 1 || oldArray[idx] != oldArray[idx + 1]){
		newArray[idx - cntArray[idx]] = oldArray[idx];
		additionNewArray[idx - cntArray[idx]] = additionOldArray[idx];
	}
}

__global__ void
compressArray(int oldSize, int * newArray, int *cntArray, int * oldArray, int *additionOldArray, int *additionNewArray, int *additionOldArray1, int *additionNewArray1){
	LEN_CHK(oldSize);
	if (idx == oldSize - 1 || oldArray[idx] != oldArray[idx + 1]){
		newArray[idx - cntArray[idx]] = oldArray[idx];
		additionNewArray[idx - cntArray[idx]] = additionOldArray[idx];
		additionNewArray1[idx - cntArray[idx]] = additionOldArray1[idx];
	}
}

__global__ void
compressArray(int oldSize, int * newArray, int *cntArray, int * oldArray, int *additionOldArray, int *additionNewArray, int *additionOldArray1, int *additionNewArray1, int *len, int *lenCompressed){
	LEN_CHK(oldSize);
	if (idx == oldSize - 1 || oldArray[idx] != oldArray[idx + 1]){
		newArray[idx - cntArray[idx]] = oldArray[idx];
		additionNewArray[idx - cntArray[idx]] = additionOldArray[idx];
		additionNewArray1[idx - cntArray[idx]] = additionOldArray1[idx];

		if (idx != oldSize-1)
			lenCompressed[idx - cntArray[idx]] = len[idx];
	}
}

__global__ void
countFaceFaceHashTable(int numface, int faceFaceHashTableSize, int *faceFaceHashTableCnt, int totalCompare, int *lenTable, int lenCompressed, int *lenTableCompressed, int *faceHashTableCompressed, int *faceHashTablePCompressed, int *faceHlenCompressed,
int faceHashTableSize, const int *faceHashTable, const int *faceHashTableVal, const CellIdType *faceCellIdVal, int DIMX, int DIMY, int DIMZ, const int *faceHashTableP, const int *faceHashTablePVal, const CellIdType * faceCellIdValP, const int *facetable, const REAL *vertex, g_box *worldBox, g_box *faceBox, REAL *averageSide, int *complist, bool outputlist, int *pNum, FFPair *ps){
	LEN_CHK(totalCompare);
	int key = find(lenTableCompressed, lenCompressed, idx);
	int offset = idx - lenTableCompressed[key];
	int start = faceHashTableCompressed[key];
	int startP = faceHashTablePCompressed[key];
	int Hlen = faceHlenCompressed[key];
	int offset1 = offset - Hlen * (Hlen - 1) / 2;
	if (offset1 < 0){
		int i, j;
		int fid0, fid1, cellId0, cellId1;

		decode(offset, i, j);
		fid0 = faceHashTableVal[start + i];
		fid1 = faceHashTableVal[start + j];
		cellId0 = faceCellIdVal[start + i];
		cellId1 = faceCellIdVal[start + j];

		//filter
		if (fid0 > fid1) swapT(fid0, fid1);
		if (outputlist){
			complist[2 * idx] = fid0;
			complist[2 * idx + 1] = fid1;
		}
		//new
		if (cellId0 != cellId1) return;
		if (fid0 != fid1 && (!commonVertex(fid0, fid1, facetable) && (faceBox[fid0].overlaps(faceBox[fid1])))){
			REAL3 corner = lowCross(faceBox[fid0], faceBox[fid1]);
			REAL3 d = corner - worldBox[0].minV();
			int cx = d.x / averageSide[0];
			int cy = d.y / averageSide[0];
			int cz = d.z / averageSide[0];
			if (cellId0 == ((CellIdType)cz + (CellIdType)cy * (CellIdType)DIMZ + (CellIdType)cx * (CellIdType)DIMZ * (CellIdType)DIMY)){
#if 1 
				int findex = atomicAdd(pNum, 1);
				ps[findex].fid0 = fid0;
				ps[findex].fid1 = fid1;
#endif
			}
		}
	}
	else if(1){
		int i, j;
		int fid0, fid1, cellId0, cellId1;

		i = offset1 % Hlen;
		j = offset1 / Hlen;

		fid0 = faceHashTableVal[start + i];
		fid1 = faceHashTablePVal[startP + j];
		cellId0 = faceCellIdVal[start + i];
		cellId1 = faceCellIdValP[startP + j];

		if (fid0 > fid1) swapT(fid0, fid1);
		if (outputlist){
			complist[2 * idx] = fid0;
			complist[2 * idx + 1] = fid1;
		}
		if (cellId0 != cellId1) return;
		if (fid0 != fid1 && (!commonVertex(fid0, fid1, facetable) && (faceBox[fid0].overlaps(faceBox[fid1])))){
			REAL3 corner = lowCross(faceBox[fid0], faceBox[fid1]);
			REAL3 d = corner - worldBox[0].minV();
			int cx = d.x / averageSide[0];
			int cy = d.y / averageSide[0];
			int cz = d.z / averageSide[0];
			if (cellId0 == ((CellIdType)cz + (CellIdType)cy * (CellIdType)DIMZ + (CellIdType)cx * (CellIdType)DIMZ * (CellIdType)DIMY)){
#if 1 
				int findex = atomicAdd(pNum, 1);
				ps[findex].fid0 = fid0;
				ps[findex].fid1 = fid1;
#endif
			}
		}
	}
}
