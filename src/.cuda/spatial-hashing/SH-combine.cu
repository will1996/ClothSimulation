#include "SpatialHashCD.h"
#include "SpatialHashHelper.cuh"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <stdio.h>
#include <iostream>

using std::cout;
using std::endl;

template<typename T>
T getVal(T *val){
	T t;
	checkCudaErrors(cudaMemcpy(&t, val, sizeof(T), cudaMemcpyDeviceToHost));
	return t;
}

__global__ void
computeAABBBox2(int numface, int numC, int numO, g_box *bxC, g_box *bxO, g_box * aabbs)
{
	LEN_CHK(numface);

	g_box *bx = (idx < numC ? bxC + idx : bxO + (idx - numC));
	aabbs[idx] = *bx;
}

void SpatialHashCD::initFaceBoxs(int cNum, g_box *cBxs, int oNum, g_box *oBxs)
{
	numCFace = cNum;
	numOFace = oNum;
	totalFace = cNum + oNum;

	if (oBxs == NULL) {
		numOFace = 0;
	}
	numFace = numOFace + numCFace;

	if (faceBox == NULL) {
		checkCudaErrors(cudaMalloc((void **)&faceBox, totalFace * sizeof(g_box)));
		checkCudaErrors(cudaMalloc((void **)&worldBox, sizeof(g_box)));
		checkCudaErrors(cudaMalloc((void **)&averageSide, sizeof(REAL)));
	}

	{
		GPUTimer g;
		g.tick();
		int B, T;
		computeBT(numFace, B, T);
		computeAABBBox2 << <B, T >> >(numFace, numCFace, numOFace, cBxs, oBxs, faceBox);
		getLastCudaError("computeAABBBox2");
		g.tock("face box compute");
	}
	//cout << "computeAABBBox" << endl;
	{
		GPUTimer g;
		g.tick();
		computeWorldBox << <1, 128 >> >(numFace, faceBox, worldBox, averageSide);
		getLastCudaError("computeWorldBox ");
		g.tock("world box comput2");
	}
	//cout << "computeWorldAABBBox" << endl;
}

__device__ void getPairs(int idx,
	int numface, int faceFaceHashTableSize, int *faceFaceHashTableCnt,
	int totalCompare, int *lenTable, int lenCompressed, int *lenTableCompressed,
	int *faceHashTableCompressed, int *faceHashTablePCompressed, int *faceHlenCompressed,
	int faceHashTableSize, const int *faceHashTable, const int *faceHashTableVal, const CellIdType *faceCellIdVal,
	int DIMX, int DIMY, int DIMZ, const int *faceHashTableP, const int *faceHashTablePVal, const CellIdType * faceCellIdValP,
	int numCFace, tri3f *Atris, g_box *worldBox, g_box *faceBox,
	REAL *averageSide, int *complist, bool outputlist,
	int2 *pairSelf, uint *pairSelfIdx, int2 *pairInter, uint *pairInterIdx, int *triParents)
{
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

		// all object triangles, just skip them ...
		if (fid0 >= numCFace && fid1 >= numCFace) return;

		if (fid0 != fid1 && (faceBox[fid0].overlaps(faceBox[fid1])))
		{
			bool self = (fid0 < numCFace && fid1 < numCFace); // self-collision

			if (self && covertex(fid0, fid1, Atris))
				return;

#if 1
			if (triParents) {
				int p1 = triParents[fid0];
				int p2 = triParents[fid1];
				if (p1 != 0 && p2 != 0 && p1 == p2) // normal cone culling
					return;
			}
#endif

			REAL3 corner = lowCross(faceBox[fid0], faceBox[fid1]);
			REAL3 d = corner - worldBox[0].minV();
			int cx = d.x / averageSide[0];
			int cy = d.y / averageSide[0];
			int cz = d.z / averageSide[0];
			if (cellId0 == ((CellIdType)cz + (CellIdType)cy * (CellIdType)DIMZ + (CellIdType)cx * (CellIdType)DIMZ * (CellIdType)DIMY))
			{
#if 1 
				if (self)
					addPair(fid0, fid1, pairSelf, pairSelfIdx);
				else
					addPair(fid0, fid1 - numCFace, pairInter, pairInterIdx);
#endif
			}
		}
	}
	else {
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

		// all object triangles, just skip them ...
		if (fid0 >= numCFace && fid1 >= numCFace) return;

		if (fid0 != fid1 && (faceBox[fid0].overlaps(faceBox[fid1])))
		{
			bool self = (fid0 < numCFace && fid1 < numCFace); // self-collision

			if (self && covertex(fid0, fid1, Atris))
				return;

			REAL3 corner = lowCross(faceBox[fid0], faceBox[fid1]);
			REAL3 d = corner - worldBox[0].minV();
			int cx = d.x / averageSide[0];
			int cy = d.y / averageSide[0];
			int cz = d.z / averageSide[0];
			if (cellId0 == ((CellIdType)cz + (CellIdType)cy * (CellIdType)DIMZ + (CellIdType)cx * (CellIdType)DIMZ * (CellIdType)DIMY))
			{
#if 1 
				if (self)
					addPair(fid0, fid1, pairSelf, pairSelfIdx);
				else
					addPair(fid0, fid1 - numCFace, pairInter, pairInterIdx);
#endif
			}
		}
	}
}


__global__ void kernelGetPairs(
	int numface, int faceFaceHashTableSize, int *faceFaceHashTableCnt,
	int totalCompare, int *lenTable, int lenCompressed, int *lenTableCompressed,
	int *faceHashTableCompressed, int *faceHashTablePCompressed, int *faceHlenCompressed,
	int faceHashTableSize, const int *faceHashTable, const int *faceHashTableVal, const CellIdType *faceCellIdVal,
	int DIMX, int DIMY, int DIMZ, const int *faceHashTableP, const int *faceHashTablePVal, const CellIdType * faceCellIdValP,
	int numCFace, tri3f *Atris, g_box *worldBox, g_box *faceBox,
	REAL *averageSide, int *complist, bool outputlist,
	int2 *pairSelf, uint *pairSelfIdx, int2 *pairInter, uint *pairInterIdx, int stride, int *triParents)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i < stride; i++) {
		int j = idx*stride + i;
		if (j >= totalCompare)
			return;

		getPairs(j, numface, faceFaceHashTableSize, faceFaceHashTableCnt,
			totalCompare, lenTable, lenCompressed, lenTableCompressed,
			faceHashTableCompressed, faceHashTablePCompressed, faceHlenCompressed,
			faceHashTableSize, faceHashTable, faceHashTableVal, faceCellIdVal,
			DIMX, DIMY, DIMZ, faceHashTableP, faceHashTablePVal, faceCellIdValP,
			numCFace, Atris, worldBox, faceBox, averageSide, complist, outputlist,
			pairSelf, pairSelfIdx, pairInter, pairInterIdx, triParents);
	}
}


void SpatialHashCD::getCollisionPair(tri3f *Atris, int2 *pairSelf, uint *pairSelfIdx, int2 *pairInter, uint *pairInterIdx, int *triParents)
{
	ffTest = true;
	g_box world = getVal<g_box>(worldBox);
	REAL aver = getVal<REAL>(averageSide);
	REAL3 d = world.maxV() - world.minV();
	unsigned int Dimx = d.x / aver;
	unsigned int Dimy = d.y / aver;
	unsigned int Dimz = d.z / aver;
	Dimx += 1;
	Dimy += 1;
	Dimz += 1;

	GPUTimer g;
	g.tick();
	{//build face to hash
		faceHashTableSize = (5 * totalFace / 100 + 1) * 100 - 27; // a test number;

		if (faceHashTableCnt == NULL) {
			checkCudaErrors(cudaMalloc((void **)&faceHashTableCnt, faceHashTableSize * sizeof(int)));
			checkCudaErrors(cudaMalloc((void **)&faceHlen, faceHashTableSize * sizeof(int)));
			checkCudaErrors(cudaMalloc((void **)&faceHashTableCntP, faceHashTableSize * sizeof(int)));
			checkCudaErrors(cudaMalloc((void **)&faceHashTable, (faceHashTableSize + 1) * sizeof(int)));

			checkCudaErrors(cudaMalloc((void **)&tmptr, (faceHashTableSize + 1) * sizeof(int)));
			checkCudaErrors(cudaMalloc((void **)&tmptrP, (faceHashTableSize + 1) * sizeof(int)));

			_totalNode = totalFace * 10;

			checkCudaErrors(cudaMalloc((void **)&faceHashTableVal, _totalNode * sizeof(int)));
			// cellid
			checkCudaErrors(cudaMalloc((void **)&faceCellIdVal, _totalNode * sizeof(CellIdType)));
			checkCudaErrors(cudaMalloc((void **)&faceHashTableValP, _totalNode * sizeof(int)));
			checkCudaErrors(cudaMalloc((void **)&faceCellIdValP, _totalNode * sizeof(CellIdType)));
			checkCudaErrors(cudaMalloc((void **)&faceHashTableP, (faceHashTableSize + 1) * sizeof(int)));

			checkCudaErrors(cudaMalloc((void **)&lenTable, (faceHashTableSize + 1) * sizeof(int)));
			checkCudaErrors(cudaMalloc((void**)&cntArray, sizeof(int)* (faceHashTableSize + 1)));

			int _lenCompressed = faceHashTableSize + 1;
			checkCudaErrors(cudaMalloc((void**)&lenTableCompressed, sizeof(int)* _lenCompressed));
			checkCudaErrors(cudaMalloc((void**)&faceHashTableCompressed, sizeof(int)* _lenCompressed));
			checkCudaErrors(cudaMalloc((void**)&faceHashTablePCompressed, sizeof(int)* _lenCompressed));
			checkCudaErrors(cudaMalloc((void**)&faceHlenCompressed, sizeof(int)* (_lenCompressed - 1)));
		}

		checkCudaErrors(cudaMemset(faceHashTableCnt, 0, faceHashTableSize * sizeof(int)));
		checkCudaErrors(cudaMemset(faceHlen, 0, faceHashTableSize * sizeof(int)));
		//for P
		checkCudaErrors(cudaMemset(faceHashTableCntP, 0, faceHashTableSize * sizeof(int)));
		{//count space
			int B, T;
			computeBT(numFace, B, T);
			countFaceHashTable << <B, T >> >(numFace, faceHashTableSize, faceHashTableCnt, faceBox, worldBox, averageSide, Dimx, Dimy, Dimz);
			getLastCudaError("countFaceHashTable");
			//g.tock("1. count face hash");
			//printArray<int>(faceHashTableCnt, faceHashTableSize, "./output/hash/everyslotcnt.txt");
		}
		checkCudaErrors(cudaMemcpy(faceHlen, faceHashTableCnt, sizeof(int) * faceHashTableSize, cudaMemcpyDeviceToDevice));
		{//cout P space
			int B, T;
			computeBT(numFace, B, T);
			countFaceHashTableP << <B, T >> >(numFace, faceHashTableSize, faceHashTableCnt, faceHashTableCntP, faceBox, worldBox, averageSide, Dimx, Dimy, Dimz);
			getLastCudaError("countFaceHashTable");
		}
		thrust::device_ptr<int> dev_data(faceHashTableCnt);
		thrust::inclusive_scan(dev_data, dev_data + faceHashTableSize, dev_data);
		int totalNode;
		checkCudaErrors(cudaMemcpy(&totalNode, faceHashTableCnt + faceHashTableSize - 1, sizeof(int), cudaMemcpyDeviceToHost));
		valTotal = totalNode;

		if (totalNode > _totalNode) {
			printf("totalNode is bigger than preset value! (%d, %d)\n", totalNode, _totalNode);
			abort();
		}

		//g2.tock("1. face hash malloc");
		int tmp = 0;
		checkCudaErrors(cudaMemcpy(faceHashTable, &tmp, sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(faceHashTable + 1, faceHashTableCnt, faceHashTableSize * sizeof(int), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(tmptr, faceHashTable, (faceHashTableSize + 1)* sizeof(int), cudaMemcpyDeviceToDevice));
		//**********************************for P table
		{
			thrust::device_ptr<int> dev_data(faceHashTableCntP);
			thrust::inclusive_scan(dev_data, dev_data + faceHashTableSize, dev_data);
			int totalNodeP;
			checkCudaErrors(cudaMemcpy(&totalNodeP, faceHashTableCntP + faceHashTableSize - 1, sizeof(int), cudaMemcpyDeviceToHost));
			valTotalP = totalNodeP;
			checkCudaErrors(cudaMemset(faceHashTableValP, 0, totalNodeP * sizeof(int)));

			int tmp = 0;
			checkCudaErrors(cudaMemcpy(faceHashTableP, &tmp, sizeof(int), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(faceHashTableP + 1, faceHashTableCntP, faceHashTableSize * sizeof(int), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(tmptrP, faceHashTableP, (faceHashTableSize + 1)* sizeof(int), cudaMemcpyDeviceToDevice));
			{// fill P faceHashTable P
				int B, T;
				computeBT(numFace, B, T);
				fillFaceHashTableP << <B, T >> >(numFace, faceHashTableSize, tmptrP, faceHashTable/*for judge*/, faceHashTableValP, faceBox, worldBox, averageSide, faceCellIdValP, Dimx, Dimy, Dimz);
				getLastCudaError("fillfaceHashTable");
				//printHash<int>(faceHashTableSize, totalNodeP, faceHashTableP, faceHashTableValP, "./output/hash2/faceHashTableP.txt");
			}
		}
		{//fill
			int B, T;
			computeBT(numFace, B, T);
			fillFaceHashTable << <B, T >> >(numFace, faceHashTableSize, tmptr, faceHashTableVal, faceBox, worldBox, averageSide, faceCellIdVal, Dimx, Dimy, Dimz);
			getLastCudaError("fillfaceHashTable");

		}
	}
	{
		{
			{
				checkCudaErrors(cudaMemset(lenTable, 0, (faceHashTableSize + 1) * (sizeof(int))));
				{
					int B, T;
					computeBT<512>(faceHashTableSize + 1, B, T);
					countLenTable << <B, T >> >(faceHashTableSize + 1, lenTable, faceHashTable, faceHashTableP);
					thrust::device_ptr<int> dev_data(lenTable);
					thrust::inclusive_scan(dev_data, dev_data + faceHashTableSize + 1, dev_data);
					checkCudaErrors(cudaMemcpy(&totalCompare, lenTable + faceHashTableSize, sizeof(int), cudaMemcpyDeviceToHost));
				}
			}
			{// compress lenTable
				{
					int B, T;
					computeBT<512>(faceHashTableSize + 1, B, T);
					countCompressNum << <B, T >> >(faceHashTableSize + 1, cntArray, lenTable);
					getLastCudaError("countCompressNum");
				}
				//printArray<int>(lenTable, faceHashTableSize + 1, "./output/hash2/lenTable.txt");
				//printArray<int>(cntArray, faceHashTableSize + 1, "./output/hash2/cntArray.txt");
				thrust::device_ptr<int> dev_data(cntArray);
				thrust::inclusive_scan(dev_data, dev_data + faceHashTableSize + 1, dev_data);
				//printArray<int>(cntArray, faceHashTableSize + 1, "./output/hash2/cntArray2.txt");
				checkCudaErrors(cudaMemcpy(&lenCompressed, cntArray + faceHashTableSize, sizeof(int), cudaMemcpyDeviceToHost));//now : lenCompressed meam how many elements should be removed
				//cout << lenCompressed << endl;
				lenCompressed = faceHashTableSize + 1 - lenCompressed; // now : lenCompressed mean after compress the len should be

				checkCudaErrors(cudaMemset(lenTableCompressed, 0, sizeof(int)* lenCompressed));
				checkCudaErrors(cudaMemset(faceHashTableCompressed, 0, sizeof(int)* lenCompressed));
				checkCudaErrors(cudaMemset(faceHashTablePCompressed, 0, sizeof(int)* lenCompressed));
				checkCudaErrors(cudaMemset(faceHlenCompressed, 0, sizeof(int)* (lenCompressed - 1)));
				{
					int B, T;
					computeBT<512>(faceHashTableSize + 1, B, T);
					compressArray << <B, T >> >
						(faceHashTableSize + 1, lenTableCompressed, cntArray, lenTable,
						faceHashTable, faceHashTableCompressed, faceHashTableP,
						faceHashTablePCompressed, faceHlen, faceHlenCompressed);
					getLastCudaError("compressArray");
				}
			}
		}

		faceFaceHashTableSize = numFace;
		{//count how many vertex to each face
			//int B, T;
			//computeBT<256>(totalCompare, B, T);

			int stride = 4;
			BLK_PAR2(totalCompare, stride);
			//printf("here!! stride = %d\n", stride);

			kernelGetPairs << <B, T >> > (
				numFace, faceFaceHashTableSize, NULL,
				totalCompare, lenTable, lenCompressed, lenTableCompressed,
				faceHashTableCompressed, faceHashTablePCompressed, faceHlenCompressed,
				faceHashTableSize, faceHashTable, faceHashTableVal, faceCellIdVal,
				Dimx, Dimy, Dimz, faceHashTableP, faceHashTableValP, faceCellIdValP,
				numCFace, Atris, worldBox, faceBox, averageSide, NULL, false,
				pairSelf, pairSelfIdx, pairInter, pairInterIdx, stride, triParents);

			getLastCudaError("countFaceFaceHashTable");
		}

		g.tock("time cost");

	}
}

void SpatialHashCD::destroy()
{
	checkCudaErrors(cudaFree(faceBox));
	checkCudaErrors(cudaFree(worldBox));
	checkCudaErrors(cudaFree(averageSide));

	checkCudaErrors(cudaFree(faceHashTableCnt));
	checkCudaErrors(cudaFree(faceHlen));
	checkCudaErrors(cudaFree(faceHashTable));
	checkCudaErrors(cudaFree(faceHashTableVal));
	checkCudaErrors(cudaFree(faceCellIdVal));

	checkCudaErrors(cudaFree(faceHashTableCntP));
	checkCudaErrors(cudaFree(faceHashTableP));
	checkCudaErrors(cudaFree(faceHashTableValP));
	checkCudaErrors(cudaFree(faceCellIdValP));

	checkCudaErrors(cudaFree(lenTable));

	checkCudaErrors(cudaFree(cntArray));
	checkCudaErrors(cudaFree(lenTableCompressed));
	checkCudaErrors(cudaFree(faceHlenCompressed));
	checkCudaErrors(cudaFree(faceHashTableCompressed));
	checkCudaErrors(cudaFree(faceHashTablePCompressed));

	checkCudaErrors(cudaFree(tmptr));
	checkCudaErrors(cudaFree(tmptrP));

	faceBox = NULL;
	worldBox = NULL;
	averageSide = NULL;

	faceHashTableCnt = NULL;
	faceHlen = NULL;
	faceHashTable = NULL;
	faceHashTableVal = NULL;
	faceCellIdVal = NULL;

	faceHashTableCntP = NULL;
	faceHashTableP = NULL;
	faceHashTableValP = NULL;
	faceCellIdValP = NULL;

	lenTable = NULL;

	cntArray = NULL;
	lenTableCompressed = NULL;
	faceHlenCompressed = NULL;
	faceHashTableCompressed = NULL;
	faceHashTablePCompressed = NULL;

	tmptr = NULL;
	tmptrP = NULL;
}
