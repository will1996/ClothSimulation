#ifndef __SHCD__
#define __SHCD__

#include "CollisionPair.h"
#include "../box.cuh"
#include "../tri3f.cuh"

#define COLLISIONMAX 10000000

class SpatialHashCD{
	int _totalNode;

	int numFace, totalFace;
	int numCFace, numOFace;

	g_box * faceBox;
	g_box * worldBox;
	REAL *averageSide;

	bool ffTest;
	//build face to hash
	int faceHashTableSize; // face hash table size;
	int *faceHashTableCnt;
	int *faceHlen;
	int *faceHashTable; // index
	int valTotal;
	int *faceHashTableVal; // value
	CellIdType *faceCellIdVal; // for every H cell 
	
	//build face to hash P
	int *faceHashTableCntP;
	int *faceHashTableP; // index
	int valTotalP;
	int *faceHashTableValP; // value
	CellIdType *faceCellIdValP; // for every H cell 

	int totalCompare;
	int *lenTable;

	int *cntArray;
	int lenCompressed;
	int *lenTableCompressed;
	int *faceHlenCompressed;
	int *faceHashTableCompressed;
	int *faceHashTablePCompressed;

	//put vertex to hashtable find vf pair
	int faceFaceHashTableSize;// face num

	//TM
	int *tmptr, *tmptrP;

		
public:
	SpatialHashCD(){
		ffTest = false;

		faceBox = NULL;
		faceHashTableCnt = NULL;
		_totalNode = 0;
	};

	g_box* getAABBBox(){
		return faceBox;
	}

	void initFaceBoxs(int cNum, g_box *cBxs, int oNum, g_box *oBxs);
	void getCollisionPair(tri3f *Atris, int2 *pairSelf, uint *pairSelfIdx, int2 *pairInter, uint *pairInterIdx, int *triParents);

	~SpatialHashCD() {}
	void destroy();
};

#endif
