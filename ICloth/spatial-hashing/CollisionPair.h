#ifndef __CPAIR__
#define __CPAIR__

#include "./cuda/def.cuh"
#include "CudaBasic.cuh"

typedef int vertexid;
typedef int faceid;
typedef int sideid;
typedef unsigned int CellIdType;

struct VFPair{
	vertexid vid;
	faceid fid;
};

struct FFPair{
	faceid fid0;
	faceid fid1;
};

struct SSPair{
	sideid sid0;
	sideid sid1;
};


#endif