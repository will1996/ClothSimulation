#ifndef __CUDABASIC__
#define __CUDABASIC__

#include <cuda_runtime.h>
#include "helper_cuda.h"

#define TPBlock 64

template<int ThreadPerBlock = TPBlock>
void computeBT(int totalThreadNum, int & B, int &T){
	int blocksPerGrid = (totalThreadNum + ThreadPerBlock - 1) / (ThreadPerBlock);
	if (blocksPerGrid > 65536) {
		printf("blocksPerGrid is larger than 65536, aborting ... (N=%d, TPB=%d, BPG=%d)\n", totalThreadNum, ThreadPerBlock, blocksPerGrid);
		exit(0);
	}
	B = blocksPerGrid;
	T = ThreadPerBlock;
}

#define LEN_CHK(l) \
	int idx = blockDim.x * blockIdx.x + threadIdx.x; \
if (idx >= l) return;

template<typename T> inline __host__ __device__ void
swapT(T & i, T &j){
	T tmp = i;
	i = j;
	j = tmp;
}

struct GPUTimer {
private:
	cudaEvent_t start, stop;
	float		accum;

public:
	GPUTimer() :accum(0) {}

#ifdef USE_TIMER
	void tick() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
	}
	void tock() {
		//cudaDeviceSynchronize();
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
	}
	float tock(std::string msg) {
		//cudaDeviceSynchronize();
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		printTime(msg);
		float tmp;
		cudaEventElapsedTime(&tmp, start, stop);
		return tmp;
	}
	void printTime(std::string msg = std::string()) {
		float	costtime;
		cudaEventElapsedTime(&costtime, start, stop);
		accum += costtime;
		printf("%s %.6f ms\n", msg.c_str(), costtime);
	}
	void reset() {
		accum = 0;
	}
	void reset(std::string msg) {
		printf("%s %.6f ms\n", msg.c_str(), accum);
		accum = 0;
	}
#else
	void tick() {
	}
	void tock() {
	}
	float tock(std::string msg) {
		return 0;
	}
	void printTime(std::string msg = std::string()) {
	}
	void reset() {
	}
	void reset(std::string msg) {
	}
#endif
};

#endif
