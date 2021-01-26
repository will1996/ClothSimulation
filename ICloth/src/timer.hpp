#pragma once
#include <cuda_runtime.h>

#include <time.h>
#include <cstdlib>
#include <iostream>
#ifdef USE_BOOST
#include <boost/date_time/posix_time/posix_time.hpp>
#endif
#include <omp.h>

struct Timer {
#ifdef USE_BOOST
	boost::posix_time::ptime then;
#endif
	double last, total, then;

	Timer();
	void tick(), tock();
	double tock2();
};

struct GPUTimer2 {
private:
	cudaEvent_t start, stop;
	float		accum;

public:
	GPUTimer2() :accum(0) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}
	~GPUTimer2() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

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
	float tock2() {
		//cudaDeviceSynchronize();
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
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
	float tock2() {
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

#ifdef USE_TIMER
# define	TIMING_BEGIN \
	{Timer c; c.tick(); GPUTimer2 g; g.tick();

# define	TIMING_END(message) \
	{float gpuT = g.tock2();\
	double cpuT = c.tock2();\
	printf("%s: %3.5f s (%3.5f ms) \n", (message), cpuT, gpuT);}}

#else
# define	TIMING_BEGIN {
# define	TIMING_END(message) }
#endif

# define	TIMING_BEGIN1 \
	{Timer c; c.tick(); GPUTimer2 g; g.tick();

# define	TIMING_END1(message) \
	{float gpuT = g.tock2();\
	double cpuT = c.tock2();\
	printf("%s: %3.5f s (%3.5f ms) \n", (message), cpuT, gpuT);}}
