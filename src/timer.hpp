/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

// Armin's timer class
#ifndef __TIMER_H
#define __TIMER_H

#pragma once
#include <cuda_runtime.h>

#include <time.h>
#include <cstdlib>
#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>

struct Timer {
    boost::posix_time::ptime then;
    double last, total;
    Timer ();
    void tick (), tock ();
};


# define	TIMING_BEGIN {
# define	TIMING_END(message) }

# define	TIMING_BEGIN1 \
	{Timer c; c.tick(); GPUTimer2 g; g.tick();

# define	TIMING_END1(message) \
	{float gpuT = g.tock2();\
	double cpuT = c.tock2();\
	printf("%s: %3.5f s (%3.5f ms) \n", (message), cpuT, gpuT);}}

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


#endif
