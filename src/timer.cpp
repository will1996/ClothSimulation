
#include <iostream>
#include "timer.hpp"
#include "simulation.hpp"
using namespace std;

#ifdef USE_BOOST
using namespace boost::posix_time;
#endif

Timer::Timer() : last(0), total(0) {
	tick();
}

void Timer::tick() {
#ifdef USE_BOOST
	then = microsec_clock::local_time();
#endif
	then = omp_get_wtime();
}

void Timer::tock() {
#ifdef USE_BOOST
	ptime now = microsec_clock::local_time();
	last = (now - then).total_microseconds()*1e-6;
	total += last;
	then = now;
#endif

	double now = omp_get_wtime();
	last = now - then;
	total += last;
	then = now;
}

double Timer::tock2() {
#ifdef USE_BOOST
	ptime now = microsec_clock::local_time();
	return (now - then).total_microseconds()*1e-6;
#else
	double now = omp_get_wtime();
	return now - then;
#endif
}
