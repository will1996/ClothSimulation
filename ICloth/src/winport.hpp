
#ifndef WINPORT_HPP
#define WINPORT_HPP
// MS Windows bindings, etc

#if defined(_WIN32) && !defined(__CYGWIN__)

#pragma warning(disable:4018) // signed/unsigned mismatch
#pragma warning(disable:4244) // conversion from 'double' to 'float', possible loss of data
#pragma warning(disable:4996) // this function or variable may be unsafe
#pragma warning(disable:4251) // class needs to have dll-interface to be used by clients
#pragma warning(disable:4800) // forcing value to bool 'true' or 'false'
#pragma warning(disable:161)  // unrecognized #pragma
//#pragma warning(disable:1011) // missing return statement

#define _USE_MATH_DEFINES // just to have M_PI
#include <cmath>

#include <windows.h>
#undef min
#undef max
#include <stdio.h>
#define snprintf _snprintf

#ifdef USE_BOOST
#include <boost/math/special_functions/fpclassify.hpp> 
template <class T> inline bool isfinite(const T& number) { return boost::math::isfinite(number); }
template <class T> inline bool   finite(const T& number) { return boost::math::isfinite(number); }
#endif

inline double sqrt(int n) { return sqrt(double(n)); }

template <class T> inline T log2(const T& number) { return log(number)/log(T(2)); }

#include <iostream>

extern std::ostream cdbg;

#endif

#endif
