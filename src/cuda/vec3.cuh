/////////////////////////////////////////////////////////////////////////////////////////////
// write based on cutil_math.h
// for double3
#ifndef VEC3_CUH
#define VEC3_CUH

inline __host__ __device__ double3 make_double3(double s)
{
    return make_double3(s, s, s);
}

inline __host__ __device__ double3 make_double3(const double s[])
{
    return make_double3(s[0], s[1], s[2]);
}

inline __host__ __device__ double getI(const double3 &a, int i)
{
	if (i == 0)
		return a.x;
	else if (i == 1)
		return a.y;
	else
		return a.z;
}

inline __host__ __device__ double3 zero3f()
{
	return make_double3(0, 0, 0);
}

inline __host__ __device__ void fswap(double &a, double &b)
{
	double t = b;
	b = a;
	a = t;
}

inline  __host__ __device__ double fminf(double a, double b)
{
	return a < b ? a : b;
}

inline  __host__ __device__ double fmaxf(double a, double b)
{
	return a > b ? a : b;
}

inline __host__ __device__ double3 fminf(const double3 &a, const double3 &b)
{
	return make_double3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

inline __host__ __device__ double3 fmaxf(const double3 &a, const double3 &b)
{
	return make_double3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

inline __host__ __device__ double3 operator-(const double3 &a, const double3 &b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ double2 operator-(const double2 &a, const double2 &b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ void operator-=(double3 &a, const double3 &b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __host__ __device__ double3 cross(const double3 &a, const double3 &b)
{ 
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

inline __host__ __device__ double dot(const double3 &a, const double3 &b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ double dot(const double2 &a, const double2 &b)
{ 
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ double stp(const double3 &u, const double3 &v, const double3 &w)
{
	return dot(u,cross(v,w));
}

inline __host__ __device__ double3 operator+(const double3 &a, const double3 &b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ double2 operator+(const double2 &a, const double2 &b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(double3 &a, double3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __host__ __device__ void operator*=(double3 &a, double3 b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __host__ __device__ void operator*=(double2 &a, double b)
{
    a.x *= b; a.y *= b;
}

inline __host__ __device__ double3 operator*(const double3 &a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ double2 operator*(const double2 &a, double b)
{
    return make_double2(a.x * b, a.y * b);
}

inline __host__ __device__ double2 operator*(double b, const double2 &a)
{
    return make_double2(a.x * b, a.y * b);
}

inline __host__ __device__ double3 operator*(double b, const double3 &a)
{
    return make_double3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ void operator*=(double3 &a, double b)
{
    a.x *= b; a.y *= b; a.z *= b;
}

inline __host__ __device__ double3 operator/(const double3 &a, double b)
{
    return make_double3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ void operator/=(double3 &a, double b)
{
    a.x /= b; a.y /= b; a.z /= b;
}

inline __host__ __device__ double3 operator-(const double3 &a)
{
    return make_double3(-a.x, -a.y, -a.z);
}

//inline __device__ double atomicAdd(double* address, double val)
//{
//	unsigned long long int* address_as_ull =  (unsigned long long int*)address;
//	unsigned long long int old = *address_as_ull, assumed; 
//	
//	do {
//		assumed = old;
//		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val + __longlong_as_double(assumed)));
//	} while (assumed != old);
//	
//	return __longlong_as_double(old);
//}

inline __host__ __device__ double norm2(const double3 &v)
{
	return dot(v, v);
}

inline __host__ __device__ double length(const double3 &v)
{
    return sqrt(dot(v, v));
}


inline __host__ __device__ double3 normalize(const double3 &v)
{
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}

inline __device__ __host__ double3 lerp(const double3 &a, const double3 &b, double t)
{
    return a + t*(b-a);
}

inline __device__ __host__ double clamp(double x, double a, double b)
{
    return fminf(fmaxf(x, a), b);
}

inline __device__ __host__ double distance (const double3 &x, const double3 &a, const double3 &b) {
    double3 e = b-a;
    double3 xp = e*dot(e, x-a)/dot(e,e);
    // return norm((x-a)-xp);
    return max(length((x-a)-xp), 1e-3*length(e));
}

inline __device__ __host__ double2 barycentric_weights (const double3 &x, const double3 &a, const double3 &b) {
    double3 e = b-a;
    double t = dot(e, x-a)/dot(e,e);
    return make_double2(1-t, t);
}

inline __device__ void atomicAdd(double3 *address, const double3 &val)
{
	atomicAdd(&address->x, val.x);
	atomicAdd(&address->y, val.y);
	atomicAdd(&address->z, val.z);
}
#endif
