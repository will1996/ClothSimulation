
#include "spline.hpp"
#include "util.hpp"

using namespace std;

// binary search, returns keyframe immediately *after* given time
// range of output: 0 to a.keyfs.size() inclusive
template<typename T>
static int find (const Spline<T> &s, REAL t) {
    int l = 0, u = s.points.size();
    while (l != u) {
         int m = (l + u)/2;
         if (t < s.points[m].t) u = m;
         else l = m + 1;
    }
    return l; // which is equal to u
}

template<typename T>
T Spline<T>::pos (REAL t) const {
    int i = find(*this, t);
    if (i == 0) {
         const Point &p1 = points[i];
         return p1.x;
    } else if (i == points.size()) {
         const Point &p0 = points[i-1];
         return p0.x;
    } else {
         const Point &p0 = points[i-1], &p1 = points[i];
         REAL s = (t - p0.t)/(p1.t - p0.t), s2 = s*s, s3 = s2*s;
         return p0.x*(2*s3 - 3*s2 + 1) + p1.x*(-2*s3 + 3*s2)
             + (p0.v*(s3 - 2*s2 + s) + p1.v*(s3 - s2))*(p1.t - p0.t);
    }
}

template <typename T>
T Spline<T>::vel (REAL t) const {
    int i = find(*this, t);
    if (i == 0 || i == points.size()) {
        return T(0);
    } else {
        const Point &p0 = points[i-1], &p1 = points[i];
        REAL s = (t - p0.t)/(p1.t - p0.t), s2 = s*s;
        return (p0.x*(6*s2 - 6*s) + p1.x*(-6*s2 + 6*s))/(p1.t - p0.t)
            + p0.v*(3*s2 - 4*s + 1) + p1.v*(3*s2 - 2*s);
    }
}

vector<REAL> operator+ (const vector<REAL> &x, const vector<REAL> &y) {
    vector<REAL> z(min(x.size(), y.size()));
    for (int i = 0; i < z.size(); i++) z[i] = x[i] + y[i];
    return z;
}
vector<REAL> operator- (const vector<REAL> &x, const vector<REAL> &y) {
    vector<REAL> z(min(x.size(), y.size()));
    for (int i = 0; i < z.size(); i++) z[i] = x[i] - y[i];
    return z;
}
vector<REAL> operator* (const vector<REAL> &x, REAL a) {
    vector<REAL> y(x.size());
    for (int i = 0; i < y.size(); i++) y[i] = x[i]*a;
    return y;
}
vector<REAL> operator/ (const vector<REAL> &x, REAL a) {return x*(1/a);}

template class Spline<Vec3>;
template class Spline<Transformation>;
template class Spline<REAL>;
template class Spline< vector<REAL> >;
