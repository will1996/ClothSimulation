
#ifndef AUGLAG_HPP
#define AUGLAG_HPP

#include "sparse.hpp"
#include "vectors.hpp"

struct NLOpt { // nonlinear optimization problem
    // minimize objective s.t. constraints = or <= 0
    int nvar, ncon;
    virtual void initialize (double *x) const = 0;
    virtual void precompute (const double *x) const {}
    virtual double objective (const double *x) const = 0;
    virtual void obj_grad (const double *x, double *grad) const = 0; // set
    virtual double constraint (const double *x, int j, int &sign) const = 0;
    virtual void con_grad (const double *x, int j, double factor,
                           double *grad) const = 0; // add factor*gradient
    virtual void finalize (const double *x) const = 0;
};

void augmented_lagrangian_method (const NLOpt &problem, bool verbose=false);

// convenience functions for when optimization variables are Vec3-valued

inline Vec3 get_subvec (const double *x, int i) {
    return Vec3(x[i*3+0], x[i*3+1], x[i*3+2]);}
inline void set_subvec (double *x, int i, const Vec3 &xi) {
    for (int j = 0; j < 3; j++) x[i*3+j] = xi[j];}
inline void add_subvec (double *x, int i, const Vec3 &xi) {
    for (int j = 0; j < 3; j++) x[i*3+j] += xi[j];}

#endif
