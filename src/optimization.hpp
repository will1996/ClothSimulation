
#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

#include "sparse.hpp"
#include "vectors.hpp"

// Problems

struct NLOpt { // nonlinear optimization problem
    // minimize objective
    int nvar;
    virtual void initialize (REAL *x) const = 0;
    virtual REAL objective (const REAL *x) const = 0;
    virtual void precompute (const REAL *x) const {}
    virtual void gradient (const REAL *x, REAL *g) const = 0;
    virtual bool hessian (const REAL *x, SpMat<REAL> &H) const {
        return false; // should return true if implemented
    };
    virtual void finalize (const REAL *x) const = 0;
};

struct NLConOpt { // nonlinear constrained optimization problem
    // minimize objective s.t. constraints = or <= 0
    int nvar, ncon;
    virtual void initialize (double *x) const = 0;
	virtual void precompute(const double *x) const {}
	virtual REAL objective(const double *x) const = 0;
	virtual void obj_grad(const double *x, double *grad) const = 0; // set
	virtual REAL constraint(const double *x, int j, int &sign) const = 0;
	virtual void con_grad(const double *x, int j, double factor,
		double *grad) const = 0; // add factor*gradient
    virtual void finalize (const double *x) const = 0;
};

// Algorithms

struct OptOptions {
    int _max_iter;
    REAL _eps_x, _eps_f, _eps_g;
    OptOptions (): _max_iter(100), _eps_x(1e-6), _eps_f(1e-12), _eps_g(1e-6) {}
    // Named parameter idiom
    // http://www.parashift.com/c++-faq-lite/named-parameter-idiom.html
    OptOptions &max_iter (int n) {_max_iter = n; return *this;}
    OptOptions &eps_x (REAL e) {_eps_x = e; return *this;}
    OptOptions &eps_f (REAL e) {_eps_f = e; return *this;}
    OptOptions &eps_g (REAL e) {_eps_g = e; return *this;}
    int max_iter () {return _max_iter;}
    REAL eps_x () {return _eps_x;}
    REAL eps_f () {return _eps_f;}
    REAL eps_g () {return _eps_g;}
};

void l_bfgs_method (const NLOpt &problem,
                    OptOptions opts=OptOptions(),
                    bool verbose=false);

void line_search_newtons_method (const NLOpt &problem,
                                 OptOptions opts=OptOptions(),
                                 bool verbose=false);

void nonlinear_conjugate_gradient_method (const NLOpt &problem,
                                          OptOptions opts=OptOptions(),
                                          bool verbose=false);

void trust_region_method (const NLOpt &problem,
                          OptOptions opts=OptOptions(),
                          bool verbose=false);

void augmented_lagrangian_method (const NLConOpt &problem,
                                  OptOptions opts=OptOptions(),
                                  bool verbose=false);

// convenience functions for when optimization variables are Vec3-valued

inline Vec3 get_subvec (const double *x, int i) {
    return Vec3(x[i*3+0], x[i*3+1], x[i*3+2]);}
inline void set_subvec(double *x, int i, const Vec3 &xi) {
    for (int j = 0; j < 3; j++) x[i*3+j] = xi[j];}
inline void add_subvec(double *x, int i, const Vec3 &xi) {
    for (int j = 0; j < 3; j++) x[i*3+j] += xi[j];}

template <int n> Vec<n> get_subvec(const double *x, int i) {
    Vec<n> v; for (int j = 0; j < n; j++) v[j] = x[i*n+j]; return v;}
template <int n> void set_subvec (double *x, int i, const Vec<n> &xi) {
    for (int j = 0; j < n; j++) x[i*n+j] = xi[j];}
template <int n> void add_subvec(double *x, int i, const Vec<n> &xi) {
    for (int j = 0; j < n; j++) x[i*n+j] += xi[j];}

#endif
