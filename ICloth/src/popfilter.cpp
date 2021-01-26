
#include "popfilter.hpp"

#include "magic.hpp"
#include "optimization.hpp"
#include "physics.hpp"
#include <utility>
using namespace std;

// "rubber band" stiffness to stop vertices moving too far from initial position
static REAL mu;

struct PopOpt: public NLOpt {
    // we want F(x) = m a0, but F(x) = -grad E(x)
    // so grad E(x) + m a0 = 0
    // let's minimize E(x) + m a0 . (x - x0)
    // add spring to x0: minimize E(x) + m a0 . (x - x0) + mu (x - x0)^2/2
    // gradient: -F(x) + m a0 + mu (x - x0)
    // hessian: -J(x) + mu
    Cloth &cloth;
    Mesh &mesh;
    const vector<Constraint*> &cons;
    vector<Vec3> x0, a0;
    mutable vector<Vec3> f;
    mutable SpMat<Mat3x3> J;
    PopOpt (Cloth &cloth, const vector<Constraint*> &cons):
        cloth(cloth), mesh(cloth.mesh), cons(cons) {
        int nn = mesh.nodes.size();
        nvar = nn*3;
        x0.resize(nn);
        a0.resize(nn);
        for (int n = 0; n < nn; n++) {
            const Node *node = mesh.nodes[n];
            x0[n] = node->x;
            a0[n] = node->acceleration;
        }
        f.resize(nn);
        J = SpMat<Mat3x3>(nn,nn);
    }
    virtual void initialize (REAL *x) const;
    virtual void precompute (const REAL *x) const;
    virtual REAL objective (const REAL *x) const;
    virtual void gradient (const REAL *x, REAL *g) const;
    virtual bool hessian (const REAL *x, SpMat<REAL> &H) const;
    virtual void finalize (const REAL *x) const;
};

void subtract_rigid_acceleration (const Mesh &mesh);

void apply_pop_filter (Cloth &cloth, const vector<Constraint*> &cons,
                       REAL regularization) {
    ::mu = regularization;
    // subtract_rigid_acceleration(cloth.mesh);
    // trust_region_method(PopOpt(cloth, cons), true);
    line_search_newtons_method(PopOpt(cloth, cons), OptOptions().max_iter(10));
    compute_ws_data(cloth.mesh);
}


inline void _set_subvec(REAL *x, int i, const Vec3 &xi)
{
	for (int j = 0; j < 3; j++) x[i * 3 + j] = xi[j];
}

inline Vec3 _get_subvec(const REAL*x, int i) {
	return Vec3(x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2]);
}

void PopOpt::initialize (REAL *x) const {
    for (int n = 0; n < mesh.nodes.size(); n++)
        _set_subvec(x, n, Vec3(0));
}

void PopOpt::precompute (const REAL *x) const {
    for (int n = 0; n < mesh.nodes.size(); n++) {
        mesh.nodes[n]->x = x0[n] + _get_subvec(x, n);
        f[n] = Vec3(0);
        for (int jj = 0; jj < J.rows[n].entries.size(); jj++)
            J.rows[n].entries[jj] = Mat3x3(0);
    }
    add_internal_forces<WS>(cloth, J, f, 0);
    add_constraint_forces(cloth, cons, J, f, 0);
}

REAL PopOpt::objective (const REAL *x) const {
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->x = x0[n] + _get_subvec(x, n);
    REAL e = internal_energy<WS>(cloth);
    e += constraint_energy(cons);
    for (int n = 0; n < mesh.nodes.size(); n++) {
        const Node *node = mesh.nodes[n];
        e += node->m*dot(node->acceleration, node->x - x0[n]);
        e += ::mu*norm2(node->x - x0[n])/2.;
    }
    return e;
}

void PopOpt::gradient (const REAL *x, REAL *g) const {
    for (int n = 0; n < mesh.nodes.size(); n++) {
        const Node *node = mesh.nodes[n];
        _set_subvec(g, n, -f[n] + node->m*a0[n]
                         + ::mu*(node->x - x0[n]));
    }
}

static Mat3x3 get_submat (SpMat<REAL> &A, int i, int j) {
    Mat3x3 Aij;
    for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++)
        Aij(ii,jj) = A(i*3+ii, j*3+jj);
    return Aij;
}
static void set_submat (SpMat<REAL> &A, int i, int j, const Mat3x3 &Aij) {
    for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++)
        A(i*3+ii, j*3+jj) = Aij(ii,jj);
}
static void add_submat (SpMat<REAL> &A, int i, int j, const Mat3x3 &Aij) {
    for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++)
        A(i*3+ii, j*3+jj) += Aij(ii,jj);
}

bool PopOpt::hessian (const REAL *x, SpMat<REAL> &H) const {
    for (int i = 0; i < mesh.nodes.size(); i++) {
        const SpVec<Mat3x3> &Ji = J.rows[i];
        for (int jj = 0; jj < Ji.indices.size(); jj++) {
            int j = Ji.indices[jj];
            const Mat3x3 &Jij = Ji.entries[jj];
            set_submat(H, i, j, Jij);
        }
        add_submat(H, i, i, Mat3x3(::mu));
    }
    return true;
}

void PopOpt::finalize (const REAL *x) const {
    for (int n = 0; n < mesh.nodes.size(); n++)
        mesh.nodes[n]->x = x0[n] + _get_subvec(x, n);
}
