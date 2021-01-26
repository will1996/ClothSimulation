
#include "constraint.hpp"

#include "magic.hpp"
using namespace std;

REAL EqCon::value (int *sign) {
    if (sign) *sign = 0;
    return dot(n, node->x - x);
}
MeshGrad EqCon::gradient () {MeshGrad grad; grad[node] = n; return grad;}
MeshGrad EqCon::project () {return MeshGrad();}
REAL EqCon::energy (REAL value) {return stiff*sq(value)/2.;}
REAL EqCon::energy_grad (REAL value) {return stiff*value;}
REAL EqCon::energy_hess (REAL value) {return stiff;}
MeshGrad EqCon::friction (REAL dt, MeshHess &jac) {return MeshGrad();}

REAL GlueCon::value (int *sign) {
    if (sign) *sign = 0;
    return dot(n, nodes[1]->x - nodes[0]->x);
}
MeshGrad GlueCon::gradient () {
    MeshGrad grad;
    grad[nodes[0]] = -n;
    grad[nodes[1]] = n;
    return grad;
}
MeshGrad GlueCon::project () {return MeshGrad();}
REAL GlueCon::energy (REAL value) {return stiff*sq(value)/2.;}
REAL GlueCon::energy_grad (REAL value) {return stiff*value;}
REAL GlueCon::energy_hess (REAL value) {return stiff;}
MeshGrad GlueCon::friction (REAL dt, MeshHess &jac) {return MeshGrad();}

REAL IneqCon::value (int *sign) {
    if (sign)
        *sign = 1;
    REAL d = 0;
    for (int i = 0; i < 4; i++)
        d += w[i]*dot(n, nodes[i]->x);
    d -= ::magic.repulsion_thickness;
    return d;
}

MeshGrad IneqCon::gradient () {
    MeshGrad grad;
    for (int i = 0; i < 4; i++)
        grad[nodes[i]] = w[i]*n;
    return grad;
}

MeshGrad IneqCon::project () {
    REAL d = value() + ::magic.repulsion_thickness - ::magic.projection_thickness;
    if (d >= 0)
        return MeshGrad();
    REAL inv_mass = 0;
    for (int i = 0; i < 4; i++)
        if (free[i])
            inv_mass += sq(w[i])/nodes[i]->m;
    MeshGrad dx;
    for (int i = 0; i < 4; i++)
        if (free[i])
            dx[nodes[i]] = -(w[i]/nodes[i]->m)/inv_mass*n*d;
    return dx;
}

REAL violation(REAL value) { return std::max(-value, REAL(0.)); }

REAL IneqCon::energy (REAL value) {
    REAL v = violation(value);
    return stiff*v*v*v/::magic.repulsion_thickness/6;
}
REAL IneqCon::energy_grad (REAL value) {
    return -stiff*sq(violation(value))/::magic.repulsion_thickness/2;
}
REAL IneqCon::energy_hess (REAL value) {
    return stiff*violation(value)/::magic.repulsion_thickness;
}

MeshGrad IneqCon::friction (REAL dt, MeshHess &jac) {
    if (mu == 0)
        return MeshGrad();
    REAL fn = abs(energy_grad(value()));
    if (fn == 0)
        return MeshGrad();
    Vec3 v = Vec3(0);
    REAL inv_mass = 0;
    for (int i = 0; i < 4; i++) {
        v += w[i]*nodes[i]->v;
        if (free[i])
            inv_mass += sq(w[i])/nodes[i]->m;
    }
    Mat3x3 T = Mat3x3(1) - outer(n,n);
    REAL vt = norm(T*v);
    REAL f_by_v = fmin(mu*fn/vt, 1/(dt*inv_mass));
    // REAL f_by_v = mu*fn/max(vt, 1e-1);
    MeshGrad force;
    for (int i = 0; i < 4; i++) {
        if (free[i]) {
            force[nodes[i]] = -w[i]*f_by_v*T*v;
            for (int j = 0; j < 4; j++) {
                if (free[j]) {
                    jac[make_pair(nodes[i],nodes[j])] = -w[i]*w[j]*f_by_v*T;
                }
            }
        }
    }
    return force;
}
