
#include "handle.hpp"
#include "magic.hpp"
using namespace std;

static Vec3 directions[3] = {Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1)};

void add_position_constraints (const Node *node, const Vec3 &x, REAL stiff,
                               vector<Constraint*> &cons);

Transformation normalize (const Transformation &T) {
    Transformation T1 = T;
    T1.rotation = normalize(T1.rotation);
    return T1;
}

vector<Constraint*> NodeHandle::get_constraints (REAL t) {
    REAL s = strength(t);
    if (!s)
        return vector<Constraint*>();
    if (!activated) {
        // handle just got started, fill in its original position
        x0 = motion ? inverse(normalize(motion->pos(t))).apply(node->x) : node->x;
        activated = true;
    }
    Vec3 x = motion ? normalize(motion->pos(t)).apply(x0) : x0;
    vector<Constraint*> cons;
    add_position_constraints(node, x, s*::magic.handle_stiffness, cons);
    return cons;
}

#include "simulation.hpp"

extern Simulation *glSim;

vector<Constraint*> AttachHandle::get_constraints(REAL t) {
	REAL s = strength(t);
	if (!s)
		return vector<Constraint*>();

	Node *no = glSim->obstacle_meshes[oid]->nodes[id2];
	Node *nc = glSim->cloth_meshes[cid]->nodes[id1];

	if (!init) {
		offset = nc->x0 - no->x0;
		init = true;
	}

	Vec3 now = no->x0 + no->v * glSim->step_time;
	now += offset;

	vector<Constraint*> cons;
	add_position_constraints(nc, now, s*::magic.handle_stiffness, cons);
	return cons;
}

vector<Constraint*> CircleHandle::get_constraints (REAL t) {
    REAL s = strength(t);
    if (!s)
        return vector<Constraint*>();
    vector<Constraint*> cons;
    for (int n = 0; n < mesh->nodes.size(); n++) {
        Node *node = mesh->nodes[n];
        if (node->label != label)
            continue;
        REAL theta = 2*M_PI*dot(node->verts[0]->u, u)/c;
        Vec3 x = xc + (dx0*cos(theta) + dx1*sin(theta))*c/REAL(2*M_PI);
        if (motion)
            x = motion->pos(t).apply(x);
        REAL l = 0;
        for (int e = 0; e < node->adje.size(); e++) {
            const Edge *edge = node->adje[e];
            if (edge->n[0]->label != label || edge->n[1]->label != label)
                continue;
            l += edge->l;
        }
        add_position_constraints(node, x, s*::magic.handle_stiffness*l, cons);
    }
    return cons;
}

inline Vec3 Interp(Vec3 &x0, Vec3 &x1, REAL w)
{
	return x0*(1 - w) + x1*w;
}

vector<Constraint*> GlueHandle::get_constraints (REAL t) {
	// will be proceeded seperately ...
	return vector<Constraint*>();

    REAL s = strength(t);
    if (!s)
        return vector<Constraint*>();

    vector<Constraint*> cons;
	Vec3 x0 = nodes[0]->x0 + nodes[0]->v * glSim->step_time;
	Vec3 x1 = nodes[1]->x0 + nodes[1]->v * glSim->step_time;
	Vec3 x = (x0 + x1)*REAL(0.5);

	for (int i = 0; i < 3; i++) {
		EqCon *con = new EqCon;
		con->node = nodes[0];
		con->x = x; // Interp(x0, x, 0.95);
		con->n = directions[i];
		con->stiff = ::magic.handle_stiffness;
		cons.push_back(con);
	}
	for (int i = 0; i < 3; i++) {
		EqCon *con = new EqCon;
		con->node = nodes[1];
		con->x = x; // Interp(x1, x, 0.95);
		con->n = directions[i];
		con->stiff = ::magic.handle_stiffness;
		cons.push_back(con);
	}
	return cons;
}

void add_position_constraints (const Node *node, const Vec3 &x, REAL stiff,
                               vector<Constraint*> &cons) {
    for (int i = 0; i < 3; i++) {
        EqCon *con = new EqCon;
        con->node = (Node*)node;
        con->x = x;
        con->n = directions[i];
        con->stiff = stiff;
        cons.push_back(con);
    }
}
