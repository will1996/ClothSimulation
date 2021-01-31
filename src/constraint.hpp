#pragma once

#include "mesh.hpp"
//#include "sparse.hpp"
#include "spline.hpp"
#include "util.hpp"
#include "vectors.hpp"
#include <map>
#include <vector>

typedef std::map<Node*, Vec3> MeshGrad;
typedef std::map<std::pair<Node*, Node*>, Mat3x3> MeshHess;

struct Constraint {
	virtual ~Constraint() {};
	virtual REAL value(int *sign = NULL) = 0;
	virtual MeshGrad gradient() = 0;
	virtual MeshGrad project() = 0;
	// energy function
	virtual REAL energy(REAL value) = 0;
	virtual REAL energy_grad(REAL value) = 0;
	virtual REAL energy_hess(REAL value) = 0;
	// frictional force
	virtual MeshGrad friction(REAL dt, MeshHess &jac) = 0;

	//re-enforce
	virtual void apply() {}
};

struct EqCon : public Constraint {
	// n . (node->x - x) = 0
	Node *node;
	Vec3 x, n;
	REAL stiff;
	REAL value(int *sign = NULL);
	MeshGrad gradient();
	MeshGrad project();
	REAL energy(REAL value);
	REAL energy_grad(REAL value);
	REAL energy_hess(REAL value);
	MeshGrad friction(REAL dt, MeshHess &jac);
	void apply() { node->x = x; }
};

struct GlueCon : public Constraint {
	Node *nodes[2];
	Vec3 n;
	REAL stiff;
	REAL value(int *sign = NULL);
	MeshGrad gradient();
	MeshGrad project();
	REAL energy(REAL value);
	REAL energy_grad(REAL value);
	REAL energy_hess(REAL value);
	MeshGrad friction(REAL dt, MeshHess &jac);
};

struct IneqCon : public Constraint {
	// n . sum(w[i] verts[i]->x) >= 0
	Node *nodes[4];
	REAL w[4];
	bool free[4];
	Vec3 n;
	REAL a; // area
	REAL mu; // friction
	REAL stiff;
	REAL value(int *sign = NULL);
	MeshGrad gradient();
	MeshGrad project();
	REAL energy(REAL value);
	REAL energy_grad(REAL value);
	REAL energy_hess(REAL value);
	MeshGrad friction(REAL dt, MeshHess &jac);
};
