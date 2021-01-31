
#include "geometry.hpp"
#include <cstdlib>

using namespace std;

REAL unwrap_angle(REAL theta, REAL theta_ref) {
	if (theta - theta_ref > M_PI)
		theta -= 2 * M_PI;
	if (theta - theta_ref < -M_PI)
		theta += 2 * M_PI;
	return theta;
}


template <> const Vec3 &pos<PS>(const Node *node) { return node->y; }
template <> const Vec3 &pos<WS>(const Node *node) { return node->x; }
template <> Vec3 &pos<PS>(Node *node) { return node->y; }
template <> Vec3 &pos<WS>(Node *node) { return node->x; }

template <Space s> Vec3 nor(const Face *face) {
	const Vec3 &x0 = pos<s>(face->v[0]->node),
		&x1 = pos<s>(face->v[1]->node),
		&x2 = pos<s>(face->v[2]->node);
	return normalize(cross(x1 - x0, x2 - x0));
}
template Vec3 nor<PS>(const Face *face);
template Vec3 nor<WS>(const Face *face);

template <Space s> REAL dihedral_angle(const Edge *edge) {
	// if (!hinge.edge[0] || !hinge.edge[1]) return 0;
	// const Edge *edge0 = hinge.edge[0], *edge1 = hinge.edge[1];
	// int s0 = hinge.s[0], s1 = hinge.s[1];
	if (!edge->adjf[0] || !edge->adjf[1])
		return 0;
	Vec3 e = normalize(pos<s>(edge->n[0]) - pos<s>(edge->n[1]));
	if (norm2(e) == 0) return 0;
	Vec3 n0 = nor<s>(edge->adjf[0]), n1 = nor<s>(edge->adjf[1]);
	if (norm2(n0) == 0 || norm2(n1) == 0) return 0;
	REAL cosine = dot(n0, n1), sine = dot(e, cross(n0, n1));
	REAL theta = atan2(sine, cosine);
	return unwrap_angle(theta, edge->reference_angle);
}
template REAL dihedral_angle<PS>(const Edge *edge);
template REAL dihedral_angle<WS>(const Edge *edge);
