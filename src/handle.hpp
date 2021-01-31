#pragma once

#include "constraint.hpp"
#include "cloth.hpp"
#include "mesh.hpp"
#include <assert.h>
#include <vector>
using namespace std;

struct Handle {
	REAL start_time, end_time, fade_time;
	virtual ~Handle() {};
	virtual std::vector<Constraint*> get_constraints(REAL t) = 0;
	virtual std::vector<Node*> get_nodes() = 0;
	bool active(REAL t) { return t >= start_time && t <= end_time; }
	REAL strength(REAL t) {
		if (t < start_time || t > end_time + fade_time) return 0;
		if (t <= end_time) return 1;
		REAL s = 1 - (t - end_time) / (fade_time + 1e-6);
		return sq(sq(s));
	}

	virtual void backup(const vector<Cloth>&) = 0;
	virtual void resume(vector<Mesh *>&) = 0;
};

struct NodeHandle : public Handle {
	Node *node;
	const Motion *motion;
	bool activated;
	Vec3 x0;
	NodeHandle() : activated(false) {}
	std::vector<Constraint*> get_constraints(REAL t);
	std::vector<Node*> get_nodes() { return std::vector<Node*>(1, node); }

	int index;
	int mid;
	void backup(const vector<Cloth>&ms) {
		index = node->index;

		mid = -1;
		for (int i = 0; i < ms.size(); i++) {
			const Mesh *m = &ms[i].mesh;
			for (int j = 0; j < m->nodes.size(); j++)
				if (node == m->nodes[j]) {
					mid = i;
					break;
				}
		}
		assert(mid != -1);
	}
	void resume(vector<Mesh *>&ms) {
		node = ms[mid]->find_node(index);
	}
};

struct CircleHandle : public Handle {
	Mesh *mesh;
	int label;
	const Motion *motion;
	REAL c; // circumference
	Vec2 u;
	Vec3 xc, dx0, dx1;
	std::vector<Constraint*> get_constraints(REAL t);
	std::vector<Node*> get_nodes() { return std::vector<Node*>(); }

	void backup(const vector<Cloth>&ms) {}
	void resume(vector<Mesh *>&ms) {
		//mesh = m;
	}
};

struct GlueHandle : public Handle {
	Node* nodes[2];
	std::vector<Constraint*> get_constraints(REAL t);
	std::vector<Node*> get_nodes() {
		std::vector<Node*> ns;
		ns.push_back(nodes[0]);
		ns.push_back(nodes[1]);
		return ns;
	}

	int indices[2];
	int mid[2];
	void backup(const vector<Cloth>&ms) {
		indices[0] = nodes[0]->index;
		indices[1] = nodes[1]->index;

		mid[0] = -1;
		for (int i = 0; i < ms.size(); i++) {
			const Mesh *m = &ms[i].mesh;
			for (int j = 0; j < m->nodes.size(); j++) {
				if (nodes[0] == m->nodes[j]) {
					mid[0] = i;
					break;
				}
			}
		}
		assert(mid[0] != -1);

		mid[1] = -1;
		for (int i = 0; i < ms.size(); i++) {
			const Mesh *m = &ms[i].mesh;
			for (int j = 0; j < m->nodes.size(); j++) {
				if (nodes[1] == m->nodes[j]) {
					mid[1] = i;
					break;
				}
			}
		}
		assert(mid[1] != -1);
	}

	void resume(vector<Mesh *>&ms) {
		nodes[0] = ms[mid[0]]->find_node(indices[0]);
		nodes[1] = ms[mid[1]]->find_node(indices[1]);
	}
};

struct AttachHandle : public Handle {
	int id1, id2, cid, oid;
	bool init;
	Vec3 offset;

	std::vector<Node*> get_nodes() { return std::vector<Node*>(); }

	std::vector<Constraint*> get_constraints(REAL t);
	void backup(const vector<Cloth>&ms) {}
	void resume(vector<Mesh *>&ms) {
		//mesh = m;
	}
};

