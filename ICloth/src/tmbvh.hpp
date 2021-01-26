#pragma once

#include "mesh.hpp"
#include "util.hpp"
#include <float.h>
#include <stdlib.h>

#pragma once

typedef Vec3 vec3f;
typedef Vec<3, int> vec3i;

FORCEINLINE void
vmin(vec3f &a, const vec3f &b)
{
	a = vec3f(
		fmin(a[0], b[0]),
		fmin(a[1], b[1]),
		fmin(a[2], b[2]));
}

FORCEINLINE void
vmax(vec3f &a, const vec3f &b)
{
	a = vec3f(
		fmax(a[0], b[0]),
		fmax(a[1], b[1]),
		fmax(a[2], b[2]));
}


class aabb {
	FORCEINLINE void init() {
		_max = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		_min = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
	}

public:
	vec3f _min;
	vec3f _max;

	FORCEINLINE aabb() {
		init();
	}

	FORCEINLINE aabb(const vec3f &v) {
		_min = _max = v;
	}

	FORCEINLINE aabb(const vec3f &a, const vec3f &b) {
		_min = a;
		_max = a;
		vmin(_min, b);
		vmax(_max, b);
	}

	FORCEINLINE bool overlaps(const aabb& b) const
	{
		if (_min[0] > b._max[0]) return false;
		if (_min[1] > b._max[1]) return false;
		if (_min[2] > b._max[2]) return false;

		if (_max[0] < b._min[0]) return false;
		if (_max[1] < b._min[1]) return false;
		if (_max[2] < b._min[2]) return false;

		return true;
	}

	FORCEINLINE bool overlaps(const aabb &b, aabb &ret) const
	{
		if (!overlaps(b))
			return false;

		ret._min = vec3f(
			fmax(_min[0],  b._min[0]),
			fmax(_min[1],  b._min[1]),
			fmax(_min[2],  b._min[2]));

		ret._max = vec3f(
			fmin(_max[0], b._max[0]),
			fmin(_max[1], b._max[1]),
			fmin(_max[2], b._max[2]));

		return true;
	}

	FORCEINLINE bool inside(const vec3f &p) const
	{
		if (p[0] < _min[0] || p[0] > _max[0]) return false;
		if (p[1] < _min[1] || p[1] > _max[1]) return false;
		if (p[2] < _min[2] || p[2] > _max[2]) return false;

		return true;
	}

	FORCEINLINE aabb &operator += (const vec3f &p)
	{
		vmin(_min, p);
		vmax(_max, p);
		return *this;
	}

	FORCEINLINE aabb &operator += (const aabb &b)
	{
		vmin(_min, b._min);
		vmax(_max, b._max);
		return *this;
	}

	FORCEINLINE aabb operator + ( const aabb &v) const
	{ aabb rt(*this); return rt += v; }

	FORCEINLINE REAL width()  const { return _max[0] - _min[0]; }
	FORCEINLINE REAL height() const { return _max[1] - _min[1]; }
	FORCEINLINE REAL depth()  const { return _max[2] - _min[2]; }
	FORCEINLINE vec3f center() const { return (_min+_max)*REAL(0.5); }
	FORCEINLINE REAL volume() const { return width()*height()*depth(); }


	FORCEINLINE bool empty() const {
		return _max[0] < _min[0];
	}

	FORCEINLINE void enlarge(REAL thickness) {
		_max += vec3f(thickness, thickness, thickness);
		_min -= vec3f(thickness, thickness, thickness);
	}

	vec3f getMax() { return _max; }
	vec3f getMin() { return _min; }

	void print(FILE *fp) {
		//fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	}

	void visualize();
};

#define MAX(a,b)	((a) > (b) ? (a) : (b))
#define MIN(a,b)	((a) < (b) ? (a) : (b))
#define BOX aabb

class bvh;
class bvh_node;
class front_list;

class front_node {
public:
	bvh_node *_left, *_right;
	unsigned int _flag; // vailid or not
	unsigned int _ptr; // self-coliding parent;

	FORCEINLINE front_node(bvh_node *l, bvh_node *r, unsigned int ptr)
	{
		_left = l, _right = r, _flag = 0;
		_ptr = ptr;
	}

	void update (front_list &appended) {
#ifdef XXXXXXXX
		if (_flag != 0)
			return;

		if (_left->isLeaf() && _right->isLeaf()) {
			collide_leaves(_left, _right);
			return;
		}

		if (!_left->box().overlaps(_right->box()))
			return;

		// need to be spouted
		_flag = 1; // set to be invalid

		if (_left->isLeaf()) {
			_left->sprouting(_right->left(), appended);
			_left->sprouting(_right->right(), appended);
		} else {
			_left->left()->sprouting(_right, appended);
			_left->right()->sprouting(_right, appended);
		}
#endif
	}
};

#include <vector>
using namespace std;

#include "contour.h"


bool covertex(int tri1, int tri2);
void self_mesh(vector<Mesh *> &);

class front_list : public vector<front_node> {
public:
	//void propogate();
	void push2GPU(bvh_node *r1, bvh_node *r2 = NULL);
};

class bvh_node {
	BOX _box;
	int _child; // >=0 leaf with tri_id, <0 left & right
	int _parent;
	contour _bound;

	void setParent(int p) { _parent = p; }

public:
	int get_bound_length()
	{
		if (isLeaf()) return 0;
		if (!_bound.single_ring()) return 0;
		return _bound.ring_length(0);
	}

	int get_bound_idx(int r, int i){
		return _bound.ring_index(r, i);
	}

public:
	bvh_node() {
		_child = 0;
		_parent = 0;
	}

	~bvh_node() {
		NULL;
	}

	void collide(bvh_node *other, front_list &f, int level, int ptr)
	{
		if (isLeaf() && other->isLeaf()) {
			if (!covertex(this->triID(), other->triID()) )
				f.push_back(front_node(this, other, ptr));

			return;
		}

		if (!_box.overlaps(other->box()) || level > 100) {
			f.push_back(front_node(this, other, ptr));
			return;
		}

		if (isLeaf()) {
			collide(other->left(), f, level++, ptr);
			collide(other->right(), f, level++, ptr);
		} else {
			left()->collide(other, f, level++, ptr);
			right()->collide(other, f, level++, ptr);
		}
	}

	void self_collide(front_list &lst, bvh_node *r) {
		if (isLeaf())
			return;

		left()->self_collide(lst, r);
		right()->self_collide(lst, r);
		left()->collide(right(), lst, 0, this-r);
	}

	void construct(unsigned int id);
	void construct(unsigned int *lst, unsigned int num);

//	void collide(bvh_node *, front_list &, int level=0);
//	void self_collide(front_list &);
//	void sprouting(bvh_node *, front_list &, mesh *, mesh *);
	void visualize(int level);
	void visualizeBound(bool);
	void refit(bool bound);
	void resetParents(bvh_node *root);

	FORCEINLINE BOX &box() { return _box; }
	FORCEINLINE bvh_node *left() { return this - _child; }
	FORCEINLINE bvh_node *right() { return this - _child + 1; }
	FORCEINLINE int triID() { return _child; }
	FORCEINLINE int isLeaf() { return _child >= 0; }
	FORCEINLINE int parentID() { return _parent; }

	FORCEINLINE void getLevel(int current, int &max_level) {
		if (current > max_level)
			max_level = current;

		if (isLeaf()) return;
		left()->getLevel(current+1, max_level);
		right()->getLevel(current+1, max_level);
	}

	FORCEINLINE void getLevelIdx(int current, unsigned int *idx) {
		idx[current]++;

		if (isLeaf()) return;
		left()->getLevelIdx(current+1, idx);
		right()->getLevelIdx(current+1, idx);
	}

	friend class bvh;
};

struct Mesh;

class bvh {
	int _num; // all face num
	bvh_node *_nodes;

	void construct(std::vector<Mesh*> &, const char *, bool cloth);
	void construct(std::vector<Mesh*> &, bool cloth);
	void refit(bool bound = false);
	void reorder(); // for breath-first refit
	void resetParents();

public:
	bvh(std::vector<Mesh*> &ms, bool cloth);

	~bvh() {
		if (_nodes)
			delete [] _nodes;
	}
	
	bvh_node *root() { return _nodes; }

	void push2GPU(bool);

	void collide(bvh *other, front_list &f) {
		f.clear();

		vector<Mesh *> c;
		self_mesh(c);
		
		if (other)
		root()->collide(other->root(), f, 0, -1);
	}

	void self_collide(front_list &f, vector<Mesh *> &c) {
		f.clear();

		self_mesh(c);
		root()->self_collide(f, root());
	}
};