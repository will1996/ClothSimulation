#if defined(_WIN32)
#include <Windows.h>
#endif

#include <GL/gl.h>

#include <stdlib.h>
#include <assert.h>

#include "tmbvh.hpp"
#include "mesh.hpp"
#include <climits>
#include <utility>
#include <cstdarg>

using namespace std;

//#include "magic.hpp"

extern string outprefix;

static vector<Mesh *>  *ptCloth;

void self_mesh(vector<Mesh *> &meshes)
{
	ptCloth = &meshes;
}


class aap {
public:
	char _xyz;
	float _p;

	FORCEINLINE aap(const BOX &total) {
		vec3f center = total.center();
		char xyz = 2;

		if (total.width() >= total.height() && total.width() >= total.depth()) {
			xyz = 0;
		} else
			if (total.height() >= total.width() && total.height() >= total.depth()) {
				xyz = 1;
			}

			_xyz = xyz;
			_p = center[xyz];
	}

	FORCEINLINE bool inside(const vec3f &mid) const {
		return mid[_xyz]>_p;
	}
};

bvh::bvh(std::vector<Mesh*> &ms, bool cloth)
{
	_num = 0;
	_nodes = NULL;

	if (cloth) {
		string fname = outprefix + "/bvh.txt";
		FILE *fp = fopen(fname.c_str(), "rt");
		if (fp != NULL) {
			fclose(fp);
			construct(ms, fname.c_str(), cloth);
		}
		else
			construct(ms, cloth);

		/*
		int which = ::magic.tm_load_bvh;

		switch (which) {
		case 1:
			construct(ms, "c:\\temp\\br3-bvh-200k.txt", cloth);
			break;

		case 2:
			construct(ms, "c:\\temp\\andy-bvh-2.txt", cloth);
			break;

		case 3:
			construct(ms, "c:\\temp\\andy-bvh.txt", cloth);
			break;

		case 4:
			construct(ms, "c:\\temp\\qman-bvh.txt", cloth);
			break;

		case 5:
			construct(ms, "c:\\temp\\br-3l-16k-bvh.txt", cloth);
			break;

		case 6:
			construct(ms, "c:\\temp\\bishop-bvh.txt", cloth);
			break;

		case 7:
			construct(ms, "c:\\temp\\qman-21k-bvh.txt", cloth);
			break;

		default:
			construct(ms, cloth);
		}
		*/
	}
	else
		construct(ms, cloth);

	reorder();
	resetParents(); //update the parents after reorder ...
}

static vec3f *s_fcenters;
static BOX *s_fboxes;
static unsigned int *s_idx_buffer;
static bvh_node *s_current;
static vec3i *s_nidx;

void bvh::construct(std::vector<Mesh*> &ms, const char *fname, bool cloth)
{
	_num = 0;
	BOX total;

	for (int i = 0; i < ms.size(); i++) {
		_num += ms[i]->faces.size();
		for (int j = 0; j < ms[i]->verts.size(); j++) {
			if (ms[i]->verts[j]->node == NULL)
				continue;

			total += ms[i]->verts[j]->node->x;
		}
	}

	s_fboxes = new BOX[_num];
	s_nidx = new vec3i[_num];

	int tri_idx = 0;
	int vtx_offset = 0;

	for (int i = 0; i < ms.size(); i++) {
		for (int j = 0; j < ms[i]->faces.size(); j++) {
			Face *f = ms[i]->faces[j];
			vec3f &p1 = f->v[0]->node->x;
			vec3f &p2 = f->v[1]->node->x;
			vec3f &p3 = f->v[2]->node->x;

			s_fboxes[tri_idx] += p1;
			s_fboxes[tri_idx] += p2;
			s_fboxes[tri_idx] += p3;

			s_nidx[tri_idx] = vec3i(
				f->v[0]->node->index + vtx_offset,
				f->v[1]->node->index + vtx_offset,
				f->v[2]->node->index + vtx_offset);
			tri_idx++;
		}

		vtx_offset += ms[i]->nodes.size();
	}

	_nodes = new bvh_node[_num * 2 - 1];
	_nodes[0]._box = total;

	{
		FILE *fp = fopen(fname, "rt");
		char buff[512];
		int idx = 0;
		while (fgets(buff, 512, fp)) {
			if (buff[0] == '#')
				continue;

			int id;
			sscanf(buff, "%d", &id);
			_nodes[idx++]._child = id;
		}
		assert(idx == (_num * 2 - 1));
	}

#ifdef USE_NC
	if (cloth)
		refit(true);
	else
		refit();
#else
	refit();
#endif

	delete[] s_fboxes;
	delete[] s_nidx;
}

void bvh::construct(std::vector<Mesh*> &ms, bool cloth)
{
	BOX total;

	for (int i=0; i<ms.size(); i++)
		for (int j = 0; j<ms[i]->verts.size(); j++) {
			if (ms[i]->verts[j]->node == NULL)
				continue;

			total += ms[i]->verts[j]->node->x;
		}

	_num = 0;
	for (int i=0; i<ms.size(); i++)
		_num += ms[i]->faces.size();

	s_fcenters = new vec3f[_num];
	s_fboxes = new BOX[_num];
	s_nidx = new vec3i[_num];

	int tri_idx = 0;
	int vtx_offset = 0;

	for (int i = 0; i < ms.size(); i++) {
		for (int j = 0; j < ms[i]->faces.size(); j++) {
			Face *f = ms[i]->faces[j];
			vec3f &p1 = f->v[0]->node->x;
			vec3f &p2 = f->v[1]->node->x;
			vec3f &p3 = f->v[2]->node->x;

			s_fboxes[tri_idx] += p1;
			s_fboxes[tri_idx] += p2;
			s_fboxes[tri_idx] += p3;

			s_fcenters[tri_idx] = (p1 + p2 + p3) / REAL(3.0);
			s_nidx[tri_idx] = vec3i(
				f->v[0]->node->index + vtx_offset,
				f->v[1]->node->index + vtx_offset,
				f->v[2]->node->index + vtx_offset);
			tri_idx++;
		}
		vtx_offset += ms[i]->nodes.size();
	}

	aap pln(total);
	s_idx_buffer = new unsigned int[_num];
	unsigned int left_idx = 0, right_idx = _num;

	tri_idx = 0;
	for (int i=0; i<ms.size(); i++)
		for (int j=0; j<ms[i]->faces.size(); j++) {
		if (pln.inside(s_fcenters[tri_idx]))
			s_idx_buffer[left_idx++] = tri_idx;
		else
			s_idx_buffer[--right_idx] = tri_idx;

		tri_idx++;
	}

	_nodes = new bvh_node[_num*2-1];
	_nodes[0]._box = total;
	s_current = _nodes+3;

	if (_num == 1)
		_nodes[0]._child = 0;
	else {
		_nodes[0]._child = -1;

		if (left_idx == 0 || left_idx == _num)
			left_idx = _num/2;

		_nodes[0].left()->construct(s_idx_buffer, left_idx);
		_nodes[0].right()->construct(s_idx_buffer+left_idx, _num-left_idx);
	}

	delete [] s_idx_buffer;
	delete [] s_fcenters;

#ifdef USE_NC
	if (cloth)
		refit(true);
	else
		refit();
#else
	refit();
#endif

	delete[] s_nidx;
	delete[] s_fboxes;
}

void bvh::resetParents()
{
	root()->resetParents(root());
}

void bvh::refit(bool bound)
{
	root()->refit(bound);
}

#include <queue>
using namespace std;

void bvh::reorder()
{
	if (true) 
	{
		queue<bvh_node *> q;

		// We need to perform a breadth-first traversal to fill the ids

		// the first pass get idx for each node ...
		int *buffer = new int[_num*2-1];
		int idx = 0;
		q.push(root());
		while (!q.empty()) {
			bvh_node *node = q.front();
			buffer[node-_nodes] = idx++;
			q.pop();

			if (!node->isLeaf()) {
				q.push(node->left());
				q.push(node->right());
			}
		}

		// the 2nd pass, get right nodes ...
		bvh_node *new_nodes = new bvh_node[_num*2-1];
		idx=0;
		q.push(root());
		while (!q.empty()) {
			bvh_node *node = q.front();
			q.pop();

			new_nodes[idx] = *node;
			if (!node->isLeaf()) {
				int loc = node->left()-_nodes;
				new_nodes[idx]._child = idx-buffer[loc];
			}
			idx++;

			if (!node->isLeaf()) {
				q.push(node->left());
				q.push(node->right());
			}
		}

		delete [] buffer;
		delete [] _nodes;
		_nodes = new_nodes;
	}
}

#ifdef USE_NC

extern void refitBVH(bool);
extern void pushBVH(unsigned int length, int *ids, bool isCloth);
extern void pushBVHLeaf(unsigned int length, int *idf, bool isCloth);
extern void pushBVHIdx(int max_level, unsigned int *level_idx, bool isCloth);
extern void pushBVHContour(bool isCloth, unsigned int *ctIdx, unsigned int *ctLst,
	int ctNum, int length, int triNum);

void bvh::push2GPU(bool isCloth)
{
	unsigned int length = _num*2-1;
	int *ids = new int[length*2];

	for (unsigned int i=0; i<length; i++) {
		ids[i] = (root()+i)->triID();
		ids[length+i] = (root()+i)->parentID();
	}

	pushBVH(length, ids, isCloth);
	delete [] ids;

	unsigned int leafNum = 0;
	int *idf = new int[_num];
	for (unsigned int i = 0; i < length; i++) {
		if ((root() + i)->isLeaf()) {
			int idx = (root() + i)->triID();
			idf[idx] = i;
			leafNum++;
		}
	}
	assert(leafNum == _num);
	pushBVHLeaf(leafNum, idf, isCloth);
	delete []idf;

	if (isCloth) //push contour information of NCC
	{
		uint *ctIdx = new uint[length];
		uint idx = 0;
		for (int i = 0; i < length; i++) {
			bvh_node *node = root() + i;
			idx += node->get_bound_length();
			ctIdx[i] = idx;
		}

		uint *ctLst = new uint[idx];
		idx = 0;
		for (int i = 0; i < length; i++) {
			bvh_node *node = root() + i;
			for (int j = 0; j < node->get_bound_length(); j++)
				ctLst[idx++] = node->get_bound_idx(0, j);
		}

		pushBVHContour(isCloth, ctIdx, ctLst, idx, length, _num);

		delete[] ctIdx;
		delete[] ctLst;
	}
	else
		pushBVHContour(isCloth, NULL, NULL, 0, 0, 0);

	//if (isCloth) 
	{// push information for refit
		int max_level = 0;
		root()->getLevel(0, max_level);
		max_level++;

		unsigned int *level_idx = new unsigned int [max_level];
		unsigned int *level_buffer = new unsigned int [max_level];
		for (int i=0; i<max_level; i++)
			level_idx[i] = level_buffer[i] = 0;

		root()->getLevelIdx(0, level_buffer);
		for (int i=1; i<max_level; i++)
			for (int j=0; j<i; j++)
				level_idx[i] += level_buffer[j];

		delete [] level_buffer;
		pushBVHIdx(max_level, level_idx, isCloth);
		delete [] level_idx;
	}

	//refitBVH_Serial(isCloth); // will cause runtime-out-error at 1Kx1K resolution
	refitBVH(isCloth);
}
#endif

void
bvh_node::construct(unsigned int id)
{
	_child = id;
	_box = s_fboxes[id];
}

void
bvh_node::construct(unsigned int *lst, unsigned int num)
{
	for (unsigned int i=0; i<num; i++)
		_box += s_fboxes[lst[i]];

	if (num == 1) {
		_child = lst[0];
		return;
	}

	// try to split them
	_child = int(this-s_current);
	s_current += 2;

	if (num == 2) {
		left()->construct(lst[0]);
		right()->construct(lst[1]);
		return;
	}

	aap pln(_box);
	unsigned int left_idx=0, right_idx=num-1;
	for (unsigned int t=0; t<num; t++) {
		int i=lst[left_idx];

		if (pln.inside( s_fcenters[i]))
			left_idx++;
		else {// swap it
			unsigned int tmp=lst[left_idx];
			lst[left_idx] = lst[right_idx];
			lst[right_idx--] = tmp;
		}
	}

	int half = num/2;

	if (left_idx == 0 || left_idx == num) {
		left()->construct(lst, half);
		right()->construct(lst+half, num-half);
	} else {
		left()->construct(lst, left_idx);
		right()->construct(lst+left_idx, num-left_idx);
	}
}

REAL *__getVtx(int idx)
{
	int c = 0;
	while (idx >= (*ptCloth)[c]->nodes.size()) {
		idx -= (*ptCloth)[c]->nodes.size();
		c++;
	}

	return &(*ptCloth)[c]->nodes[idx]->x[0];
}

void
contour::visualize(bool spec)
{
	for (int i = 0; i < _contours.size(); i++)
	{ 
		if (spec)
			glColor3f(1, 0, 0);
		else
			glColor3f(0.6, 0.6, 0);

		glBegin(GL_LINE_LOOP);
		for (int j = 0; j < _contours[i].size(); j++) {
			int idx = _contours[i][j];
			REAL *pt = __getVtx(idx);

#ifdef USE_DOUBLE
			glVertex3dv(pt);
#else
			glVertex3fv(pt);
#endif
		}
		glEnd();
	}
}

void
bvh_node::visualizeBound(bool spec)
{
	_bound.visualize(spec);
}

void
bvh_node::visualize(int level)
{
	if (isLeaf()) {
		_bound.visualize(false);
	}
	else
		if ((level > 0)) {
			if (level == 1) {
				_bound.visualize(false);
			}
			else {
				if (left()) left()->visualize(level - 1);
				if (right()) right()->visualize(level - 1);
			}
		}
}

void
bvh_node::refit(bool bound)
{
	if (isLeaf()) {
		_box = s_fboxes[_child];

		if (bound) {
			vec3i ids = s_nidx[_child];

			_bound.build(ids[0], ids[1], ids[2]);
		}

	} else {
		left()->refit(bound);
		right()->refit(bound);

		_box = left()->_box + right()->_box;

		if (bound) {
			_bound.build(left()->_bound, right()->_bound);
		}
	}
}

void
bvh_node::resetParents(bvh_node *root)
{
	if (this == root)
		setParent(-1);

	if (isLeaf())
		return;

	left()->resetParents(root);
	right()->resetParents(root);

	left()->setParent(this - root);
	right()->setParent(this - root);
}

#ifdef USE_NC
extern void pushFront(bool, int, unsigned int *);

void 
front_list::push2GPU(bvh_node *r1, bvh_node *r2 )
{
	bool self = (r2 == NULL);

	if (r2 == NULL)
		r2 = r1;

	int num = size();
	if (num) {
		int idx = 0;
		unsigned int *buffer = new unsigned int [num*4];
		for (vector<front_node>::iterator it=begin();
			it != end(); it++)
		{
			front_node n = *it;
			buffer[idx++] = n._left - r1;
			buffer[idx++] = n._right-r2;
			buffer[idx++] = 0;
			buffer[idx++] = n._ptr;
		}

		pushFront(self, num, buffer);
		delete [] buffer;
	} else
		pushFront(self, 0, NULL);
}
#endif

void mesh_id(int id, vector<Mesh *> &m, int &mid, int &fid)
{
	fid = id;
	for (mid=0; mid<m.size(); mid++)
		if (fid < m[mid]->faces.size()) {
			return;
		} else {
			fid -= m[mid]->faces.size();
		}

	assert(false);
	fid = -1;
	mid = -1;
	printf("mesh_id error!!!!\n");
	abort();
}

bool covertex(int id1, int id2)
{
	if ((*ptCloth).empty())
		return false;

	int mid1, fid1, mid2, fid2;

	mesh_id(id1, *ptCloth, mid1, fid1);
	mesh_id(id2, *ptCloth, mid2, fid2);

	if (mid1 != mid2)
		return false;

	Vert ** v1 = (*ptCloth)[mid1]->faces[fid1]->v;
	Vert ** v2 = (*ptCloth)[mid2]->faces[fid2]->v;
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			if (v1[i]->index == v2[j]->index)
				return true;

	return false;

/*
	Vert** v1 =ptCloth->faces[id1]->v;
	Vert** v2 =ptCloth->faces[id2]->v;

	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			if (v1[i]->index == v2[j]->index)
				return true;

	return false;
*/
}