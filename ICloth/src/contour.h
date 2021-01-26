#pragma once
#include <stdio.h>
#include <string.h>

#include <vector>
#include <set>
using namespace std;

typedef vector<unsigned int> ring;

typedef Vec2 vec2f;

FORCEINLINE void
vmin(vec2f &a, const vec2f &b)
{
	a = vec2f(
		fmin(a[0], b[0]),
		fmin(a[1], b[1]));
}

FORCEINLINE void
vmax(vec2f &a, const vec2f &b)
{
	a = vec2f(
		fmax(a[0], b[0]),
		fmax(a[1], b[1]));
}

class aabb2 {
	FORCEINLINE void init() {
		_max = vec2f(-FLT_MAX, -FLT_MAX);
		_min = vec2f(FLT_MAX, FLT_MAX);
	}

public:
	vec2f _min;
	vec2f _max;

	FORCEINLINE aabb2() {
		init();
	}

	FORCEINLINE aabb2(const vec2f &v) {
		_min = _max = v;
	}

	FORCEINLINE aabb2(const vec2f &a, const vec2f &b) {
		_min = a;
		_max = a;
		vmin(_min, b);
		vmax(_max, b);
	}

	FORCEINLINE bool overlaps(const aabb2& b) const
	{
		if (_min[0] > b._max[0]) return false;
		if (_min[1] > b._max[1]) return false;

		if (_max[0] < b._min[0]) return false;
		if (_max[1] < b._min[1]) return false;

		return true;
	}

	FORCEINLINE bool overlaps(const aabb2 &b, aabb2 &ret) const
	{
		if (!overlaps(b))
			return false;

		ret._min = vec2f(
			fmax(_min[0], b._min[0]),
			fmax(_min[1], b._min[1]));

		ret._max = vec2f(
			fmin(_max[0], b._max[0]),
			fmin(_max[1], b._max[1]));

		return true;
	}

	FORCEINLINE bool inside(const vec2f &p) const
	{
		if (p[0] < _min[0] || p[0] > _max[0]) return false;
		if (p[1] < _min[1] || p[1] > _max[1]) return false;

		return true;
	}

	FORCEINLINE aabb2 &operator += (const vec2f &p)
	{
		vmin(_min, p);
		vmax(_max, p);
		return *this;
	}

	FORCEINLINE aabb2 &operator += (const aabb2 &b)
	{
		vmin(_min, b._min);
		vmax(_max, b._max);
		return *this;
	}

	FORCEINLINE aabb2 operator + (const aabb2 &v) const
	{
		aabb2 rt(*this); return rt += v;
	}

	FORCEINLINE REAL width()  const { return _max[0] - _min[0]; }
	FORCEINLINE REAL height() const { return _max[1] - _min[1]; }
	FORCEINLINE vec2f center() const { return (_min + _max)*REAL(0.5); }
	FORCEINLINE REAL area() const { return width()*height(); }


	FORCEINLINE bool empty() const {
		return _max[0] < _min[0];
	}

	FORCEINLINE void enlarge(REAL thickness) {
		_max += vec2f(thickness, thickness);
		_min -= vec2f(thickness, thickness);
	}

	vec2f getMax() { return _max; }
	vec2f getMin() { return _min; }

	void print(FILE *fp) {
		//fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	}

	//void visualize();
};

typedef vector<unsigned int> ring;

// for contour list of a BVH node
class contour {
	// now assuming only one perfect contour in each node
	// sorted node indices
	vector<ring> _contours;

public:
	inline bool single_ring() { return _contours.size() == 1; }

	inline int ring_num() { return _contours.size(); }
	inline int ring_length(int i) { return _contours[i].size(); }
	inline int ring_index(int r, int i) { return _contours[r][i]; }

	inline unsigned int get(int r, int i) const {
		int len = _contours[r].size();

		if (i >= len) i -= len;
		if (i < 0) i += len;

		return _contours[r][i];
	}

	FORCEINLINE contour()
	{}

	// build for leaf nodes
	FORCEINLINE void build(unsigned int id0, unsigned int id1, unsigned id2)
	{
		_contours.clear();

		ring tmp;
		tmp.push_back(id0);
		tmp.push_back(id1);
		tmp.push_back(id2);

		_contours.push_back(tmp);
	}

	bool check() const;
	bool check(const ring &r) const;
	void relink();
	void relink(ring &, vector<ring> &);

	void build(const contour &left, const contour &right);
	bool build_one(const ring &left, const ring &right, ring &result);

	void visualize(bool);
};



template <class T>
static inline T vecget(const vector<T>& v, int idx){
	int len = v.size();

	while (idx >= len) idx -= len;
	while (idx < 0) idx += len;

	return v[idx];
}

// build for internal nodes by merging ...
inline bool
contour::build_one(const ring &left, const ring &right, ring &result)
{
	//	idx++;
	//	if (idx == 13)
	//		printf("here");

	//_contours.clear();

	// It is two loop merging problem, so we first find out the duplications: a - b for left, c - d for right
	// then we merge the left as a whole

	int a, b, c, d;
	int llen = left.size();
	int rlen = right.size();
	bool shareVtx = false;

	//if (llen == 3 && left.get(0) == 31089 && left.get(1) == 3395 && left.get(2) == 10951)
	//	printf("here!");
	//		if (llen == 5)
	//			printf("here!");

	/*
	if (left.size() == 18 && right.size() == 3)
		if (right[0] == 13162 && right[1] == 13171 && right[2] == 13227)
			if (left[0] == 13171 && left[1] == 13162 && left[2] == 13170)
				printf("here!");
	*/

	// assuming the 1st segement is duplicate
	for (a = 0; a < llen; a++) {
		for (c = 0; c < rlen; c++) {
			if ((vecget(right, c) == vecget(left, a) &&
				vecget(right, c - 1) == vecget(left, a + 1)) &&
				vecget(right, c + 1) != vecget(left, a - 1))
				goto find;
		}
	}

	//printf("two contours not share an edge...\n");
	shareVtx = true;

	if (!shareVtx) {
	find:
		// now determine b and d
		for (int i = 1; i < llen && i <rlen; i++) {
			b = a + i;
			d = c - i;

			if (vecget(right, d) != vecget(left, b)) {
				b--;
				d++;
				break;
			}
		}

		if (b < 0) b += llen;
		if (b >= llen) b -= llen;

		if (d < 0) d += rlen;
		if (d >= rlen) d -= rlen;
	}
	else {
		for (a = 0; a < llen; a++) {
			for (c = 0; c < rlen; c++) {
				if (vecget(right, c) == vecget(left, a))
					goto findVtx;
			}
		}

		if (true) {
			//printf("two contours not share a vertex...\n");
			//exit(0);
			return false;
		}

	findVtx:
		b = a;
		d = c;
	}

	if (b < a) {
		for (int i = b; i <= a; i++)
			result.push_back(vecget(left, i));
	}
	else {
		for (int i = 0; i < llen; i++) {
			int l = b + i;
			if (l >= llen) {
				l -= llen;

				if (l <= a)
					result.push_back(vecget(left, l));
			}
			else
				result.push_back(vecget(left, l));
		}
	}

	// special case: insert a duplicate vertex here!
	if (a == b)
		result.push_back(vecget(left, a));

	if (c < d)
		for (int i = c + 1; i < d; i++)
			result.push_back(vecget(right, i));
	else {
		for (int j = 0; j < rlen; j++) {
			int r = c + 1 + j;

			if (r >= rlen) {
				r -= rlen;

				if (r < d)
					result.push_back(vecget(right, r));
			}
			else
				result.push_back(vecget(right, r));
		}
	}

	return true;
	/*
	// it will be ok, if only share a vertex
	// check for duplicated
	int len = _contours.size();
	for (int i = 0; i < len; i++) {
	int id = _contours[i];

	for (int j = i + 1; j < len; j++)
	if (_contours[j] == id) {
	printf("Bad ..., idx=%d\n", idx);
	}
	}
	*/
	//if (_contours.size() == 5)
	//	printf("Here!");
}

inline bool
contour::check() const
{
	for (int i = 0; i < _contours.size(); i++)
		if (false == check(_contours[i]))
			return false;

	return true;
}

inline bool
contour::check(const ring &r) const
{
	for (int i = 0; i < r.size(); i++) {
		int s = r[i];

		int I = i + 1;
		if (I >= r.size()) I -= r.size();

		int t = r[I];

		for (int j = i + 1; j < r.size(); j++) {
			int u = r[j];
			int J = j + 1;
			if (J >= r.size()) J -= r.size();
			int v = r[J];

			if (s == u && t == v)
				return false;

			if (s == v && t == u)
				return false;
		}

	}
	return true;
}


inline bool _Eflag(int i, bool *flags, int len)
{
	if (i < 0) i += len;
	if (i >= len) i -= len;
	return flags[i];
}

inline int _Dec(int i, int len)
{
	i--;
	if (i < 0) i += len;
	return i;
}

inline int _Inc(int i, int len)
{
	i++;
	if (i >= len) i -= len;
	return i;
}

inline void
contour::relink(ring &r, vector<ring> &split)
{
	int numV = r.size();
	int numE = numV;
	bool *flag = new bool[numE];
	memset(flag, 0, numE);

	//mark repeated edges ...
	for (int i = 0; i<numE; i++) {
		if (flag[i] == 0) {// unchecked yet...
			for (int j = i + 1; j <numE; j++) {
				int s = i;
				int t = i + 1;
				int u = j;
				int v = j + 1;

				if (s >= numV) s -= numV;
				if (t >= numV) t -= numV;
				if (u >= numV) u -= numV;
				if (v >= numV) v -= numV;

				if (r[s] == r[v] && r[t] == r[u]) {
					flag[i] = flag[j] = 1;
				}
			}
		}
	}

	//pickout new rings from unmarked edges...
	ring nr;

	do {
		nr.clear();

		for (int i = 0; i < numE; i++) {
			if (flag[i] == 0) { //find a remained new edge ...
				int st = i;
				int ed = _Inc(i, numE);
				flag[st] = 1;

				nr.push_back(r[st]);
				//nr.push_back(st);


				while (_Eflag(st - 1, flag, numE) == 0) {
					st = _Dec(st, numE);
					nr.insert(nr.begin(), r[st]);
					//nr.insert(nr.begin(), st);
					flag[st] = 1;
				}

				while (_Eflag(ed, flag, numE) == 0) {
					flag[ed] = 1;
					nr.push_back(r[ed]);
					//nr.push_back(ed);
					ed = _Inc(ed, numE);
				}

				//assert()
				if (r[ed] != r[st])
					; //printf("Unfinshed work!\n");

				break;
			}
		}

		if (nr.size())
			split.push_back(nr);
		else
			break;
	} while (1);

	delete[] flag;
}

inline void
contour::relink()
{
	vector<ring>tmp;

	for (int i = 0; i < _contours.size(); i++) {
		relink(_contours[i], tmp);
	}

	_contours = tmp;
}


// build for internal nodes by merging ...
//merge maybe not only loop
inline void
contour::build(const contour &left, const contour &right)
{
	//if (left._contours.size() > 1 || right._contours.size() > 1)
	//	printf("here!\n");

	//start merge maybe not
	_contours.clear();

	//printf("into it........11111\n");
	const vector<ring>& tmpleft = left._contours;
	const vector<ring>& tmpright = right._contours;


	int llen = tmpleft.size();
	int rlen = tmpright.size();
	bool *lmark = new bool[llen];
	for (int i = 0; i < llen; i++){
		lmark[i] = false;
	}
	bool *rmark = new bool[rlen];
	for (int i = 0; i < rlen; i++){
		rmark[i] = false;
	}
	//printf("into it........22222\n");
	for (int i = 0; i < llen; i++){
		bool mark = false;
		for (int j = 0; j < rlen; j++){
			if (rmark[j]){
				continue;
			}
			vector<unsigned int> tmp;
			//printf("%d %d.......\n", i, j);
			bool cb = build_one(tmpleft[i], tmpright[j], tmp);
			//printf(".......777777777\n");
			if (cb){
				mark = true;
				rmark[j] = true;
				_contours.push_back(tmp);
				break;
			}
		}
		if (mark){
			lmark[i] = true;
		}
	}
	//printf("into it........33333\n");
	int csz = _contours.size();
	for (int i = 0; i < csz; i++){
		bool change = false;
		for (int j = 0; j < llen; j++){
			if (!lmark[j]){
				vector<unsigned int> tmp;
				bool cb = build_one(_contours[i], tmpleft[j], tmp);
				if (cb){
					change = true;
					_contours[i] = tmp;
					lmark[j] = true;
					break;
				}
			}
		}
		if (!change){
			for (int j = 0; j < rlen; j++){
				if (!rmark[j]){
					vector<unsigned int> tmp;
					bool cb = build_one(_contours[i], tmpright[j], tmp);
					if (cb){
						change = true;
						_contours[i] = tmp;
						rmark[j] = true;
						break;
					}
				}
			}
		}
		if (!change){
			break;
		}
	}

	for (int j = 0; j < llen; j++){
		if (!lmark[j]){
			_contours.push_back(tmpleft[j]);
		}
	}
	for (int j = 0; j < rlen; j++){
		if (!rmark[j]){
			_contours.push_back(tmpright[j]);
		}
	}

	//	if (_contours[0].size() == 76)
	//		printf("here!\n");

	if (check() == false) {
		//printf("self topology intersection ...\n");
		relink();
	}

	//printf("run out!!!\n");
	delete[] lmark;
	delete[] rmark;
}
