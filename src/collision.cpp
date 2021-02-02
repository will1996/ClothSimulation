#include "collision.hpp"
#include "collisionutil.hpp"
#include "geometry.hpp"
#include "magic.hpp"
#include "optimization.hpp"
#include "simulation.hpp"
#include "timer.hpp"
#include <algorithm>
#include <fstream>
#include <omp.h>
using namespace std;

#include "eigen.hpp"

static const int max_iter = 100;
static const REAL &thickness = ::magic.projection_thickness;

static REAL obs_mass;
static bool deform_obstacles;

static vector<Vec3> xold;
static vector<Vec3> xold_obs;

void impact_zone(std::vector<Mesh*> &meshes, const std::vector<Mesh*> &obs_meshes, REAL dt);

double get_mass(const Node *node) { return is_free(node) ? node->m : obs_mass; }

typedef enum {
	I_NULL = -1,
	I_VF = 0,
	I_EE = 1
} ImpactType;


struct tmImpactInfo {
	uint _nodes[4];
	bool _frees[4];
	REAL _w[4];

	ImpactType _type;
	REAL _t;
	Vec3 _n;

	void print() {
		printf("(%d, %d, %d, %d), (%d, %d, %d, %d), %d, (%lf, %lf, %lf, %lf), (%lf, %lf, %lf)\n",
			_nodes[0], _nodes[1], _nodes[2], _nodes[3],
			_frees[0], _frees[1], _frees[2], _frees[3],
			_type,
			_w[0], _w[1], _w[2], _w[3],
			_n[0], _n[1], _n[2]
			);
	}
};

struct g_impactNode {
	uint _n;
	bool _f;
	Vec3 _ox;
	Vec3 _x;
	REAL _m;

	g_impactNode() {
		_n = -1;
	}

	g_impactNode(uint n, bool f) {
		_n = n;
		_f = f;
	}
};

vector<g_impactNode> *currentNodes = NULL;

struct g_impact {
	uint _nodes[4];
	bool _frees[4];
	REAL _w[4];

	ImpactType _type;
	REAL _t;
	Vec3 _n;

	g_impact()
	{
		_nodes[0] = _nodes[1] = _nodes[2] = _nodes[3] = -1;
		_frees[0] = _frees[1] = _frees[2] = _frees[3] = false;
		_w[0] = _w[1] = _w[2] = _w[3] = 0.0;
		_type = I_NULL;
		_t = 0.0;
		_n = Vec3(0.0);
	}

	g_impact(ImpactType type, uint n0, uint n1, uint n2, uint n3, bool f0, bool f1, bool f2, bool f3)
	{
		_type = type;
		_nodes[0] = n0, _nodes[1] = n1, _nodes[2] = n2, _nodes[3] = n3;
		_frees[0] = f0, _frees[1] = f1, _frees[2] = f2, _frees[3] = f3;
	}

	void print() {
		printf("(%d, %d, %d, %d), (%d, %d, %d, %d), %d, (%lf, %lf, %lf, %lf), (%lf, %lf, %lf)\n",
			_nodes[0], _nodes[1], _nodes[2], _nodes[3],
			_frees[0], _frees[1], _frees[2], _frees[3],
			_type,
			_w[0], _w[1], _w[2], _w[3],
			_n[0], _n[1], _n[2]
			);
	}
};

struct g_impactZone {
	vector<uint> _nodes;
	vector<bool>  _frees;
	vector<g_impact> _impacts;
	bool _active;

	void print() {
		printf("impacts = %d\n", _impacts.size());
		for (int i = 0; i<_impacts.size(); i++) {
			_impacts[i].print();
		}
	}
};

vector<g_impact> independent_impacts(const vector<g_impact> &impacts);
void independent_impacts(tmImpactInfo *imps, int num, vector<tmImpactInfo> &indep);

void add_impacts(const vector<g_impact> &impacts, vector<g_impactZone*> &zones, bool);

void apply_inelastic_projection(g_impactZone *zone, REAL dt);

extern int get_impacts_gpu(REAL dt, REAL mu, REAL mu_obs, REAL mrt, REAL mcs, bool self_cd);
extern void get_impact_data_gpu(void *data, void *nodes, int num);
extern void put_impact_node_gpu(void *nodes, int num, REAL mrt);
extern void put_impact_vel_gpu(void *nodes, int num, REAL dt);

void push_node_data(vector<g_impactNode> &nodes, REAL mrt)
{
	int num = nodes.size();
	g_impactNode *buffer = new g_impactNode[num];
	for (int i = 0; i<num; i++)
		buffer[i] = nodes[i];

	put_impact_node_gpu(buffer, num, mrt);
	delete[] buffer;
}

void push_node_vel(vector<g_impactNode> &nodes, REAL dt)
{
	int num = nodes.size();
	g_impactNode *buffer = new g_impactNode[num];
	for (int i = 0; i<num; i++)
		buffer[i] = nodes[i];

	put_impact_vel_gpu(buffer, num, dt);
	delete[] buffer;
}

extern void load_x_gpu();

void collision_response_gpu(REAL dt, REAL mu, REAL mu_obs, REAL mrt, REAL mcs)
{
	REAL obs_mass = 1e3;
	int max_iter = 100, iter = 0;
	bool updated = false;

	::obs_mass = 1e3;

	vector<g_impactZone *> zones;
	vector<g_impactNode> iNodes;
	currentNodes = &iNodes;

	for (int deform = 0; deform <= 1; deform++)
	{
		for (int z = 0; z<zones.size(); z++)
			delete zones[z];
		zones.clear();

		for (iter = 0; iter<max_iter; iter++)
		{
			int impNum = get_impacts_gpu(dt, mu, mu_obs, mrt, mcs, ::magic.tm_self_cd);
			if (impNum == 0)
				break;

			g_impact *imps = new g_impact[impNum];
			g_impactNode *impNodes = new g_impactNode[impNum * 4];
			get_impact_data_gpu(imps, impNodes, impNum);
			//printf("total impacts = %d\n", impNum);
			::sort(imps, imps + impNum);

			vector<g_impact> impacts;
			for (int i = 0; i<impNum; i++) {
				impacts.push_back(imps[i]);
			}
			delete[] imps;

			for (int i = 0; i<impNum * 4; i++) {
				if (find(g_impactNode(impNodes[i]._n, impNodes[i]._f), *currentNodes) != -1)
					continue;
				else {
					impNodes[i]._ox = impNodes[i]._x; // backup the _x, the _ox never used

					iNodes.push_back(impNodes[i]);
				}
			}
			delete[] impNodes;

			::sort(iNodes.begin(), iNodes.end());
			iNodes.erase(::unique(iNodes.begin(), iNodes.end()), iNodes.end());


			impacts = independent_impacts(impacts);
			if (impacts.empty())
				break;

			updated = true;

			add_impacts(impacts, zones, deform);
			for (int i = 0; i<zones.size(); i++) {
				g_impactZone *z = zones[i];

				REAL tx = omp_get_wtime();
				apply_inelastic_projection(z, dt);
				push_node_data(iNodes, ::magic.repulsion_thickness); // need to update x, and update bounding volumes ...
			}

			if (deform)
				obs_mass *= 0.5;

		}

		if (iter < max_iter) //success!
			break;
	}

	if (iter == max_iter) {
		cerr << "Collision resolution failed to converge!" << endl;
		//exit(1);
	}

	if (updated) {
		push_node_vel(iNodes, dt);
	}

	for (int z = 0; z<zones.size(); z++)
		delete zones[z];
	zones.clear();
}

// returns pair of (i) is_free(vert), and
// (ii) index of mesh in ::meshes or ::obs_meshes that contains vert
pair<bool,int> find_in_meshes (const Node *node) {
    int m = find_mesh(node, *::meshes);
    if (m != -1)
        return make_pair(true, m);
    else
        return make_pair(false, find_mesh(node, *::obs_meshes));
}

struct Impact {
    enum Type {VF, EE} type;
    double t;
    Node *nodes[4];
    double w[4];
    Vec3 n;
    Impact () {}
    Impact (Type type, const Node *n0, const Node *n1, const Node *n2,
            const Node *n3): type(type) {
        nodes[0] = (Node*)n0;
        nodes[1] = (Node*)n1;
        nodes[2] = (Node*)n2;
        nodes[3] = (Node*)n3;
    }
};

struct ImpactZone {
    vector<Node*> nodes;
    vector<Impact> impacts;
    bool active;
};

void update_active (const vector<AccelStruct*> &accs,
                    const vector<AccelStruct*> &obs_accs,
                    const vector<ImpactZone*> &zones);

vector<Impact> find_impacts (const vector<AccelStruct*> &acc,
                             const vector<AccelStruct*> &obs_accs);
vector<Impact> independent_impacts (const vector<Impact> &impacts);

void add_impacts (const vector<Impact> &impacts, vector<ImpactZone*> &zones);

void apply_inelastic_projection (ImpactZone *zone,
                                 const vector<Constraint*> &cons);

vector<Constraint> impact_constraints (const vector<ImpactZone*> &zones);

ostream &operator<< (ostream &out, const Impact &imp);
ostream &operator<< (ostream &out, const ImpactZone *zone);

void collision_response (vector<Mesh*> &meshes, const vector<Constraint*> &cons,
                         const vector<Mesh*> &obs_meshes, bool verbose) {
    return;
}

static int nthreads = 0;

// Independent impacts
bool operator < (const g_impact &imp0, const g_impact &imp1) {
	//	return imp0._t < imp1._t;


	if (imp0._type != imp1._type)
		return imp0._type < imp1._type;

	for (int i = 0; i<4; i++) {
		if (imp0._frees[i] != imp1._frees[i])
			return imp0._frees[i] < imp1._frees[i];

		if (imp0._nodes[i] != imp1._nodes[i])
			return imp0._nodes[i] < imp1._nodes[i];
	}

	return true;
}

bool operator == (const g_impactNode &n0, const g_impactNode &n1) {
	return (n0._f == n1._f) && (n0._n == n1._n);
}


bool operator < (const g_impactNode &n0, const g_impactNode &n1) {
	if (n0._f != n1._f)
		return n0._f < n1._f;
	else
		return n0._n < n1._n;
}

// Impacts


bool conflict(const g_impact &impact0, const g_impact &impact1);

vector<g_impact> independent_impacts(const vector<g_impact> &impacts) {
	vector<g_impact> sorted = impacts;
	sort(sorted.begin(), sorted.end());
	vector<g_impact> indep;
	for (int e = 0; e < sorted.size(); e++) {
		const g_impact &impact = sorted[e];
		bool con = false;
		for (int e1 = 0; e1 < indep.size(); e1++)
			if (conflict(impact, indep[e1]))
				con = true;
		if (!con)
			indep.push_back(impact);
	}
	return indep;
}

bool is_in(uint id, const uint ids[], const bool frees[])
{
	for (int i = 0; i<4; i++)
		if (ids[i] == id && frees[i])
			return true;

	return false;
}

bool conflict(const g_impact&i0, const g_impact&i1) {
	return
		(i0._frees[0] && is_in(i0._nodes[0], i1._nodes, i1._frees)) ||
		(i0._frees[1] && is_in(i0._nodes[1], i1._nodes, i1._frees)) ||
		(i0._frees[2] && is_in(i0._nodes[2], i1._nodes, i1._frees)) ||
		(i0._frees[3] && is_in(i0._nodes[3], i1._nodes, i1._frees));
}

// Impact zones
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
g_impactZone *find_or_create_zone(uint node, bool free, vector<g_impactZone*> &zones);
void merge_zones(g_impactZone* zone0, g_impactZone *zone1,
	vector<g_impactZone*> &zones);

void add_impacts(const vector<g_impact> &impacts, vector<g_impactZone*> &zones, bool deform) {
	for (int z = 0; z < zones.size(); z++)
		zones[z]->_active = false;

	int num_active = 0;
	for (int i = 0; i < impacts.size(); i++) {
		const g_impact &impact = impacts[i];
		uint node = impact._nodes[impact._frees[0] ? 0 : 3];
		bool free = impact._frees[impact._frees[0] ? 0 : 3];
		assert(free);

		g_impactZone *zone = find_or_create_zone(node, free, zones);
		for (int n = 0; n < 4; n++)
			if (impact._frees[n] || deform)
				merge_zones(zone, find_or_create_zone(impact._nodes[n], impact._frees[n], zones), zones);

		zone->_impacts.push_back(impact);
		zone->_active = true;
		num_active++;
	}

	//printf("all/active zones = %d/%d\n", zones.size(), num_active);
}

bool is_in(uint node, bool free, g_impactZone *z)
{
	assert(z->_nodes.size() == z->_frees.size());
	int num = z->_nodes.size();
	for (int i = 0; i<num; i++) {
		if (z->_nodes[i] == node && z->_frees[i] == free)
			return true;
	}

	return false;
}

g_impactZone *find_or_create_zone(uint node, bool free, vector<g_impactZone*> &zones) {
	for (int z = 0; z < zones.size(); z++)
		if (is_in(node, free, zones[z]))
			return zones[z];

	g_impactZone *zone = new g_impactZone;
	zone->_nodes.push_back(node);
	zone->_frees.push_back(free);

	zones.push_back(zone);
	return zone;
}

void merge_zones(g_impactZone* zone0, g_impactZone *zone1)
{
	append(zone0->_nodes, zone1->_nodes);
	append(zone0->_frees, zone1->_frees);
	append(zone0->_impacts, zone1->_impacts);
}

void merge_zones(g_impactZone* zone0, g_impactZone *zone1,
	vector<g_impactZone*> &zones)
{
	if (zone0 == zone1)
		return;
	append(zone0->_nodes, zone1->_nodes);
	append(zone0->_frees, zone1->_frees);
	append(zone0->_impacts, zone1->_impacts);
	exclude(zone1, zones);
	delete zone1;
}

// Response
inline REAL get_mass(uint n, bool f)
{
	if (f == false)
		return obs_mass;

	int id = find(g_impactNode(n, f), *currentNodes);
	assert(id != -1);
	return (*currentNodes)[id]._m;
}

inline Vec3 get_x(uint n, bool f)
{
	int id = find(g_impactNode(n, f), *currentNodes);
	assert(id != -1);
	return (*currentNodes)[id]._x;
}

inline Vec3 get_x0(uint n, bool f)
{
	int id = find(g_impactNode(n, f), *currentNodes);
	assert(id != -1);
	return (*currentNodes)[id]._ox;
}

inline Vec3 get_xold(uint n, bool f)
{
	int id = find(g_impactNode(n, f), *currentNodes);
	assert(id != -1);
	return (*currentNodes)[id]._ox;
}

inline void set_x(uint n, bool f, Vec3 v)
{
	if (f == false)
		return;

	int id = find(g_impactNode(n, f), *currentNodes);
	assert(id != -1);
	(*currentNodes)[id]._x = v;
}

struct gNormalOpt : public NLConOpt {
	g_impactZone *_zone;
	REAL _inv_m;
	gNormalOpt() : _zone(NULL), _inv_m(0) { nvar = ncon = 0; }
	gNormalOpt(g_impactZone *zone) : _zone(zone), _inv_m(0) {
		nvar = _zone->_nodes.size() * 3;
		ncon = _zone->_impacts.size();
		for (int n = 0; n < _zone->_nodes.size(); n++)
			_inv_m += 1 / get_mass(_zone->_nodes[n], _zone->_frees[n]);
		_inv_m /= _zone->_nodes.size();
	}
	void initialize(double *x) const;
	void precompute(const double *x) const;
	double objective(const double *x) const;
	void obj_grad(const double *x, double *grad) const;
	double constraint(const double *x, int i, int &sign) const;
	void con_grad(const double *x, int i, double factor, double *grad) const;
	void finalize(const double *x) const;
};

void apply_inelastic_projection(g_impactZone *zone, REAL dt)
{
	if (!zone->_active)
		return;

	augmented_lagrangian_method(gNormalOpt(zone));
}


void gNormalOpt::initialize(double *x) const {
	for (int n = 0; n < _zone->_nodes.size(); n++) {
		set_subvec(x, n, get_x(_zone->_nodes[n], _zone->_frees[n]));
	}
}

void gNormalOpt::precompute(const double *x) const {
	for (int n = 0; n < _zone->_nodes.size(); n++)
		//zone->nodes[n]->x = get_subvec(x, n);
		set_x(_zone->_nodes[n], _zone->_frees[n], get_subvec(x, n));

}

double gNormalOpt::objective(const double *x) const {
	REAL e = 0;
	for (int n = 0; n < _zone->_nodes.size(); n++) {
		uint nd = _zone->_nodes[n];
		bool f = _zone->_frees[n];
		Vec3 dx = get_x(nd, f) - get_xold(nd, f);
		e += _inv_m*get_mass(nd, f)*norm2(dx) / 2;
	}
	return e;
}

void gNormalOpt::obj_grad(const double *x, double *grad) const {
	for (int n = 0; n < _zone->_nodes.size(); n++) {
		uint nd = _zone->_nodes[n];
		bool f = _zone->_frees[n];
		Vec3 dx = get_x(nd, f) - get_xold(nd, f);
		set_subvec(grad, n, double(_inv_m)*get_mass(nd, f)*dx);
	}
}

double gNormalOpt::constraint(const double *x, int j, int &sign) const
{
	sign = 1;
	REAL c = -::thickness;
	const g_impact &impact = _zone->_impacts[j];
	for (int n = 0; n < 4; n++) {
		uint nd = impact._nodes[n];
		bool f = impact._frees[n];
		Vec3 x = get_x(nd, f);
		c += impact._w[n] * dot(impact._n, x);
	}
	return c;
}

void
gNormalOpt::con_grad(const double *x, int j, double factor, double *grad) const
{
	const g_impact &impact = _zone->_impacts[j];

	for (int n = 0; n < 4; n++) {
		int i = find(impact._nodes[n], _zone->_nodes);
		if (i != -1)
			add_subvec(grad, i, double(double(factor)*impact._w[n]) * impact._n);
	}
}

void gNormalOpt::finalize(const double *x) const {
	precompute(x);
}
