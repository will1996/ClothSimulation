#include "simulation.hpp"
#include "magic.hpp"
#include "real.hpp"
#include "vectors.hpp"
#include "save.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <helper_cuda.h>

#define ONE_PASS_CD
#ifndef _WIN32
#define FORCEINLINE inline
#endif

std::vector<Vec3> totalVert;
static int tidx;

extern Simulation *glSim;
extern void init_aux_gpu();
extern void set_cache_perf_gpu();
extern void push_eq_cstr_gpu(int nid, REAL *x, REAL *n, REAL stiff, int i);
extern void check_eq_cstrs_gpu();
void save_gpu(const Simulation &sim, int frame);
void physics_step_gpu(int max_iter, REAL dt, REAL mrt, REAL mpt, bool dynamic, bool jacobi);
void get_collisions_gpu(REAL dt, REAL mu, REAL mu_obs, REAL mrt, REAL mcs, bool self_cd);
void collision_response_gpu(REAL dt, REAL mu, REAL mu_obs, REAL mrt, REAL mcs);
void next_step_mesh_gpu(REAL mrt);
void check_gpu();

void check_step(Simulation &sim) {
	NULL;
}

extern void init_glues_gpu(int);
extern void push_glue_gpu(int, int, REAL *, REAL *, REAL, int);

extern void init_constraints_gpu();
extern void init_impacts_gpu();
extern void init_handles_gpu(int);
extern void push_num_gpu(int, int);
extern void set_current_gpu(int, bool);
extern void init_pairs_gpu();

extern void pop_cloth_gpu(int, REAL *, REAL *);
extern void push_node_gpu(int, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, int, int *, int *);
extern void push_node_gpu(int, REAL *, REAL);
extern void push_face_gpu(int num, void *nods, void *vrts, void *edgs, REAL *nrms, REAL *a, REAL *m, REAL *dm, REAL *idm, int *midx, int mn);
extern void push_edge_gpu(int num, void *n, void *f, REAL *t, REAL *l, REAL *i, REAL *r);
extern void push_vert_gpu(int, REAL *, int *, int *, int *, int);
extern void merge_obstacles_gpu();
extern void merge_clothes_gpu();

extern void inc_node_gpu(int, int, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *);
extern void inc_edge_gpu(int idx, int num, REAL *theta);
extern void inc_face_gpu(int idx, int num, REAL *fn);

void update_mesh_gpu(int idx, Mesh &mesh)
{
	{
		int num = mesh.edges.size();
		REAL *t = new REAL[num];

		for (int i = 0; i<num; i++)
			t[i] = mesh.edges[i]->theta;

		inc_edge_gpu(idx, num, t);
		delete[] t;
	}
	{
		int num = mesh.faces.size();
		Vec3 *fn = new Vec3[num];

		for (int i = 0; i<num; i++)
			fn[i] = mesh.faces[i]->n;

		inc_face_gpu(idx, num, (REAL *)fn);
		delete[] fn;
	}
	{
		int num = mesh.nodes.size();
		Vec3 *x = new Vec3[num];
		Vec3 *x0 = new Vec3[num];
		Vec3 *v = new Vec3[num];
		REAL *a = new REAL[num];
		REAL *m = new REAL[num];
		Vec3 *nn = new Vec3[num];

		for (int i = 0; i<num; i++) {
			Node *n = mesh.nodes[i];
			x[i] = n->x;
			x0[i] = n->x0;
			v[i] = n->v;
			a[i] = n->a;
			m[i] = n->m;
			nn[i] = n->n;
		}

		// these are updated in compute_ws_data
		//fn, edge->theta, nn,
		inc_node_gpu(idx, num, (REAL *)x, (REAL *)x0, (REAL *)v, a, m, (REAL *)nn);

		delete[] x;
		delete[] x0;
		delete[] v;
		delete[] a;
		delete[] m;
		delete[] nn;
	}
}

void push_node_gpu(Mesh &mesh, REAL dt)
{
	// nodes
	{
		int num = mesh.nodes.size();
		if (mesh._x == NULL) {
			mesh._x = new Vec3[num];
			mesh._x0 = new Vec3[num];
		}

		for (int i = 0; i<num; i++) {
			Node *n = mesh.nodes[i];
			mesh._x[i] = n->x;
		}

		push_node_gpu(num, (REAL *)mesh._x, dt);
	}
}

void pop_mesh_gpu(Mesh &mesh)
{
	int num = mesh.nodes.size();
	Vec3 *hx = new Vec3[num];

	pop_cloth_gpu(num, (REAL *)hx, NULL);

	for (int i = 0; i<num; i++) {
		Node *n = mesh.nodes[i];
		n->x = hx[i];
	}

	delete[] hx;
}

void push_mesh_gpu(Mesh &mesh, int mtrNum)
{
	// faces
	{
		int num = mesh.faces.size();

		int *nods = new int[num * 3];
		int *vrts = new int[num * 3];
		int *edgs = new int[num * 3];
		Vec3 *nrms = new Vec3[num];
		REAL *a = new REAL[num];
		REAL *m = new REAL[num];
		Mat2x2 *dm = new Mat2x2[num];
		Mat2x2 *idm = new Mat2x2[num];
		int *midx = new int[num];
		int mn = mtrNum;

		for (int i = 0; i<num; i++) {
			Face *f = mesh.faces[i];

			nods[i * 3 + 0] = f->v[0]->node->index;
			nods[i * 3 + 1] = f->v[1]->node->index;
			nods[i * 3 + 2] = f->v[2]->node->index;
			vrts[i * 3 + 0] = f->v[0]->index;
			vrts[i * 3 + 1] = f->v[1]->index;
			vrts[i * 3 + 2] = f->v[2]->index;
			edgs[i * 3 + 0] = f->adje[0]->index;
			edgs[i * 3 + 1] = f->adje[1]->index;
			edgs[i * 3 + 2] = f->adje[2]->index;
			nrms[i] = f->n;
			a[i] = f->a;
			m[i] = f->m;
			dm[i] = f->Dm;
			idm[i] = f->invDm;
			midx[i] = f->label;
		}

		push_face_gpu(num, nods, vrts, edgs, (REAL *)nrms, a, m, (REAL *)dm, (REAL *)idm, midx, mn);

		delete[] nods;
		delete[] vrts;
		delete[] edgs;
		delete[] nrms;
		delete[] a;
		delete[] m;
		delete[] dm;
		delete[] idm;
		delete[] midx;
	}

	// edges
	{
		int num = mesh.edges.size();
		int *n = new int[num * 2];
		int *f = new int[num * 2];
		REAL *theta = new REAL[num];
		REAL *len = new REAL[num];
		REAL *itheta = new REAL[num];
		REAL *ref = new REAL[num];

		for (int i = 0; i<num; i++) {
			Edge *e = mesh.edges[i];

			n[i * 2 + 0] = e->n[0]->index;
			n[i * 2 + 1] = e->n[1]->index;
			f[i * 2 + 0] = e->adjf[0] ? e->adjf[0]->index : -1;
			f[i * 2 + 1] = e->adjf[1] ? e->adjf[1]->index : -1;
			theta[i] = e->theta;
			len[i] = e->l;
			itheta[i] = e->theta_ideal;
			ref[i] = e->reference_angle;
		}

		push_edge_gpu(num, n, f, theta, len, itheta, ref);
		delete[] n;
		delete[] f;
		delete[] theta;
		delete[] len;
		delete[] itheta;
		delete[] ref;
	}

	// nodes
	{
		int num = mesh.nodes.size();
		Vec3 *x = new Vec3[num];
		Vec3 *x0 = new Vec3[num];
		Vec3 *v = new Vec3[num];
		REAL *a = new REAL[num];
		REAL *m = new REAL[num];
		Vec3 *nn = new Vec3[num];

		int *n2vIdx = new int[num];
		int idx = 0;
		for (int i = 0; i<num; i++) {
			idx += mesh.nodes[i]->verts.size();
			n2vIdx[i] = idx;
		}

		int *n2vList = new int[idx];
		idx = 0;
		for (int i = 0; i<num; i++) {
			Node *n = mesh.nodes[i];
			for (int j = 0; j<n->verts.size(); j++)
				n2vList[idx++] = n->verts[j]->index;
		}

		for (int i = 0; i<num; i++) {
			Node *n = mesh.nodes[i];
			x[i] = n->x;
			x0[i] = n->x0;
			v[i] = n->v;
			a[i] = n->a;
			m[i] = n->m;
			nn[i] = n->n;
		}

		push_node_gpu(num, (REAL *)x, (REAL *)x0, (REAL *)v, a, m, (REAL *)nn, idx, n2vIdx, n2vList);

		delete[] x;
		delete[] x0;
		delete[] v;
		delete[] a;
		delete[] m;
		delete[] nn;
		delete[] n2vIdx;
		delete[] n2vList;
	}

	// vertices
	{
		int num = mesh.verts.size();
		Vec2 *vu = new Vec2[num];
		int *vn = new int[num];

		// index: l[0], l[0]+l[1], ..., l[0]+...+l[n-1]
		int *adjIdx = new int[num];
		int idx = 0;
		for (int i = 0; i<num; i++) {
			idx += mesh.verts[i]->adjf.size();
			adjIdx[i] = idx;
		}

		// all the adjacent faces 
		int *adjList = new int[idx];

		idx = 0;
		for (int i = 0; i<num; i++) {
			Vert *v = mesh.verts[i];
			vu[i] = v->u;
			vn[i] = v->node->index;
			for (int j = 0; j<v->adjf.size(); j++) {
				adjList[idx++] = v->adjf[j]->index;
			}
		}

		push_vert_gpu(num, (REAL *)vu, vn, adjIdx, adjList, idx);

		delete[] adjList;
		delete[] adjIdx;

		delete[] vu;
		delete[] vn;
	}
}

void push_node_gpu(int idx, Cloth &c, REAL dt)
{
	set_current_gpu(idx, true);
	push_node_gpu(c.mesh, dt);
}

void push_cloth_gpu(int idx, Cloth &c)
{
	set_current_gpu(idx, true);
	push_mesh_gpu(c.mesh, c.materials.size());
}

void push_obstacle_gpu(int idx, Obstacle &o)
{
	set_current_gpu(idx, false);
	push_mesh_gpu(o.curr_state_mesh, 0);
}

extern void update_bvs_gpu(bool is_cloth, bool ccd, REAL mrt);

void update_obstacle_gpu(int idx, Obstacle &o)
{
	update_mesh_gpu(idx, o.curr_state_mesh);
}


extern void push_material_gpu(const void *, const void *, int, const void *, const void *, REAL, REAL, REAL);
void push_material_gpu(const std::vector<Cloth> &c, const void *w, const void *g)
{
	int total = 0;

	for (int i = 0; i < c.size(); i++)
	{
		total += c[i].materials.size();
	}

	StretchingSamples *s = new StretchingSamples[total];
	BendingData *b = new BendingData[total];

	for (int i = 0, k = 0; i < c.size(); i++)
	{
		for (int j = 0; j < c[i].materials.size(); j++, k++)
		{
			s[k] = c[i].materials[j]->stretching;
			b[k] = c[i].materials[j]->bending;
		}
	}

	push_material_gpu(s, b, total, w, g, 0, 0, 0);

	delete[] s;
	delete[] b;
}

extern void build_mask_gpu();

void pop_cloth_gpu(Simulation &sim)
{
	static Vec3 *nodeBuffer = NULL;
	static Vec3 *forceBuffer = NULL;

	int total = 0;
	for (int n = 0; n<sim.cloths.size(); n++)
		total += sim.cloths[n].mesh.nodes.size();

	if (nodeBuffer == NULL) {
		nodeBuffer = new Vec3[total];
		forceBuffer = new Vec3[total];
	}

	pop_cloth_gpu(total, (REAL *)nodeBuffer, (REAL *)forceBuffer);

	int idx = 0;
	for (int n = 0; n<sim.cloths.size(); n++) {
		Mesh &m = sim.cloths[n].mesh;

		for (int i = 0; i<m.nodes.size(); i++) {
			Node *n = m.nodes[i];
			n->x0 = n->x;
			n->x = nodeBuffer[idx];
			n->f = forceBuffer[idx];
			idx++;
		}
	}
}

void push_nodes_gpu(Simulation &sim)
{
	for (int n = 0; n<sim.cloths.size(); n++)
		push_node_gpu(n, sim.cloths[n], sim.step_time);
}

extern void self_mesh(std::vector<Mesh *> &meshes);

#ifdef USE_NC
#include "tmbvh.hpp"
bvh *bvhC = NULL;
extern uint *cone_front_buffer;
extern uint cone_front_len;


void visualize_cones(Simulation &sim)
{
#ifdef USE_NC
	if (!bvhC) return;

	self_mesh(sim.cloth_meshes);
	bvh_node *root = bvhC->root();
	for (int i = 0; i < cone_front_len; i++) {
		uint *ptr = cone_front_buffer + i * 3;
		if (*(ptr + 2) != 0) //invalid
			continue;

		uint idx = *ptr;
		(root + idx)->visualizeBound(false);
	}

	//(root + 2000)->visualizeBound(true);
#endif
}

void visualize_cones()
{
	visualize_cones(*glSim);
}
#else
void visualize_cones()
{

}

#endif

void push_data_gpu(Simulation &sim)
{
	push_num_gpu(sim.cloths.size(), sim.obstacles.size());
	for (int n = 0; n<sim.cloths.size(); n++) {
		push_cloth_gpu(n, sim.cloths[n]);
	}
	merge_clothes_gpu();

	for (int n = 0; n<sim.obstacles.size(); n++) {
		push_obstacle_gpu(n, sim.obstacles[n]);
	}
	merge_obstacles_gpu();

	build_mask_gpu();

	push_material_gpu(sim.cloths, &sim.wind, &sim.gravity);

	if (::magic.tm_with_cd) // for collision detection stuff
	{
		std::vector<Mesh *> cMeshes, oMeshes;
		for (int n = 0; n<sim.cloths.size(); n++)
			cMeshes.push_back(&sim.cloths[n].mesh);
		for (int n = 0; n<sim.obstacles.size(); n++)
			oMeshes.push_back(&sim.obstacles[n].get_mesh());

		update_bvs_gpu(true, true, ::magic.repulsion_thickness);
		update_bvs_gpu(false, true, ::magic.repulsion_thickness);

#ifdef USE_NC
		assert(cMeshes.size() >= 1);
		bvhC = new bvh(cMeshes, true);
		bvhC->push2GPU(true);
#endif

		init_pairs_gpu();
	}
}

void update_obstacles_gpu(Simulation &sim)
{
	for (int n = 0; n<sim.obstacles.size(); n++) {
		update_obstacle_gpu(n, sim.obstacles[n]);
	}
}

extern Vec3 *getObstaclePts();

void update_obstacles(Simulation &sim, bool update_positions) {
	bool loadAnimation = (totalVert.size() > 0);

	REAL decay_time = 0.1,
		blend = sim.step_time / decay_time;
	blend = blend / (1 + blend);
	for (int o = 0; o < sim.obstacles.size(); o++) {
		if (loadAnimation){
			Mesh &mesh = sim.obstacles[o].get_mesh();

			for (int n = 0; n < mesh.nodes.size(); n++) {
				Node *node = mesh.nodes[n];
				node->x = totalVert[tidx++];
			}
			//compute_ws_data(mesh);
		}
		else {
			sim.obstacles[o].get_mesh(sim.time);

			Vec3 *pts = getObstaclePts();
			if (pts)
				sim.obstacles[o].load_mesh(pts);
			else {
				if (::magic.tm_load_ob)
					sim.obstacles[o].load_mesh(sim.time, sim.step_time, blend);
				else
					sim.obstacles[o].blend_with_previous(sim.time, sim.step_time, blend);
			}
		}

		if (!update_positions) {
			// put positions back where they were
			Mesh &mesh = sim.obstacles[o].get_mesh();
			REAL tmp = 1 / sim.step_time;
			for (int n = 0; n < mesh.nodes.size(); n++) {
				Node *node = mesh.nodes[n];
				node->v = (node->x - node->x0)*double(tmp);
				node->x = node->x0;
			}
		}
	}
}

extern void delete_constraints(const std::vector<Constraint*> &cons);

int getNodeID(Simulation &sim, Node *n)
{
	int offset = 0;
	for (int i = 0; i < sim.cloth_meshes.size(); i++)
	{
		Mesh *m = sim.cloth_meshes[i];
		for (int j = 0; j < m->nodes.size(); j++)
			if (n == m->nodes[j])
				return n->index + offset;

		offset += m->nodes.size();
	}
	assert(0);
	return -1;
}

extern void add_position_constraints(const Node *node, const Vec3 &x, REAL stiff,
	std::vector<Constraint*> &cons);

void init_handles(Simulation &sim)
{
	glSim = &sim;


	std::vector<Constraint*> eqCons;
	for (int h = 0; h < sim.handles.size(); h++)
		append(eqCons, sim.handles[h]->get_constraints(sim.time));
	//getLastCudaError("42");

	init_handles_gpu(eqCons.size());
	//getLastCudaError("12");
	for (int i = 0; i<eqCons.size(); i++) {
		EqCon *eq = (EqCon *)eqCons[i];

		push_eq_cstr_gpu(getNodeID(sim, eq->node),
			(REAL *)&(eq->x), (REAL *)&(eq->n), eq->stiff, i);
	}
	//delete_constraints(eqCons);
	//getLastCudaError("32");

	// for stitch
	std::vector<Node *> glues;
	for (int h = 0; h < sim.handles.size(); h++) {
		std::vector<Node*> nodes =
			sim.handles[h]->get_nodes();

		if (nodes.size() == 2) {// glue
			glues.push_back(nodes[0]);
			glues.push_back(nodes[1]);
		}
	}

	init_glues_gpu(glues.size() / 2);
	for (int i = 0; i < glues.size() / 2; i++) {
		Node *n0 = glues[i * 2];
		Node *n1 = glues[i * 2 + 1];
		push_glue_gpu(getNodeID(sim, n0), getNodeID(sim, n1), (REAL *)&(n0->x), (REAL *)&(n1->x), ::magic.collision_stiffness, i);
	}
}


void step_mesh(Mesh &mesh, REAL dt) {
	for (int n = 0; n < mesh.nodes.size(); n++)
		mesh.nodes[n]->x += mesh.nodes[n]->v*double(dt);
}

void collision_step_gpu(Simulation &sim)
{
	if (!sim.enabled[Simulation::Collision])
		return;
	sim.timers[Simulation::Collision].tick();

	collision_response_gpu(sim.step_time, sim.friction, sim.obs_friction, ::magic.repulsion_thickness, ::magic.collision_stiffness);

	sim.timers[Simulation::Collision].tock();
}

extern std::string outprefix;

int load_obj_vertices(const std::string &filename, std::vector<Vec3> &verts);
void load_obj(Mesh &mesh, const std::string &filename);

void load_animation()
{
	char buffer[512];

//#define READ_TEX_DATA
#ifdef READ_TEX_DATA
	Mesh m;
	int frame = 0;

	load_obj(m, stringf("%s/%04d_ob.obj", outprefix.c_str(), frame));
	int numVert = m.nodes.size() * 900;
	for (int i = 0; i < m.nodes.size(); i++)
		totalVert.push_back(m.nodes[i]->x);


	for (int frame = 1; frame<900; frame++) {
		load_obj_vertices(stringf("%s/%04d_ob.obj", outprefix.c_str(), frame), totalVert);
	}

	FILE *fp = fopen(stringf("%s/vert.dat", outprefix.c_str()).c_str(), "wb");
	int sz = totalVert.size();
	fwrite(&sz, sizeof(int), 1, fp);
	fwrite(reinterpret_cast<const char*>(&totalVert[0]), sizeof(Vec3), sz, fp);
	fclose(fp);
#else
	sprintf(buffer, "%s/vert.dat", outprefix.c_str());
	FILE *fp = fopen(buffer, "rb");
	if (!fp) return;

	int sz;
	fread(&sz, sizeof(int), 1, fp);
	totalVert.resize(sz);
	fread(reinterpret_cast<char*>(&totalVert[0]), sizeof(Vec3), sz, fp);
	fclose(fp);
#endif
}

void clear_data_gpu();

void advance_step_gpu(Simulation &sim)
{
	if (!sim.is_in_gpu) {
		clear_data_gpu();

		push_data_gpu(sim);
		init_constraints_gpu();
		init_impacts_gpu();
		init_aux_gpu();

		save_gpu(sim, sim.frame);

		set_cache_perf_gpu();
		load_animation();
	}

	sim.time += sim.step_time;
	sim.step++;

	TIMING_BEGIN
	//if (!sim.is_in_gpu) 
	{
		TIMING_BEGIN
		update_obstacles(sim, false);
		//getLastCudaError("0");
		TIMING_END("%%%update_obstacles1")

		TIMING_BEGIN
		update_obstacles_gpu(sim);
		//getLastCudaError("1");
		TIMING_END("%%%update_obstacles2")
	}
	TIMING_END("###update_obstacles")

	TIMING_BEGIN
	init_handles(sim);
	//getLastCudaError("2");

#ifdef ONE_PASS_CD
	if (!sim.is_in_gpu)
#endif
	if (sim.enabled[Simulation::Proximity])
		get_collisions_gpu(sim.step_time, sim.friction, sim.obs_friction, ::magic.repulsion_thickness, ::magic.collision_stiffness, ::magic.tm_self_cd);

	TIMING_END("###get_collisions_gpu")

	TIMING_BEGIN
	if (::magic.tm_with_ti)
		physics_step_gpu(::magic.tm_iterations, sim.step_time, ::magic.repulsion_thickness, 
			::magic.projection_thickness, true, ::magic.tm_jacobi_preconditioner);

	// updating x of obstacles...
	for (int o = 0; o < sim.obstacle_meshes.size(); o++)
		step_mesh(*sim.obstacle_meshes[o], sim.step_time);
	TIMING_END("###physics_step")

	TIMING_BEGIN
	if (sim.enabled[Simulation::Collision])
		collision_step_gpu(sim);
	TIMING_END("###collision_step")

	TIMING_BEGIN
	// updating ws & x0, as the final step
	next_step_mesh_gpu(::magic.repulsion_thickness);

	// updating x0 of obstacles ...
	for (int o = 0; o < sim.obstacle_meshes.size(); o++) {
		compute_ws_data(*sim.obstacle_meshes[o]);
		update_x0(*sim.obstacle_meshes[o]);
	}

	// for rendering ...
	pop_cloth_gpu(sim);

	check_step(sim);
	check_gpu();
	TIMING_END("###recap ...")

	sim.is_in_gpu = true;
	if (sim.step % sim.frame_steps == 0) {
		sim.frame++;
	}
}

#ifdef USE_NC
#else
//#########################################################
class vec3f {
public:
	union {
		struct {
			double x, y, z;
		};
		struct {
			double v[3];
		};
	};
public:

	FORCEINLINE vec3f()
	{
		x = 0; y = 0; z = 0;
	}

	FORCEINLINE vec3f(const vec3f &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}

	FORCEINLINE vec3f(const double *v)
	{
		x = v[0];
		y = v[1];
		z = v[2];
	}

	FORCEINLINE vec3f(double x, double y, double z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	FORCEINLINE double operator [] (int i) const { return v[i]; }
	FORCEINLINE double &operator [] (int i) { return v[i]; }

#ifdef VEC_OPS
	FORCEINLINE vec3f &operator += (const vec3f &v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	FORCEINLINE vec3f &operator -= (const vec3f &v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	FORCEINLINE vec3f &operator *= (double t) {
		x *= t;
		y *= t;
		z *= t;
		return *this;
	}

	FORCEINLINE vec3f &operator /= (double t) {
		x /= t;
		y /= t;
		z /= t;
		return *this;
	}

	FORCEINLINE void negate() {
		x = -x;
		y = -y;
		z = -z;
	}

	FORCEINLINE vec3f operator - () const {
		return vec3f(-x, -y, -z);
	}

	FORCEINLINE vec3f operator+ (const vec3f &v) const
	{
		return vec3f(x + v.x, y + v.y, z + v.z);
	}

	FORCEINLINE vec3f operator- (const vec3f &v) const
	{
		return vec3f(x - v.x, y - v.y, z - v.z);
	}

	FORCEINLINE vec3f operator *(double t) const
	{
		return vec3f(x*t, y*t, z*t);
	}

	FORCEINLINE vec3f operator /(double t) const
	{
		return vec3f(x / t, y / t, z / t);
	}

	// cross product
	FORCEINLINE const vec3f cross(const vec3f &vec) const
	{
		return vec3f(y*vec.z - z*vec.y, z*vec.x - x*vec.z, x*vec.y - y*vec.x);
	}

	FORCEINLINE double dot(const vec3f &vec) const {
		return x*vec.x + y*vec.y + z*vec.z;
	}

	FORCEINLINE void normalize()
	{
		double sum = x*x + y*y + z*z;
		if (sum > GLH_EPSILON_2) {
			double base = double(1.0 / sqrt(sum));
			x *= base;
			y *= base;
			z *= base;
		}
	}

	FORCEINLINE double length() const {
		return double(sqrt(x*x + y*y + z*z));
	}

	FORCEINLINE vec3f getUnit() const {
		return (*this) / length();
	}

	FORCEINLINE bool isUnit() const {
		return isEqual(squareLength(), 1.f);
	}

	//! max(|x|,|y|,|z|)
	FORCEINLINE double infinityNorm() const
	{
		return fmax(fmax(fabs(x), fabs(y)), fabs(z));
	}

	FORCEINLINE vec3f & set_value(const double &vx, const double &vy, const double &vz)
	{
		x = vx; y = vy; z = vz; return *this;
	}

	FORCEINLINE bool operator == (const vec3f &other) const {
		return equal_abs(other);
	}

	FORCEINLINE bool equal_abs(const vec3f &other) const {
		return x == other.x && y == other.y && z == other.z;
	}

	FORCEINLINE double squareLength() const {
		return x*x + y*y + z*z;
	}

	static vec3f zero() {
		return vec3f(0.f, 0.f, 0.f);
	}

	//! Named constructor: retrieve vector for nth axis
	static vec3f axis(int n) {
		assert(n < 3);
		switch (n) {
		case 0: {
			return xAxis();
		}
		case 1: {
			return yAxis();
		}
		case 2: {
			return zAxis();
		}
		}
		return vec3f();
	}

	//! Named constructor: retrieve vector for x axis
	static vec3f xAxis() { return vec3f(1.f, 0.f, 0.f); }
	//! Named constructor: retrieve vector for y axis
	static vec3f yAxis() { return vec3f(0.f, 1.f, 0.f); }
	//! Named constructor: retrieve vector for z axis
	static vec3f zAxis() { return vec3f(0.f, 0.f, 1.f); }
#endif
};
#endif

class tri3f {
public:
	unsigned int _ids[3];

	FORCEINLINE tri3f() {
		_ids[0] = _ids[1] = _ids[2] = -1;
	}

	FORCEINLINE tri3f(unsigned int id0, unsigned int id1, unsigned int id2) {
		set(id0, id1, id2);
	}

	FORCEINLINE void set(unsigned int id0, unsigned int id1, unsigned int id2) {
		_ids[0] = id0;
		_ids[1] = id1;
		_ids[2] = id2;
	}

	FORCEINLINE unsigned int id(int i) { return _ids[i]; }
	FORCEINLINE unsigned int id0() { return _ids[0]; }
	FORCEINLINE unsigned int id1() { return _ids[1]; }
	FORCEINLINE unsigned int id2() { return _ids[2]; }

	FORCEINLINE bool operator == (const tri3f &a) {
		return _ids[0] == a._ids[0] && _ids[1] == a._ids[1] && _ids[2] == a._ids[2];
	}
};

#ifdef USE_NC
#else
//remove unused vertices ...
bool compressobjfile(const char *path, const char *opath)
{
	std::vector<tri3f> triset, vtriset;
	std::vector<vec3f> vtxset;
	std::vector<vec3f> nodset;

	{ // reading
		FILE *fp = fopen(path, "rt");
		if (fp == NULL) return false;

		char buf[1024];
		while (fgets(buf, 1024, fp)) {
			if (buf[0] == 'v' && buf[1] == 't') {
				double x, y, z = 0;
				sscanf(buf + 3, "%lf%lf%lf", &x, &y);
				vtxset.push_back(vec3f(x, y, z));
			}
			else

				if (buf[0] == 'v' && buf[1] == ' ') {
					double x, y, z;
					sscanf(buf + 2, "%lf%lf%lf", &x, &y, &z);
					nodset.push_back(vec3f(x, y, z));
				}
				else
					if (buf[0] == 'f' && buf[1] == ' ') {
						int id0, id1, id2;
						int vid0, vid1, vid2;

						sscanf(buf + 2, "%d/%d", &id0, &vid0);
						char *nxt = strchr(buf + 2, ' ');
						sscanf(nxt + 1, "%d/%d", &id1, &vid1);
						nxt = strchr(nxt + 1, ' ');
						sscanf(nxt + 1, "%d/%d", &id2, &vid2);

						triset.push_back(tri3f(id0 - 1, id1 - 1, id2 - 1));
						vtriset.push_back(tri3f(vid0 - 1, vid1 - 1, vid2 - 1));
					}
		}

		fclose(fp);
	}

	if (triset.size() == 0 || vtxset.size() == 0)
		return false;

	unsigned int numVtx = vtxset.size();
	bool *vflags = new bool[numVtx];
	unsigned int *vidxs = new unsigned int[numVtx];

	for (unsigned int i = 0; i<numVtx; i++) {
		vflags[i] = false;
	}

	unsigned int numNode = nodset.size();
	bool *nflags = new bool[numNode];
	unsigned int *nidxs = new unsigned int[numNode];

	for (unsigned int i = 0; i<numNode; i++) {
		nflags[i] = false;
	}

	unsigned int numTri = triset.size();
	for (unsigned int i = 0; i<numTri; i++) {
		tri3f &nt = triset[i];
		nflags[nt.id0()] = true;
		nflags[nt.id1()] = true;
		nflags[nt.id2()] = true;

		tri3f &vt = vtriset[i];
		vflags[vt.id0()] = true;
		vflags[vt.id1()] = true;
		vflags[vt.id2()] = true;
	}

	int count = 0;
	for (unsigned int i = 0; i<numVtx; i++) {
		if (vflags[i])
			vidxs[i] = count++;
		else
			vidxs[i] = -1;
	}

	count = 0;
	for (unsigned int i = 0; i<numNode; i++) {
		if (nflags[i])
			nidxs[i] = count++;
		else
			nidxs[i] = -1;
	}

	// output stage
	{
		FILE *fp = fopen(opath, "wt");
		if (fp == NULL) return false;

		for (unsigned int i = 0; i<numVtx; i++) {
			if (vflags[i])
				fprintf(fp, "vt %f %f\n", vtxset[i].x, vtxset[i].y);
		}

		for (unsigned int i = 0; i<numNode; i++) {
			if (nflags[i])
				fprintf(fp, "v %f %f %f\n", nodset[i].x, nodset[i].y, nodset[i].z);
		}

		for (unsigned int i = 0; i<numTri; i++) {
			tri3f &nt = triset[i];
			unsigned int nid0 = nidxs[nt.id0()];
			unsigned int nid1 = nidxs[nt.id1()];
			unsigned int nid2 = nidxs[nt.id2()];

			tri3f &vt = vtriset[i];
			unsigned int vid0 = vidxs[vt.id0()];
			unsigned int vid1 = vidxs[vt.id1()];
			unsigned int vid2 = vidxs[vt.id2()];

			fprintf(fp, "f %d/%d %d/%d %d/%d\n", nid0 + 1, vid0 + 1, nid1 + 1, vid1 + 1, nid2 + 1, vid2 + 1);
		}

		fclose(fp);
	}
	return true;
}

void mergeVtxs(const char *ipath, const char *opath)
{
	std::map<int, int> pairs;

	{//reading pairs
		for (int h = 0; h < glSim->handles.size(); h++) {
			Handle *ptr = glSim->handles[h];
			std::vector<Node *> nodes = ptr->get_nodes();
			if (nodes.size() == 2) {
				int id1 = nodes[0]->index;
				int id2 = nodes[1]->index;

				assert(id1 != id2);

				if (id1 < id2)
					std::swap(id1, id2);

				if (pairs.find(id1) != pairs.end())
					printf("Duplicated items\n");

				pairs[id1] = id2;
			}
		}
	}

	{// replacing
		FILE *ifp = fopen(ipath, "rt");
		assert(ifp != NULL);

		FILE *ofp = fopen(opath, "wt");
		assert(ofp != NULL);

		char buf[1024];
		while (fgets(buf, 1024, ifp)) {
			if (buf[0] == 'f' && buf[1] == ' ') {
				int id0, id1, id2;
				int t0, t1, t2;
				sscanf(buf + 2, "%d/%d", &id0, &t0);
				char *nxt = strchr(buf + 2, ' ');
				sscanf(nxt + 1, "%d/%d", &id1, &t1);
				nxt = strchr(nxt + 1, ' ');
				sscanf(nxt + 1, "%d/%d", &id2, &t2);

				fprintf(ofp, "f %d/%d %d/%d %d/%d\n",
					(pairs.find(id0 - 1) != pairs.end()) ? pairs[id0 - 1] + 1 : id0, t0,
					(pairs.find(id1 - 1) != pairs.end()) ? pairs[id1 - 1] + 1 : id1, t1,
					(pairs.find(id2 - 1) != pairs.end()) ? pairs[id2 - 1] + 1 : id2, t2);
			}
			else
				fputs(buf, ofp);
		}

		fclose(ofp);
		fclose(ifp);
	}
}

void save_obj(const Mesh &mesh, const std::string &filename);
void load_obj(Mesh &mesh, const std::string &filename);
void clear_data_gpu();
void init_resume(char *path, int st);
bool sim_step_gpu();

extern Simulation sim;

void moveMesh()
{
	Mesh &mesh = sim.cloths[0].mesh;

	for (int n = 0; n < mesh.nodes.size(); n++) {
		Node *node = mesh.nodes[n];
		Vec3 &pt = node->x;
		pt[2] += 0.01;
		node->x0 = node->x;
	}
}

bool
LineTriangleIntersect(Vec3 &start, Vec3 &end, Vec3 &v0, Vec3 &v1, Vec3 &v2)
{
	const float epsilon = 0.000001f;

	Vec3 dir = end - start;
	Vec3 e1 = v1 - v0;
	Vec3 e2 = v2 - v0;
	Vec3 nrm = cross(dir, e2);
	REAL tmp = dot(nrm, e1);
	if (tmp > -epsilon && tmp < epsilon)
		return false;

	tmp = 1.0f / tmp;
	Vec3 s = start - v0;
	REAL u = tmp*dot(s, nrm);
	if (u < 0.0 || u > 1.0)
		return false;

	Vec3 q = cross(s, e1);
	REAL v = tmp*dot(dir, q);
	if (v < 0.0 || v > 1.0)
		return false;

	if (u + v > 1.0)
		return false;

	REAL t = tmp * dot(e2, q);
	if (t < 0.0 || t > 1.0)
		return false;

	return true;
}

bool bodyColliding(Vec3 &pt0, Vec3 &pt1)
{
	for (int o = 0; o < sim.obstacle_meshes.size(); o++) {
		Mesh *ob = sim.obstacle_meshes[o];
		for (int f = 0; f < ob->faces.size(); f++) {
			Face *of = ob->faces[f];
			Vec3 v0 = of->v[0]->node->x;
			Vec3 v1 = of->v[1]->node->x;
			Vec3 v2 = of->v[2]->node->x;
			
			if (LineTriangleIntersect(pt0, pt1, v0, v1, v2))
				return true;
		}
	}

	return false;
}

#endif
