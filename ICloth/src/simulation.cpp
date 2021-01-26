#if defined(_WIN32)
#include <Windows.h>
#endif

#include <GL/gl.h>

#include <stdio.h>
#include <string>
#include <stdarg.h>

#include "simulation.hpp"
#include "conf.hpp"
#include "io.hpp"
#include "save.hpp"
#include "geometry.hpp"
#include "magic.hpp"
#include <math.h>


//#include "separateobs.hpp"
//#include "equilibration.hpp"

Simulation sim;
static vector<Mesh*> &meshes = sim.cloth_meshes;

Simulation *glSim;

string outprefix;

// Helper functions

template <typename Prim> int size(const vector<Mesh*> &meshes) {
	int np = 0;
	for (int m = 0; m < meshes.size(); m++) np += get<Prim>(*meshes[m]).size();
	return np;
}
template int size<Vert>(const vector<Mesh*>&);
template int size<Node>(const vector<Mesh*>&);
template int size<Edge>(const vector<Mesh*>&);
template int size<Face>(const vector<Mesh*>&);

template <typename Prim> int get_index(const Prim *p,
	const vector<Mesh*> &meshes) {
	int i = 0;
	for (int m = 0; m < meshes.size(); m++) {
		const vector<Prim*> &ps = get<Prim>(*meshes[m]);
		if (p->index < ps.size() && p == ps[p->index])
			return i + p->index;
		else i += ps.size();
	}
	return -1;
}
template int get_index(const Vert*, const vector<Mesh*>&);
template int get_index(const Node*, const vector<Mesh*>&);
template int get_index(const Edge*, const vector<Mesh*>&);
template int get_index(const Face*, const vector<Mesh*>&);

template <typename Prim> Prim *get(int i, const vector<Mesh*> &meshes) {
	for (int m = 0; m < meshes.size(); m++) {
		const vector<Prim*> &ps = get<Prim>(*meshes[m]);
		if (i < ps.size())
			return ps[i];
		else
			i -= ps.size();
	}
	return NULL;
}
template Vert *get(int, const vector<Mesh*>&);
template Node *get(int, const vector<Mesh*>&);
template Edge *get(int, const vector<Mesh*>&);
template Face *get(int, const vector<Mesh*>&);

vector<Vec3> node_positions(const vector<Mesh*> &meshes) {
	vector<Vec3> xs(size<Node>(meshes));
	for (int n = 0; n < xs.size(); n++)
		xs[n] = get<Node>(n, meshes)->x;
	return xs;
}

//#######################################

void prepare(Simulation &sim)
{
	sim.cloth_meshes.resize(sim.cloths.size());
	for (int c = 0; c < sim.cloths.size(); c++) {
		compute_masses(sim.cloths[c]);
		sim.cloth_meshes[c] = &sim.cloths[c].mesh;
		update_x0(*sim.cloth_meshes[c]);
	}
	sim.obstacle_meshes.resize(sim.obstacles.size());
	for (int o = 0; o < sim.obstacles.size(); o++) {
		sim.obstacle_meshes[o] = &sim.obstacles[o].get_mesh();
		update_x0(*sim.obstacle_meshes[o]);
	}

	//for (int h=0; h<sim.handles.size(); h++)
	//	sim.handles[h]->resume(sim.cloth_meshes[0]);

	for (int h = 0; h<sim.handles.size(); h++)
		sim.handles[h]->resume(sim.cloth_meshes);
}

void init_physics(const string &json_file, string outprefix, bool is_reloading)
{
	::outprefix = outprefix;

	TIMING_BEGIN
	load_json(json_file, sim);
	prepare(sim);
	TIMING_END("Init_physics")
}

void init_resume(char *path, int st)
{
	char buffer[512];
	string outprefix(path);
	string jsfile = outprefix + "/conf.json";
	init_physics(jsfile, path, true);
	//printf("after init_physics\n");

	// Get the initialization information
	sim.frame = st;
	sim.time = sim.frame * sim.frame_time;
	sim.step = sim.frame * sim.frame_steps;

	if (::magic.tm_load_ob) {
		sprintf(buffer, "%s/%04d_ob.obj", outprefix.c_str(), sim.frame);

		load_obj(*sim.obstacle_meshes[0], buffer);
	}
	else
	for (int i = 0; i<sim.obstacles.size(); ++i)
		sim.obstacles[i].get_mesh(sim.time);

	sprintf(buffer, "%s/%04d", outprefix.c_str(), sim.frame);
	load_objs(sim.cloth_meshes, buffer);
	//printf("after load_objs\n");

	prepare(sim); // re-prepare the new cloth meshes
	//printf("after prepare\n");

	sim.is_in_gpu = false;
}

extern void cudaQuit();
extern bool b[];

extern double g_timing_start;

bool sim_step_gpu()
{
	TIMING_BEGIN1
	advance_step_gpu(sim);

	printf("sim.step = %d\n", sim.step);
	TIMING_END1("#########advance_step")

	if (::magic.tm_output_file
		&& sim.step % sim.frame_steps == 0) {
		cout << "saving frame " << sim.frame << " ..." << endl;

		save(sim, sim.frame);
		save_gpu(sim, sim.frame);
	}

	if (sim.time >= sim.end_time || sim.frame >= sim.end_frame) {
		//double timing_finish = omp_get_wtime(); 
		//double timing_duration = timing_finish - g_timing_start;
		//printf("%d: %2.5f seconds\n", sim.frame-1, timing_duration);

		exit(EXIT_SUCCESS);
	}

	return true;
}

//############################################
// display functions

Vec3 area_color2(const Face *face) {
	Vec3 x0 = face->v[0]->node->x;
	Vec3 x1 = face->v[1]->node->x;
	Vec3 x2 = face->v[2]->node->x;

	Vec3 tmp = cross(x1 - x0, x2 - x0);
	REAL area = sqrt(dot(tmp, tmp))*0.5;

	REAL cr = area / face->a;
	cr = clamp(cr, REAL(1.), REAL(1.2)) - REAL(1.0);

	if (cr < 0.1)
		return Vec3(interp(1., 0., (0.1 - cr) * 10), 1, 0);
	else
		return Vec3(1, interp(0., 1., (0.2 - cr) * 10), 0);
}

Vec3 friction_color(const Face *face) {
	REAL c0 = norm2(face->v[0]->node->f)*10e9;
	REAL c1 = norm2(face->v[1]->node->f)*10e9;
	REAL c2 = norm2(face->v[2]->node->f)*10e9;

	REAL cr = fmaxf(c0, fmaxf(c1, c2));
	cr = clamp(cr, REAL(1.), REAL(1.2)) - REAL(1.0);

	if (cr < 0.1)
		return Vec3(interp(1., 0., (0.1 - cr) * 10), 1, 0);
	else
		return Vec3(1, interp(0., 1., (0.2 - cr) * 10), 0);
}

void node_color(const Node *node) {
	REAL cr = norm2(node->f)*10e10;

	cr = clamp(cr, REAL(1.), REAL(1.2)) - REAL(1.0);

	if (cr < 0.1)
		glColor3f(interp(1., 0., (0.1 - cr) * 10), 1, 0);
	else
		glColor3f(1, interp(0., 1., (0.2 - cr) * 10), 0);
}

void normal(const Vec3 &n) {
	glNormal3d(n[0], n[1], n[2]);
}


void vertex(const Vec3 &x) {
	glVertex3d(x[0], x[1], x[2]);
}

//#define SHOW_FRICTION
extern bool b[];

template <Space s>
void draw_mesh(const Mesh &mesh, bool set_color = false) {
#ifdef SHOW_FRICTION
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_LIGHTING);

	glBegin(GL_TRIANGLES);
	for (int i = 0; i < mesh.faces.size(); i++) {
		Face *face = mesh.faces[i];
		if (i % 256 == 0) {
			glEnd();
			glBegin(GL_TRIANGLES);
		}
		normal(nor<s>(face));
		for (int v = 0; v < 3; v++) {
			node_color(face->v[v]->node);
			vertex(pos<s>(face->v[v]->node));
		}
	}
	glEnd();


	glEnable(GL_LIGHTING);

#else
	if (set_color)
		glDisable(GL_COLOR_MATERIAL);
	glBegin(GL_TRIANGLES);
	for (int i = 0; i < mesh.faces.size(); i++) {
		Face *face = mesh.faces[i];
		if (i % 256 == 0) {
			glEnd();
			glBegin(GL_TRIANGLES);
		}
		if (set_color) {
			if (!b['Z']) {
				int c = find((Mesh*)&mesh, ::meshes);
				static const float phi = (1 + sqrt(5)) / 2;
				REAL hue = c*(2 - phi) * 2 * M_PI; // golden angle
				hue = -0.6*M_PI + hue; // looks better this way :/
				if (face->label % 2 == 1) hue += M_PI;
				static Vec3 a = Vec3(0.92, -0.39, 0), b = Vec3(0.05, 0.12, -0.99);
				Vec3 frt = Vec3(0.7, 0.7, 0.7) + (a*cos(hue) + b*sin(hue))*REAL(0.3),
					bak = frt*REAL(0.5) + Vec3(0.5, 0.5, 0.5);
				float front[4] = { frt[0], frt[1], frt[2], 1 },
					back[4] = { bak[0], bak[1], bak[2], 1 };
				glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, front);
				glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, back);
				// color(area_color(face));
			}
			else {
				Vec3 cr = b['X'] ? friction_color(face) : area_color2(face);

				float front[4] = { cr[0], cr[1], cr[2], 1 };
				float back[4] = { cr[0], cr[1], cr[2], 1 };
				glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, front);
				glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, back);
			}
		}
		normal(nor<s>(face));
		for (int v = 0; v < 3; v++) {
			//node_color(face->v[v]->node);
			vertex(pos<s>(face->v[v]->node));
		}
	}
	glEnd();
	if (set_color)
		glEnable(GL_COLOR_MATERIAL);
#endif
}

template <Space s>
void draw_meshes(bool set_color = false) {
	for (int m = 0; m < meshes.size(); m++)
		draw_mesh<s>(*meshes[m], set_color);
}


template <Space s>
void draw_seam_or_boundary_edges() {
	glColor3f(0, 0, 0);
	glBegin(GL_LINES);
	for (int m = 0; m < meshes.size(); m++) {
		const Mesh &mesh = *meshes[m];
		for (int e = 0; e < mesh.edges.size(); e++) {
			const Edge *edge = mesh.edges[e];
			if (!is_seam_or_boundary(edge))
				continue;
			vertex(pos<s>(edge->n[0]));
			vertex(pos<s>(edge->n[1]));
		}
	}
	glEnd();
}


void draw_handles()
{
	glPointSize(5.f);

	for (int h = 0; h < sim.handles.size(); h++) {
		vector<Node*> nodes = sim.handles[h]->get_nodes();
		int num = nodes.size();
		glColor3f(1, 0, 0);
		glBegin(GL_POINTS);
		for (int n = 0; n < nodes.size(); n++) {
			const Node *node = nodes[n];
			vertex(nodes[n]->x);
		}
		glEnd();

		if (num == 2) {
			glColor3f(0, 0, 1);
			glBegin(GL_LINES);
			vertex(nodes[0]->x);
			vertex(nodes[1]->x);
			glEnd();
		}
	}
}

//############################################
// interface with GLUT
void checkModel()
{

}

#ifdef USE_NC
void visualize_cones(Simulation &sim);
#endif

void drawModel(bool tt, bool cone, bool mdl, bool, int level)
{
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1, 1);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	if (mdl)
	draw_meshes<WS>(true);

	if (tt) {
		glEnable(GL_CULL_FACE);
		glColor3f(0.8, 0.8, 0.8);
		for (int o = 0; o < sim.obstacles.size(); o++)
			draw_mesh<WS>(sim.obstacles[o].get_mesh());
		glDisable(GL_CULL_FACE);
	}

	glColor4d(0, 0, 0, 0.2);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	//draw_meshes<WS>();
	//draw_seam_or_boundary_edges<WS>();

	if (cone) {
#ifdef USE_NC
		glLineWidth(3.0f);
		visualize_cones(sim);
#endif

		draw_handles();
	}

	glEnable(GL_LIGHTING);


}

bool dynamicModel(char *, bool, bool)
{
	return sim_step_gpu();
}

void key_s()
{
	//printf("on s\n");
	dynamicModel(NULL, false, false);
}

void Simulation::reset()
{
	is_in_gpu = false;
	// variables
	time = 0.0;
	frame = step = 0;

	for (int i = 0; i < cloths.size(); i++)
	{
		Cloth &cloth = cloths[i];
		for (int j = 0; j < cloth.materials.size(); j++)
		{
			delete cloth.materials[j];
		}

		Mesh &mesh = cloth.mesh;
		for (int j = 0; j < mesh.verts.size(); j++)
		{
			delete mesh.verts[j];
		}
		for (int j = 0; j < mesh.nodes.size(); j++)
		{
			delete mesh.nodes[j];
		}
		for (int j = 0; j < mesh.edges.size(); j++)
		{
			delete mesh.edges[j];
		}
		for (int j = 0; j < mesh.faces.size(); j++)
		{
			delete mesh.faces[j];
		}
	}
	//cloths.swap(vector<Cloth>());
	cloths.clear();

	// constants
	frame_steps = 1;
	frame_time = step_time = 0.005f;
	end_time = end_frame = infinity;
	//motions.swap(vector<Motion>());
	motions.clear();
	for (int i = 0; i < handles.size(); i++)
	{
		if (handles[i] != NULL) delete handles[i];
	}
	//handles.swap(vector<Handle*>());
	handles.clear();

	for (int i = 0; i < obstacles.size(); i++)
	{
		Obstacle &obs = obstacles[i];

		Mesh &mesh = obs.base_mesh;
		for (int j = 0; j < mesh.verts.size(); j++)
		{
			delete mesh.verts[j];
		}
		for (int j = 0; j < mesh.nodes.size(); j++)
		{
			delete mesh.nodes[j];
		}
		for (int j = 0; j < mesh.edges.size(); j++)
		{
			delete mesh.edges[j];
		}
		for (int j = 0; j < mesh.faces.size(); j++)
		{
			delete mesh.faces[j];
		}

		mesh = obs.curr_state_mesh;
		for (int j = 0; j < mesh.verts.size(); j++)
		{
			delete mesh.verts[j];
		}
		for (int j = 0; j < mesh.nodes.size(); j++)
		{
			delete mesh.nodes[j];
		}
		for (int j = 0; j < mesh.edges.size(); j++)
		{
			delete mesh.edges[j];
		}
		for (int j = 0; j < mesh.faces.size(); j++)
		{
			delete mesh.faces[j];
		}
	}
	//obstacles.swap(vector<Obstacle>());
	obstacles.clear();

	//morphs.swap(vector<Morph>());
	morphs.clear();

	gravity = Vec3(0, 0, -9.8);
	wind.density = 0.0;
	wind.drag = 0.0;
	wind.velocity = Vec3(0, 0, 0);
	friction = 0.6;
	obs_friction = 0.3;

	// handy pointers
	//cloth_meshes.swap(vector<Mesh*>());
	cloth_meshes.clear();
	//obstacle_meshes.swap(vector<Mesh*>());
	obstacle_meshes.clear();

	for (int i = 0; i < nModules; i++) enabled[i] = true;
}

