
#include "io.hpp"
#include "obstacle.hpp"
#include "util.hpp"
#include <cstdio>
#include "magic.hpp"
#include <assert.h>

using namespace std;
extern int stFrame;

Mesh& Obstacle::get_mesh() {
    return curr_state_mesh;
}

const Mesh& Obstacle::get_mesh() const {
    return curr_state_mesh;
}

Mesh& Obstacle::get_mesh(REAL time) {
    if (time > end_time)
        delete_mesh(curr_state_mesh);
    if (time < start_time || time > end_time)
        return curr_state_mesh;
    if (!activated)
        curr_state_mesh = deep_copy(base_mesh);
    if (transform_spline) {
        DTransformation dtrans = get_dtrans(*transform_spline, time);
        Mesh &mesh = curr_state_mesh;
        for (int n = 0; n < curr_state_mesh.nodes.size(); n++)
            mesh.nodes[n]->x = apply_dtrans(dtrans, base_mesh.nodes[n]->x,
                                            &mesh.nodes[n]->v);
        compute_ws_data(mesh);
    }
    if (!activated)
        update_x0(curr_state_mesh);
    activated = true;
    return curr_state_mesh;
}

void Obstacle::blend_with_previous (REAL t, REAL dt, REAL blend) {
    const Motion *spline = transform_spline;

	if (spline == NULL)
		return;

    Transformation trans = (spline)
                         ? get_trans(*spline, t)
                           * inverse(get_trans(*spline, t-dt))
                         : identityT();
    Mesh &mesh = curr_state_mesh;
    for (int n = 0; n < mesh.nodes.size(); n++) {
        Node *node = mesh.nodes[n];
        Vec3 x0 = trans.apply(node->x0);
        node->x = x0 + blend*(node->x - x0);
    }
    compute_ws_data(mesh);
}

void
Obstacle::load_mesh(REAL t, REAL dt, REAL blend)
{
#define READ_FRAME_BY_FRAME
#ifdef READ_FRAME_BY_FRAME
	Mesh &mesh = curr_state_mesh;
	char obuf[512];
	vector<Vec3> verts;

	int obstacle = ::magic.tm_load_ob;
	assert(obstacle != 0);

	if (obstacle == -1)
		return;

	if (obstacle == 1) {
		// for bishop
		//static int id = 400;
		static int id = 0;
		//sprintf(obuf, "E:\\work\\cudaCloth-6\\data\\meshes\\bishop-meshes-10split\\body%03d.obj", id++);
		sprintf(obuf, ".\\meshes\\bishop-meshes-30split\\body%03d.obj", id++);
	}
	else if (obstacle == 2) {
		// for andy
		static int id = 0;
		sprintf(obuf, ".\\meshes\\Andy-karate-meshes-8split\\body%03d.obj", id++);
	}
	else if (obstacle == 3) {
		//for kneel_man
		static int id = 0;
		sprintf(obuf, ".\\meshes\\man_kneeling_5split\\body%03d.obj", id++);
	}
	else if (obstacle == 4) {
		// for qman
		//static int id = 130;
		static int id = 200;
		sprintf(obuf, ".\\meshes\\qman3-3split\\body%03d.obj", id++);
	}
	else if (obstacle == 5) {
		// for victor-woman
		static int id = 0;
		//sprintf(obuf, "E:\\work\\cudaCloth-5.5\\meshes\\qwoman2-trimmed\\body%03d.obj", id++);
		sprintf(obuf, ".\\meshes\\qwoman2-trimmed-2split\\body%03d.obj", id++);
	}
	else if (obstacle == 6) {
		// for bridson-5x
		static int id;
		static bool first = true;

		if (first){
			id = stFrame;
			first = false;
		}

		sprintf(obuf, ".\\meshes\\bridson-sphere-scaled-8x\\%04d_ob.obj", id++);
	}
	else if (obstacle == 7) {
		static int id;
		static bool first = true;

		if (first){
			id = stFrame;
			first = false;
		}

		sprintf(obuf, ".\\meshes\\bodymodel\\body%04d.obj", id++);
	}


	load_obj_vertices(obuf, verts);
	if (verts.size() != mesh.nodes.size()) {
		printf("loading obstacle error!\n...");
		exit(0);
	}
	else
		printf("loading obstacle %s\n...", obuf);


	for (int n = 0; n < mesh.nodes.size(); n++) {
		Node *node = mesh.nodes[n];
		node->x = verts[n];
	}
	compute_ws_data(mesh);
#else
	static vector<Vec3> verts;
	static bool first=true;
	static int idx = 0;

	Mesh &mesh = curr_state_mesh;
	int numVert = mesh.nodes.size();

	if (first) {
		int obstacle = ::magic.tm_load_ob;
		assert(obstacle != 0);

		int sid=0;
		char obuf[512];

		if (obstacle == 1) {
			// for bishop
			sid = 0;
			strcpy(obuf, ".\\meshes\\bishop-meshes-30split\\body%03d.obj");
		}
		else if (obstacle == 2) {
			// for andy
			sid = 0;
			strcpy(obuf, ".\\meshes\\Andy-karate-meshes-8split\\body%03d.obj");
		}
		else if (obstacle == 3) {
			//for kneel_man
			sid = 0;
			strcpy(obuf, ".\\meshes\\man_kneeling_5split\\body%03d.obj");
		}
		else if (obstacle == 4) {
			// for qman
			//static int id = 130;
			sid = 200;
			strcpy(obuf, ".\\meshes\\qman3-3split\\body%03d.obj");
		}
		else if (obstacle == 5) {
			// for victor-woman
			sid = 0;
			strcpy(obuf, ".\\meshes\\qwoman2-trimmed-2split\\body%03d.obj");
		}


		int id = 0;
		char oobuf[512];
		do {
			sprintf(oobuf, obuf, sid+(id++));
			int count = load_obj_vertices(oobuf, verts);
			if (count == 0) break; //ending ...
			if (count != numVert) {
				printf("loading obstacle error!\n...");
				exit(0);
			}
			else
				printf("loading obstacle %s\n...", oobuf);
		} while (1);

		first = false;
	}


	for (int n = 0; n < numVert; n++) {
		Node *node = mesh.nodes[n];
		node->x = verts[n+idx];
	}

	idx += numVert;
	compute_ws_data(mesh);
#endif
}


void
Obstacle::load_mesh(Vec3 *pts)
{
	Mesh &mesh = curr_state_mesh;

	for (int n = 0; n < mesh.nodes.size(); n++) {
		Node *node = mesh.nodes[n];
		node->x = pts[n];
	}
	compute_ws_data(mesh);
}
