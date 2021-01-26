#pragma once

#include "real.hpp"
#include "dde.hpp"
#include "mesh.hpp"

struct Cloth {
	Mesh mesh;
	struct Material {
		REAL density; // area density
		StretchingSamples stretching;
		BendingData bending;
		REAL damping; // stiffness-proportional damping coefficient
		REAL strain_min, strain_max; // strain limits
		REAL yield_curv, weakening; // plasticity parameters
	};
	std::vector<Material *> materials;
	struct Remeshing {
		REAL refine_angle, refine_compression, refine_velocity;
		REAL size_min, size_max; // size limits
		REAL aspect_min; // aspect ratio control
	} remeshing;
};

inline void compute_masses(Cloth &cloth)
{
	for (int v = 0; v < cloth.mesh.verts.size(); v++)
		cloth.mesh.verts[v]->m = 0;
	for (int n = 0; n < cloth.mesh.nodes.size(); n++)
		cloth.mesh.nodes[n]->m = 0;
	for (int f = 0; f < cloth.mesh.faces.size(); f++) {
		Face *face = cloth.mesh.faces[f];
		face->m = face->a * cloth.materials[face->label]->density;
		for (int v = 0; v < 3; v++) {
			face->v[v]->m += face->m / 3.;
			face->v[v]->node->m += face->m / 3.;
		}
	}
}
