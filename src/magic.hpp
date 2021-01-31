#pragma once
#include "real.hpp"

// Magic numbers and other hacks

struct Magic {
	bool fixed_high_res_mesh;
	REAL handle_stiffness, collision_stiffness;
	REAL repulsion_thickness, projection_thickness;
	REAL edge_flip_threshold;
	REAL rib_stiffening;
	bool combine_tensors;
	bool preserve_creases;

	// add by TangMin
	int tm_load_ob;
	bool tm_with_cd;
	bool tm_with_ti;
	bool tm_jacobi_preconditioner;
	bool tm_use_gpu;
	bool tm_output_file;
	int tm_iterations;
	bool tm_self_cd;
	bool tm_strain_limiting;

	Magic() :
		fixed_high_res_mesh(false),
		handle_stiffness(1e3),
		collision_stiffness(1e9),
		repulsion_thickness(1e-3),
		projection_thickness(1e-4),
		edge_flip_threshold(1e-2),
		rib_stiffening(1),
		combine_tensors(true),
		preserve_creases(false),
		tm_load_ob(0),
		tm_with_cd(true),
		tm_with_ti(true),
		tm_jacobi_preconditioner(true),
		tm_use_gpu(true),
		tm_output_file(true),
		tm_self_cd(true),
		tm_strain_limiting(false),
		tm_iterations(500) {}
};

extern Magic magic;
