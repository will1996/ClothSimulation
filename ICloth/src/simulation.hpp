#pragma once

#include <string>
#include <vector>
#include "real.hpp"
#include "vectors.hpp"
#include "cloth.hpp"
#include "handle.hpp"
#include "obstacle.hpp"
#include "morph.hpp"
#include "timer.hpp"

struct Wind {
	REAL density;
	Vec3 velocity;
	REAL drag;
};

struct Simulation {
	// variables
	REAL time;
	int frame, step;
	std::vector<Cloth> cloths;
	// constants
	int frame_steps;
	REAL frame_time, step_time;
	REAL end_time, end_frame;
	std::vector<Motion> motions;
	std::vector<Handle*> handles;
	std::vector<Obstacle> obstacles;
	std::vector<Morph> morphs;
	Vec3 gravity;
	Wind wind;
	REAL friction, obs_friction;
	enum {
		Proximity, Physics, StrainLimiting, Collision, Remeshing, Separation,
		PopFilter, Plasticity, nModules
	};
	bool enabled[nModules];
	Timer timers[nModules];
	// handy pointers
	std::vector<Mesh*> cloth_meshes, obstacle_meshes;

	// restart
	bool is_in_gpu;
	void reset();
};

void prepare(Simulation &sim);
void advance_step_gpu(Simulation &sim);

// Helper functions

template <typename Prim> int size(const std::vector<Mesh*> &meshes);
template <typename Prim> int get_index(const Prim *p,
	const std::vector<Mesh*> &meshes);
template <typename Prim> Prim *get(int i, const std::vector<Mesh*> &meshes);

std::vector<Vec3> node_positions(const std::vector<Mesh*> &meshes);
