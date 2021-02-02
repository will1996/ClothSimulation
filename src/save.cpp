#include "simulation.hpp"
#include "io.hpp"
#include "transformation.hpp"
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern std::string outprefix;

static void save_obstacle_transforms(const std::vector<Obstacle> &obs, int frame,
	double time) {
	if (!outprefix.empty() && frame < 10000) {
		for (int o = 0; o < obs.size(); o++) {
			Transformation trans = identity();
			if (obs[o].transform_spline)
				trans = get_dtrans(*obs[o].transform_spline, time).first;

			char buffer[512];
			sprintf(buffer, "%s/%04dobs%02d.txt", outprefix.c_str(), frame, o);
			save_transformation(trans, buffer);
		}
	}
}

static void save(const std::vector<Mesh*> &meshes, int frame) {
	if (!outprefix.empty() && frame < 10000) {
		char buffer[512];
		sprintf(buffer, "%s/%04d", outprefix.c_str(), frame);
		save_objs(meshes, buffer);
	}
}

void save(const Simulation &sim, int frame)
{
	save(sim.cloth_meshes, frame);
	save_obstacle_transforms(sim.obstacles, frame, sim.time);
}

extern "C" void save_objs_gpu(const std::string &prefix);

static void save_gpu(const std::vector<Mesh*> &meshes, int frame) {
	if (!outprefix.empty() && frame < 10000) {
		char buffer[512];
		sprintf(buffer, "%s/%04d", outprefix.c_str(), frame);

		save_objs_gpu(buffer);
	}
}

void save_gpu(const Simulation &sim, int frame)
{
	save_gpu(sim.cloth_meshes, frame);
}
