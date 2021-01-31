#pragma once

#include "mesh.hpp"
#include "spline.hpp"
#include "util.hpp"

// A class which holds both moving and static meshes.
// Note that moving meshes MUST retain their structure across frames with only
// positions changing.
struct Obstacle {
public:
	REAL start_time, end_time;
	bool activated;
	// Gets the last-returned mesh or its transformation
	Mesh& get_mesh();
	const Mesh& get_mesh() const;

	// Gets the state of the mesh at a given time, and updates the internal
	// meshes
	Mesh& get_mesh(REAL time_sec);

	// lerp with previous mesh at time t - dt
	void blend_with_previous(REAL t, REAL dt, REAL blend);

	// loading external mesh vertices
	void load_mesh(REAL t, REAL dt, REAL blend);
	void load_mesh(Vec3 *);

	const Motion *transform_spline;

	// A mesh containing the original, untransformed object
	Mesh base_mesh;
	// A mesh containing the correct mesh structure
	Mesh curr_state_mesh;

	Obstacle() : start_time(0), end_time(infinity), activated(false) {}
};

// // Default arguments imply it's a static obstacle
// // An obstacle mesh may have multiple parts, so when you read one in,
// // you get a vector of obstacles back, each representing one part.
// std::vector<Obstacle> make_obstacle
//     (std::string filename, Transformation overall_transform = identity(),
//      std::vector<Transformation> global_transforms = std::vector<Transformation>(),
//      REAL fps = 1, REAL start_time = 0, REAL pause_time = 0);
