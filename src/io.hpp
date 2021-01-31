#pragma once

#include "mesh.hpp"

void load_obj(Mesh &mesh, const std::string &filename);
void load_objs(std::vector<Mesh*> &meshes, const std::string &prefix);
int load_obj_vertices(const std::string &filename, std::vector<Vec3> &verts);

void save_obj(const Mesh &mesh, const std::string &filename);
void save_objs(const std::vector<Mesh*> &meshes, const std::string &prefix);

void save_transformation(const Transformation &tr,
	const std::string &filename);

// w_crop and h_crop specify a multiplicative crop window
void save_screenshot(const std::string &filename);

// check that output directory exists; if not, create it
void ensure_existing_directory(const std::string &path);
