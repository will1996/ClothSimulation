#pragma once

#include "mesh.hpp"
#include "util.hpp"

enum Space {PS, WS}; // plastic space, world space

template <Space s> const Vec3 &pos (const Node *node);
template <Space s> Vec3 &pos (Node *node);
template <Space s> Vec3 nor (const Face *face);
template <Space s> REAL dihedral_angle (const Edge *edge);

REAL unwrap_angle (REAL theta, REAL theta_ref);

