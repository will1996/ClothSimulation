#ifndef STRAINLIMITING_HPP
#define STRAINLIMITING_HPP

#include "cloth.hpp"
#include "constraint.hpp"

std::vector<Vec2> get_strain_limits (const std::vector<Cloth> &cloths);

void strain_limiting (std::vector<Mesh*> &meshes,
                      const std::vector<Vec2> &strain_limits,
                      const std::vector<Constraint*> &cons);

#endif
