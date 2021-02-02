/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#ifndef OBSTACLE_HPP
#define OBSTACLE_HPP

#include "real.hpp"
#include "mesh.hpp"
#include "spline.hpp"
#include "util.hpp"

// A class which holds both moving and static meshes.
// Note that moving meshes MUST retain their structure across frames with only
// positions changing.
struct Obstacle {
public:
    double start_time, end_time;
    bool activated;
    // Gets the last-returned mesh or its transformation
    Mesh& get_mesh();
    Mesh& get_mesh(REAL time_sec);
    const Mesh& get_mesh() const;

    // Gets the state of the mesh at a given time, and updates the internal
    // meshes
    Mesh& get_mesh(double time_sec, int cur_frame, double percent, double frame_time);
    std::string motion_obj_file;
    int motion_type, slow;//0 for regular; 1 for interpolating mocap data;

    // lerp with previous mesh at time t - dt
    void blend_with_previous (double t, double dt, double blend);

    	// loading external mesh vertices
    void load_mesh(REAL t, REAL dt, REAL blend);
    void load_mesh(Vec3 *);

    const Motion *transform_spline;

    // A mesh containing the original, untransformed object
    Mesh base_mesh;
    // A mesh containing the correct mesh structure
    Mesh curr_state_mesh, last_frame_mesh, next_frame_mesh;

    Obstacle (): start_time(0), end_time(infinity), activated(false), slow(1) {}
};

// // Default arguments imply it's a static obstacle
// // An obstacle mesh may have multiple parts, so when you read one in,
// // you get a vector of obstacles back, each representing one part.
// std::vector<Obstacle> make_obstacle
//     (std::string filename, Transformation overall_transform = identity(),
//      std::vector<Transformation> global_transforms = std::vector<Transformation>(),
//      double fps = 1, double start_time = 0, double pause_time = 0);

#endif
