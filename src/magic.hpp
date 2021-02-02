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

#ifndef MAGIC_HPP
#define MAGIC_HPP

#pragma once
#include "real.hpp"

// Magic numbers and other hacks

struct Magic {
    bool fixed_high_res_mesh;
    double handle_stiffness, collision_stiffness;
    double repulsion_thickness, projection_thickness;
    double edge_flip_threshold;
    double rib_stiffening;
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

    Magic ():
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

#endif
