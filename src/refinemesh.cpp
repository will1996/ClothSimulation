/*************************************************************************\

Copyright 2016 The University of North Carolina at Chapel Hill.
All Rights Reserved.

Permission to use, copy, modify and distribute this software and its
documentation for educational, research and non-profit purposes, without
fee, and without a written agreement is hereby granted, provided that the
above copyright notice and the following three paragraphs appear in all
copies.

IN NO EVENT SHALL THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL BE
LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY
OF NORTH CAROLINA HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH
DAMAGES.

THE UNIVERSITY OF NORTH CAROLINA SPECIFICALLY DISCLAIM ANY
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
NORTH CAROLINA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

The authors may be contacted via:

US Mail:             GAMMA Research Group at UNC
Department of Computer Science
Sitterson Hall, CB #3175
University of N. Carolina
Chapel Hill, NC 27599-3175

Phone:               (919)962-1749

EMail:              tamert@cs.unc.edu


\**************************************************************************/

#include "refinemesh.hpp"

#include <vector>
#include "mesh.hpp"
#include "remesh.hpp"

extern int combine_label(int l0, int l1);

void refine_mesh(Simulation &sim, int k) {
    // For now, does a single step for a single mesh
    Mesh &m = *sim.cloth_meshes[k];

    // Allocate arrays for edges, vertices, nodes, and faces
    Mesh newMesh;
    newMesh.edges.resize(m.edges.size() * 2 + m.faces.size() * 3);
    newMesh.nodes.resize(m.nodes.size() + m.edges.size());
    newMesh.verts.resize(m.verts.size() + m.edges.size());
    newMesh.faces.resize(m.faces.size() * 4);

    // Deal with seams
    std::vector<bool> hasSeamVert(m.edges.size(), false);
    std::vector<int> seamVertIdx(m.edges.size(), -1);
    std::vector<Vert*> seamVerts;
    int numSeamVerts = 0;

    // For each edge, create the new node and vertex for its midpoint
    for (int eidx = 0; eidx < m.edges.size(); eidx++) {
        Node *n0 = m.edges[eidx]->n[0],
             *n1 = m.edges[eidx]->n[1];
        Edge *e = m.edges[eidx];

        // Create the new node
        Node *newNode = new Node((n0->y + n1->y) / 2.0,
                                 (n0->x + n1->x) / 2.0,
                                 (n0->v + n1->v) / 2.0,
                                 combine_label(n0->label, n1->label));
        newNode->x0 = (n0->x0 + n1->x0) / 2.0;
        newNode->acceleration = (n0->acceleration + n1->acceleration) / 2.0;
        newNode->preserve = n0->preserve && n1->preserve;
        newNode->index = m.nodes.size() + eidx;
        newNode->n = (n0->n + n1->n) / 2.0;
newNode->dx = (n0->dx + n1->dx) / 2.0;
newNode->dv = (n0->dv + n1->dv) / 2.0;
        // a covered by compute_ms_data
        // n updated by compute_ws_data
        // m covered by compute_masses
        // adje, verts below

        // Create the new edges
        Edge *newEdge0 = new Edge(n0, newNode, e->theta_ideal, e->label);
        Edge *newEdge1 = new Edge(newNode, n1, e->theta_ideal, e->label);
        newEdge0->index = e->index;
        newEdge1->index = m.edges.size() + eidx;
        newEdge0->adjf[0] = newEdge0->adjf[1] = NULL;
        newEdge1->adjf[0] = newEdge1->adjf[1] = NULL;
        // l covered by compute_ms_data
        // theta covered by compute_ws_data
        // adjf below

        // Create the new vertex
        Vert *vnew[2] = { NULL, NULL };
        for (int s = 0; s < 2; s++) {
            if (!e->adjf[s])
                continue;
            Vert *v0 = edge_vert(e, s, s), // requires face to be setup
                *v1 = edge_vert(e, s, 1 - s),
                *v2 = edge_opp_vert(e, s);
            if (s == 0 || is_seam_or_boundary(e)) {
                vnew[s] = new Vert((v0->u + v1->u) / 2.,
                    combine_label(v0->label, v1->label));
                connect(vnew[s], newNode); // connect the new node to this vertex (handles verts)
                vnew[s]->index = m.verts.size() + eidx;
                if (s == 0) {
                    vnew[s]->index = m.verts.size() + eidx;
                }
                else {
                    vnew[s]->index = m.verts.size() + m.edges.size() + numSeamVerts++;
                    hasSeamVert[eidx] = true;
                    seamVertIdx[eidx] = vnew[s]->index;
                    seamVerts.push_back(vnew[s]);
                }
                // a covered by compute_ms_data
                // m covered by compute_masses
                // adjf below
            }
            else {
                vnew[s] = vnew[0];
            }
        }

        // Update new lists
        newMesh.nodes[newNode->index] = newNode;
        newMesh.edges[newEdge0->index] = newEdge0;
        newMesh.edges[newEdge1->index] = newEdge1;
        newMesh.verts[vnew[0]->index] = vnew[0];

        newMesh.nodes[n0->index] = n0;
        newMesh.nodes[n1->index] = n1;
        for (int vidx = 0; vidx < n0->verts.size(); vidx++) {
            newMesh.verts[n0->verts[vidx]->index] = n0->verts[vidx];
        }
        for (int vidx = 0; vidx < n1->verts.size(); vidx++) {
            newMesh.verts[n1->verts[vidx]->index] = n1->verts[vidx];
        }
    }

    // Add the seam vertices to vert list
    for (int svidx = 0; svidx < numSeamVerts; svidx++) {
        newMesh.verts.push_back(seamVerts[svidx]);
    }

    // Remove existing faces from the adjf lists of vertices
    for (int vidx = 0; vidx < m.verts.size(); vidx++) {
        m.verts[vidx]->adjf.clear();
    }

    // Faces
    // Dm, invDm, a covered by compute_ms_data
    // n updated by compute_ws_data
    // m covered by compute_masses
    for (int fidx = 0; fidx < m.faces.size(); fidx++) {
        Face *f = m.faces[fidx];

        // Create the three new edges
        int e_a1_index, e_b1_index, e_c1_index;
        if ((f->adje[0]->n[0] == f->v[0]->node && f->adje[0]->n[1] == f->v[1]->node) ||
            (f->adje[0]->n[1] == f->v[0]->node && f->adje[0]->n[0] == f->v[1]->node)) {
            e_a1_index = f->adje[0]->index;
        }
        else if ((f->adje[1]->n[0] == f->v[0]->node && f->adje[1]->n[1] == f->v[1]->node) ||
            (f->adje[1]->n[1] == f->v[0]->node && f->adje[1]->n[0] == f->v[1]->node)) {
            e_a1_index = f->adje[1]->index;
        }
        else {
            e_a1_index = f->adje[2]->index;
        }

        if ((f->adje[0]->n[0] == f->v[1]->node && f->adje[0]->n[1] == f->v[2]->node) ||
            (f->adje[0]->n[1] == f->v[1]->node && f->adje[0]->n[0] == f->v[2]->node)) {
            e_b1_index = f->adje[0]->index;
        }
        else if ((f->adje[1]->n[0] == f->v[1]->node && f->adje[1]->n[1] == f->v[2]->node) ||
            (f->adje[1]->n[1] == f->v[1]->node && f->adje[1]->n[0] == f->v[2]->node)) {
            e_b1_index = f->adje[1]->index;
        }
        else {
            e_b1_index = f->adje[2]->index;
        }

        if ((f->adje[0]->n[0] == f->v[2]->node && f->adje[0]->n[1] == f->v[0]->node) ||
            (f->adje[0]->n[1] == f->v[2]->node && f->adje[0]->n[0] == f->v[0]->node)) {
            e_c1_index = f->adje[0]->index;
        }
        else if ((f->adje[1]->n[0] == f->v[2]->node && f->adje[1]->n[1] == f->v[0]->node) ||
            (f->adje[1]->n[1] == f->v[2]->node && f->adje[1]->n[0] == f->v[0]->node)) {
            e_c1_index = f->adje[1]->index;
        }
        else {
            e_c1_index = f->adje[2]->index;
        }

        Edge *e_a1 = newMesh.edges[e_a1_index]; // first half of edge 0
        Edge *e_a2 = newMesh.edges[m.edges.size() + e_a1_index]; // second half of edge 0
        Edge *e_b1 = newMesh.edges[e_b1_index]; // first half of edge 1
        Edge *e_b2 = newMesh.edges[m.edges.size() + e_b1_index]; // second half of edge 1
        Edge *e_c1 = newMesh.edges[e_c1_index]; // first half of edge 2
        Edge *e_c2 = newMesh.edges[m.edges.size() + e_c1_index]; // second half of edge 2

        // Might need to swap the half-edges
        Node *a2 = e_a2->n[0];
        Node *b2 = e_b2->n[0];
        Node *c2 = e_c2->n[0];
        if (m.edges[e_a1_index]->adjf[1] == f) {
            Edge *tmp = e_a2;
            e_a2 = e_a1;
            e_a1 = tmp;
            a2 = e_a2->n[1];
        }
        if (m.edges[e_b1_index]->adjf[1] == f) {
            Edge *tmp = e_b2;
            e_b2 = e_b1;
            e_b1 = tmp;
            b2 = e_b2->n[1];
        }
        if (m.edges[e_c1_index]->adjf[1] == f) {
            Edge *tmp = e_c2;
            e_c2 = e_c1;
            e_c1 = tmp;
            c2 = e_c2->n[1];
        }

        Edge *newEdge1 = new Edge(a2, c2, combine_label(e_a1->label, e_c1->label));
        Edge *newEdge2 = new Edge(c2, b2, combine_label(e_c1->label, e_b1->label));
        Edge *newEdge3 = new Edge(b2, a2, combine_label(e_b1->label, e_a1->label));
        newEdge1->index = 2 * m.edges.size() + 3 * fidx;
        newEdge2->index = 2 * m.edges.size() + 3 * fidx + 1;
        newEdge3->index = 2 * m.edges.size() + 3 * fidx + 2;
        newEdge1->adjf[0] = newEdge1->adjf[1] = NULL;
        newEdge2->adjf[0] = newEdge2->adjf[1] = NULL;
        newEdge3->adjf[0] = newEdge3->adjf[1] = NULL;

        // Create the four new faces
        Vert *va2 = a2->verts[0];
        Vert *vb2 = b2->verts[0];
        Vert *vc2 = c2->verts[0];

        bool aHasSeamVert = hasSeamVert[va2->index - m.verts.size()];
        bool bHasSeamVert = hasSeamVert[vb2->index - m.verts.size()];
        bool cHasSeamVert = hasSeamVert[vc2->index - m.verts.size()];
        double matSpaceEps = 0.00000001;
        if (aHasSeamVert) {
            Edge *_ea1 = m.edges[e_a1_index];

            // If the face is the 0th face of the edge, use the first vertex;
            // otherwise, use the second vertex
            if (_ea1->adjf[0] == f) {
                va2 = newMesh.verts[m.verts.size() + _ea1->index];
            }
            else {
                va2 = newMesh.verts[seamVertIdx[_ea1->index]];
            }
        }
        if (bHasSeamVert) {
            Edge *_eb1 = m.edges[e_b1_index];

            // If the face is the 0th face of the edge, use the first vertex;
            // otherwise, use the second vertex
            if (_eb1->adjf[0] == f) {
                vb2 = newMesh.verts[m.verts.size() + _eb1->index];
            }
            else {
                vb2 = newMesh.verts[seamVertIdx[_eb1->index]];
            }
        }
        if (cHasSeamVert) {
            Edge *_ec1 = m.edges[e_c1_index];

            // If the face is the 0th face of the edge, use the first vertex;
            // otherwise, use the second vertex
            if (_ec1->adjf[0] == f) {
                vc2 = newMesh.verts[m.verts.size() + _ec1->index];
            }
            else {
                vc2 = newMesh.verts[seamVertIdx[_ec1->index]];
            }
        }

        Face *newFace1 = new Face(f->v[0], va2, vc2, f->label);
        Face *newFace2 = new Face(f->v[1], vb2, va2, f->label);
        Face *newFace3 = new Face(f->v[2], vc2, vb2, f->label);
        Face *newFace4 = new Face(va2, vb2, vc2, f->label); // reverse direction!
        newFace1->index = fidx * 4;
        newFace2->index = fidx * 4 + 1;
        newFace3->index = fidx * 4 + 2;
        newFace4->index = fidx * 4 + 3;

        // Add the faces to the adjf lists of the outer six edges
        e_a1->adjf[m.edges[e_a1_index]->adjf[0] == f ? 0 : 1] = newFace1;
        e_a2->adjf[m.edges[e_a1_index]->adjf[0] == f ? 0 : 1] = newFace2;
        e_b1->adjf[m.edges[e_b1_index]->adjf[0] == f ? 0 : 1] = newFace2;
        e_b2->adjf[m.edges[e_b1_index]->adjf[0] == f ? 0 : 1] = newFace3;
        e_c1->adjf[m.edges[e_c1_index]->adjf[0] == f ? 0 : 1] = newFace3;
        e_c2->adjf[m.edges[e_c1_index]->adjf[0] == f ? 0 : 1] = newFace1;

        // Add the faces to the adjf lists of the inner three edges
        newEdge1->adjf[0] = newFace1;
        newEdge1->adjf[1] = newFace4;
        newEdge2->adjf[0] = newFace3;
        newEdge2->adjf[1] = newFace4;
        newEdge3->adjf[0] = newFace2;
        newEdge3->adjf[1] = newFace4;

        // Add the edges to the adje lists of the faces
        newFace1->adje[0] = e_a1;
        newFace1->adje[1] = newEdge1;
        newFace1->adje[2] = e_c2;
        newFace2->adje[0] = e_b1;
        newFace2->adje[1] = newEdge3;
        newFace2->adje[2] = e_a2;
        newFace3->adje[0] = e_c1;
        newFace3->adje[1] = newEdge2;
        newFace3->adje[2] = e_b2;
        newFace4->adje[0] = newEdge3;
        newFace4->adje[1] = newEdge2;
        newFace4->adje[2] = newEdge1;

        // Update new lists
        newMesh.edges[newEdge1->index] = newEdge1;
        newMesh.edges[newEdge2->index] = newEdge2;
        newMesh.edges[newEdge3->index] = newEdge3;
        newMesh.faces[newFace1->index] = newFace1;
        newMesh.faces[newFace2->index] = newFace2;
        newMesh.faces[newFace3->index] = newFace3;
        newMesh.faces[newFace4->index] = newFace4;
    }

    // Update nodes' adjacent edges
    for (int eidx = 0; eidx < newMesh.edges.size(); eidx++) {
        Edge *e = newMesh.edges[eidx];
        e->n[0]->adje.push_back(e);
        e->n[1]->adje.push_back(e);
    }

    // Update verts' adjacent faces
    for (int fidx = 0; fidx < newMesh.faces.size(); fidx++) {
        Face *f = newMesh.faces[fidx];
        f->v[0]->adjf.push_back(f);
        f->v[1]->adjf.push_back(f);
        f->v[2]->adjf.push_back(f);
    }

    // Free the old edges
    for (int eidx = 0; eidx < m.edges.size(); eidx++) {
        delete m.edges[eidx];
    }

    // Free the old faces
    for (int fidx = 0; fidx < m.faces.size(); fidx++) {
        //delete m.faces[fidx];
    }

    sim.cloths[k].mesh = newMesh;
    sim.cloth_meshes[k] = &sim.cloths[k].mesh;

    compute_ms_data(newMesh);
    prepare(sim); // calls compute_masses
}

void refine_mesh(Simulation &sim) {
for (int i = 0; i < sim.cloths.size(); ++i)
refine_mesh(sim, i);
}
