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

#include "runphysics.hpp"
#include "conf.hpp"
#include "io.hpp"
#include "misc.hpp"
#include "real.hpp"
#include "separateobs.hpp"
#include "simulation.hpp"
#include "timer.hpp"
#include "util.hpp"
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <cstdio>
#include <fstream>
#include <json/json.h>

using namespace std;

string outprefix;
static fstream timingfile;

Simulation sim;
Simulation *glSim;
Timer fps;

extern bool USE_GPU;
void copy_file (const string &input, const string &output);
extern void parse_handles (vector<Handle*>&, const Json::Value&,
                     const vector<Cloth>&, const vector<Motion>&);

void init_physics (const string &json_file, string outprefix,
                   bool is_reloading, int rank) {
    load_json(json_file, sim);
    ::outprefix = outprefix;
    if (rank != 0)
        return;
    if (!outprefix.empty()) {
        ::timingfile.open(stringf("%s/timing", outprefix.c_str()).c_str(),
                          is_reloading ? ios::out|ios::app : ios::out);
        // Make a copy of the config file for future use
        copy_file(json_file.c_str(), stringf("%s/conf.json",outprefix.c_str()));
        // And copy over all the obstacles
        vector<Mesh*> base_meshes(sim.obstacles.size());
        for (int o = 0; o < sim.obstacles.size(); o++)
            base_meshes[o] = &sim.obstacles[o].base_mesh;
        save_objs(base_meshes, stringf("%s/obs", outprefix.c_str()));
    }
    prepare(sim);
    for (int i = 0 ; i < sim.cloths.size(); ++i)
    {
        double tmp = sim.cloths[i].remeshing.tmp_min;
        tmp *= _coarseFineRatio;
        sim.cloths[i].remeshing.size_min = tmp;
        cout << "current size=" << sim.cloths[i].remeshing.size_min << endl;
    }
    if (!is_reloading) {
        separate_obstacles(sim.obstacle_meshes, sim.cloth_meshes);
        remeshing_step(sim);
        relax_initial_state(sim);
    }
}

static void save (const vector<Mesh*> &meshes, int frame, int level, int step) {
    if (!outprefix.empty() && frame < 10000)
        save_objs(meshes, stringf("%s/%d_%04d_%02d", outprefix.c_str(), level, frame, step), level == 1);
}

static void save_obstacle_transforms (const vector<Obstacle> &obs, int frame,
                                      double time, int level = 0) {
    if (!outprefix.empty() && frame < 10000) {
        for (int o = 0; o < obs.size(); o++) {
            Transformation trans = identity();
            if (obs[o].transform_spline)
                trans = get_dtrans(*obs[o].transform_spline, time).first;
            save_transformation(trans, stringf("%s/%d_%04dobs%02d.txt",
                                               outprefix.c_str(), level, frame, o));
        }
    }
}

static void save_timings () {
    static double old_totals[Simulation::nModules] = {};
    if (!::timingfile)
        return; // printing timing data to stdout is getting annoying
    ostream &out = ::timingfile ? ::timingfile : cout;
    for (int i = 0; i < Simulation::nModules; i++) {
        out << sim.timers[i].total - old_totals[i] << " ";
        old_totals[i] = sim.timers[i].total;
    }
    out << endl;
}

void save (Simulation &sim, int frame, int level, int step) {
    save(sim.cloth_meshes, frame, level, step);
    //save_obstacle_transforms(sim.obstacles, frame, sim.time, level);
/*        vector<Mesh*> base_meshes(sim.obstacles.size());
        for (int o = 0; o < sim.obstacles.size(); o++)
            base_meshes[o] = &sim.obstacles[o].get_mesh();
        save_objs(base_meshes, stringf("%s/%04d", outprefix.c_str(), frame));
*/
}

extern void advance_step_gpu (Simulation &sim);

void sim_step(int level, int rest_level, Timer *timer, int largeStep, int reload) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //    cout << "sim " << sim.step<<" from proc " << rank << endl;
    if (reload && rank != 0) {
        int nextStep = reload;
        sim.futureStep = reload;
        int nextFrame = reload / sim.frame_steps;
        nextStep %= sim.frame_steps;
        cout << "reload " << stringf("%s/%d_%04d_%02d",outprefix.c_str(),level-1,
            nextFrame,nextStep%sim.frame_steps) << endl;
        load_objs(sim.coarseMesh, stringf("%s/%d_%04d_%02d",outprefix.c_str(),level-1,
            nextFrame,nextStep%sim.frame_steps));
        //printf("reload %d\n", sim.coarseMesh.size());
        cout << "done" << endl;
    }
    fps.tick();
    if (USE_GPU) {
        advance_step_gpu(sim);
    }
    else {
        advance_step(sim, rest_level, level == 1, largeStep);
    }
    if (sim.step % sim.frame_steps == 0 && rest_level == 0) {
        save(sim, sim.frame, level, sim.step % sim.frame_steps);
        //save_timings();
    }
    fps.tock();
    if (sim.time >= sim.end_time || sim.frame >= sim.end_frame || sim.step >= sim.end_step)
    {
        if (timer != NULL)
        {
            timer->tock();
            cout << "final level proc "<< rank << " take time " << timer->last << endl;
        }
        MPI_Finalize();
        exit(EXIT_SUCCESS);
    }
}

void offline_loop() {
    while (true)
        sim_step();
}

void run_physics (const vector<string> &args) {
    if (args.size() != 1 && args.size() != 2) {
        cout << "Runs the simulation in batch mode." << endl;
        cout << "Arguments:" << endl;
        cout << "    <scene-file>: JSON file describing the simulation setup"
             << endl;
        cout << "    <out-dir> (optional): Directory to save output in" << endl;
        exit(EXIT_FAILURE);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    string json_file = args[0];
    string outprefix = args.size()>1 ? args[1] : "";
    if (rank == 0 && !outprefix.empty())
        ensure_existing_directory(outprefix);
    init_physics(json_file, outprefix, false, rank);
    /*
    offline_loop();
    */
    sim.add_pos = false;
    sim.finer = false;
    if (rank == 0)
    {
        if (!outprefix.empty())
            save(sim, 0, 0, 0);
        int ansk = 0;
        for (int i = 0; i < sim.cloths.size(); ++i)
        {
          double tmp = log(sim.cloths[i].remeshing.size_max/sim.cloths[i].remeshing.size_min)/
          log(2);
          if (ansk < ceil(tmp)) ansk = ceil(tmp);
        }
        do_simulation(sim, 0, sim.end_frame*sim.frame_steps, 1, 1, 0, size-1,-1,outprefix);//1,ansk-1
    }
    else
    {
        int param[10];
        MPI_Status status;
        MPI_Recv(param, 6, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        do_simulation(sim, param[0],param[1],param[2],param[3],param[4],param[5],status.MPI_SOURCE,outprefix);
    }
}

void init_resume(const vector<string> &args) {
    assert(args.size() == 2);
    string outprefix = args[0];
    string start_frame_str = args[1];
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Load like we would normally begin physics
    init_physics(stringf("%s/conf.json", outprefix.c_str()), outprefix, true, rank);
    // Get the initialization information
    sim.frame = atoi(start_frame_str.c_str());
    sim.time = sim.frame * sim.frame_time;
    sim.step = sim.frame * sim.frame_steps;
    for(int i=0; i<sim.obstacles.size(); ++i)
        sim.obstacles[i].get_mesh(sim.time, sim.frame, 
            (double)(sim.step-sim.frame*sim.frame_steps)/sim.frame_steps, 
            sim.frame_steps*sim.step_time);
    load_objs(sim.cloth_meshes, stringf("%s/%04d",outprefix.c_str(),sim.frame));
    prepare(sim); // re-prepare the new cloth meshes
    separate_obstacles(sim.obstacle_meshes, sim.cloth_meshes);
    sim.is_in_gpu = false;
}

void resume_physics (const vector<string> &args) {
    if (args.size() != 2) {
        cout << "Resumes an incomplete simulation in batch mode." << endl;
        cout << "Arguments:" << endl;
        cout << "    <out-dir>: Directory containing simulation output files"
             << endl;
        cout << "    <resume-frame>: Frame number to resume from" << endl;
        exit(EXIT_FAILURE);
    }
    init_resume(args);
    offline_loop();
}

void copy_file (const string &input, const string &output) {
    if(input == output) {
        return;
    }
    if(boost::filesystem::exists(output)) {
        boost::filesystem::remove(output);
    }
    boost::filesystem::copy_file(
        input, output);
}
