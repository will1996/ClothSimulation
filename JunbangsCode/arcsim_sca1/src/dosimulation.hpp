
#ifndef DO_SIMULATION_HPP
#define DO_SIMULATION_HPP

#include "simulation.hpp"
#include <string>
#include <vector>
#include <mpi.h>

const double _coarseFineRatio = 5;

void do_simulation(Simulation &sim, int start_step, int end_step, int cur_level, int rest_level, 
    int pstart, int pend, int source, std::string outprefix);

#endif