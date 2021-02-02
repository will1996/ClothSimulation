#include "simulation.hpp"
#include "magic.hpp"
#include "real.hpp"
#include "vectors.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void advance_step_gpu (Simulation &sim, int level, double save, int ratio);