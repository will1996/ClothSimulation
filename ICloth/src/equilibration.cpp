#include "simulation.hpp"
#include "magic.hpp"
//#include "tmbvh.hpp"
#include "proximity.hpp"
#include "collision.hpp"
#include "popfilter.hpp"

extern Simulation *glSim;

void validate_handles(const Simulation &sim) {
	return;
	/*
	for (int h = 0; h < sim.handles.size(); h++) {
		vector<Node*> nodes = sim.handles[h]->get_nodes();
		for (int n = 0; n < nodes.size(); n++) {
			if (!nodes[n]->preserve) {
				cout << "Constrained node " << nodes[n]->index << " will not be preserved by remeshing" << endl;
				abort();
			}
		}
	}
	*/
}

vector<Constraint*> get_constraints(Simulation &sim, bool include_proximity) {
	glSim = &sim;
	vector<Constraint*> cons;
	for (int h = 0; h < sim.handles.size(); h++)
		append(cons, sim.handles[h]->get_constraints(sim.time));
	if (include_proximity && sim.enabled[Simulation::Proximity]) {
		append(cons, proximity_constraints(sim.cloth_meshes,
			sim.obstacle_meshes,
			sim.friction, sim.obs_friction));
	}
	return cons;
}

void delete_constraints(const vector<Constraint*> &cons) {
	for (int c = 0; c < cons.size(); c++)
		delete cons[c];
}

void equilibration_step(Simulation &sim) {
	vector<Constraint*> cons;
	for (int c = 0; c < sim.cloths.size(); c++) {
		Mesh &mesh = sim.cloths[c].mesh;
		for (int n = 0; n < mesh.nodes.size(); n++)
			mesh.nodes[n]->acceleration = Vec3(0);
		apply_pop_filter(sim.cloths[c], cons, 1);
	}
	printf("after apply_pop_filter\n");
	delete_constraints(cons);

	cons = get_constraints(sim, false);
	collision_response(sim.cloth_meshes, cons, sim.obstacle_meshes);
	//delete_constraints(cons);
	printf("after collision_response\n");
}

void relax_initial_state(Simulation &sim) {
	validate_handles(sim);
	for (int i = 0; i < 2; i++)
	equilibration_step(sim);
}