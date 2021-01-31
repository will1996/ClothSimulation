#include "CCSManager.h"
#include "src/simulation.hpp"
#include <helper_cuda.h>

extern void init_cuda(int argc, char **argv);
extern void load_material_data(Cloth::Material&, const string &filename);
extern Simulation sim;

static bool init = false;

string material_table[10] = {
	".\\materials\\11oz-black-denim.json",
	".\\materials\\camel-ponte-roma.json",
	".\\materials\\gray-interlock.json",
	".\\materials\\ivory-rib-knit.json",
	".\\materials\\navy-sparkle-sweat.json",
	".\\materials\\pink-ribbon-brown.json",
	".\\materials\\royal-target.json",
	".\\materials\\tango-red-jet-set.json",
	".\\materials\\white-dots-on-blk.json",
	".\\materials\\white-swim-solid.json" };

CCSManager* CCSManager::p = new CCSManager;

CCSManager* CCSManager::getInstance()
{
	return p;
}

void CCSManager::initialize(int argc, char *argv[])
{
	if (!init)
	{
		//gpuDeviceInit(0);
		printf("init......\n");

		init_cuda(argc, argv);
		init = true;
	}
}


CCSManager::CCSManager() : state(SIMULATE)
{
	endSimulation();
}

int CCSManager::addCloth()
{
	if (state == PREPARE)
	{
		sim.cloths.resize(++totalCloth);
		return totalCloth - 1;
	}
	return -1;
}

void CCSManager::addClothMaterial(int idx, int materialType)
{
	if (state == PREPARE)
	{
		Cloth::Material *material = new Cloth::Material;
		memset(material, 0, sizeof(Cloth::Material));
		sim.cloths[idx].materials.push_back(material);
		load_material_data(*material, material_table[materialType]);
	}
}

void CCSManager::addClothNode(int idx, REAL x, REAL y, REAL z)
{
	if (state == PREPARE)
		sim.cloths[idx].mesh.add(new Node(Vec3(x, y, z), Vec3(0)));
}

void CCSManager::addClothVert(int idx, REAL u, REAL v)
{
	if (state == PREPARE)
		sim.cloths[idx].mesh.add(new Vert(Vec2(u, v)));
}

void CCSManager::addClothFace(int idx, 
	int na, int nb, int nc,
	int va, int vb, int vc)
{
	if (state == PREPARE)
	{
		Mesh &mesh = sim.cloths[idx].mesh;
		connect(mesh.verts[va], mesh.nodes[na]);
		connect(mesh.verts[vb], mesh.nodes[nb]);
		connect(mesh.verts[vc], mesh.nodes[nc]);
		mesh.add(new Face(mesh.verts[va], mesh.verts[vb], mesh.verts[vc]));
	}
}

int CCSManager::addObs()
{
	if (state == PREPARE)
	{
		sim.obstacles.resize(++totalObs);
		sim.obstacles[totalObs - 1].transform_spline = NULL;

		return totalObs - 1;
	}
	return -1;
}

void CCSManager::addObsNode(int idx, REAL x, REAL y, REAL z)
{
	if (state == PREPARE)
	{
		sim.obstacles[idx].curr_state_mesh.add(new Node(Vec3(x, y, z), Vec3(0)));
		sim.obstacles[idx].curr_state_mesh.add(new Vert(Vec2(0)));
	}
}

void CCSManager::addObsFace(int idx, int a, int b, int c)
{
	if (state == PREPARE)
	{
		Mesh &mesh = sim.obstacles[idx].curr_state_mesh;
		connect(mesh.verts[a], mesh.nodes[a]);
		connect(mesh.verts[b], mesh.nodes[b]);
		connect(mesh.verts[c], mesh.nodes[c]);
		mesh.add(new Face(mesh.verts[a], mesh.verts[b], mesh.verts[c]));
	}
}

void CCSManager::addNodeHandle(int clothIdx, int nodeIdx, REAL startTime, REAL endTime)
{
	if (state == PREPARE)
	{
		NodeHandle *han = new NodeHandle;
		han->node = sim.cloths[clothIdx].mesh.nodes[nodeIdx];
		han->node->preserve = true;
		han->motion = NULL;
		han->start_time = startTime;
		han->end_time = endTime;
		han->fade_time = 0.f;
		han->backup(sim.cloths);
		sim.handles.push_back(han);
	}
}

void CCSManager::addAttachHandle(int clothIdx, int clothNodeIdx, int obsIdx, int obsNodeIdx, REAL startTime, REAL endTime)
{
	if (state == PREPARE)
	{
		AttachHandle *han = new AttachHandle;
		han->cid = clothIdx;
		han->oid = obsIdx;
		han->id1 = clothNodeIdx;
		han->id2 = obsNodeIdx;
		han->init = false;
		han->start_time = startTime;
		han->end_time = endTime;
		han->fade_time = 0.f;
		han->backup(sim.cloths);
		sim.handles.push_back(han);
	}
}

void CCSManager::setTimeStep(REAL time)
{
	if (state == PREPARE)
		sim.step_time = time;
}

void CCSManager::setStartStep(int start)
{
	if (state == PREPARE)
	{
		step = startStep = start;
	}
}

void CCSManager::setEndStep(int end)
{
	if (state == PREPARE)
	{
		sim.end_time = infinity;
		endStep = end;
	}
}

void CCSManager::setGravity(REAL x, REAL y, REAL z)
{
	if (state == PREPARE)
		sim.gravity = Vec3(x, y, z);
}

void CCSManager::setFriction(REAL fri)
{
	if (state == PREPARE)
		sim.friction = fri;
}

void CCSManager::setObsFriction(REAL fri)
{
	if (state == PREPARE)
		sim.obs_friction = fri;
}

void CCSManager::initSimulation()
{
	if (state == PREPARE)
	{
		state = SIMULATE;

		for (int i = 0; i < sim.cloths.size(); i++)
			compute_ms_data(sim.cloths[i].mesh);

		for (int i = 0; i < sim.obstacles.size(); i++) {
			sim.obstacles[i].base_mesh = deep_copy(sim.obstacles[i].curr_state_mesh);
			compute_ws_data(sim.obstacles[i].curr_state_mesh);
		}
		prepare(sim);
		//printf("cloth size = %d\n", sim.cloth_meshes.size());

	}
}

static Vec3 *obstaclePts = NULL;

Vec3 *getObstaclePts() { return obstaclePts; }

void CCSManager::runOneStep(void *pts)
{
	if (state == SIMULATE && step < endStep)
	{
		if (pts) {
			obstaclePts = (Vec3 *)pts;
		}

		advance_step_gpu(sim);
		step++;
		obstaclePts = NULL;
	}
}

void CCSManager::endSimulation()
{
	if (state == SIMULATE)
	{
		state = PREPARE;
		sim.reset();
		step = startStep = 0;
		endStep = INT_MAX;
		curCloth = totalCloth = curObs = totalObs = 0;
	}
	
}

int CCSManager::getStep()
{
	return step;
}

int CCSManager::getStartStep()
{
	return startStep;
}

int CCSManager::getEndStep()
{
	return endStep;
}

CCSManager::State CCSManager::getState()
{
	return state;
}

void CCSManager::updateObsNode(int obsIdx, int nodeIdx, REAL x, REAL y, REAL z)
{
	if (state == SIMULATE)
		sim.obstacles[obsIdx].curr_state_mesh.nodes[nodeIdx]->x = Vec3(x, y, z);
}

void CCSManager::getClothNode(int clothIdx, int nodeIdx, REAL *x, REAL *y, REAL *z)
{
	Vec3 nodeX = sim.cloths[clothIdx].mesh.nodes[nodeIdx]->x;
	*x = nodeX[0];
	*y = nodeX[1];
	*z = nodeX[2];
}
