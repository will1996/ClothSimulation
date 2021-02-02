#pragma once

#define REAL float

#ifdef WIN32
#ifdef DLL_EXPORT
#define DLL_PUBLIC __declspec(dllexport)
#else
#define DLL_PUBLIC __declspec(dllimport)
#endif
#else
#if __GNUC__ >= 4
#define DLL_PUBLIC __attribute__ ((visibility ("default")))
#define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
#else
#define DLL_PUBLIC
#define DLL_LOCAL
#endif
#endif

class DLL_PUBLIC CCSManager
{
	enum State{PREPARE, SIMULATE};
public:
	static CCSManager* getInstance();

private:
	CCSManager();
	static CCSManager* p;

public:
	// Prepare for cuda library
	void initialize(int argc, char *argv[]);

	//PREPARE only
	int addCloth();
	void addClothMaterial(int idx, int materialType);
	void addClothNode(int idx, REAL x, REAL y, REAL z);
	void addClothVert(int idx, REAL u, REAL v);
	void addClothFace(int idx, int na, int nb, int nc, int va, int vb, int vc);

	int addObs();
	void addObsNode(int idx, REAL x, REAL y, REAL z);
	void addObsFace(int idx, int na, int nb, int nc);

	void addNodeHandle(int clothIdx, int nodeIdx, REAL startTime = 0, REAL endTime = 1e10f);
	void addAttachHandle(int clothIdx, int clothNodeIdx, int obsIdx, int obsNodeIdx, REAL startTime = 0, REAL endTime = 1e10f);

	void setTimeStep(REAL time);
	void setStartStep(int start);
	void setEndStep(int end);
	void setGravity(REAL x, REAL y, REAL z);
	void setFriction(REAL fri);
	void setObsFriction(REAL fri);
	void initSimulation(); //PREPARE -> SIMULATE

	//SIMULATE only
	void runOneStep(void *pts);
	void updateObsNode(int obsIdx, int nodeIdx, REAL x, REAL y, REAL z); // update before step
	void endSimulation(); //SIMULATE -> PREPARE

	//PREPARE and SIMULATE
	void getClothNode(int clothIdx, int nodeIdx, REAL *x, REAL *y, REAL *z); // output nodes after step
	int getStep();
	int getStartStep();
	int getEndStep();
	State getState();

private:
	State state;
	int step;
	int startStep;
	int endStep;
	int curCloth;
	int totalCloth;
	int curObs;
	int totalObs;
};
