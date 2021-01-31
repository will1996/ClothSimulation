#include "CCSManager.h"

#include <time.h>
#include <stdarg.h>

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include <vector>
using namespace std;

#include "vec3f.h"

class MeshInfo;

static vector<int>  clothIDs;
static vector<MeshInfo *>cloths;
static vector<MeshInfo *>obstacles;

class HandleInfo {
public:
	int cid, nid;
	int oid, onid;

	HandleInfo(int cid, int nid) {
		this->cid = cid;
		this->nid = nid;
		this->oid = -1;
		this->onid = -1;
	}

	HandleInfo(int cid, int nid, int oid, int onid){
		this->cid = cid;
		this->nid = nid;
		this->oid = oid;
		this->onid = onid;
	}
};
static vector<HandleInfo> handles;

void exportMeshes(char *path, int frame);

/*
class NodeInfo {
float x, y, z;

public:
NodeInfo(float x, float y, float z) {
this->x = x, this->y = y, this->z = z;
}

vec3f pt() { return vec3f(x, y, z); }

friend class MeshInfo;
//friend void MeshInfo::PushCloth(CCSManager *mgr);
//friend void MeshInfo::PushObstacle(CCSManager *mgr);
};
*/

typedef vec3f NodeInfo;

class VertInfo {
	float u, v;

public:
	VertInfo(float u, float v) {
		this->u = u, this->v = v;
	}

	float getU() { return u; }

	float getV() { return v; }

	friend class MeshInfo;
	//friend void MeshInfo::PushCloth(CCSManager *mgr);
	//friend void MeshInfo::PushObstacle(CCSManager *mgr);
};

class FaceInfo {
	int a, b, c;

public:
	FaceInfo(int a, int b, int c)
	{
		this->a = a, this->b = b, this->c = c;
	}

	int getA() { return a; }

	int getB() { return b; }

	int getC() { return c; }

	friend class MeshInfo;
	//friend void MeshInfo::PushCloth(CCSManager *mgr);
	//friend void MeshInfo::PushObstacle(CCSManager *mgr);
};


inline vec3f normal(vec3f &v1, vec3f &v2, vec3f &v3)
{
	vec3f s = (v2 - v1);
	return s.cross(v3 - v1);
}

void drawMeshInfo(float *vtxs, float *nrms, int *tris, int numTri, int i, bool w);

class MeshInfo {
	vector<NodeInfo> _nodes;
	vector<VertInfo> _verts;
	vector<FaceInfo> _faceNodes, _faceVerts;

	vector<vec3f> _nrms;

public:
	MeshInfo(const string &filename);

	vector<NodeInfo> getNodes() { return _nodes; }

	vector<VertInfo> getVerts() { return _verts; }

	vector<FaceInfo> getFaceNodes() { return _faceNodes; }

	vector<FaceInfo> getFaceVerts() { return _faceVerts; }

	int nodeNum() { return _nodes.size(); }

	vec3f getNode(int nid) { return _nodes[nid]; }

	void setNode(int nidx, float x, float y, float z) {
		_nodes[nidx] = NodeInfo(x, y, z);
	}

	void setNodes(vector<vec3f> pts) {
		if (pts.size() == _nodes.size()) {
			for (int i = 0; i < pts.size(); i++)
				_nodes[i] = pts[i];
		}
		else {
			printf("Error, unmatched!\n");
			exit(0);
		}
	}

	void draw(int type, bool w) {
		//drawMeshInfo((float *)_nodes.data(), (float *)_nrms.data(), (int *)_faceNodes.data(), _faceNodes.size(), type, w);

		float *buffer1 = new float[_nodes.size() * 3];
		float *buffer2 = new float[_nrms.size() * 3];
		int *buffer3 = new int[_faceNodes.size() * 3];

		int idx = 0;
		for (int i = 0; i < _nodes.size(); i++) {
			buffer1[idx++] = _nodes[i].x;
			buffer1[idx++] = _nodes[i].y;
			buffer1[idx++] = _nodes[i].z;
		}

		idx = 0;
		for (int i = 0; i < _nrms.size(); i++) {
			buffer2[idx++] = _nrms[i].x;
			buffer2[idx++] = _nrms[i].y;
			buffer2[idx++] = _nrms[i].z;
		}

		idx = 0;
		for (int i = 0; i < _faceNodes.size(); i++) {
			buffer3[idx++] = _faceNodes[i].getA();
			buffer3[idx++] = _faceNodes[i].getB();
			buffer3[idx++] = _faceNodes[i].getC();
		}

		drawMeshInfo(buffer1, buffer2, buffer3, _faceNodes.size(), type, w);

		delete[] buffer1;
		delete[] buffer2;
		delete[] buffer3;

	}

	void updateNrms()
	{
		_nrms.clear();
		//_nrms.resize(nodeNum());

		for (unsigned int i = 0; i < nodeNum(); i++)
			//_nrms[i] = vec3f::zero();
			_nrms.push_back(vec3f::zero());

		for (unsigned int i = 0; i<_faceNodes.size(); i++) {
			vec3f n = normal(
				_nodes[_faceNodes[i].a],
				_nodes[_faceNodes[i].b],
				_nodes[_faceNodes[i].c]);
			n.normalize();

			_nrms[_faceNodes[i].a] += n;
			_nrms[_faceNodes[i].b] += n;
			_nrms[_faceNodes[i].c] += n;
		}

		for (unsigned int i = 0; i<nodeNum(); i++)
			_nrms[i].normalize();
	}

	int pushCloth(CCSManager *mgr)
	{
		if (nodeNum() == 0) return -1;

		int idx = mgr->addCloth();
		mgr->addClothMaterial(idx, 1);

		for (int i = 0; i < _nodes.size(); i++)
			mgr->addClothNode(idx, _nodes[i].x, _nodes[i].y, _nodes[i].z);

		for (int i = 0; i < _verts.size(); i++)
			mgr->addClothVert(idx, _verts[i].u, _verts[i].v);

		for (int i = 0; i < _faceNodes.size(); i++)
			mgr->addClothFace(idx,
			_faceNodes[i].a, _faceNodes[i].b, _faceNodes[i].c,
			_faceVerts[i].a, _faceVerts[i].b, _faceVerts[i].c);

		updateNrms();
		return idx;
	}

	void pushObstacle(CCSManager *mgr)
	{
		if (nodeNum() == 0) return;

		int idx = mgr->addObs();

		for (int i = 0; i < _nodes.size(); i++)
			mgr->addObsNode(idx, _nodes[i].x, _nodes[i].y, _nodes[i].z);

		for (int i = 0; i < _faceNodes.size(); i++)
			mgr->addObsFace(idx, _faceNodes[i].a, _faceNodes[i].b, _faceNodes[i].c);

		updateNrms();
	}
};

static void get_valid_line(istream &in, string &line) {
	do
	getline(in, line);
	while (in && (line.length() == 0 || line[0] == '#'));
}

int load_obj_vertices(const string &filename, vector<vec3f> &verts)
{
	fstream file(filename.c_str(), ios::in);
	if (!file) {
		return 0;
	}

	int count = 0;
	while (file) {
		string line;
		get_valid_line(file, line);
		stringstream linestream(line);
		string keyword;
		linestream >> keyword;

		if (keyword == "v") {
			//occationally, double output -> float input error!
			double xx, yy, zz;
			linestream >> xx >> yy >> zz;
			verts.push_back(vec3f(xx, yy, zz));
			count++;
		}
	}

	return count;
}

MeshInfo::MeshInfo(const string &filename)
{
	CCSManager *mgr = CCSManager::getInstance();

	fstream file(filename.c_str(), ios::in);
	if (!file.is_open()) {
		//		cout << "Error: failed to open file " << filename << endl;
		return;
	}

	while (file) {
		string line;
		get_valid_line(file, line);
		stringstream linestream(line);
		string keyword;
		linestream >> keyword;

		if (keyword == "vt") {
			float u, v;
			linestream >> u >> v;

			_verts.push_back(VertInfo(u, v));
		}
		else if (keyword == "v") {
			//occationally, double output -> float input error!
			double xx, yy, zz;
			linestream >> xx >> yy >> zz;

			_nodes.push_back(NodeInfo(xx, yy, zz));
		}
		else if (keyword == "f") {
			string w;

			int n[3], v[3];
			for (int i = 0; i < 3; i++) {
				linestream >> w;
				stringstream wstream(w);
				char c;
				v[i] = -1;
				wstream >> n[i] >> c >> v[i];
			}

			_faceNodes.push_back(FaceInfo(n[0] - 1, n[1] - 1, n[2] - 1));
			if (v[0] != -1)
				_faceVerts.push_back(FaceInfo(v[0] - 1, v[1] - 1, v[2] - 1));

		}
	}
}


void initHandles(char *path)
{
	CCSManager *mgr = CCSManager::getInstance();
	handles.clear();

	string outprefix(path);
	char filename[512];
	sprintf(filename, "%s\\nodes.txt", outprefix.c_str());
	fstream file(filename, ios::in);
	if (!file.is_open()) {
		return; //no handles...
	}

	while (file) {
		string line;
		get_valid_line(file, line);
		stringstream linestream(line);
		string keyword;
		linestream >> keyword;

		if (keyword == "Fixed") {
			int cid, nid;
			float t1, t2;
			linestream >> cid >> nid >> t1 >> t2;
			mgr->addNodeHandle(cid, nid, t1, t2);
			handles.push_back(HandleInfo(cid, nid));
		}

		if (keyword == "Attach") {
			int cid, nid;
			int oid, onid;
			float t1, t2;
			linestream >> cid >> nid >> oid >> onid >> t1 >> t2;
			mgr->addAttachHandle(cid, nid, oid, onid, t1, t2);
			handles.push_back(HandleInfo(cid, nid, oid, onid));
		}
	}
}

void init_resume2(char *path, int startFrame)
{
	CCSManager *mgr = CCSManager::getInstance();
	string outprefix(path);
	char filename[512];

	sprintf(filename, "%s\\%04d_ob.obj", outprefix.c_str(), startFrame);

	MeshInfo *mObj = new MeshInfo((filename));// "%s/%04d_ob.obj", outprefix.c_str(), startFrame));
	mObj->pushObstacle(mgr);
	if (mObj->nodeNum())
		obstacles.push_back(mObj);

	for (int m = 0; m < 99; m++) {
		sprintf(filename, "%s\\%04d_%02d.obj", outprefix.c_str(), startFrame, m);
		MeshInfo *mCloth = new MeshInfo((filename));//"%s/%04d_%02d.obj", outprefix.c_str(), startFrame, m));
		if (mCloth->nodeNum() == 0)
			break;

		int idx = mCloth->pushCloth(mgr);

		clothIDs.push_back(idx);
		cloths.push_back(mCloth);
	}

	initHandles(path);
}

void initModel(int argc, char ** argv, char *path, int startFrame)
{
	CCSManager *mgr = CCSManager::getInstance();

	mgr->initialize(argc, argv);
	init_resume2(path, startFrame);

	mgr->setGravity(0, 0, -0.98);
	mgr->initSimulation();
}

void quitModel()
{
	for (int i = 0; i < cloths.size(); i++)
		delete cloths[i];

	for (int i = 0; i < obstacles.size(); i++)
		delete obstacles[i];

	cloths.clear();
	clothIDs.clear();
	obstacles.clear();
}

int load_obj_vertices(const string &filename, vector<vec3f> &verts);
extern float tick();

bool dynamicModel(char *dataPath, bool output, int frame, char *animPath)
{
	CCSManager *mgr = CCSManager::getInstance();

	if (animPath == NULL) {
		float last = tick();
		mgr->runOneStep(NULL);
		float delta = tick() - last;
		printf("Step %d: %3.5f s\n", frame, delta);
	}
	else {
		char obuf[512];
		sprintf(obuf, "%s\\body%03d.obj", animPath, frame);
		printf("Loading %s...\n", obuf);

		vector<vec3f> pts;
		load_obj_vertices((obuf), pts);
		if (pts.size() != obstacles[0]->nodeNum()) {
			printf("obstacle obj file (%s) unmatch!\n", obuf);
			exit(1);
		}

		float last = tick();
		mgr->runOneStep(pts.data());
		float delta = tick() - last;
		printf("Step %d: %3.5f s\n", frame, delta);

		{
			MeshInfo *mOb = obstacles[0];
			mOb->setNodes(pts);
		}

	}

	for (int i = 0; i < cloths.size(); i++) {
		int idx = clothIDs[i];
		MeshInfo *mCloth = cloths[i];

		float x, y, z;
		for (int nid = 0; nid < mCloth->nodeNum(); nid++) {
			mgr->getClothNode(idx, nid, &x, &y, &z);
			mCloth->setNode(nid, x, y, z);
		}

		mCloth->updateNrms();
	}

	if (output) {
		exportMeshes(dataPath, frame);
	}
	return true;
}

void drawPt(float x, float y, float z);

void drawHandles()
{
	for (int h = 0; h < handles.size(); h++) {
		HandleInfo &hdl = handles[h];

		vec3f pt = cloths[hdl.cid]->getNode(hdl.nid);
		drawPt(pt[0], pt[1], pt[2]);
	}
}

void drawMeshes(bool w)
{
	for (int i = 0; i < obstacles.size(); i++) {
		obstacles[i]->draw(0, false);
	}

	for (int i = 0; i < cloths.size(); i++) {
		cloths[i]->draw(i + 1, w);
	}
}

void writeFile(MeshInfo* mesh, char filename[])
{
	vector<NodeInfo> mNodes = mesh->getNodes();
	vector<VertInfo> mVerts = mesh->getVerts();
	vector<FaceInfo> mFaceNodes = mesh->getFaceNodes();
	vector<FaceInfo> mFaceVerts = mesh->getFaceVerts();

	ofstream ofile;
	ofile.open(filename);
	if (!ofile.is_open()) {
		cout << "Cannot open file to write." << endl;
		return;
	}
	//write vt
	for (vector<VertInfo>::iterator t = mVerts.begin(); t != mVerts.end(); t++) {
		ofile << "vt " << t->getU() << " " << t->getV() << endl;
	}

	//write v
	for (vector<NodeInfo>::iterator v = mNodes.begin(); v != mNodes.end(); v++) {
		vec3f temp = v[0];
		float a = temp[0], b = temp[1], c = temp[2];
		ofile << "v " << a << " " << b << " " << c << endl;
	}

	//write f
	for (vector<FaceInfo>::iterator fn = mFaceNodes.begin(), fv = mFaceVerts.begin();
		fn != mFaceNodes.end(); fn++) {
		if (fv != mFaceVerts.end()) {
			ofile << "f " << fn->getA() + 1 << "/" << fv->getA() + 1 << " "
				<< fn->getB() + 1 << "/" << fv->getB() + 1 << " "
				<< fn->getC() + 1 << "/" << fv->getC() + 1 << endl;
			fv++;
		}
		else {
			ofile << "f " << fn->getA() + 1 << " " << fn->getB() + 1 << " " << fn->getC() + 1 << endl;
		}
	}

	ofile.close();
	cout << "Export to file " << filename << "." << endl;
}

void exportMeshes(char *path, int frame)
{
	if (frame == 0) return; //avoid overwrite the 0 files

	char filename[512];
	for (int i = 0; i < obstacles.size(); i++) {
		sprintf(filename, "%s/%04d_%02dob.obj", path, frame, i);
		MeshInfo *mObstacles = obstacles[i];
		writeFile(mObstacles, filename);
	}
	for (int j = 0; j < cloths.size(); j++) {
		int idx = clothIDs[j];
		sprintf(filename, "%s/%04d_%02d.obj", path, frame, j);
		MeshInfo *mCloth = cloths[idx];
		writeFile(mCloth, filename);
	}
}

