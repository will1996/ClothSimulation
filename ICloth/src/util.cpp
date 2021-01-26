
#include "util.hpp"
#include "io.hpp"
#include "mesh.hpp"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
using namespace std;

template <typename T> string name(const T *p) {
	stringstream ss;
	ss << setw(3) << setfill('0') << hex << ((size_t)p / sizeof(T)) % 0xfff;
	return ss.str();
}

ostream &operator<< (ostream &out, const Vert *vert) {
	out << "v:" << name(vert); return out;
}

ostream &operator<< (ostream &out, const Node *node) {
	out << "n:" << name(node) << node->verts; return out;
}

ostream &operator<< (ostream &out, const Edge *edge) {
	out << "e:" << name(edge) << "(" << edge->n[0] << "-" << edge->n[1] << ")"; return out;
}

ostream &operator<< (ostream &out, const Face *face) {
	out << "f:" << name(face) << "(" << face->v[0] << "-" << face->v[1] << "-" << face->v[2] << ")"; return out;
}

const REAL infinity = numeric_limits<REAL>::infinity();

int solve_quadratic(REAL a, REAL b, REAL c, REAL x[2]) {
	// http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
	REAL d = b*b - 4 * a*c;
	if (d < 0) {
		x[0] = -b / (2 * a);
		return 0;
	}
	REAL q = -(b + sgn(b)*sqrt(d)) / 2;
	int i = 0;
	if (abs(a) > 1e-12*abs(q))
		x[i++] = q / a;
	if (abs(q) > 1e-12*abs(c))
		x[i++] = c / q;
	if (i == 2 && x[0] > x[1])
		swap(x[0], x[1]);
	return i;
}

bool is_seam_or_boundary(const Vert *v) {
	return is_seam_or_boundary(v->node);
}

bool is_seam_or_boundary(const Node *n) {
	for (int e = 0; e < n->adje.size(); e++)
		if (is_seam_or_boundary(n->adje[e]))
			return true;
	return false;
}

bool is_seam_or_boundary(const Edge *e) {
	return !e->adjf[0] || !e->adjf[1] || edge_vert(e, 0, 0) != edge_vert(e, 1, 0);
}

bool is_seam_or_boundary(const Face *f) {
	return is_seam_or_boundary(f->adje[0])
		|| is_seam_or_boundary(f->adje[1])
		|| is_seam_or_boundary(f->adje[2]);
}

void debug_save_meshes(const vector<Mesh*> &meshvec, const string &name,
	int n) {
	static map<string, int> savecount;
	if (n == -1)
		n = savecount[name];
	else
		savecount[name] = n;

	char buffer[512];
	sprintf(buffer, "tmp/%s%04d", name.c_str(), n);
	save_objs(meshvec, buffer);
	savecount[name]++;
}

void debug_save_mesh(const Mesh &mesh, const string &name, int n) {
	static map<string, int> savecount;
	if (n == -1)
		n = savecount[name];
	else
		savecount[name] = n;

	char buffer[512];
	sprintf(buffer, "tmp/%s%04d", name.c_str(), n);
	save_obj(mesh, buffer);
	savecount[name]++;
}

void output1(const char *fname, REAL *data, int len)
{
	FILE *fp = fopen(fname, "wt");
	if (!fp) return;

	for (int i = 0; i<len / 3; i++) {
		fprintf(fp, "%lg, %lg, %lg\n", data[i * 3] * 1000, data[i * 3 + 1] * 1000, data[i * 3 + 2] * 1000);
	}
	fclose(fp);
}

void input2(const char *fname, REAL *data, int len)
{
	FILE *fp = fopen(fname, "rb");
	if (!fp) return;

	fread(data, sizeof(REAL), len, fp);
	fclose(fp);
}

void output2(const char *fname, REAL *data, int len)
{
	FILE *fp = fopen(fname, "wb");
	if (!fp) return;

	fwrite(data, sizeof(REAL), len, fp);
	fclose(fp);
}

void output(char *fname, int *data, int len)
{
	FILE *fp = fopen(fname, "wt");
	for (int i = 0; i<len; i++) {
		fprintf(fp, "%d ", data[i]);
		if (i != 0 && (i % 10 == 0))
			fprintf(fp, "\n");
	}
	fclose(fp);
}

void output(char *fname, float *data, int len)
{
#ifdef TXT
	FILE *fp = fopen(fname, "wt");
	if (!fp) return;

	for (int i = 0; i<len; i++) {
		fprintf(fp, "%lg ", data[i] * 1000);
		if (i != 0 && (i % 10 == 0))
			fprintf(fp, "\n");
	}
	fclose(fp);
#else
	FILE *fp = fopen(fname, "wb");
	fwrite(&len, sizeof(int), 1, fp);
	fwrite(data, sizeof(float), len, fp);
	fclose(fp);
#endif
}

void output(char *fname, double *data, int len)
{
#ifdef TXT
	FILE *fp = fopen(fname, "wt");
	if (!fp) return;

	for (int i = 0; i<len; i++) {
		fprintf(fp, "%lg ", data[i] * 1000);
		if (i != 0 && (i % 10 == 0))
			fprintf(fp, "\n");
	}
	fclose(fp);
#else
	FILE *fp = fopen(fname, "wb");
	fwrite(&len, sizeof(int), 1, fp);
	fwrite(data, sizeof(double), len, fp);
	fclose(fp);
#endif
}

void output9(char *fname, REAL *data, int len, bool txt)
{
	if (txt) {
		FILE *fp = fopen(fname, "wt");
		if (!fp) return;

		for (int i = 0; i<len * 9; i++) {
			if (data[i] == 0)
				data[i] = 0;

			fprintf(fp, "%lg ", data[i]);
			if (i % 9 == 8)
				fprintf(fp, "\n");
		}
		fclose(fp);
	}
	else {
		FILE *fp = fopen(fname, "wb");
		fwrite(data, sizeof(REAL), len * 9, fp);
		fclose(fp);
	}
}

void output12(char *fname, REAL *data, int len)
{
	FILE *fp = fopen(fname, "wt");
	if (!fp) return;

	for (int i = 0; i<len * 12; i++) {
		//		if (i == 410*12)
		//			fprintf(fp, "###");

		if (data[i] == 0)
			data[i] = 0;

		fprintf(fp, "%lg ", data[i]);
		if (i % 12 == 11)
			fprintf(fp, "\n");
	}
	fclose(fp);
}

void output3x9(REAL *data, int st, FILE *fp)
{
	REAL *idx = data + st;

	for (int k = 0; k<3; k++) {
		for (int i = 0; i<9; i++) {
			REAL v = idx[i * 3];
			if (v == 0)
				v = 0;
			fprintf(fp, "%lg ", v);
		}

		fprintf(fp, "\n");
		idx += 1;
	}
}

void output9x9(char *fname, REAL *data, int len, bool txt)
{
	if (txt) {
		FILE *fp = fopen(fname, "wt");
		if (!fp) return;

		for (int i = 0; i<len; i++) {
			output3x9(data, 0, fp);
			output3x9(data, 27, fp);
			output3x9(data, 54, fp);

			data += 81;
		}

		fclose(fp);
	}
	else {
		FILE *fp = fopen(fname, "wb");
		fwrite(data, sizeof(REAL), len * 9 * 9, fp);
		fclose(fp);
	}
}


void output3x12(REAL *data, int st, FILE *fp)
{
	REAL *idx = data + st;

	for (int k = 0; k<3; k++) {
		for (int i = 0; i<12; i++) {
			REAL v = idx[i * 3];
			if (v == 0)
				v = 0;
			fprintf(fp, "%lg ", v);
		}

		fprintf(fp, "\n");
		idx += 1;
	}
}

void output12x12(char *fname, REAL *data, int len)
{
	FILE *fp = fopen(fname, "wt");
	if (!fp) return;

	for (int i = 0; i<len; i++) {
		output3x12(data, 0, fp);
		output3x12(data, 36, fp);
		output3x12(data, 72, fp);
		output3x12(data, 108, fp);

		data += 144;
	}

	fclose(fp);
}
