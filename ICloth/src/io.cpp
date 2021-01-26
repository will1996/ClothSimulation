
#include "io.hpp"

//#include "display.hpp"
//#include "opengl.hpp"
#include "util.hpp"
#ifdef USE_BOOST
#include <boost/filesystem.hpp>
#endif
#include <cassert>
#include <cfloat>
#include <json/json.h>
#include <fstream>
//#include <png.h>
#include <sstream>
using namespace std;

// OBJ meshes

void get_valid_line(istream &in, string &line) {
	do
		getline(in, line);
	while (in && (line.length() == 0 || line[0] == '#'));
}


vector<Face*> triangulate(const vector<Vert*> &verts);

int load_obj_vertices(const string &filename, vector<Vec3> &verts)
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
			verts.push_back(Vec3(xx, yy, zz));
			count++;
		}
	}

	return count;
}

void load_obj(Mesh &mesh, const string &filename)
{
	fstream file(filename.c_str(), ios::in);
	if (!file) {
		cout << "Error: failed to open file " << filename << endl;
		return;
	}
	else
		delete_mesh(mesh);

	while (file) {
		string line;
		get_valid_line(file, line);
		stringstream linestream(line);
		string keyword;
		linestream >> keyword;
		if (keyword == "vt") {
			Vec2 u;
			linestream >> u[0] >> u[1];

			//#define ADJUST_VT
#ifdef ADJUST_VT
			u *= 200.0;
#endif

			mesh.add(new Vert(u));
		}
		else if (keyword == "vl") {
			linestream >> mesh.verts.back()->label;
		}
		else if (keyword == "v") {
			//occationally, double output -> float input error!
			double xx, yy, zz;
			linestream >> xx >> yy >> zz;
			mesh.add(new Node(Vec3(xx, yy, zz), Vec3(0)));

/*			Vec3 x;
			linestream >> x[0] >> x[1] >> x[2];
			mesh.add(new Node(x, Vec3(0)));*/
		}
		else if (keyword == "ny") {
			Vec3 &y = mesh.nodes.back()->y;
			linestream >> y[0] >> y[1] >> y[2];
		}
		else if (keyword == "nv") {
			Vec3 &v = mesh.nodes.back()->v;
			linestream >> v[0] >> v[1] >> v[2];
		}
		else if (keyword == "nl") {
			linestream >> mesh.nodes.back()->label;
		}
		else if (keyword == "e") {
			int n0, n1;
			linestream >> n0 >> n1;
			mesh.add(new Edge(mesh.nodes[n0 - 1], mesh.nodes[n1 - 1]));
		}
		else if (keyword == "ea") {
			linestream >> mesh.edges.back()->theta_ideal;
		}
		else if (keyword == "ed") {
			linestream >> mesh.edges.back()->damage;
		}
		else if (keyword == "el") {
			linestream >> mesh.edges.back()->label;
		}
		else if (keyword == "f") {
			vector<Vert*> verts;
			vector<Node*> nodes;
			string w;
			while (linestream >> w) {
				stringstream wstream(w);
				int v, n;
				char c;
				wstream >> n >> c >> v;
				nodes.push_back(mesh.nodes[n - 1]);
				if (wstream)
					verts.push_back(mesh.verts[v - 1]);
				else if (!nodes.back()->verts.empty())
					verts.push_back(nodes.back()->verts[0]);
				else {
					verts.push_back(new Vert(project<2>(nodes.back()->x),
						nodes.back()->label));
					mesh.add(verts.back());
				}
			}
			for (int v = 0; v < verts.size(); v++)
				connect(verts[v], nodes[v]);
			vector<Face*> faces = triangulate(verts);
			for (int f = 0; f < faces.size(); f++)
				mesh.add(faces[f]);
		}
		else if (keyword == "tl" || keyword == "fl") {
			linestream >> mesh.faces.back()->label;
		}
		else if (keyword == "ts" || keyword == "fs") {
			Mat2x2 &S = mesh.faces.back()->S_plastic;
			linestream >> S(0, 0) >> S(0, 1) >> S(1, 0) >> S(1, 1);
		}
		else if (keyword == "td" || keyword == "fd") {
			linestream >> mesh.faces.back()->damage;
		}
	}
	mark_nodes_to_preserve(mesh);
	compute_ms_data(mesh);
}

void load_objs(vector<Mesh*> &meshes, const string &prefix) {
	char buffer[512];

	for (int m = 0; m < meshes.size(); m++) {
		sprintf(buffer, "%s_%02d.obj", prefix.c_str(), m);
		load_obj(*meshes[m], buffer);
	}
}

static double angle(const Vec3 &x0, const Vec3 &x1, const Vec3 &x2) {
	Vec3 e1 = normalize(x1 - x0);
	Vec3 e2 = normalize(x2 - x0);
	return acos(clamp(dot(e1, e2), REAL(-1.), REAL(1.)));
}

vector<Face*> triangulate(const vector<Vert*> &verts) {
	int n = verts.size();
	double best_min_angle = 0;
	int best_root = -1;
	for (int i = 0; i < n; i++) {
		double min_angle = infinity;
		const Vert *vert0 = verts[i];
		for (int j = 2; j < n; j++) {
			const Vert *vert1 = verts[(i + j - 1) % n], *vert2 = verts[(i + j) % n];
			min_angle = min(min_angle,
				angle(vert0->node->x, vert1->node->x, vert2->node->x),
				angle(vert1->node->x, vert2->node->x, vert0->node->x),
				angle(vert2->node->x, vert0->node->x, vert1->node->x));
		}
		if (min_angle > best_min_angle) {
			best_min_angle = min_angle;
			best_root = i;
		}
	}

	// added by Tangmin, 2014/04/06 00:17 AM
	if (best_root == -1) {
		printf("$$$$$$$$$ degenerated model, triangle best min angle = 0\n");
		best_root = 0;
	}

	int i = best_root;
	Vert* vert0 = verts[i];
	vector<Face*> tris;
	for (int j = 2; j < n; j++) {
		Vert *vert1 = verts[(i + j - 1) % n], *vert2 = verts[(i + j) % n];
		tris.push_back(new Face(vert0, vert1, vert2));
	}
	return tris;
}

void save_obj(const Mesh &mesh, const string &filename) {
	fstream file(filename.c_str(), ios::out);
	for (int v = 0; v < mesh.verts.size(); v++) {
		const Vert *vert = mesh.verts[v];
		file << "vt " << vert->u[0] << " " << vert->u[1] << endl;
		if (vert->label)
			file << "vl " << vert->label << endl;
	}
	for (int n = 0; n < mesh.nodes.size(); n++) {
		const Node *node = mesh.nodes[n];
		file << "v " << node->x[0] << " " << node->x[1] << " "
			<< node->x[2] << endl;
		if (norm2(node->x - node->y))
			file << "ny " << node->y[0] << " " << node->y[1] << " "
			<< node->y[2] << endl;
		if (norm2(node->v))
			file << "nv " << node->v[0] << " " << node->v[1] << " "
			<< node->v[2] << endl;
		if (node->label)
			file << "nl " << node->label << endl;
	}
	for (int e = 0; e < mesh.edges.size(); e++) {
		const Edge *edge = mesh.edges[e];
		if (edge->theta_ideal || edge->label) {
			file << "e " << edge->n[0]->index + 1 << " " << edge->n[1]->index + 1
				<< endl;
			if (edge->theta_ideal)
				file << "ea " << edge->theta_ideal << endl;
			if (edge->damage)
				file << "ed " << edge->damage << endl;
			if (edge->label)
				file << "el " << edge->label << endl;
		}
	}
	for (int f = 0; f < mesh.faces.size(); f++) {
		const Face *face = mesh.faces[f];
		file << "f " << face->v[0]->node->index + 1 << "/" << face->v[0]->index + 1
			<< " " << face->v[1]->node->index + 1 << "/" << face->v[1]->index + 1
			<< " " << face->v[2]->node->index + 1 << "/" << face->v[2]->index + 1
			<< endl;
		if (face->label)
			file << "tl " << face->label << endl;
		if (norm2_F(face->S_plastic)) {
			const Mat2x2 &S = face->S_plastic;
			file << "ts " << S(0, 0) << " " << S(0, 1) << " " << S(1, 0) << " "
				<< S(1, 1) << endl;
		}
		if (face->damage)
			file << "td " << face->damage << endl;
	}
}

void save_objs(const vector<Mesh*> &meshes, const string &prefix) {
	char buffer[512];

	for (int m = 0; m < meshes.size(); m++) {
		sprintf(buffer, "%s_%02d.obj", prefix.c_str(), m);
		save_obj(*meshes[m], buffer);
	}
}

void save_transformation(const Transformation &tr, const string &filename) {
	FILE* file = fopen(filename.c_str(), "w");
	pair<Vec3, double> axis_angle = tr.rotation.to_axisangle();
	Vec3 axis = axis_angle.first;
	double angle = axis_angle.second * 180 / M_PI;
	fprintf(file, "<rotate angle=\"%f\" x=\"%f\" y=\"%f\" z=\"%f\"/>\n",
		angle, axis[0], axis[1], axis[2]);
	fprintf(file, "<scale value=\"%f\"/>\n", tr.scale);
	fprintf(file, "<translate x=\"%f\" y=\"%f\" z=\"%f\"/>\n",
		tr.translation[0], tr.translation[1], tr.translation[2]);
	fclose(file);
}

// Images

void flip_image(int w, int h, unsigned char *pixels);

void save_png(const char *filename, int width, int height,
	unsigned char *pixels, bool has_alpha = false);

#ifndef NO_OPENGL
void save_screenshot(const string &filename) {
#ifdef  XXXXXXXXXXXXXXXXX
	int w = 0, h = 0;
	for (int s = 0; s < 3; s++) {
		glutSetWindow(subwindows[s]);
		w += glutGet(GLUT_WINDOW_WIDTH);
		h = max(h, glutGet(GLUT_WINDOW_HEIGHT));
	}
	unsigned char *pixels = new unsigned char[w*h * 3];
	int x = 0;
	for (int s = 0; s < 3; s++) {
		glutSetWindow(subwindows[s]);
		int wsub = glutGet(GLUT_WINDOW_WIDTH),
			hsub = glutGet(GLUT_WINDOW_HEIGHT);
		unsigned char *pixelsub = new unsigned char[wsub*hsub * 3];
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, wsub, hsub, GL_RGB, GL_UNSIGNED_BYTE, pixelsub);
		for (int j = 0; j < hsub; j++)
			for (int i = 0; i < wsub; i++)
				for (int c = 0; c < 3; c++)
					pixels[(x + i + w*j) * 3 + c] = pixelsub[(i + wsub*j) * 3 + c];
		x += wsub;
		delete[] pixelsub;
	}
	flip_image(w, h, pixels);
	save_png(filename.c_str(), w, h, pixels);
	delete[] pixels;
#endif
}
#endif

void flip_image(int w, int h, unsigned char *pixels) {
	for (int j = 0; j < h / 2; j++)
		for (int i = 0; i < w; i++)
			for (int c = 0; c < 3; c++)
				swap(pixels[(i + w*j) * 3 + c], pixels[(i + w*(h - 1 - j)) * 3 + c]);
}

void save_png(const char *filename, int width, int height,
	unsigned char *pixels, bool has_alpha) {
#ifdef XXXXXXXXXXX
	FILE* file = fopen(filename, "wb");
	if (!file) {
		printf("Couldn't open file %s for writing.\n", filename);
		return;
	}
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL,
		NULL, NULL);
	if (!png_ptr) {
		printf("Couldn't create a PNG write structure.\n");
		fclose(file);
		return;
	}
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		printf("Couldn't create a PNG info structure.\n");
		png_destroy_write_struct(&png_ptr, NULL);
		fclose(file);
		return;
	}
	if (setjmp(png_jmpbuf(png_ptr))) {
		printf("Had a problem writing %s.\n", filename);
		png_destroy_write_struct(&png_ptr, &info_ptr);
		fclose(file);
		return;
	}
	png_init_io(png_ptr, file);
	png_set_IHDR(png_ptr, info_ptr, width, height, 8,
		has_alpha ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB,
		PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT);
	int channels = has_alpha ? 4 : 3;
	png_bytep* row_pointers = (png_bytep*) new unsigned char*[height];
	for (int y = 0; y < height; y++)
		row_pointers[y] = (png_bytep)&pixels[y*width*channels];
	png_set_rows(png_ptr, info_ptr, row_pointers);
	png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
	delete[] row_pointers;
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(file);
#endif
}

#ifdef USE_BOOST

void ensure_existing_directory(const std::string &path) {
#ifdef _DEBUG
	return;
#endif

	using namespace boost::filesystem;
	if (!exists(path))
		create_directory(path);
	if (!is_directory(path)) {
		cout << "Error: " << path << " is not a directory!" << endl;
		abort();
	}
}
#endif
