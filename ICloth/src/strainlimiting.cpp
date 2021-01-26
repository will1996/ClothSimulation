#include "strainlimiting.hpp"

#include "optimization.hpp"
#include "simulation.hpp"

using namespace std;

vector<Vec2> get_strain_limits (const vector<Cloth> &cloths) {
    vector<Vec2> strain_limits;
    for (int c = 0; c < cloths.size(); c++) {
        const Cloth &cloth = cloths[c];
        const Mesh &mesh = cloth.mesh;
        int f0 = strain_limits.size();
        strain_limits.resize(strain_limits.size() + cloth.mesh.faces.size());
        for (int f = 0; f < mesh.faces.size(); f++) {
            const Cloth::Material *material =
                cloth.materials[mesh.faces[f]->label];
            strain_limits[f0+f] = Vec2(material->strain_min,
                                       material->strain_max);
        }
    }
    return strain_limits;
}

struct SLOpt: public NLConOpt {
    vector<Mesh*> meshes;
    int nn, nf;
    const vector<Vec2> &strain_limits;
    const vector<Constraint*> &cons;
    vector<Vec3> xold;
    vector<double> conold;
    mutable vector<double> s;
    mutable vector<Mat3x3> sg;
    REAL inv_m;
    SLOpt (vector<Mesh*> &meshes, const vector<Vec2> &strain_limits,
           const vector<Constraint*> &cons):
          meshes(meshes), nn(size<Node>(meshes)), nf(size<Face>(meshes)),
          strain_limits(strain_limits), cons(cons),
          xold(node_positions(meshes)), s(nf*2), sg(nf*2) {
        nvar = nn*3;
        ncon = cons.size() + nf*4;
        conold.resize(cons.size());
        for (int j = 0; j < cons.size(); j++)
            conold[j] = cons[j]->value();
        inv_m = 0;
        for (int n = 0; n < nn; n++)
            inv_m += 1/get<Node>(n, meshes)->m;
        inv_m /= nn;
    }
    void initialize (double *x) const;
    REAL objective (const double *x) const;
    void obj_grad (const double *x, double *grad) const;
    void precompute (const double *x) const;
    REAL constraint (const double *x, int j, int &sign) const;
    void con_grad (const double *x, int j, double factor, double *grad) const;
    void finalize (const double *x) const;
};

extern void augmented_lagrangian_method2(const NLConOpt &problem);
extern void augmented_lagrangian_method3(const NLConOpt &problem);

void strain_limiting_opt
	(vector<Mesh*> &meshes, const vector<Vec2> &strain_limits,
                      const vector<Constraint*> &cons) {
	OptOptions opts;
	opts.max_iter(20000);
	augmented_lagrangian_method(SLOpt(meshes, strain_limits, cons), opts);
}


#include "io.hpp"

void strain_limiting_jaccobi
	(Mesh &mesh, Vec2 strain_limits, const vector<Constraint*> &cons)
{
	REAL max_strain = strain_limits[0];
	REAL min_strain = strain_limits[1];

	int numVerts = mesh.verts.size();
	int numFaces = mesh.faces.size();

	int *weightSum = new int[numVerts];
	Vec3 *tempSum = new Vec3[numVerts];

	bool done = false;
	int iterations = 0;

	while (!done) {
		memset(weightSum, 0, sizeof(int)*numVerts);
		memset(tempSum, 0, sizeof(Vec3)*numVerts);
		iterations++;

		int find = 0;
		for (int face_i = 0; face_i < numFaces; face_i++)
		{
			Face *face = mesh.faces[face_i];
			Vert* v0 = face->v[0];
			Vert* v1 = face->v[1];
			Vert* v2 = face->v[2];

			{
				Mat3x2 F = derivative(v0->node->x, v1->node->x, v2->node->x, face);
				SVD<3, 2> svd = singular_value_decomposition(F);

				if (svd.s[0] >= min_strain && svd.s[0] <= max_strain &&
					svd.s[1] >= min_strain && svd.s[1] <= max_strain)
					continue;

				find++;
				svd.s[0] = clamp(svd.s[0], min_strain, max_strain);
				svd.s[1] = clamp(svd.s[1], min_strain, max_strain);
				Mat3x2 S = Mat3x2(Vec3(svd.s[0], 0, 0), Vec3(0, svd.s[1], 0));
				Mat3x2 Fnew = svd.U*S*svd.Vt;
				Vec2 c = (v0->u + v1->u + v2->u) / REAL(3.);
				for (int i = 0; i < 3; i++) {
					Vec2 u = face->v[i]->u - c;
					Vec3 x = F*u, xnew = Fnew*u;
					tempSum[face->v[i]->index] += xnew - x;
					weightSum[face->v[i]->index]++;
				}
				continue;
			}
		}

		printf("%d: %d triangles need to be adjusted.\n", iterations, find);

		if (find == 0)
			break;

		for (int v = 0, v3 = 0; v<numVerts; v++, v3 += 3) {
			if (weightSum[v] != 0) {
				// Each core wrote to a different spot in memory
				// Add all of these up
				mesh.verts[v]->node->x += tempSum[v] / REAL(weightSum[v]);
			}
		}

		// re-enforce constraints
		for (int c = 0; c < cons.size(); c++)
			cons[c]->apply();

		if (iterations > 0 && (iterations % 1000 == 0)) {
			char buffer[512];
			sprintf(buffer, "meshes/debug%05d.obj", iterations);
			save_obj(mesh, buffer);
		}

		if (iterations > 10000)
			done = true;
	}
}

void strain_limiting
	(vector<Mesh*> &meshes, const vector<Vec2> &strain_limits, const vector<Constraint*> &cons)
{
	for (int i = 0; i < meshes.size(); i++)
		strain_limiting_jaccobi(*meshes[i], strain_limits[i], cons);
}

void SLOpt::initialize (double *x) const {
    for (int n = 0; n < nn; n++) {
        const Node *node = get<Node>(n, meshes);
        set_subvec(x, n, node->x);
    }
}

void SLOpt::precompute (const double *x) const {
#pragma omp parallel for
    for (int n = 0; n < nn; n++)
        get<Node>(n, meshes)->x = get_subvec(x, n);
#pragma omp parallel for
    for (int f = 0; f < nf; f++) {
        const Face *face = get<Face>(f, meshes);
        Mat3x2 F = derivative(face->v[0]->node->x, face->v[1]->node->x,
                              face->v[2]->node->x, face);
        SVD<3,2> svd = singular_value_decomposition(F);
        s[f*2+0] = svd.s[0];
        s[f*2+1] = svd.s[1];
        Mat3x2 Vt_ = Mat3x2(0);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Vt_(i,j) = svd.Vt(i,j);
        Mat3x2 Delta = Mat3x2(Vec3(-1,1,0),Vec3(-1,0,1));
        Mat3x3 &Left = svd.U;
        Mat3x3 Right = Delta*face->invDm*Vt_.t();
        sg[f*2+0] = outer(Left.col(0), Right.col(0));
        sg[f*2+1] = outer(Left.col(1), Right.col(1));
    }
}

REAL SLOpt::objective (const double *x) const {
    double f = 0;
#pragma omp parallel for reduction (+: f)
    for (int n = 0; n < nn; n++) {
        const Node *node = get<Node>(n, meshes);
        Vec3 dx = node->x - xold[n];
        f += inv_m*node->m*norm2(dx)/2.;
    }
    return f;
}

void SLOpt::obj_grad (const double *x, double *grad) const {
#pragma omp parallel for
    for (int n = 0; n < nn; n++) {
        const Node *node = get<Node>(n, meshes);
        Vec3 dx = node->x - xold[n];
        set_subvec(grad, n, inv_m*node->m*dx);
    }
}

REAL strain_con (const SLOpt &sl, const double *x, int j, int &sign);
void strain_con_grad (const SLOpt &sl, const double *x, int j, double factor,
                      double *grad);

REAL SLOpt::constraint (const double *x, int j, int &sign) const {
    if (j < cons.size())
        return cons[j]->value(&sign) - conold[j];
    else
        return strain_con(*this, x, j-cons.size(), sign);
}

void SLOpt::con_grad (const double *x, int j, double factor,
                      double *grad) const {
    if (j < cons.size()) {
        MeshGrad mgrad = cons[j]->gradient();
        for (MeshGrad::iterator it=mgrad.begin(); it!=mgrad.end(); it++) {
            int n = get_index(it->first, meshes);
            if (n == -1)
                continue;
            const Vec3 &g = it->second;
            for (int i = 0; i < 3; i++)
                grad[n*3+i] += factor*g[i];
        }
    } else
        strain_con_grad(*this, x, j-cons.size(), factor, grad);
}

REAL strain_con (const SLOpt &sl, const double *x, int j, int &sign) {
    int f = j/4;
    int a = j/2; // index into s, sg
    const Face *face = get<Face>(f, sl.meshes);
    double strain_min = sl.strain_limits[f][0],
           strain_max = sl.strain_limits[f][1];
    double c;
    double w = sqrt(face->a);
    if (strain_min == strain_max) {
        sign = 0;
        c = (j%2 == 0) ? w*(sl.s[a] - strain_min) : 0;
    } else {
        if (j%2 == 0) { // lower bound
            sign = 1;
            c = w*(sl.s[a] - strain_min);
        } else { // upper bound
            sign = -1;
            c = w*(sl.s[a] - strain_max);
        }
    }
    return c;
}

void add_strain_row (const Mat3x3 &sg, const Face *face,
                     const vector<Mesh*> &meshes, double factor, double *grad);

void strain_con_grad (const SLOpt &sl, const double *x, int j, double factor,
                      double *grad) {
    int f = j/4;
    int a = j/2; // index into s, sg
    const Face *face = get<Face>(f, sl.meshes);
    REAL strain_min = sl.strain_limits[f][0],
           strain_max = sl.strain_limits[f][1];
    REAL w = sqrt(face->a);
    if (strain_min == strain_max) {
        if (j%2 == 0)
            add_strain_row(w*sl.sg[a], face, sl.meshes, factor, grad);
    } else
        add_strain_row(w*sl.sg[a], face, sl.meshes, factor, grad);
}

void add_strain_row (const Mat3x3 &sg, const Face *face,
                     const vector<Mesh*> &meshes, double factor, double *grad) {
    for (int i = 0; i < 3; i++) {
        int n = get_index(face->v[i]->node, meshes);
        for (int j = 0; j < 3; j++)
            grad[n*3+j] += factor*sg(j,i);
    }
}

void SLOpt::finalize (const double *x) const {
    for (int n = 0; n < nn; n++)
        get<Node>(n, meshes)->x = get_subvec(x, n);
}

// DEBUG

void debug_cpu (char *ofile)
{
    Mesh mesh;
    //load_obj(mesh, "meshes/square35785.obj");
	load_obj(mesh, ofile);

    Mat3x3 M = diag(Vec3(2., 0.5, 1.));
    for (int n = 0; n < mesh.nodes.size(); n++) {
        mesh.nodes[n]->x[0] *= 2;
        mesh.nodes[n]->x[1] *= 0.5;
        mesh.nodes[n]->m = mesh.nodes[n]->a;
    }
	save_obj(mesh, "meshes/debug1.obj");

    vector<Mesh*> meshes(1, &mesh);
    vector<Vec2> strain_limits(mesh.faces.size(), Vec2(0.95, 1.05));
    Timer timer;
    strain_limiting(meshes, strain_limits, vector<Constraint*>());
    timer.tock();
    cout << "cpu total time: " << timer.total << endl;
    save_obj(mesh, "meshes/debug2.obj");
}

extern void strain_limiting_gpu(double, double, double, int);
extern void strain_limiting_gpu_jaccobi(REAL, REAL, REAL, int, REAL);

extern void init_data_gpu(Mesh &);
extern void pop_data_gpu(Mesh &m);
	
static Mesh mesh;

void debug_gpu(char *ofile)
{
	/*
	//load_obj(mesh, "meshes/square35785.obj");
	load_obj(mesh, ofile);

	Mat3x3 M = diag(Vec3(2., 0.5, 1.));
	for (int n = 0; n < mesh.nodes.size(); n++) {
		mesh.nodes[n]->x[0] *= 2;
		mesh.nodes[n]->x[1] *= 0.5;
		mesh.nodes[n]->m = mesh.nodes[n]->a;
	}
	save_obj(mesh, "meshes/debug3.obj");

	init_data_gpu(mesh);

	Timer timer;
	//strain_limiting_gpu(0.95, 1.05, 0.001);
	strain_limiting_gpu_jaccobi(0.95, 1.05, 0.001, 10000);
	timer.tock();
	cout << "gpu total time: " << timer.total << endl;

	pop_data_gpu(mesh);
	save_obj(mesh, "meshes/debug4.obj");
	*/
}

void output_gpu(int i)
{
	/*
	pop_data_gpu(mesh);
	char buffer[512];
	sprintf(buffer, "meshes/gdebug%05d.obj", i);
	save_obj(mesh, buffer);
	*/
}

#define OBJ_FILE "meshes/square-9600.obj"
//#define OBJ_FILE "meshes/square-600.obj"

void debug(const vector<string> &args)
{
//	debug_cpu(OBJ_FILE);
	debug_gpu(OBJ_FILE);
}