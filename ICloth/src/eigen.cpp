#include "real.hpp"
#include "eigen.hpp"

#include "Eigen/Dense"
#include "Eigen/Sparse"

using namespace std;
using namespace Eigen;

typedef SparseMatrix<REAL> EigenSpMat;
typedef Triplet<REAL> T;

#ifdef USE_DOUBLE
typedef VectorXd EigenVec;
#else
typedef VectorXf EigenVec;
#endif

void
sparse_to_eigen(const SpMat<REAL> &As, EigenSpMat &Ae, int n)
{
	std::vector< T> triplets;
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < As.rows[i].indices.size(); k++) {
			int j = As.rows[i].indices[k];
			REAL v = As.rows[i].entries[k];
			triplets.push_back(T(i, j, v));
		}
	}

	Ae.setFromTriplets(triplets.begin(), triplets.end());
	Ae.makeCompressed();
}

std::vector<REAL> eigen_linear_solve(const SpMat<REAL> &A,
	const std::vector<REAL> &b)
{
	vector<REAL> x(b.size());

	EigenSpMat EA(b.size(), b.size());
	sparse_to_eigen(A, EA, b.size());

	//Eigen::SparseQR<EigenSpMat> solver;
	Eigen::SparseLU<EigenSpMat, COLAMDOrdering<int> >   solver;
	solver.compute(EA);

	if (solver.info() != Success) {
		cout << "decomposition failed" << endl;
		return x;
	}

	EigenVec EB(b.size());
	for (int i = 0; i < b.size(); i++)
		EB(i) = b[i];

	// Solve system
	EigenVec EX = solver.solve(EB);

	if (solver.info() != Success) {
		cout << "solving failed" << endl;
		return x;
	}

	for (int i = 0; i<b.size(); i++)
		x[i] = EX(i);

	return x;
}
