
#include "vectors.hpp"
#include "blockvectors.hpp"

using namespace std;

#if defined(_WIN32)

//#include <mkl_lapacke.h>

#else

// LAPACK stuff
extern "C" {

#ifdef _WIN32
#define LAPACKE_dgesvd dgesvd_
#define LAPACKE_dsyev  dsyev_
#endif

#define lapack_int int
#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102
lapack_int LAPACKE_dsyev( int matrix_order, char jobz, char uplo, lapack_int n,
                          REAL* a, lapack_int lda, REAL* w );
lapack_int LAPACKE_dgesvd( int matrix_order, char jobu, char jobvt,
                           lapack_int m, lapack_int n, REAL* a,
                           lapack_int lda, REAL* s, REAL* u, lapack_int ldu,
                           REAL* vt, lapack_int ldvt, REAL* superb );

}
#endif

template <int n> Eig<n> eigen_decomposition (const Mat<n,n> &A) {
    Eig<n> eig;
    Vec<n*n> a = mat_to_vec(A);
    Vec<n> &w = eig.l;
    int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, &a[0], n, &w[0]);
    if (info != 0)
        cout << "LAPACKE_dsyev failed with return value " << info << " on matrix " << A << endl;
    // SSYEV overwrites a with the eigenvectors
    eig.Q = vec_to_mat<n,n>(a);
    for (int i = 0; i < n/2; i++) {
        swap(eig.l[i], eig.l[n-i-1]);
        swap(eig.Q.col(i), eig.Q.col(n-i-1));
    }
    return eig;
}

template<> Eig<2> eigen_decomposition<2>(const Mat2x2 &A) {
#if 0
    Eig<2> eig0;
    {
      Vec<2*2> a = mat_to_vec(A);
      Vec<2> &w = eig0.l;
      int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', 2, &a[0], 2, &w[0]);
      if (info != 0)
        cout << "LAPACKE_dsyev failed with return value " << info << " on matrix " << A << endl;
      // SSYEV overwrites a with the eigenvectors
      eig0.Q = vec_to_mat<2,2>(a);
      for (int i = 0; i < 2/2; i++) {
        swap(eig0.l[i], eig0.l[2-i-1]);
        swap(eig0.Q.col(i), eig0.Q.col(2-i-1));
      }
    }
	return eig0;
#else
    // http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
    // http://en.wikipedia.org/wiki/Eigenvalue_algorithm
    Eig<2> eig;
    REAL a   = A(0,0), b = A(1,0), d = A(1,1); // A(1,0) == A(0,1)
    REAL amd = a - d;
    REAL apd = a + d;
    REAL b2  = b * b;
    REAL det = sqrt(4*b2 + amd*amd);
    REAL l1  = 0.5 * (apd + det);
    REAL l2  = 0.5 * (apd - det);

    eig.l[0] = l1;
    eig.l[1] = l2;

    REAL v0, v1, vn;
    if (b) {
      v0 = l1 - d;
      v1 = b;
      vn = sqrt(v0*v0 + b2);
      eig.Q(0,0) = v0/vn;
      eig.Q(1,0) = v1/vn;

      v0 = l2 - d;
      vn = sqrt(v0*v0 + b2);
      eig.Q(0,1) = v0/vn;
      eig.Q(1,1) = v1/vn;
    } else if (a >= d) {
      eig.Q(0,0) = 1;
      eig.Q(1,0) = 0;
      eig.Q(0,1) = 0;
      eig.Q(1,1) = 1;
    } else {
      eig.Q(0,0) = 0;
      eig.Q(1,0) = 1;
      eig.Q(0,1) = 1;
      eig.Q(1,1) = 0;
    }

	return eig;
#endif
}

template <int m, int n> SVD<m,n> singular_value_decomposition (const Mat<m,n> &A) {
    SVD<m,n> svd;
    Vec<m*n> a = mat_to_vec(A);
    Vec<m*m> u;
    Vec<n> &s = svd.s;
    Vec<n*n> vt;
    Vec<n> superb;
    int info  = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, &a[0], m,
                               &s[0], &u[0], m, &vt[0], n, &superb[0]);
    if (info != 0)
        cout << "LAPACKE_dgesvd failed with return value " << info << " on matrix " << A << endl;
    svd.U = vec_to_mat<m,m>(u);
    svd.Vt = vec_to_mat<n,n>(vt);
    return svd;
}

template<> SVD<3,2> singular_value_decomposition<3,2> (const Mat<3,2> &A) {
  //SVD<3,2> svd0 = singular_value_decomposition0(A);
	SVD<3,2> svd;
	const Vec<3>& c0 = A.col(0);
	const Vec<3>& c1 = A.col(1);
	REAL a0   = dot(c0, c0); //        |a0  b|
	REAL b    = dot(c0, c1); // A*A' = |     |
	REAL a1   = dot(c1, c1); //        |b  a1|
	REAL am   = a0 - a1;
	REAL ap   = a0 + a1;
	REAL det  = sqrt(am*am + 4*b*b);
  // eigen values
	REAL ev0  = sqrt(0.5 * (ap + det));
	REAL ev1  = sqrt(0.5 * (ap - det));
	svd.s[0]    = ev0;
	svd.s[1]    = ev1;
	// http://en.wikipedia.org/wiki/Trigonometric_identities
	REAL sina, cosa;
	if (b == 0) {
		sina = 0;
		cosa = 1;
	} else {
		REAL tana = (am - det) / (2*b);
		cosa = 1.0/sqrt(1 + tana*tana);
		sina = tana * cosa;
	}
  // 2x2
	svd.Vt(0,0) = -cosa;
	svd.Vt(1,0) =  sina;
	svd.Vt(0,1) =  sina;
	svd.Vt(1,1) =  cosa;
  // 3x3
  REAL t00  = -cosa/ev0;
  REAL t10  =  sina/ev0;
	svd.U(0,0)  = t00*c0[0] + t10*c1[0];
	svd.U(1,0)  = t00*c0[1] + t10*c1[1];
	svd.U(2,0)  = t00*c0[2] + t10*c1[2];
  REAL t01  =  sina/ev1;
  REAL t11  =  cosa/ev1;
	svd.U(0,1)  = t01*c0[0] + t11*c1[0];
	svd.U(1,1)  = t01*c0[1] + t11*c1[1];
	svd.U(2,1)  = t01*c0[2] + t11*c1[2];
  svd.U.col(2)= cross(svd.U.col(0), svd.U.col(1));
  return svd;
}
