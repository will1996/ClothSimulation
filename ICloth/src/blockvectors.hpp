#pragma once

#include "vectors.hpp"

template <int m, int n, typename T>
Vec<m*n, T> mat_to_vec(const Mat<m, n, T> &A) {
	Vec<m*n, T> a;
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			a[i + j*m] = A(i, j);
	return a;
}

template <int m, int n, typename T>
Mat<m, n, T> vec_to_mat(const Vec<m*n, T> &a) {
	Mat<m, n, T> A;
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			A(i, j) = a[i + j*m];
	return A;
}

template <int bn, int m, int n, typename T> Mat<m*bn, n*bn, T> blockdiag(const Mat<m, n, T> &A) {
	Mat<m*bn, n*bn, T> B = 0;
	for (int b = 0; b < bn; b++)
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				B(b*m + i, b*n + j) = A(i, j);
	return B;
}

template <int m, int n> Mat<m*n, m*n, REAL> transpose() {
	Mat<m*n, m*n, double> T = 0;
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			T(n*i + j, i + j*m) = 1;
	return T;
}

template <int n> Mat<n*(n + 1) / 2, n*n> symmetrize();

template <> inline Mat<3, 4> symmetrize<2>() {
	Mat<3, 4> S = Mat<3, 4>(0);
	S(0, 0) = 1.f;
	S(1, 3) = 1.f;
	S(2, 1) = S(2, 2) = 1 / 2.f;
	return S;
}
