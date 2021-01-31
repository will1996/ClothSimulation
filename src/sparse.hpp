#ifndef SPARSE_HPP
#define SPARSE_HPP

#include "util.hpp"
#include <fstream>
#include <utility>
#include <vector>

inline size_t find_index (int i, const std::vector<int> &indices) {
    for (size_t ii = 0; ii < indices.size(); ii++)
        if (indices[ii] == i)
            return ii;
    return indices.size();
}

template <typename T> void insert_index (int i, int j,
                                         std::vector<int> &indices,
                                         std::vector<T> &entries) {
    indices.insert(indices.begin() + j, i);
    entries.insert(entries.begin() + j, T(0));
}

template <typename T> struct SpVec {
    std::vector<int> indices;
    std::vector<T> entries;
    T operator[] (int i) const {
        size_t j = find_index(i, indices);
        if (j >= indices.size() || indices[j] != i)
            return T(0);
        else
            return entries[j];
    }
    T &operator[] (int i) {// inserts entry as side-effect
        size_t j = find_index(i, indices);
        if (j >= indices.size() || indices[j] != i)
            insert_index(int(i), int(j), indices, entries);
        return entries[j];
    }
};

template <typename T>
std::ostream &operator<< (std::ostream &out, const SpVec<T> &v) {
    out << "[";
    for (int k = 0; k < v.indices.size(); k++)
        out << (k==0 ? "" : ", ") << v.indices[k] << ": " << v.entries[k];
    out << "]";
    return out;
}

template <typename T> struct SpMat {
    int m, n;
    std::vector< SpVec<T> > rows;
    SpMat (): m(0), n(0), rows() {}
    explicit SpMat (int m, int n): m(m), n(n), rows(m) {}
    T operator() (int i, int j) const {
        return rows[i][j];
    }
    T &operator() (int i, int j) {// inserts entry as side-effect
        return rows[i][j];
    }

	void apply(const std::vector<T> &x, std::vector <T> &y) const
	{
		for (int i = 0; i<m; ++i){
			double d = 0;
			const SpVec<T> &r = rows[i];

			for (int jj = 0; jj < r.indices.size(); jj++) {
				int p_index = r.indices[jj];
				const double &p_value = r.entries[jj];

				d += p_value*x[p_index];
			}

			y[i] = d;
		}
	}

	void apply_transpose(const std::vector<T> &x, std::vector<T> &y) const
	{
		//BLAS::set_zero(n, y);
		// input must be reset before

		for (int i = 0; i<m; ++i){
			const SpVec<T> &r = rows[i];
			double xi = x[i];

			for (int jj = 0; jj < r.indices.size(); jj++) {
				int p_index = r.indices[jj];
				const double &p_value = r.entries[jj];

				y[p_index] += p_value*xi;
			}
		}
	}

};

template <typename T>
std::ostream &operator<< (std::ostream &out, const SpMat<T> &A) {
    out << "[";
    for (int i = 0; i < A.m; i++) {
        const SpVec<T> &row = A.rows[i];
        for (int jj = 0; jj < row.indices.size(); jj++) {
            int j = row.indices[jj];
            const T &aij = row.entries[jj];
            out << (i==0 && jj==0 ? "" : ", ") << "(" << i << "," << j
                << "): " << aij;
        }
    }
    out << "]";
    return out;
}
#endif
