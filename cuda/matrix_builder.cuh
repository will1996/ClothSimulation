struct CooMatrix {
	int _num;
	int *_rows;
	int *_cols;
	REAL *_vals;
	bool _bsr;

	void init(int nn, int nvtx, bool bsr)
	{
		_bsr = bsr;
		_num = nn;

		checkCudaErrors(cudaMalloc((void **)&_rows, (nvtx + 1)*sizeof(int)));
		checkCudaErrors(cudaMemset(_rows, 0, (nvtx + 1)*sizeof(int)));

		checkCudaErrors(cudaMalloc((void **)&_cols, nn*sizeof(int)));
		checkCudaErrors(cudaMemset(_cols, 0, nn*sizeof(int)));

		checkCudaErrors(cudaMalloc((void **)&_vals, nn * 9 * sizeof(REAL)));
		checkCudaErrors(cudaMemset(_vals, 0, nn * 9 * sizeof(REAL)));
	}

	void resetValues(int nn, int nvtx, bool bsr)
	{
		_bsr = bsr;
		_num = nn;
		checkCudaErrors(cudaMemset(_vals, 0, nn * 9 * sizeof(REAL)));
	}

	void destroy() {
		checkCudaErrors(cudaFree(_rows));
		checkCudaErrors(cudaFree(_cols));
		checkCudaErrors(cudaFree(_vals));
	}

};

typedef struct {
	int *_colLen; // for counting, used for alloc matIdx
	int *_rowIdx; // for adding, inc with atomicAdd
	int *_rowInc; // for fast locating
	int **_matIdx;
	int _matDim;

	int *_tItems, *_cItems; // all the nodes & compressed nodes
	int _cNum; // compressed length

	int *_hBuffer; // read back buffer on host, used for total only, can be removed later

	//location data
	int *_diaLocs;
	// jocobi vals
	REAL *_jbs;

	void init(int nn) {
		_matDim = nn;
		checkCudaErrors(cudaMalloc((void **)&_colLen, nn*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&_rowInc, nn*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&_rowIdx, nn*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&_matIdx, nn*sizeof(int *)));

		checkCudaErrors(cudaMemset(_colLen, 0, nn*sizeof(int)));
		checkCudaErrors(cudaMemset(_rowInc, 0, nn*sizeof(int)));
		checkCudaErrors(cudaMemset(_rowIdx, 0, nn*sizeof(int)));
		checkCudaErrors(cudaMemset(_matIdx, 0, nn*sizeof(int *)));

		_hBuffer = new int[nn];

		_tItems = _cItems = NULL;
		_cNum = 0;

		//location data
		checkCudaErrors(cudaMalloc(&_diaLocs, nn*sizeof(int)));
		checkCudaErrors(cudaMemset(_diaLocs, 0, nn*sizeof(int)));
		checkCudaErrors(cudaMalloc(&_jbs, nn*9*sizeof(REAL)));
		checkCudaErrors(cudaMemset(_jbs, 0, nn*9*sizeof(REAL)));
	}

	int length() {
		return _cNum;
	}

	void destroy() {
		checkCudaErrors(cudaFree(_colLen));
		checkCudaErrors(cudaFree(_rowInc));
		checkCudaErrors(cudaFree(_rowIdx));
		checkCudaErrors(cudaFree(_matIdx));
		delete[] _hBuffer;

		if (_tItems)
			checkCudaErrors(cudaFree(_tItems));
		if (_cItems)
			checkCudaErrors(cudaFree(_cItems));

		checkCudaErrors(cudaFree(_diaLocs));
		checkCudaErrors(cudaFree(_jbs));
	}

	void	mat_fill_constraint_forces(REAL dt, CooMatrix &A, REAL mrt)
	{
		{
			int num = getHandleNum();
			if (num) {
				BLK_PAR(num);
				kernel_fill_handle_forces << <B, T >> >
					(getHandles(), dt, A._vals, A._bsr, num, _matIdx, _rowInc, totalAux._db,
					currentCloth._dx, currentCloth._dv);
				getLastCudaError("kernel_fill_handle_forces");
			}
		}
		{
			int num = getConstraintNum();
			if (num) {
				BLK_PAR(num);
				kernel_fill_constraint_forces << <B, T >> >
					(getConstraints(), dt, A._vals, A._bsr, num, _matIdx, _rowInc, totalAux._db,
					currentCloth._dx, currentObj._dx, currentCloth._dv, currentObj._dv, mrt);
				getLastCudaError("kernel_fill_constraint_forces");
			}
		}
	}

	void mat_fill_friction_forces(REAL dt, CooMatrix &A, REAL mrt)
	{
		int num = getConstraintNum();
		if (num) {
			BLK_PAR(num);
			kernel_fill_friction_forces << <B, T >> >
				(getConstraints(), dt, A._vals, A._bsr, num, _matIdx, _rowInc, totalAux._db, totalAux._df,
				currentCloth._dx, currentObj._dx, currentCloth._dv, currentObj._dv,
				currentCloth._dm, currentObj._dm, mrt);
			getLastCudaError("kernel_fill_friction_forces");
		}
	}

	void mat_fill_diagonals(REAL dt, CooMatrix &A)
	{
		int num = _matDim;
		BLK_PAR(num);

		if (A._bsr) {
			kernel_mat_fill_bsr << <B, T >> >(dt, A._vals, num, _matIdx, _rowInc,
				totalAux._db, currentCloth._dm, totalAux.dFext, totalAux.dJext, _diaLocs);
			getLastCudaError("kernel_mat_fill_bsr");
		}
		else {
			kernel_mat_fill << <B, T >> >(dt, A._vals, num, _matIdx, _rowInc,
				totalAux._db, currentCloth._dm, totalAux.dFext, totalAux.dJext);
			getLastCudaError("kernel_mat_fill");
		}
	}

	void mat_fill_internal_forces(REAL dt, CooMatrix &A, REAL damping, REAL weakening, REAL damage)
	{
		{
			int num = currentCloth.numFace;

			REAL9 *dF = NULL;
			REAL9x9 *dJ = NULL;

			BLK_PAR(num);
			kernel_internal_face_forces << <B, T >> >(
				currentCloth._dfnod, currentCloth._dv,
				currentCloth.dfa, currentCloth._dx,
				currentCloth._dfidm,
				dt, A._vals, A._bsr, num, _matIdx, _rowInc,
				totalAux._db, currentCloth._dm, 
				totalAux.dFext, totalAux.dJext,
				dMaterialStretching, currentCloth._dfmtr, dF, dJ, damping, weakening, damage);
			getLastCudaError("kernel_internal_face_forces");
		}

		{
			int num = currentCloth.numEdge;
			REAL12 *dF = NULL;
			REAL12x12 *dJ = NULL;

			BLK_PAR(num);
			kernel_internal_edge_forces << <B, T >> >(
				currentCloth.den, currentCloth.def,
				currentCloth._dfnod, currentCloth._dv,
				currentCloth.delen, currentCloth.detheta,
				currentCloth._deitheta, currentCloth._deref,
				currentCloth.dfa, currentCloth._dfn, currentCloth._dfidm,
				currentCloth._dvu, currentCloth._dx, currentCloth._dfvrt,
				dt, A._vals, A._bsr, num, _matIdx, _rowInc,
				totalAux._db, currentCloth._dm, totalAux.dFext, totalAux.dJext, dMaterialBending, currentCloth._dfmtr, dF, dJ, damping);
			getLastCudaError("kernel_internal_edge_forces");
		}
	}

	void mat_add_diagonals(bool counting)
	{
		BLK_PAR(_matDim);
		kernel_mat_add << <B, T >> > (_colLen, _rowIdx, _matIdx, _matDim, counting);
		getLastCudaError("kernel_mat_add");
	}

	void mat_add_constraint_forces(bool counting)
	{
		int num = getConstraintNum();
		if (num == 0)
			return;

		BLK_PAR(num);
		kernel_add_constraint_forces << <B, T >> >(getConstraints(), _colLen, _rowIdx, _matIdx, num, counting);
		getLastCudaError("kernel_add_constraint_forces");
	}

	void mat_add_internal_forces(bool counting)
	{
		{
			int num = currentCloth.numFace;
			BLK_PAR(num);
			kernel_add_face_forces << <B, T >> >(currentCloth._dfnod, _colLen, _rowIdx, _matIdx, num, counting);
			getLastCudaError("kernel_add_face_forces");
		}
		{
			int num = currentCloth.numEdge;
			BLK_PAR(num);
			kernel_add_edge_forces << <B, T >> >(
				currentCloth.den, currentCloth.def, currentCloth._dfnod,
				_colLen, _rowIdx, _matIdx, num, counting);
			getLastCudaError("kernel_add_edge_forces");
		}
	}

	void mat_build_inc(bool bsr)
	{
		{
			BLK_PAR(_matDim);
			kernel_sort_idx << <B, T >> > (_matIdx, _colLen, _rowInc, _matDim);
			getLastCudaError("kernel_idx_sort");
		}

		thrust::device_ptr<int> dev_data(_rowInc);
		thrust::inclusive_scan(dev_data, dev_data + _matDim, dev_data);

		checkCudaErrors(cudaMemcpy(&_cNum, _rowInc + _matDim - 1, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMalloc((void **)&_cItems, _cNum*sizeof(int)));

		{
			int num = _matDim;

			BLK_PAR(num);
			kernel_compress_idx << <B, T >> > (_matIdx, _cItems, _rowInc, num);
			getLastCudaError("kernel_compress_idx");
		}
	}

	void mat_build_space()
	{
		int N = _matDim;

		thrust::device_ptr<int> dev_data(_colLen);
		thrust::inclusive_scan(dev_data, dev_data + N, dev_data);

		int total;
		checkCudaErrors(cudaMemcpy(&total, _colLen + N - 1, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMalloc((void **)&_tItems, total*sizeof(int)));

		{
			int num = _matDim;

			BLK_PAR(num);
			kernel_set_matIdx << <B, T >> >(_matIdx, _colLen, _tItems, num);
			getLastCudaError("kernel_set_matIdx");
		}

	}

	void getColSpace(REAL dt)
	{
		mat_add_diagonals(true);
		mat_add_internal_forces(true);
		mat_add_constraint_forces(true);

		mat_build_space();
	}

	void getColIndex(REAL dt, bool bsr) {
		mat_add_diagonals(false);
		mat_add_internal_forces(false);
		mat_add_constraint_forces(false);

		mat_build_inc(bsr);
	}

	void getJacobiDia(CooMatrix &A)
	{
		int num = _matDim;
		BLK_PAR(num);
		kernel_jacobi_val << <B, T >> >(_jbs, _diaLocs, A._vals, num);
		getLastCudaError("kernel_jacobi_val");
	}

	void generateJacobiMatrix(CooMatrix &A)
	{
		int *jdrow;
		REAL *jbs;

		int num = _matDim;
		cudaMalloc(&jbs, num*3*sizeof(REAL));
		cudaMalloc(&jdrow, A._num*sizeof(int));
		cudaCSR2COO(A._rows, A._num, num, jdrow);

		{
			BLK_PAR(num);
			kernel_jacobi_val << <B, T >> >(jbs, _diaLocs, A._vals, num);
			getLastCudaError("kernel_jacobi_val");

			if (false) {
				REAL *buffer = new REAL[num * 3];
				cudaMemcpy(buffer, jbs, (num*3)*sizeof(REAL),
					cudaMemcpyDeviceToHost);
				printf("here4!\n");

				delete[] buffer;
			}

			kernel_jacobi_applyB << <B, T >> >(totalAux._db, jbs, num);
			if (false) {
				printf("HERE\n");
				REAL *buffer = new REAL[num * 3];
				cudaMemcpy(buffer, totalAux._db, (num * 3)*sizeof(REAL),
					cudaMemcpyDeviceToHost);
				printf("here5!\n");

				delete[] buffer;
			}

			getLastCudaError("kernel_jacobi_applyB");
		}

		{
			BLK_PAR(A._num);
			kernel_jacobi_applyA << <B, T >> >(A._vals, jdrow, jbs, A._num);
			getLastCudaError("kernel_jacobi_applyA");
		}

		cudaFree(jbs);
		cudaFree(jdrow);
	}

	void generateIdx(CooMatrix &A)
	{
		int num = _matDim;

		BLK_PAR(num);
		cudaMemcpy(A._cols, _cItems, _cNum*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(A._rows + 1, _rowInc, num*sizeof(int), cudaMemcpyDeviceToDevice);
	}

	void fillValues(REAL dt, CooMatrix &A, REAL mrt, REAL damping, REAL weakening, REAL damage) {
		mat_fill_diagonals(dt, A);
		mat_fill_internal_forces(dt, A, damping, weakening, damage);
		mat_fill_constraint_forces(dt, A, mrt);
		mat_fill_friction_forces(dt, A, mrt);
	}

	void solveJacobi(int max_iter, CooMatrix &A)
	{
		getJacobiDia(A);
		gpuSolver(max_iter, A._num, A._rows, A._cols, A._vals, A._bsr,
			(REAL *)totalAux._db, (REAL *)totalAux._dx, _matDim * 3, _jbs);
		getLastCudaError("solve");
	}
} SparseMatrixBuilder;

