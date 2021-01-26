#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "./src/timer.hpp"

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverSp.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper for CUDA error checking
#include "def.cuh"

#ifdef USE_DOUBLE
#define _bsrmv cusparseDbsrmv
#define _csrmv cusparseDcsrmv
#define _copy cublasDcopy
#define _axpy cublasDaxpy
#define _asum cublasDasum
#define _dot cublasDdot
#define _csrlsvchol cusolverSpDcsrlsvchol
#define _nrm2 cublasDnrm2_v2
#else
#define _bsrmv cusparseSbsrmv
#define _csrmv cusparseScsrmv
#define _copy cublasScopy
#define _axpy cublasSaxpy
#define _asum cublasSasum
#define _dot cublasSdot
#define _csrlsvchol cusolverSpScsrlsvchol
#define _nrm2 cublasSnrm2_v2
#endif

static bool first = true;
static int *d_col, *d_row;
static int *d_col2, *d_row2;
static REAL*d_r, *d_d, *d_q, *d_qq;

const REAL EPS2 = REAL(1.0e-20);
const REAL floatone = REAL(1.0);
const REAL nfloatone = REAL(-1.0);
const REAL floatzero = REAL(0.0);

static cublasHandle_t cublasHandle = 0;
static cusparseHandle_t cusparseHandle = 0;
static cusparseMatDescr_t descr = 0;
static cusolverSpHandle_t cusolverHandle = 0;

void cudaQuit()
{
	cudaDeviceReset();
	exit(EXIT_SUCCESS);
}

cudaDeviceProp deviceProp;

void cublasInit()
{
	/* This will pick the best possible CUDA capable device */
	int devID = findCudaDevice(0, NULL);
	printf("GPU selected Device ID = %d \n", devID);

	if (devID < 0)
	{
		printf("Invalid GPU device %d selected,  exiting...\n", devID);
		exit(EXIT_SUCCESS);
	}
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	/* Statistics about the GPU device */
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
		deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	int version = (deviceProp.major * 0x10 + deviceProp.minor);

	if (version < 0x11)
	{
		printf("ConjugateGradient: requires a minimum CUDA compute 1.1 capability\n");
		cudaQuit();
	}

	/* Create CUBLAS context */
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);
	checkCudaErrors(cublasStatus);

	/* Create CUSPARSE context */
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);
	checkCudaErrors(cusparseStatus);

	/* Description of the A matrix*/
	cusparseStatus = cusparseCreateMatDescr(&descr);
	checkCudaErrors(cusparseStatus);

	/* Define the properties of the matrix */
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	// Create cuSolver context
	cusolverStatus_t cusolverStatus;
	cusolverStatus = cusolverSpCreate(&cusolverHandle);
	checkCudaErrors(cusolverStatus);
}

bool cuda_solver_init(int nVtx, int nElm, int *cooRowInd, int *cooColInd)
{
	/* Allocate required memory */
	d_col = cooColInd;
	d_row = cooRowInd;
	d_col2 = NULL;
	d_row2 = NULL;

	if (first) {
		first = false;

		checkCudaErrors(cudaMalloc((void **)&d_r, nVtx*sizeof(REAL)));
		checkCudaErrors(cudaMalloc((void **)&d_d, nVtx*sizeof(REAL)));
		checkCudaErrors(cudaMalloc((void **)&d_q, nVtx*sizeof(REAL)));
		checkCudaErrors(cudaMalloc((void **)&d_qq, nVtx*sizeof(REAL)));
	}

	return true;
}

void cuda_solver_destory()
{
	checkCudaErrors(cudaFree(d_r));
	checkCudaErrors(cudaFree(d_d));
	checkCudaErrors(cudaFree(d_q));
	checkCudaErrors(cudaFree(d_qq));
	first = true;
}

bool cuda_solver_quit()
{
	return true;
}

void
cudaCSR2COO(int *rows, int n, int m, int *all_rows)
{
	cusparseXcsr2coo(cusparseHandle, rows, n, m,
		all_rows, CUSPARSE_INDEX_BASE_ZERO);
};

extern void cublasDvmul(REAL *a, REAL *b, int num);
extern void cublasDvmul(REAL *a, REAL *b, REAL *c, int num);

//=============================================================================
static void cublasCheck(cublasStatus_t st, const char* msg = nullptr)
{
	if (CUBLAS_STATUS_SUCCESS != st)
	{
		printf("cublas error[%d]: %s", st, msg);
		abort();
		//throw std::exception(msg);
	}
}

static float __pcg_tol = 1e-4f;

// new solver: solving Ax = b on CUDA with Jacobi Preconditioner
bool
cuda_solver_jacobi(int max_iter, int nVtx, int nElm, REAL *cooData, REAL *d_b, REAL *d_x, REAL *d_jacobi)
{
	int N = nVtx;
	REAL *d_val = cooData;

	float norm_b = 0.f;
	cublasCheck(cublasSetPointerMode_v2(cublasHandle, CUBLAS_POINTER_MODE_HOST));
	_nrm2(cublasHandle, N, d_b, 1, &norm_b);

	if (norm_b == 0.f)
		return false;

	// r = b-Ax
	//LargeVector<glm::vec3> r = b - A*x;
	checkCudaErrors(
		_bsrmv(
		cusparseHandle,
		CUSPARSE_DIRECTION_ROW,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		N / 3, N / 3, nElm, &floatone,
		descr, d_val, d_row, d_col, 3, d_x, &floatzero, d_q));
	getLastCudaError("solve");

	checkCudaErrors(
		_copy(cublasHandle, N, d_b, 1, d_r, 1));
	_axpy(cublasHandle, N, &nfloatone, d_q, 1, d_r, 1);
	getLastCudaError("solve");

	int iter = 0;
	float err = 0.f;
	float rd = 0.f, old_rd = 0.f, qAq = 0.f;

	cudaMemset(d_q, 0, nVtx*sizeof(REAL)); // initialize d_q

	for (iter = 0; iter<max_iter; iter++)
	{
		// d = invD * r
		//m_A_diag_d->Mv(r.data(), z.data());
		cublasDvmul(d_r, d_jacobi, d_d, N);

		// rd = r'*d
		//pcg_dot_rz(nVal, r.data(), z.data(), pcg_orz_rz_pAp.data());
		old_rd = rd;
		_dot(cublasHandle, N, d_d, 1, d_r, 1, &rd);

		// q = d+beta*q, beta = rd/old_rd
		//pcg_update_p(nVal, z.data(), p.data(), pcg_orz_rz_pAp.data());
		float beta = (old_rd == 0.f) ? 0.f : rd / old_rd;
		_axpy(cublasHandle, N, &beta, d_q, 1, d_d, 1);
		_copy(cublasHandle, N, d_d, 1, d_q, 1);

		// Aq = A*q, qAq = q'*Aq, alpha = rd / qAq
		//m_A_d->Mv(p.data(), Ap.data());
		//pcg_dot_pAp(nVal, p.data(), Ap.data(), pcg_orz_rz_pAp.data());
		checkCudaErrors(
			_bsrmv(
			cusparseHandle,
			CUSPARSE_DIRECTION_ROW,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			N / 3, N / 3, nElm, &floatone,
			descr, d_val, d_row, d_col, 3, d_q, &floatzero, d_qq));
		float qAq = 0.f;
		_dot(cublasHandle, N, d_qq, 1, d_q, 1, &qAq);

		// x = x + alpha*q, r = r - alpha*Aq
		//pcg_update_x_r(nVal, p.data(), Ap.data(), (float*)m_dv_d.ptr(), r.data(), pcg_orz_rz_pAp.data());
		float alpha = (qAq == 0.f) ? 0.f : rd / qAq;
		float nalpha = -alpha;

		_axpy(cublasHandle, N, &alpha, d_q, 1, d_x, 1);
		_axpy(cublasHandle, N, &nalpha, d_qq, 1, d_r, 1);

		// each several iterations, we check the convergence
		if (iter % 10 == 0)
		{
			// Ap = b - A*x
			//cudaSafeCall(cudaMemcpy(Ap.data(), m_b_d.ptr(), Ap.bytes(), cudaMemcpyDeviceToDevice));
			//m_A_d->Mv((const float*)m_dv_d.ptr(), Ap.data(), -1.f, 1.f);

			checkCudaErrors(
				_bsrmv(
				cusparseHandle,
				CUSPARSE_DIRECTION_ROW,
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				N / 3, N / 3, nElm, &floatone,
				descr, d_val, d_row, d_col, 3, d_x, &floatzero, d_d));

			checkCudaErrors(
				_copy(cublasHandle, N, d_b, 1, d_qq, 1));
			_axpy(cublasHandle, N, &nfloatone, d_d, 1, d_qq, 1);


			float norm_bAx = 0.f;
			cublasCheck(cublasSetPointerMode_v2(cublasHandle, CUBLAS_POINTER_MODE_HOST));
			//cublasSnrm2_v2(m_cublasHandle, nVal, Ap.data(), 1, &norm_bAx);
			_nrm2(cublasHandle, N, d_qq, 1, &norm_bAx);
			err = norm_bAx / (norm_b + 1e-15f);
			if (err < __pcg_tol)
				break;
		}
	} // end for iter

#ifdef OUTPUT_TXT
	printf(" CUDA Jacobi-PCG sovler: iteration = %d, residual = %e \n", iter, err);
#endif

	return true;
}

void gpuSolver(int max_iter,
	int cooN, int *cooRowInd, int *cooColInd, REAL *cooData, bool bsr,
	REAL *dSolverB, REAL *dSolverX, int vtxN, REAL *jacobiDia)
{
	TIMING_BEGIN
	cuda_solver_init(vtxN, cooN, cooRowInd, cooColInd);
	cuda_solver_jacobi(max_iter, vtxN, cooN, cooData, dSolverB, dSolverX, jacobiDia);
	cuda_solver_quit();
	TIMING_END("---------------- gpuSolver");
}

