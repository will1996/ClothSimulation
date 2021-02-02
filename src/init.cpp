//############################################
// interface with GLUT
// Required to include CUDA vector types
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

extern void cublasInit();

void init_cuda(int argc, char **argv)
{
	findCudaDevice(argc, (const char **)argv);
	cublasInit();
}

