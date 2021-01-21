#include <omp.h>

# define	TIMING_BEGIN \
	{double tmp_timing_start = omp_get_wtime();

# define	TIMING_END(message) \
	{double tmp_timing_finish = omp_get_wtime();\
	double  tmp_timing_duration = tmp_timing_finish - tmp_timing_start;\
	printf("%s: %2.5f seconds\n", (message), tmp_timing_duration);}}

#define VLST_BEGIN(lstIdx, lstData, idd) \
	{int vst = (idd == 0) ? 0 : lstIdx[idd-1];\
	int vnum = lstIdx[idd] - vst;\
	for (int vi=0; vi<vnum; vi++) {\
		int vid = lstData[vi+vst];\

#define VLST_END }}

#define FLST_BEGIN(lstIdx, lstData, idd) \
	{int fst = (idd == 0) ? 0 : lstIdx[idd-1];\
	int fnum = lstIdx[idd] - fst;\
	for (int fi=0; fi<fnum; fi++) {\
		int fid = lstData[fi+fst];\

#define FLST_END }}

///////////////////////////////////////////////////////
// show memory usage of GPU
void  reportMemory()
{
	size_t free_byte;
	size_t total_byte;
	cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
	
	if ( cudaSuccess != cuda_status ) {
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
		exit(1);
	}
	
	double free_db = (double)free_byte; 
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
		used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}


///////////////////////////////////////////////////////
// show memory usage of GPU

#define BLOCK_DIM 64

int BPG(int N, int TPB)
{
	int blocksPerGrid = (N + TPB - 1) / (TPB);
	//printf("(N=%d, TPB=%d, stride=1, BPG=%d)\n", N, TPB, blocksPerGrid);
	
	if (blocksPerGrid > 65536) {
		printf("blocksPerGrid is larger than 65536, aborting ... (N=%d, TPB=%d, BPG=%d)\n", N, TPB, blocksPerGrid);
		exit(0);
	}

	return blocksPerGrid;
}

int BPG(int N, int TPB, int &stride)
{
	int blocksPerGrid = 0;
	
	do {
		blocksPerGrid = (N + TPB*stride - 1) / (TPB*stride);
		//printf("(N=%d, TPB=%d, stride=%d, BPG=%d)\n", N, TPB, stride, blocksPerGrid);
	
		if (blocksPerGrid <= 65536)
			return blocksPerGrid;

		printf("blocksPerGrid is larger than 65536, double the stride ... (N=%d, TPB=%d, stride=%d, BPG=%d)\n", N, TPB, stride, blocksPerGrid);
		stride *= 2;
	} while (1);

	assert(0);
	return 0;
}

#define LEN_CHK(l) \
    int idx = blockDim.x * blockIdx.x + threadIdx.x;\
	if (idx >= l) return;

#define BLK_PAR(l) \
   int T = BLOCK_DIM; \
    int B = BPG(l, T);

#define BLK_PAR2(l, s) \
   int T = BLOCK_DIM; \
    int B = BPG(l, T, s);

#define BLK_PAR3(l, s, n) \
   int T = n; \
    int B = BPG(l, T, s);

#define cutilSafeCall checkCudaErrors
#define M_PI       3.14159265358979323846
