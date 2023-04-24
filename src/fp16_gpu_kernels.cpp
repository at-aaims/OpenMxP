#include "fp16_gpu_kernels.h"

using namespace std;

#define TILE_DIM 64
#define TILE_DIM_TRANS 64
#define TILE_DIM_ID 64
#define BLOCK_ROWS 4

// LEFT CONVERT KERNELS
__global__ void copyCoalesced(__half *odata, const float *idata, int olda,
                              int ilda) {
    /*__shared__ float tile[TILE_DIM][TILE_DIM];

    int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
    int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
    // int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[(GPU_THREADIDX_Y + j)][GPU_THREADIDX_X] =
            idata[(long(y) +long( j)) * long(ilda) +long( x)];

    __syncthreads();

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(long(y) +long( j)) *long( olda) +long( x)] =
            __float2half(tile[(GPU_THREADIDX_Y + j)][GPU_THREADIDX_X]);*/

    int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
    int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
    // int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(long(y) +long( j)) *long( olda) +long( x)] =
            __float2half(idata[(long(y) +long( j)) * long(ilda) +long( x)]);
}

/// HUGE ASSUMTION OF ilda = olda
__global__ void copyCoalesced_lower(__half *odata, const float *idata, int olda,
                              int ilda) {
    if(GPU_BLOCKIDX_Y > GPU_BLOCKIDX_X){ return;};
    if(GPU_BLOCKIDX_X == GPU_BLOCKIDX_Y)
    {
        int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
        int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
        // int width = gridDim.x * TILE_DIM;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            if(x > y+j)
                odata[(y+j) * olda + x] =
                    __float2half(idata[(y+j) * olda + x]);
        }
        return;
    }
    
    if(GPU_BLOCKIDX_X > GPU_BLOCKIDX_Y)
    {
        int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
        int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            odata[(y+j) * olda + x] =
                __float2half(idata[(y+j) * olda + x]);
        return;
    }
}

/// HUGE ASSUMTION OF ilda = olda
__global__ void copyCoalesced_upper(__half *odata, const float *idata, int olda,
                              int ilda) {
    if(GPU_BLOCKIDX_Y < GPU_BLOCKIDX_X){ return;};
    if(GPU_BLOCKIDX_X == GPU_BLOCKIDX_Y)
    {
        int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
        int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            if(x < y+j+1)
                odata[(y+j) * olda + x] =
                    __float2half(idata[(y+j) * olda + x]);
        }
        return;
    }
    
    if(GPU_BLOCKIDX_X < GPU_BLOCKIDX_Y)
    {
        int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
        int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            odata[(y+j) * olda + x] =
                __float2half(idata[(y+j) * olda + x]);
        return;
    }
}

/////// UPCAST FOR DEBUG
/// HUGE ASSUMTION OF ilda = olda
__global__ void copyCoalesced_lower(float *odata, const __half *idata, int olda,
                              int ilda) {
    if(GPU_BLOCKIDX_Y > GPU_BLOCKIDX_X){ return;};
    if(GPU_BLOCKIDX_X == GPU_BLOCKIDX_Y)
    {
        int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
        int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
        // int width = gridDim.x * TILE_DIM;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            if(x > y+j)
                odata[(y+j) * olda + x] =
                    __half2float(idata[(y+j) * olda + x]);
        }
        return;
    }
    
    if(GPU_BLOCKIDX_X > GPU_BLOCKIDX_Y)
    {
        int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
        int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            odata[(y+j) * olda + x] =
                __half2float(idata[(y+j) * olda + x]);
        return;
    }
}

/// HUGE ASSUMTION OF ilda = olda
__global__ void copyCoalesced_upper(float *odata, const __half *idata, int olda,
                              int ilda) {
  
    if(GPU_BLOCKIDX_Y < GPU_BLOCKIDX_X){ return;};
    if(GPU_BLOCKIDX_X == GPU_BLOCKIDX_Y)
    {
        int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
        int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            if(x <= y+j)
                odata[(y+j) * olda + x] =
                    __half2float(idata[(y+j) * olda + x]);
        }
        return;
    }
    
    if(GPU_BLOCKIDX_X < GPU_BLOCKIDX_Y)
    {
        int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
        int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            odata[(y+j) * olda + x] =
                __half2float(idata[(y+j) * olda + x]);
        return;
    }
}


// RIGHT TRANSPOSE CONVERT KERNELS
__global__ void transposeCoalesced(__half *odata, const float *idata, int olda,
                                   int ilda) {
    __shared__ float tile[TILE_DIM_TRANS][TILE_DIM_TRANS+1];

    int x = GPU_BLOCKIDX_X * TILE_DIM_TRANS + GPU_THREADIDX_X;
    int y = GPU_BLOCKIDX_Y * TILE_DIM_TRANS + GPU_THREADIDX_Y;
    for (int j = 0; j < TILE_DIM_TRANS; j += BLOCK_ROWS)
        tile[GPU_THREADIDX_Y + j][GPU_THREADIDX_X] = 
            idata[(long(y) +long( j)) * long(ilda) +long( x)];

    __syncthreads();

    x = GPU_BLOCKIDX_Y * TILE_DIM_TRANS + GPU_THREADIDX_X; // transpose block offset
    y = GPU_BLOCKIDX_X * TILE_DIM_TRANS + GPU_THREADIDX_Y;

    for (int j = 0; j < TILE_DIM_TRANS; j += BLOCK_ROWS)
        odata[(long(y) + long(j)) *long( olda) +long( x)] =
            __float2half(tile[GPU_THREADIDX_X][GPU_THREADIDX_Y + j]);
}


// HOST CALLS TO LAUNCH KERNELS (LEFT)
__host__ void half_conversion_left( const float *C, int b, int plda, __half *lp, int lplda) {
	if(lplda == 0) return;
    dim3 block_dims(lplda / TILE_DIM, b / TILE_DIM, 1);
    dim3 thread_dims(TILE_DIM, BLOCK_ROWS, 1);
    copyCoalesced<<<block_dims, thread_dims>>>(lp, C, lplda, plda);
/*#elif defined(ROCM_OLCF_PLATFORM)
    hipLaunchKernelGGL(copyCoalesced, block_dims, thread_dims, 0, 0,
            lp, C, lplda, plda);
#else
    throw std::runtime_error("Error with build platform, neither CUDA nor ROCM. See CMake output.")
#endif*/

}

// HOST CALLS TO LAUNCH KERNELS (RIGHT TRANS)

__host__ void half_conversion_right_trans(const float *C, int b, int plda, __half * rp, int rplda) {
    //printf("rplda:%d\n",rplda);
	if(rplda == 0) return;
    dim3 block_dims(b / TILE_DIM_TRANS, rplda / TILE_DIM_TRANS, 1);
    dim3 thread_dims(TILE_DIM_TRANS, BLOCK_ROWS, 1);

//#ifdef CUDA_OLCF_PLATFORM
    transposeCoalesced<<<block_dims, thread_dims>>>(rp, C, rplda, plda);
/*#elif defined(ROCM_OLCF_PLATFORM)
    hipLaunchKernelGGL(transposeCoalesced, block_dims, thread_dims, 0, 0,
            rp, C, rplda, plda);
#else
    throw std::runtime_error("Error with build platform, neither CUDA nor ROCM. See CMake output.")
#endif*/

}


// Copy cast kernel
__global__ void copyFtoH(__half *odata, const float *idata, long olda,
                              long ilda) {
   /* __shared__ float tile[TILE_DIM][TILE_DIM];

    // int width = gridDim.x * TILE_DIM;
    int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
    int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
 
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[(GPU_THREADIDX_Y + j)][GPU_THREADIDX_X] =
            idata[(long(y) +long( j)) * long(ilda) +long( x)];

    __syncthreads();

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(long(y) +long( j)) *long( olda) +long( x)] =
            __float2half(tile[(GPU_THREADIDX_Y + j)][GPU_THREADIDX_X]);
    */
    int x = GPU_BLOCKIDX_X * TILE_DIM + GPU_THREADIDX_X;
    int y = GPU_BLOCKIDX_Y * TILE_DIM + GPU_THREADIDX_Y;
 
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(long(y) +long( j)) *long( olda) +long( x)] =
            idata[(long(y) +long( j)) * long(ilda) +long( x)];
}

__host__ void downCast_copy_general(__half* out, long olda, long nrow, long ncol, float* C, long ilda)
{
        dim3 dimGrid(nrow/TILE_DIM,ncol/TILE_DIM, 1);
        dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
//#ifdef CUDA_OLCF_PLATFORM
        copyFtoH<<<dimGrid, dimBlock>>>(out, C, olda, ilda);
/*#elif defined(ROCM_OLCF_PLATFORM)
    hipLaunchKernelGGL(copyFtoH, dimGrid, dimBlock,0,0,
                                                 out, C, olda, ilda);
#else
    throw std::runtime_error("Error with build platform, neither CUDA nor ROCM. See CMake output.")
#endif
*/        //GPU_DEVICE_SYNCHRONIZE();
}

__host__ void downCast_copy_lower(__half* out, long olda, long nrow, long ncol, float* C, long ilda)
{
        dim3 dimGrid(nrow/TILE_DIM,ncol/TILE_DIM, 1);
        dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
        copyCoalesced_lower<<<dimGrid, dimBlock>>>(out, C, olda, ilda);
        //GPU_DEVICE_SYNCHRONIZE();
}

__host__ void downCast_copy_upper(__half* out, long olda, long nrow, long ncol, float* C, long ilda)
{
        dim3 dimGrid(nrow/TILE_DIM,ncol/TILE_DIM, 1);
        dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
        copyCoalesced_upper<<<dimGrid, dimBlock>>>(out, C, olda, ilda);
        //GPU_DEVICE_SYNCHRONIZE();
}

// Debug purpose
__host__ void upCast_copy_lower(float* out, long olda, long nrow, long ncol, __half * C, long ilda)
{
        dim3 dimGrid(nrow/TILE_DIM,ncol/TILE_DIM, 1);
        dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
        copyCoalesced_lower<<<dimGrid, dimBlock>>>(out, C, olda, ilda);
        //GPU_DEVICE_SYNCHRONIZE();
}

__host__ void upCast_copy_upper(float* out, long olda, long nrow, long ncol, __half * C, long ilda)
{
        dim3 dimGrid(nrow/TILE_DIM,ncol/TILE_DIM, 1);
        dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
        copyCoalesced_upper<<<dimGrid, dimBlock>>>(out, C, olda, ilda);
        //GPU_DEVICE_SYNCHRONIZE();
}

__host__ void downCast_trans_general(__half* out, long olda, long nrow, long ncol , float* C, long ilda)
{
        // rplda should be passed in as npcol-start_col
        dim3 dimGrid(nrow/TILE_DIM_TRANS, ncol/TILE_DIM_TRANS, 1);
        dim3 dimBlock(TILE_DIM_TRANS, BLOCK_ROWS, 1);
//#ifdef CUDA_OLCF_PLATFORM
        transposeCoalesced<<<dimGrid, dimBlock>>>(out, C, olda, ilda);
/*#elif defined(ROCM_OLCF_PLATFORM)
    hipLaunchKernelGGL(transposeCoalesced, dimGrid, dimBlock,0,0,
                                                 out, C, olda, ilda);
#else
    throw std::runtime_error("Error with build platform, neither CUDA nor ROCM. See CMake output.")
#endif*/
        //GPU_DEVICE_SYNCHRONIZE();
}

__host__ void gen_identity_mat( float * out, long nrow, long ncol)
{
        // ASSUMING IT B has to be divisible by TILE_DIM
        dim3 dimGrid(nrow/TILE_DIM_ID, ncol/TILE_DIM_ID,1);
        dim3 dimBlock(TILE_DIM_ID, BLOCK_ROWS,1);

#ifdef CUDA_OLCF_PLATFORM
        gen_identity_mat_kernel<<<dimGrid, dimBlock>>>(out, nrow, ncol);
#elif defined(ROCM_OLCF_PLATFORM)
    hipLaunchKernelGGL(gen_identity_mat_kernel, dimGrid, dimBlock,0,0,
                                                 out, nrow, ncol);
#else
    throw std::runtime_error("Error with build platform, neither CUDA nor ROCM. See CMake output.")
#endif
        GPU_DEVICE_SYNCHRONIZE();
}

__global__ void gen_identity_mat_kernel( float * out, long nrow, long ncol) {
        int x = GPU_BLOCKIDX_X * TILE_DIM_ID + GPU_THREADIDX_X;
        int y = GPU_BLOCKIDX_Y * TILE_DIM_ID + GPU_THREADIDX_Y;
        float a = 1.0;
        float b = 0.0;
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            //y += j;
            if(x == y+j){
                out[long(y+j)*long(nrow) + long( x)] = a;
            }
            else{
                out[long(y+j)*long(nrow) + long( x)] = b;
            }
        }
}

__global__ void gen_identity_mat_kernel( __half * out, long nrow, long ncol) {
        int x = GPU_BLOCKIDX_X * TILE_DIM_ID + GPU_THREADIDX_X;
        int y = GPU_BLOCKIDX_Y * TILE_DIM_ID + GPU_THREADIDX_Y;
        __half a = 1.0;
        __half b = 0.0;
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            //y += j;
            if(x == y+j){
                out[long(y+j)*long(nrow) + long( x)] = a;
            }
            else{
                out[long(y+j)*long(nrow) + long( x)] = b;
            }
        }
}

__host__ void gen_identity_mat( __half * out, long nrow, long ncol)
{
        // ASSUMING IT B has to be divisible by TILE_DIM
        dim3 dimGrid(nrow/TILE_DIM_ID, ncol/TILE_DIM_ID,1);
        dim3 dimBlock(TILE_DIM_ID, BLOCK_ROWS,1);

#ifdef CUDA_OLCF_PLATFORM
        gen_identity_mat_kernel<<<dimGrid, dimBlock>>>(out, nrow, ncol);
#elif defined(ROCM_OLCF_PLATFORM)
    hipLaunchKernelGGL(gen_identity_mat_kernel, dimGrid, dimBlock,0,0,
                                                 out, nrow, ncol);
#else
    throw std::runtime_error("Error with build platform, neither CUDA nor ROCM. See CMake output.")
#endif
        GPU_DEVICE_SYNCHRONIZE();

}


__host__ void downCast_trans_general(__half* out, int olda, int nrow, int ncol , float* C, int ilda)
{
  // rplda should be passed in as npcol-start_col
  
  dim3 dimGrid(nrow/TILE_DIM_TRANS, ncol/TILE_DIM_TRANS, 1); 
  dim3 dimBlock(TILE_DIM_TRANS, BLOCK_ROWS, 1);
#ifdef CUDA_OLCF_PLATFORM
  transposeCoalesced<<<dimGrid, dimBlock>>>(out, C, (long)olda, (long)ilda);
#endif
 // GPU_DEVICE_SYNCHRONIZE();
}


// ORNL VECTOR OPS
template<typename F>
void copycolv_h(Panels<F>const& p, double const* x, double* y )
{
    int b = p.b;
    int j1 = p.j1;
    int i1 = p.i1;
    int nprow = p.nprow;
    int istride = p.istride;
    int jstride = p.jstride;
    for(int ii = 0; ii < nprow; ii++)
    {
        int ipos = ii * istride + i1;
        if (ipos % jstride == j1) {
            GPU_MEMCPY(y+ii*b, x+ii*b, b*sizeof(double), GPU_MEMCPY_DEVICE_TO_HOST);
        }
    }
}

__global__ void divcolv_d(int b, double const* __restrict__ x, double* __restrict__ y)
{
    size_t id = threadIdx.x;
    for (int j = 0; j + id < b; j += VECTOR_OP_THREADS) 
    {
        y[j+id] /= x[j+id];
    }
}

template<typename F>
void divcolv_h(Panels<F>const& p, double const* x, double* y )
{
    int b = p.b;
    int j1 = p.j1;
    int i1 = p.i1;
    int nprow = p.nprow;
    int istride = p.istride;
    int jstride = p.jstride;
    for(int ii = 0; ii < nprow; ii++)
    {
        size_t ipos = (size_t)ii * istride + i1;
        if (ipos % jstride == j1) {
            divcolv_d<<< 1, VECTOR_OP_THREADS>>> (b, x+ii*b, y+ii*b);
        }
    }
}

__global__ void colv2rowv_d(int b, double const* __restrict__ x, double* __restrict__ y )
{
    size_t id = threadIdx.x;
    for (int j = 0; j + id < b; j += VECTOR_OP_THREADS) 
    {
        y[j+id] = x[j+id];
    }
}

template<typename F>
void colv2rowv_h(Panels<F>const& p, double const* x, double* y )
{
    int b = p.b;
    int j1 = p.j1;
    int i1 = p.i1;
    int nprow = p.nprow;
    int istride = p.istride;
    int jstride = p.jstride;
    for(int ii = 0; ii < nprow; ii++)
    {
        size_t ipos = (size_t)ii * istride + i1;
        if (ipos % jstride == j1) {
            int jj = (ipos - j1) / jstride;
            colv2rowv_d<<< 1, VECTOR_OP_THREADS>>> (b, x+jj*b, y+ii*b);
        }
    }
}

__global__ void addcolv_d(int b, double const* __restrict__ x, double* __restrict__ y)
{
    size_t id = threadIdx.x;
    for (int j = 0; j + id < b; j += VECTOR_OP_THREADS) 
    {
        y[j+id] += x[j+id];
    }
}

template<typename F>
void addcolv_h(Panels<F>const& p, double const* x, double* y )
{
    int b = p.b;
    int j1 = p.j1;
    int i1 = p.i1;
    int nprow = p.nprow;
    int istride = p.istride;
    int jstride = p.jstride;
    for(int ii = 0; ii < nprow; ii++)
    {
        size_t ipos = (size_t)ii * istride + i1;
        if (ipos % jstride == j1) {
            addcolv_d<<< 1, VECTOR_OP_THREADS>>> (b, x+ii*b, y+ii*b);
        }
    }
}


/// Panel_Gemv
__global__ void otf_gemv_diag_d(int n, int b, double alpha, double const* __restrict__ x, double * y, 
        RandStat stat_ij, RandCoeff incl1, RandCoeff jump_thread, double const* diag)
{
    int cur_row = blockIdx.x;
    int id = threadIdx.x;
    __shared__ double sh[MV_THREADS];
    double t = 0;
    double d = diag[cur_row];

    if (cur_row < b) {
        RandCoeff jump_id = incl1.pow(id * n + cur_row);
        RandStat stat_tj = jump_id*stat_ij;
        for (int j = 0; j + id < b; j += blockDim.x) {
            double aij = (cur_row == j + id ? d : static_cast<double>(stat_tj));
            t += aij*x[j+id];;
            stat_tj = jump_thread * stat_tj;
        }
    }
    sh[id] = t;
    __syncthreads();

    if (id < 128) sh[id] += sh[id + 128];
    __syncthreads();
    if (id < 64) sh[id] += sh[id + 64];
    __syncthreads();
    if (id < 32) sh[id] += sh[id + 32];
    __syncthreads();
    if (id < 16) sh[id] += sh[id + 16];
    __syncthreads();
    if (id < 8) sh[id] += sh[id + 8];
    __syncthreads();
    if (id < 4) sh[id] += sh[id + 4];
    __syncthreads();
    if (id == 0) y[cur_row] = y[cur_row] + alpha * (sh[0] + sh[1] + sh[2] + sh[3]);
}

__global__ void otf_gemv_d(int n, int b, double alpha, double const* __restrict__ x, double * y,
        RandStat stat_ij, RandCoeff incl1, RandCoeff jump_thread)
{
    int cur_row = blockIdx.x;
    int id = threadIdx.x;
    __shared__ double sh[MV_THREADS];
    double t = 0;

    if (cur_row < b) {
        RandCoeff jump_id = incl1.pow(id * n + cur_row);
        RandStat stat_tj = jump_id*stat_ij;
        for (int j = 0; j + id < b; j += blockDim.x) {
            t += static_cast<double>(stat_tj)*x[j+id];
            stat_tj = jump_thread * stat_tj;
        }
    }
    sh[id] = t;
    __syncthreads();

    if (id < 128) sh[id] += sh[id + 128];
    __syncthreads();
    if (id < 64) sh[id] += sh[id + 64];
    __syncthreads();
    if (id < 32) sh[id] += sh[id + 32];
    __syncthreads();
    if (id < 16) sh[id] += sh[id + 16];
    __syncthreads();
    if (id < 8) sh[id] += sh[id + 8];
    __syncthreads();
    if (id < 4) sh[id] += sh[id + 4];
    __syncthreads();
    if (id == 0) y[cur_row] = y[cur_row] + alpha * (sh[0] + sh[1] + sh[2] + sh[3]);
}

template<typename FPanel>
void otf_gemv_h(Matgen<double> const& mg, Panels<FPanel>const& p,
                int rowstart, int rowend, int colstart, int colend, double alpha, double const* x, double* y)
{
    int const n = mg.n;
    int const b = p.b;
    int const i1 = p.i1;
    int const j1 = p.j1;
    int const istride = p.istride;
    int const jstride = p.jstride;
    int const nprow = p.nprow;
    int const npcol = p.npcol;
    RandCoeff incl1 = mg.incl1;
    RandStat stat_00 = RandStat::initialize(mg.seed);
    RandCoeff jump_thread = incl1.pow(MV_THREADS * n);
    double const* diag = mg.diag_dev;

    for (int pj = colstart; pj < colend; ++pj) {
        int j0 = j1 + pj * jstride;

        for (int pi = 0; pi < nprow; ++pi) {
            int i0 = i1 + pi * istride;
            RandCoeff jump_ij = mg.jump(b * static_cast<uint64_t>(i0), b * static_cast<uint64_t>(j0));
            RandStat stat_ij = jump_ij * stat_00;

            if (i0 != j0) {
                otf_gemv_d<<<b, MV_THREADS, 0>>>(n, b, alpha, x + b * pj, y + b * pi, stat_ij, incl1, jump_thread);
            }
            else {
                otf_gemv_diag_d<<<b, MV_THREADS, 0>>>(n, b, alpha, x + b * pj, y + b * pi, stat_ij, incl1, jump_thread, diag + b * pi);
            }
            GPU_THREAD_SYNCHRONIZE(0);
        }
    }
}

// Panel Norm
#define NORM_THREADS 1024

__global__ void calc_infnorm_d(int b, double* __restrict__ x, double* __restrict__ result) {
    __shared__ double sdata[NORM_THREADS];
    
    size_t id = threadIdx.x;
    sdata[id] = 0.0;

    for (int j = 0; j + id < b; j += NORM_THREADS) 
    {
        sdata[id] = fmax(sdata[id], fabs(x[id+j]));
    }
    __syncthreads();

    for (unsigned int s = NORM_THREADS / 2; s > 0; s >>= 1) {
        if (id < s) {
            sdata[id] = fmax(sdata[id], sdata[id + s]);
        }
        __syncthreads();
    }

    if (id == 0) {
        result[id] = sdata[0];
    }
}

template <typename F>
double colv_infnorm_h(Panels<F> const &p, double *dx, Grid &g, double* workspace) {
    // computes the inf-norm of the distributed column vector dx.
    // descriptros are derived from p
    int nprow = p.nprow;
    int b = p.b;
    int i1 = p.i1;
    int j1 = p.j1;
    int istride = p.istride;
    int jstride = p.jstride;
    double norm = 0.;
    double t = 0;
    for (int i = 0; i < nprow; ++i) {
        int ipos = i1 + i * istride;
        if ((ipos % jstride) == j1) {
            calc_infnorm_d<<<1,NORM_THREADS>>>(b, dx + b * i, workspace);
            GPU_DEVICE_SYNCHRONIZE();
            GPU_MEMCPY(&t, workspace, sizeof(double), GPU_MEMCPY_DEVICE_TO_HOST);
            norm = norm >= t ? norm : t;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_MAX, g.commworld);
    return norm;
}


// Trsv
template <int DIM_X, int DIM_Y, typename T, typename U>
static __global__
void gemv_kernel(const int m,
                     const int n,
                     const U   alpha,
                     const T* __restrict__ A,
                     const int lda,
                     const U* __restrict__ x,
                     const int incx,
                     const U   beta,
                     U* __restrict__ y,
                     const int incy) {
  const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

  if(!alpha) {
    if(thread_id < DIM_X * 4) {
      const int ind = blockIdx.x * DIM_X * 4 + thread_id;

      if(ind < m) y[ind * incy] = beta ? beta * y[ind * incy] : 0;
    }
    return;
  }

  // threads are all configurated locally
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  int ind;

  __shared__ U sdata[DIM_X * 4 * DIM_Y];

  U res_A[4];
  U res_x[4];

  res_A[0] = res_A[1] = res_A[2] = res_A[3] = U{0};

  ind = blockIdx.x * DIM_X * 4 + tx;

  const int n_tail = n % (4 * DIM_Y);
  int       col;

  for(col = ty * 4; col < (n - n_tail); col += 4 * DIM_Y) {
    res_x[0] = x[(col + 0) * incx];
    res_x[1] = x[(col + 1) * incx];
    res_x[2] = x[(col + 2) * incx];
    res_x[3] = x[(col + 3) * incx];

    if(ind < m) {
      res_A[0] += static_cast<U>(A[ind + (col + 0) * lda]) * res_x[0];
      res_A[0] += static_cast<U>(A[ind + (col + 1) * lda]) * res_x[1];
      res_A[0] += static_cast<U>(A[ind + (col + 2) * lda]) * res_x[2];
      res_A[0] += static_cast<U>(A[ind + (col + 3) * lda]) * res_x[3];

      if(ind + DIM_X < m) {
        res_A[1] += static_cast<U>(A[ind + DIM_X + (col + 0) * lda]) * res_x[0];
        res_A[1] += static_cast<U>(A[ind + DIM_X + (col + 1) * lda]) * res_x[1];
        res_A[1] += static_cast<U>(A[ind + DIM_X + (col + 2) * lda]) * res_x[2];
        res_A[1] += static_cast<U>(A[ind + DIM_X + (col + 3) * lda]) * res_x[3];

        if(ind + 2 * DIM_X < m) {
          res_A[2] +=
              static_cast<U>(A[ind + 2 * DIM_X + (col + 0) * lda]) * res_x[0];
          res_A[2] +=
              static_cast<U>(A[ind + 2 * DIM_X + (col + 1) * lda]) * res_x[1];
          res_A[2] +=
              static_cast<U>(A[ind + 2 * DIM_X + (col + 2) * lda]) * res_x[2];
          res_A[2] +=
              static_cast<U>(A[ind + 2 * DIM_X + (col + 3) * lda]) * res_x[3];

          if(ind + 3 * DIM_X < m) {
            res_A[3] +=
                static_cast<U>(A[ind + 3 * DIM_X + (col + 0) * lda]) * res_x[0];
            res_A[3] +=
                static_cast<U>(A[ind + 3 * DIM_X + (col + 1) * lda]) * res_x[1];
            res_A[3] +=
                static_cast<U>(A[ind + 3 * DIM_X + (col + 2) * lda]) * res_x[2];
            res_A[3] +=
                static_cast<U>(A[ind + 3 * DIM_X + (col + 3) * lda]) * res_x[3];
          }
        }
      }
    }
  }

  // if n is not multiple of (DIM_Y * 4)
  if(n_tail > 0) {
    res_x[0] = res_x[1] = res_x[2] = res_x[3] = U{0};

    if(col + 0 < n) {
      res_x[0] = x[(col + 0) * incx];

      if(col + 1 < n) {
        res_x[1] = x[(col + 1) * incx];

        if(col + 2 < n) {
          res_x[2] = x[(col + 2) * incx];

          if(col + 3 < n) res_x[3] = x[(col + 3) * incx];
        }
      }
    }

    if(ind < m) {
      res_A[0] +=
          static_cast<U>(A[ind + (col + 0) * lda * (col + 0 < n)]) * res_x[0];
      res_A[0] +=
          static_cast<U>(A[ind + (col + 1) * lda * (col + 1 < n)]) * res_x[1];
      res_A[0] +=
          static_cast<U>(A[ind + (col + 2) * lda * (col + 2 < n)]) * res_x[2];
      res_A[0] +=
          static_cast<U>(A[ind + (col + 3) * lda * (col + 3 < n)]) * res_x[3];

      if(ind + DIM_X < m) {
        res_A[1] +=
            static_cast<U>(A[ind + DIM_X + (col + 0) * lda * (col + 0 < n)]) *
            res_x[0];
        res_A[1] +=
            static_cast<U>(A[ind + DIM_X + (col + 1) * lda * (col + 1 < n)]) *
            res_x[1];
        res_A[1] +=
            static_cast<U>(A[ind + DIM_X + (col + 2) * lda * (col + 2 < n)]) *
            res_x[2];
        res_A[1] +=
            static_cast<U>(A[ind + DIM_X + (col + 3) * lda * (col + 3 < n)]) *
            res_x[3];

        if(ind + 2 * DIM_X < m) {
          res_A[2] +=
              static_cast<U>(
                  A[ind + 2 * DIM_X + (col + 0) * lda * (col + 0 < n)]) *
              res_x[0];
          res_A[2] +=
              static_cast<U>(
                  A[ind + 2 * DIM_X + (col + 1) * lda * (col + 1 < n)]) *
              res_x[1];
          res_A[2] +=
              static_cast<U>(
                  A[ind + 2 * DIM_X + (col + 2) * lda * (col + 2 < n)]) *
              res_x[2];
          res_A[2] +=
              static_cast<U>(
                  A[ind + 2 * DIM_X + (col + 3) * lda * (col + 3 < n)]) *
              res_x[3];

          if(ind + 3 * DIM_X < m) {
            res_A[3] +=
                static_cast<U>(
                    A[ind + 3 * DIM_X + (col + 0) * lda * (col + 0 < n)]) *
                res_x[0];
            res_A[3] +=
                static_cast<U>(
                    A[ind + 3 * DIM_X + (col + 1) * lda * (col + 1 < n)]) *
                res_x[1];
            res_A[3] +=
                static_cast<U>(
                    A[ind + 3 * DIM_X + (col + 2) * lda * (col + 2 < n)]) *
                res_x[2];
            res_A[3] +=
                static_cast<U>(
                    A[ind + 3 * DIM_X + (col + 3) * lda * (col + 3 < n)]) *
                res_x[3];
          }
        }
      }
    }
  }

  sdata[tx + ty * DIM_X * 4]             = res_A[0];
  sdata[tx + DIM_X + ty * DIM_X * 4]     = res_A[1];
  sdata[tx + 2 * DIM_X + ty * DIM_X * 4] = res_A[2];
  sdata[tx + 3 * DIM_X + ty * DIM_X * 4] = res_A[3];

  __syncthreads();

  if(thread_id < DIM_X * 4) {
    for(int i = 1; i < DIM_Y; i++) {
      sdata[thread_id] += sdata[thread_id + DIM_X * 4 * i];
    }

    ind = blockIdx.x * DIM_X * 4 + thread_id;

    if(ind < m)
      y[ind * incy] = beta ? alpha * sdata[thread_id] + beta * y[ind * incy]
                           : alpha * sdata[thread_id];
  }
}

template <typename T, typename U>
void HPLMXP_gemv(const int m,
                 const int n,
                 const T   alpha,
                 const U*  A,
                 const int lda,
                 const T*  x,
                 const T   beta,
                 T*        y) {
#if GEMV_USE_MED_KNL
  static constexpr int GEMVN_DIM_X = 32;
  static constexpr int GEMVN_DIM_Y = 16;
#else
  static constexpr int GEMVN_DIM_X = 64;
  static constexpr int GEMVN_DIM_Y = 16;
#endif

  const int blocks = (m + GEMVN_DIM_X * 4 - 1) / (GEMVN_DIM_X * 4);
  dim3      gemvn_grid(blocks);
  dim3      gemvn_threads(GEMVN_DIM_X, GEMVN_DIM_Y);

  gemv_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, U, T>
      <<<gemvn_grid, gemvn_threads, 0>>>(
          m, n, alpha, A, lda, x, 1, beta, y, 1);
}

template <typename Fx, typename Fy>
__global__ void convert_precision (Fy * odata, const Fx * idata, size_t olda,
                              size_t ilda) {
    __shared__ double tile[TILE_DIM_TRSV * TILE_DIM_TRSV];

    // int width = gridDim.x * TILE_DIM;
    size_t x = GPU_BLOCKIDX_X * TILE_DIM_TRSV + GPU_THREADIDX_X;
    size_t y = GPU_BLOCKIDX_Y * TILE_DIM_TRSV + GPU_THREADIDX_Y;
 
    for (int j = 0; j < TILE_DIM_TRSV; j += BLOCK_ROWS_TRSV)
        tile[(GPU_THREADIDX_Y + j) * TILE_DIM_TRSV + GPU_THREADIDX_X] = 
            static_cast<double>(idata[(size_t(y) +size_t( j)) * size_t(ilda) +size_t(x)]);

    __syncthreads();

    for (int j = 0; j < TILE_DIM_TRSV; j += BLOCK_ROWS_TRSV)
        odata[(size_t(y) +size_t( j)) *size_t(olda) + size_t( x)] =
            static_cast<Fy>(tile[(GPU_THREADIDX_Y + j) * TILE_DIM_TRSV + GPU_THREADIDX_X]);
}

template <typename Fx, typename Fy>
__host__ void cast_copy_general(Fy* out, size_t olda, size_t nrow, size_t ncol , Fx* C, size_t ilda)
{
  dim3 dimGrid(nrow/TILE_DIM_TRSV, ncol/TILE_DIM_TRSV, 1); 
  dim3 dimBlock(TILE_DIM_TRSV, BLOCK_ROWS_TRSV, 1);
  convert_precision<<<dimGrid, dimBlock>>>(out, C, olda, ilda);
  GPU_DEVICE_SYNCHRONIZE();
}

template <typename Fx, typename Fy>
__global__ void matrix_vector_multiply_0(int rows, int cols, double alpha,
        Fx const* __restrict__ A, int lda, Fy * __restrict__ x, 
        Fy * __restrict__ y)
{
    int cur_row = blockIdx.x;
    int id = threadIdx.x;
    __shared__ Fy sh[TRSV_MV_THREADS];
    Fy t = 0;

    if (cur_row < rows) {
        for (int j = 0; j + id < cols; j += blockDim.x) {
            t += static_cast<Fy>(*(A + blockIdx.x + (j+id)*lda))*x[j+id];
        }
    }
    sh[id] = t;
    __syncthreads();

    if (id < 128) sh[id] += sh[id + 128];
    __syncthreads();
    if (id < 64) sh[id] += sh[id + 64];
    __syncthreads();
    if (id < 32) sh[id] += sh[id + 32];
    __syncthreads();
    if (id < 16) sh[id] += sh[id + 16];
    __syncthreads();
    if (id < 8) sh[id] += sh[id + 8];
    __syncthreads();
    if (id < 4) sh[id] += sh[id + 4];
    __syncthreads();
    if (id == 0) y[cur_row] = (sh[0] + sh[1] + sh[2] + sh[3]);
}

// Matgem
#define MATGEM_THREADS 512
template <typename T>
__global__ void fill_panel_diag_dev(int n, int b, T * __restrict__ A, int lda, RandStat stat_ij, RandCoeff incl1,
        RandCoeff jump_thread, RandStat stat_00, int row_start, double* Diag)
{
    int cur_row = blockIdx.x;
    int id = threadIdx.x;
    __shared__ double sh[MATGEM_THREADS];

    if (cur_row < b) {
        RandCoeff jump_id = incl1.pow(id * n + cur_row);
        RandStat stat_tj = jump_id*stat_ij;
        
        for (int j = 0; j + id < b; j += blockDim.x) {
            A[(j + id) * lda + cur_row] = static_cast<T>(stat_tj);
            stat_tj = jump_thread * stat_tj;
        }
    
        // diagonal 
        double t = 0;
        jump_id = incl1.pow(id * n + cur_row + row_start); //Jump from 0 column
        stat_tj = jump_id*stat_00; //Stat_i_id

        for (int j = 0; j + id < n; j += blockDim.x) {
            if (j + id != cur_row + row_start) {
                t += fabs(static_cast<double>(stat_tj));
            }
            stat_tj = jump_thread * stat_tj;
        }
        sh[id] = t;
        __syncthreads();

        if (id < 256) sh[id] += sh[id + 256];
        __syncthreads();

        if (id < 128) sh[id] += sh[id + 128];
        __syncthreads();
        if (id < 64) sh[id] += sh[id + 64];
        __syncthreads();

        if (id < 32) sh[id] += sh[id + 32];
        __syncthreads();

        if (id < 16) sh[id] += sh[id + 16];
        __syncthreads();

        if (id < 8) sh[id] += sh[id + 8];
        __syncthreads();

        if (id < 4) sh[id] += sh[id + 4];
        __syncthreads();

        // diagonal 
        if (id == 0) {
            Diag[cur_row] = sh[0] + sh[1] + sh[2] + sh[3];
            A[cur_row * lda + cur_row] = Diag[cur_row];
        }
    }
}

template <typename T>
__global__ void fill_panel_dev(int n, int b, T * __restrict__ A,
        int lda, RandStat stat_ij, RandCoeff incl1, RandCoeff jump_thread)
{
    int cur_row = blockIdx.x;
    int id = threadIdx.x;

    if (cur_row < b) {
        RandCoeff jump_id = incl1.pow(id * n + cur_row);
        RandStat stat_tj = jump_id*stat_ij;
        for (int j = 0; j + id < b; j += blockDim.x) {
            A[(j + id) * lda + cur_row] = static_cast<T>(static_cast<double>(stat_tj));
            stat_tj = jump_thread * stat_tj;
        }
    }
}

template <typename F>
void panel_matgen_dev(Matgen<F> const& mg, Panels<F>& p, double* Diag)
{
    int const n = mg.n;
    int const b = p.b;
    int const i1 = p.i1;
    int const j1 = p.j1;
    int const istride = p.istride;
    int const jstride = p.jstride;
    int const nprow = p.nprow;
    int const npcol = p.npcol;
    size_t const lda = p.lda;
    RandCoeff incl1 = mg.incl1;
    RandStat stat_00 = RandStat::initialize(mg.seed);
    RandCoeff jump_thread = incl1.pow(MATGEM_THREADS * n);

    for (int pj = 0; pj < npcol; ++pj) {
        int j0 = j1 + pj * jstride;

        for (int pi = 0; pi < nprow; ++pi) {
            int i0 = i1 + pi * istride;
            RandCoeff jump_ij = mg.jump(b * static_cast<uint64_t>(i0), b * static_cast<uint64_t>(j0));
            //RandCoeff jump_ij = inc1.pow(b*i0 + n * static_cast<uint64_t>(j0));
            RandStat stat_ij = jump_ij * stat_00;

            if (i0 != j0) {
                fill_panel_dev<<<b, MATGEM_THREADS, 0>>>(n, b, p(pi, pj, 'd'), lda, stat_ij, incl1, jump_thread);
            }
            else {
                fill_panel_diag_dev<<<b, MATGEM_THREADS, 0>>>(n, b, p(pi, pj, 'd'), lda, stat_ij, incl1, jump_thread, stat_00, b*i0, &Diag[pi*b]);
            }
        }
    }
}

template void otf_gemv_h<float>(Matgen<double> const&, Panels<float> const&, int, int, int, int, double, double const*, double*);
template void HPLMXP_gemv<double, float>(int, int, double, float const*, int, double const*, double, double*);
template void cast_copy_general<float const, double>(double*, unsigned long, unsigned long, unsigned long, float const*, unsigned long);
template void copycolv_h<float>(Panels<float> const&, double const*, double*);
template void addcolv_h<float>(Panels<float> const&, double const*, double*);
template void divcolv_h<float>(Panels<float> const&, double const*, double*);
template void panel_matgen_dev<float>(Matgen<float> const&, Panels<float>&, double*);
template void colv2rowv_h<float>(Panels<float> const&, double const*, double*);
template double colv_infnorm_h<float>(Panels<float> const&, double*, Grid&, double*);
