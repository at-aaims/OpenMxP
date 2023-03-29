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

