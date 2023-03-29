
#include "gpu_init_kernels.h"

using namespace std;



#if 0
// Generate random entries of matrix A in single precision
void fill_random(float* A, long m, long n, int n_threads, int blocksize_x, int work_per_thread)
{
    GPURAND_GENERATOR_T generator;

    GPURAND_CREATE_GENERATOR(&generator, GPURAND_RNG_PSEUDO_DEFAULT);
    GPURAND_SET_PSEUDO_RANDOM_GENERATOR_SEED(generator, (int)time(NULL));
    GPURAND_GENERATE_UNIFORM(generator, A, (size_t) m *(size_t)n);
	// Push entries to (-0.5, 0.5) range
    int blocksize_y = n_threads / blocksize_x;
    dim3 thread_dims(blocksize_x, blocksize_y, 1);
    int block_dim_x = ceil((float)m / blocksize_x), block_dim_y = ceil((float)n / (work_per_thread * blocksize_y));
    dim3 block_dims(block_dim_x, block_dim_y, 1);

#ifdef CUDA_OLCF_PLATFORM
    minus_05<<<block_dims, thread_dims>>>(A, m, n, work_per_thread);
#elif defined(ROCM_OLCF_PLATFORM)
    hipLaunchKernelGGL(minus_05, block_dims, thread_dims, 0, 0,
            A, m, n, work_per_thread);
#else
    throw std::runtime_error("Error with build platform, neither CUDA nor ROCM. See CMake output.")
#endif

    GPURAND_DESTROY_GENERATOR(generator);
}

// Generate random entries of vector v in double precision
void fill_random(double* v, long m, int n_threads)
{
    GPURAND_GENERATOR_T generator;

    GPURAND_CREATE_GENERATOR(&generator, GPURAND_RNG_PSEUDO_DEFAULT);
    GPURAND_SET_PSEUDO_RANDOM_GENERATOR_SEED(generator, (int)time(NULL));
    GPURAND_GENERATE_UNIFORM_DOUBLE(generator, v, m);
	// Push entries to (-0.5, 0.5) range
    dim3 block_dims(ceil((float) m / n_threads), 1, 1);
    dim3 thread_dims(n_threads, 1, 1);
    
#ifdef CUDA_OLCF_PLATFORM
    minus_05<<<block_dims, thread_dims>>>(v, m, 1, 1);
#elif defined(ROCM_OLCF_PLATFORM)
    hipLaunchKernelGGL(minus_05, block_dims, thread_dims, 0, 0, v, m, 1, 1);
#else
    throw std::runtime_error("Error with build platform, neither CUDA nor ROCM. See CMake output.")
#endif

    GPURAND_DESTROY_GENERATOR(generator);
}

// Generate random entries of matrix A using Fugaku's logic
__host__  void fill_random_fugaku(uint64_t N, RandStat stat0, RandCoeff inc1, float *A, double *row_sums, long m, long n, long i1, long j1, long b, long istride, long jstride, int n_threads, int blocksize_x, int work_per_thread)
{
    int blocksize_y = n_threads / blocksize_x;
    dim3 thread_dims(blocksize_x, blocksize_y, 1);
    int matrix_dim_x = ceil((float)m / blocksize_x), matrix_dim_y = ceil((float)n / (work_per_thread * blocksize_y));
    dim3 matrix_dims(matrix_dim_x, matrix_dim_y, 1);

#ifdef CUDA_OLCF_PLATFORM
    fill_random_fugaku_d<<<matrix_dims, thread_dims>>>(N, stat0, inc1, A, row_sums, m, n, i1, j1, b, istride, jstride, work_per_thread);
#elif defined(ROCM_OLCF_PLATFORM)
    hipLaunchKernelGGL(fill_random_fugaku_d, matrix_dims, thread_dims, 0, 0,
            N, stat0, inc1, A, row_sums, m, n, i1, j1, b, istride, jstride, work_per_thread);
#else
    throw std::runtime_error("Error with build platform, neither CUDA nor ROCM. See CMake output.")
#endif

}

template <typename F> __global__ void fill_random_fugaku_d(uint64_t N, RandStat stat0, RandCoeff inc1, F *A, double *row_sums, long m, long n, long i1, long j1, long b, long istride, long jstride, int work_per_thread)
{
	int idx_x = GPU_BLOCKIDX_X * GPU_BLOCKDIM_X + GPU_THREADIDX_X;
	int idx_y = GPU_BLOCKIDX_Y * GPU_BLOCKDIM_Y + GPU_THREADIDX_Y;
    const uint64_t A_row = idx_x, global_row = (i1 + (A_row / (uint64_t)b) * istride) * (uint64_t)b + (A_row % (uint64_t)b);
    uint64_t A_col, global_col, A_idx, global_idx;
    double a_ij, row_sum_increase = 0;

    if (A_row < m)
    {
        for(int i = 0 ; i < work_per_thread; i++)
        {
            A_col = work_per_thread * idx_y + i;
            global_col = (j1 + (A_col / (uint64_t)b) * jstride) * (uint64_t)b + (A_col % (uint64_t)b);
            if (A_col < n)
            {
                A_idx = A_row + A_col * (uint64_t)m;
                global_idx = global_col * N + global_row;
                a_ij = static_cast<double>(inc1.pow(global_idx) * stat0);
                A[A_idx] = static_cast<F>(a_ij);
                row_sum_increase += fabs(a_ij);
            }
        }
        atomicAdd(&row_sums[global_row], row_sum_increase);
    }
}
#endif


__host__ void compute_row_sums(float *A, double *row_sums, long m, long n, long i1, long j1, long b, long istride, long jstride, int n_threads, int blocksize_x, int work_per_thread)
{
    int blocksize_y = n_threads / blocksize_x;
    dim3 thread_dims(blocksize_x, blocksize_y, 1);
    int block_dim_x = ceil((float)m / blocksize_x), block_dim_y = ceil((float)n / (work_per_thread * blocksize_y));
    dim3 block_dims(block_dim_x, block_dim_y, 1);

#ifdef CUDA_OLCF_PLATFORM
    compute_row_sums_d<<<block_dims, thread_dims>>>(A, row_sums, m, n, i1, j1, b, istride, jstride, work_per_thread);
#elif defined(ROCM_OLCF_PLATFORM)
    hipLaunchKernelGGL(compute_row_sums_d, block_dims, thread_dims, 0, 0,
            A, row_sums, m, n, i1, j1, b, istride, jstride, work_per_thread);
#else
    throw std::runtime_error("Error with build platform, neither CUDA nor ROCM. See CMake output.")
#endif
}

__global__ void compute_row_sums_d(float *A, double *row_sums, long m, long n, long i1, long j1, long b, long istride, long jstride, int work_per_thread)
{
    int idx_x = GPU_BLOCKIDX_X * GPU_BLOCKDIM_X + GPU_THREADIDX_X;
    int idx_y = GPU_BLOCKIDX_Y * GPU_BLOCKDIM_Y + GPU_THREADIDX_Y;
    const size_t A_row = idx_x, global_row_block = i1 + (A_row / b) * istride, global_row = global_row_block * b + (A_row % b);
    size_t A_col;
    double row_sum_increase = 0;

    if (A_row < m)
    {
        for(int i = 0 ; i < work_per_thread; i++)
        {
            A_col = work_per_thread * idx_y + i;
            if (A_col < n)
                row_sum_increase += fabs(A[A_row + A_col * m]);

        }
        atomicAdd(&row_sums[global_row], row_sum_increase);
    }

}

__host__ void fill_diag_rhs(int my_init, uint64_t N, RandStat stat0, RandCoeff inc1, float *A, double *row_sums_d, double *rhs, double *rhs_d, long m, long *diag_i_steps, long *diag_j_steps, long n_diag_blocks, long i1, long b, long istride, int n_threads)
{
    int n_diag_entries = n_diag_blocks * b;
    dim3 block_dims(ceil((float) n_diag_entries / n_threads), 1, 1);
    dim3 thread_dims(n_threads, 1, 1);

#ifdef CUDA_OLCF_PLATFORM
    fill_diag_rhs_d<<<block_dims, thread_dims>>>(my_init, N, stat0, inc1, A, row_sums_d, rhs, rhs_d, m, diag_i_steps, diag_j_steps, n_diag_entries, i1, b, istride);
#elif defined(ROCM_OLCF_PLATFORM)
    hipLaunchKernelGGL(fill_diag_rhs_d, block_dims, thread_dims, 0, 0,
            my_init, N, stat0, inc1, A, row_sums_d, rhs, rhs_d, m, diag_i_steps, diag_j_steps, n_diag_entries, i1, b, istride);
#else
    throw std::runtime_error("Error with build platform, neither CUDA nor ROCM. See CMake output.")
#endif
}


__global__ void fill_diag_rhs_d(int my_init, uint64_t N, RandStat stat0, RandCoeff inc1, float *A, double *row_sums_d, double *rhs, double *rhs_d, long m, long *diag_i_steps, long *diag_j_steps, long n_diag_entries, long i1, long b, long istride)
{
    int idx = GPU_BLOCKIDX_X * GPU_BLOCKDIM_X + GPU_THREADIDX_X;
    
    if (idx < n_diag_entries)
    {
        size_t diag_block_idx = idx / b, offset_in_block = idx % b;
        size_t diag_i_step = diag_i_steps[diag_block_idx], diag_j_step = diag_j_steps[diag_block_idx];
        size_t A_idx = (diag_j_step * m + diag_i_step) * b + offset_in_block * (m + 1);
        size_t global_diag_idx = (i1 + diag_i_step * istride) * b + offset_in_block;
        
        row_sums_d[global_diag_idx] -= A[A_idx];
        A[A_idx] = row_sums_d[global_diag_idx];
        if (my_init == 3)
        {
            size_t local_diag_idx = diag_i_step * b + offset_in_block;
            rhs_d[local_diag_idx] = static_cast<double>(inc1.pow(N*N + global_diag_idx) * stat0);
        }
    }
}

template <typename F> __global__ void minus_05(F *A, long m, long n, int work_per_thread)
{
	int idx_x = GPU_BLOCKIDX_X * GPU_BLOCKDIM_X + GPU_THREADIDX_X;
	int idx_y = GPU_BLOCKIDX_Y * GPU_BLOCKDIM_Y + GPU_THREADIDX_Y;
    const uint64_t A_row = idx_x;
    uint64_t A_col = work_per_thread * idx_y, A_idx;
    
    if (A_row < m)
    {
        for(int i = 0 ; i < work_per_thread; i++)
        {
            A_col++;
            if (A_col < n)
            {
                A_idx = A_row + A_col * (uint64_t)m;
                A[A_idx] -= 0.5;
            }
        }
    }
}

