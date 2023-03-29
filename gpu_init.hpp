#ifndef GPU_INIT
#define GPU_INIT
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "gpu_init_kernels.h"
#include "hpl_rand.hpp"
#include "panel.hpp"

//#include "device_macros.h"

using namespace std;

#if 0
template <typename F>
void gpu_pgen(int my_init, Matgen<double> &mg, Panels<F> &p, LRPanels<__half> *lr, int n, double *diag, double *rhs)
{
    F *pp_d = p(0, 0, 'd'), *pp = p(0, 0);
    long const nprow = p.nprow, npcol = p.npcol, b = p.b, i1 = p.i1, j1 = p.j1, istride = p.istride, jstride = p.jstride, p_m = nprow * b, p_n = npcol * b;
    size_t const p_size = (size_t)p_m * (size_t)p_n;
    int n_threads = 1024, blocksize_x = 32, work_per_thread = 32;
    double *local_row_sums, *local_row_sums_d = (double *)lr[0].d_p;

    // Generate matrix entries in [-0.5, 0.5] and compute local row sums and copy them to host
    if (my_init == 1)
    {
        fill_random(pp_d, p_m, p_n, n_threads, blocksize_x, work_per_thread);
        compute_row_sums(pp_d, local_row_sums_d, p_m, p_n, i1, j1, b, istride, jstride, n_threads, blocksize_x, work_per_thread);
    }
    else if (my_init == 3)
    {
        fill_random_fugaku(mg.n, RandStat::initialize(mg.seed), RandCoeff::default_vals(), pp_d, local_row_sums_d, p_m, p_n, i1, j1, b, istride, jstride, n_threads, blocksize_x, work_per_thread);
    }
    GPU_DEVICE_SYNCHRONIZE();
    checkGPU(GPU_MALLOC_HOST((void **)&local_row_sums, n * sizeof(double)),
            "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] local row sums");
    GPU_MEMCPY(local_row_sums, local_row_sums_d, n * sizeof(double), GPU_MEMCPY_DEVICE_TO_HOST);

    // Compute global row sums and copy them to device
    double *global_row_sums, *global_row_sums_d = (double *)lr[1].d_p;

    checkGPU(GPU_MALLOC_HOST((void **)&global_row_sums, n * sizeof(double)),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] global row sums");
    MPI_Allreduce(local_row_sums, global_row_sums, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    GPU_MEMCPY(global_row_sums_d, global_row_sums, n * sizeof(double), GPU_MEMCPY_HOST_TO_DEVICE);

    // Find diagonal blocks owned by the rank
    long j, j_step, local_row, local_col, global_row, diag_entry_idx;
    double diag_val;
    long n_diag_blocks = 0, *diag_i_steps, *diag_j_steps;
    checkGPU( GPU_MALLOC_MANAGED( &diag_i_steps, nprow * sizeof(long) ),
	     "<FATAL:%s> %s\n", hostRank, "Allocating managed memory - diag_i_steps" );
    checkGPU( GPU_MALLOC_MANAGED( &diag_j_steps, npcol * sizeof(long) ),
	     "<FATAL:%s> %s\n", hostRank, "Allocating managed memory - diag_j_steps" );

    for (long i_step=0; i_step < nprow; i_step++)
    {
        // j == i in diagonal blocks
        j = i1 + i_step * istride;
        j_step = (j - j1) / jstride;
        if ((j - j1) % jstride == 0 && j_step >= 0 && j_step < npcol)
        {
            // rank owns diagonal entries j*b through j*b + b - 1
            diag_i_steps[n_diag_blocks] = i_step;
            diag_j_steps[n_diag_blocks] = j_step;
            n_diag_blocks++;

        }
    }


    // If the rank owns any diagonal entries...
    if (n_diag_blocks)
    {
        // Fill in the diagonal and generate rhs
        long n_diag_entries = n_diag_blocks * b;
        double *rhs_d;
        
        checkGPU(GPU_MALLOC((void**)&rhs_d, n_diag_entries * sizeof(double)),
                 "<FATAL:%s> %s\n", hostRank, "Allocate device[gpu] rhs");
        if (my_init == 1)
        {
            fill_random(rhs_d, n_diag_entries, n_threads);        
        }
        
        fill_diag_rhs(my_init, mg.n, RandStat::initialize(mg.seed), RandCoeff::default_vals(), pp_d, global_row_sums_d, rhs, rhs_d, p_m, diag_i_steps, diag_j_steps, n_diag_blocks, i1, b, istride, n_threads);
        
        // Copy diag to host
        long i_step;
        for (long diag_block_idx = 0; diag_block_idx < n_diag_blocks; diag_block_idx++)
        {
            i_step = diag_i_steps[diag_block_idx];
            GPU_MEMCPY(diag + i_step*b, global_row_sums_d + (i1 + i_step*istride)*b, b * sizeof(double), GPU_MEMCPY_DEVICE_TO_HOST);
        }

        // Copy rhs to host
        GPU_MEMCPY(rhs, rhs_d, n_diag_entries * sizeof(double), GPU_MEMCPY_DEVICE_TO_HOST);

    }

    // Copy panel to host
    GPU_MEMCPY(pp, pp_d, p_size * sizeof(float), GPU_MEMCPY_DEVICE_TO_HOST);

    // Copy panel to permanent storage if using our generation
    if (my_init == 1)
    {
        checkGPU(GPU_MALLOC_HOST((void **)&p.p_init, p.alloc_size),
                 "<FATAL:%s> %s\n", hostRank, "Allocate host[cpu] p_init");
        memcpy(p.p_init, pp, p_size * sizeof(F));
    }
}

template <typename F>
void gpu_pgen2(Panels<F> &p, Grid& grid)
{
	F *pp_d = p(0, 0, 'd'), *pp = p(0, 0);
	int b = p.b;
	fill_random2(p(0,0,'d') , p.nprow*b, p.npcol*b);

	GPU_MEMCPY(p(0,0),p(0,0,'d'), (size_t)p.nprow* (size_t)b* (size_t)p.npcol*(size_t)b*sizeof(F),
					GPU_MEMCPY_DEVICE_TO_HOST);
	double *local_row_sums, *global_row_sums;
	checkGPU( GPU_MALLOC_MANAGED( (void**)&local_row_sums, b*sizeof(double) ),
	         "<FATAL:%s> %s\n", hostRank, "Allocate managed memory - local row sums" );
   	checkGPU( GPU_MALLOC_MANAGED( (void**)&global_row_sums, b*sizeof(double) ),
	         "<FATAL:%s> %s\n", hostRank, "Allocate managed memory - global row sums" );

	for (int k = 0; k < p.nblocks; ++k) {
		// position of the panels to decomp in process grid
		int const rootrow = k % grid.nrow;
		int const rootcol = k % grid.ncol;
		// position of the panels to decomp in local matrix
		int i = k / grid.nrow + (rootrow > grid.row ? 1 : 0);
		int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);

		if (rootrow == grid.row && rootcol == grid.col) {
			compute_row_sums2(p(i,0,'d'),local_row_sums, b, p.npcol*b, p.lda);
			GPU_DEVICE_SYNCHRONIZE();

			MPI_Allreduce(local_row_sums, global_row_sums, b, MPI_DOUBLE, MPI_SUM, grid.hcomm);
			pp = p(i,j);
			#pragma omp parallel for
			for (int i_step=0; i_step < b; i_step++)
    		{
				pp[ i_step + i_step*p.lda]= static_cast<F>(global_row_sums[i_step]);
			}
		}
		else if(rootrow != grid.row)
		{
			//chill
		}
		else if(rootrow == grid.row)
		{
			compute_row_sums2(p(i,0,'d'),local_row_sums, b, p.npcol*b, p.lda);
			GPU_DEVICE_SYNCHRONIZE();
			MPI_Allreduce(local_row_sums, global_row_sums, b, MPI_DOUBLE, MPI_SUM, grid.hcomm);
		}
	}
	GPU_FREE(local_row_sums);
	GPU_FREE(global_row_sums);
}
#endif

#endif
