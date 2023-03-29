#ifndef MATGEN_HPP
#define MATGEN_HPP
// functions to initialize matrix.
#include "hpl_rand.hpp"
#include "panel.hpp"

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

template <typename F> void pmatgen2(Matgen<F> const &mg, Panels<F> &p, double* localSum) {
    int const bs = p.b;
    size_t const lda = p.lda;
    int const nprow = p.nprow;
    int const npcol = p.npcol;

    for (int pj = 0; pj < npcol; ++pj)
#pragma omp parallel for
        for (int pi = 0; pi < nprow; ++pi) {
            F *pp = p(pi, pj);
            const int i = bs * (p.i1 + pi * p.istride);
            const int j = bs * (p.j1 + pj * p.jstride);

            fill_one_panel_with_rand2(mg.n, i, j, bs, bs, pp, lda, mg.seed,localSum);
        }

    MPI_Barrier( MPI_COMM_WORLD );

}


template <typename F> void pmatgen(Matgen<F> const &mg, Panels<F> &p) {
    int const bs = p.b;
    size_t const lda = p.lda;
    int const nprow = p.nprow;
    int const npcol = p.npcol;

    for (int pj = 0; pj < npcol; ++pj)
        for (int pi = 0; pi < nprow; ++pi) {
            F *pp = p(pi, pj);
            const int i = bs * (p.i1 + pi * p.istride);
            const int j = bs * (p.j1 + pj * p.jstride);

            fill_one_panel_with_rand(mg.n, i, j, bs, bs, pp, lda, mg.seed,
                                     true);
        }
}

template <typename F> void pmatgen(HMGen<F> const &mg, Panels<F> &p) {
    size_t const lda = p.lda;
    int const nprow = p.nprow;
    int const npcol = p.npcol;
    int const b = p.b;
    int const i1 = p.i1;
    int const j1 = p.j1;
    int const istride = p.istride;
    int const jstride = p.jstride;
    F const alpha = mg.alpha;
    F const beta = mg.beta;
    F const ab = alpha * beta;
    F const done = 1;

#pragma omp parallel for collapse(2)
    for (int pj = 0; pj < npcol; ++pj) {
        for (int j = 0; j < b; ++j) {
            int jstart = b * (j1 + pj * jstride);
            F const fpjj = jstart + j;
            for (int pi = 0; pi < nprow; ++pi) {
                int istart = b * (i1 + pi * istride);
                F *to = p(pi, pj);
                if (pi < pj) {
                    for (int i = 0; i < b; ++i) {
                        // assuming no diag.
                        F aij = beta + ab * (istart + i);
                        to[j * lda + i] = aij;
                    }
                } else if (pi > pj) {
                    for (int i = 0; i < b; ++i) {
                        // assuming no diag.
                        F aij = alpha + ab * fpjj;
                        to[j * lda + i] = aij;
                    }
                } else {
                    for (int i = 0; i < j; ++i) {
                        // assuming no diag.
                        F aij = beta + ab * (jstart + i);
                        to[j * lda + i] = aij;
                    }
                    F aij = done + ab * fpjj;
                    to[j * lda + j] = aij;
                    for (int i = j + 1; i < b; ++i) {
                        // assuming no diag.
                        F aij = alpha + ab * fpjj;
                        to[j * lda + i] = aij;
                    }
                }
            }
        }
    }
}

template <typename F> void pmatgen0(Panels<F> &p) {
    // initialize with zero
    int const bs = p.b;
    size_t const lda = p.lda;
    int const nprow = p.nprow;
    int const npcol = p.npcol;
    F const dzero = static_cast<F>(0);

    if (p.is_tile) {
#pragma omp parallel for collapse(2)
        for (int pj = 0; pj < npcol; ++pj)
            for (int pi = 0; pi < nprow; ++pi) {
                F *pp = p(pi, pj);
                for (int j = 0; j < bs; ++j)
                    for (int i = 0; i < bs; ++i)
                        pp[j * lda + i] = dzero;
            }
    } else {
        F *ptr = p(0, 0);
        size_t size = static_cast<size_t>(p.ldpp) * npcol;
#pragma omp parallel for simd
        for (size_t i = 0; i < size; ++i)
            ptr[i] = dzero;
    }
}

template <typename F> void pmatl1est(Matgen<F> const &mg, Panels<F> &p) {
    // approximation of the decomposition
    int const bs = p.b;
    size_t const lda = p.lda;
    int const nprow = p.nprow;
    int const npcol = p.npcol;

#pragma omp parallel for
    for (int pj = 0; pj < npcol; ++pj) {
        double buf[bs];
        const int j = bs * (p.j1 + pj * p.jstride);
        for (int jj = 0; jj < bs; ++jj)
            buf[jj] = 1. / calc_diag(j + jj, mg.n, mg.seed);
        for (int pi = 0; pi < nprow; ++pi) {
            F *pp = p(pi, pj);
            const int i = bs * (p.i1 + pi * p.istride);
            if (i < j)
                continue;
            if (i == j) {
                for (int jj = 0; jj < bs; ++jj) {
                    F d = buf[jj];
                    for (int ii = 0; ii < bs; ++ii) {
                        if (i + ii > j + jj) {
                            pp[jj * lda + ii] *= d;
                        }
                    }
                }
            } else {
                for (int jj = 0; jj < bs; ++jj) {
                    F d = buf[jj];
                    for (int ii = 0; ii < bs; ++ii) {
                        pp[jj * lda + ii] *= d;
                    }
                }
            }
        }
    }
}

template <typename F> void pmatl1est(HMGen<F> const &mg, Panels<F> &p) {
    // approximation of the decomposition
    int const bs = p.b;
    size_t const lda = p.lda;
    int const nprow = p.nprow;
    int const npcol = p.npcol;
    F const alpha = mg.alpha;
    F const beta = mg.beta;
    F const done = 1;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int pj = 0; pj < npcol; ++pj)
        for (int pi = 0; pi < nprow; ++pi) {
            F *pp = p(pi, pj);
            const int i = bs * (p.i1 + pi * p.istride);
            const int j = bs * (p.j1 + pj * p.jstride);
            if (i < j) {
                for (int jj = 0; jj < bs; ++jj) {
                    for (int ii = 0; ii < bs; ++ii) {
                        pp[jj * lda + ii] = beta;
                    }
                }
            } else if (i > j) {
                for (int jj = 0; jj < bs; ++jj) {
                    for (int ii = 0; ii < bs; ++ii) {
                        pp[jj * lda + ii] = alpha;
                    }
                }
            } else {
                for (int jj = 0; jj < bs; ++jj) {
                    for (int ii = 0; ii < jj; ++ii) {
                        pp[jj * lda + ii] = beta;
                    }
                    pp[jj * lda + jj] = done;
                    for (int ii = jj + 1; ii < bs; ++ii) {
                        pp[jj * lda + ii] = alpha;
                    }
                }
            }
        }
}

template <typename F, typename FPanel>
void pcolvgen(Matgen<F> const &mg, Panels<FPanel> const &p, double *dx) {
    int nprow = p.nprow;
    int b = p.b;
    int i1 = p.i1;
    int j1 = p.j1;
    int istride = p.istride;
    int jstride = p.jstride;
    for (int i = 0; i < nprow; ++i) {
        int ipos = i1 + i * istride;
        if (ipos % jstride == j1) {
            fill_one_panel_with_rand(mg.n, b * ipos, mg.n, b, 1, dx + b * i, 1,
                                     mg.seed, false);
        }
    }
}

template <typename F, typename FPanel>
void pdiaggen2(Matgen<F> const &mg, Panels<FPanel> &p, double *dx, double* localSum) {
    int nprow = p.nprow;
    int b = p.b;
    int i1 = p.i1;
    int j1 = p.j1;
    int istride = p.istride;
    int jstride = p.jstride;
	int const bs = p.b;
    size_t const lda = p.lda;
    int const npcol = p.npcol;

    double* globalSum = (double*)malloc(mg.n*sizeof(double));
    if ( globalSum == NULL )
    {
       printf( "Allocation of globalSum failed\n" );
       exit( 10 );
    }
    memset(globalSum, 0, mg.n*sizeof(double)); 
    MPI_Allreduce(localSum, globalSum, mg.n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
 
#pragma omp parallel for
    for (int i = 0; i < nprow; ++i) {
        int ipos = i1 + i * istride;
        if (ipos % jstride == j1) {
            for (int k = 0; k < b; ++k)
                dx[b * i + k] = globalSum[ b * ipos + k ];
        }
    }
// fill A back
    for (int pj = 0; pj < npcol; ++pj) {
        for (int pi = 0; pi < nprow; ++pi) {
            F *pp = p(pi, pj);
            const int i0 = bs * (p.i1 + pi * p.istride);
            const int j0 = bs * (p.j1 + pj * p.jstride);
		  	if( i0 == j0 )
			{
#pragma omp parallel for 
			   for (int ii = 0; ii < b; ii++)
                              pp[ lda*ii + ii ] = static_cast<F>( globalSum[ i0+ii ] );
			}
        }
    }
    std::free( globalSum );

}

template <typename F, typename FPanel>
void pdiaggen(Matgen<F> const &mg, Panels<FPanel> const &p, double *dx) {
    int nprow = p.nprow;
    int b = p.b;
    int i1 = p.i1;
    int j1 = p.j1;
    int istride = p.istride;
    int jstride = p.jstride;
    for (int i = 0; i < nprow; ++i) {
        int ipos = i1 + i * istride;
        if (ipos % jstride == j1) {
#pragma omp parallel for
            for (int k = 0; k < b; ++k)
                dx[b * i + k] = calc_diag(b * ipos + k, mg.n, mg.seed);
        }
    }
}

template <typename F, typename FPanel>
void pdiaggen(HMGen<F> const &mg, Panels<FPanel> const &p, double *dx) {
    int nprow = p.nprow;
    int b = p.b;
    int i1 = p.i1;
    int j1 = p.j1;
    int istride = p.istride;
    int jstride = p.jstride;
    F const ab = mg.alpha * mg.beta;
    F const done = 1;
    for (int i = 0; i < nprow; ++i) {
        int ipos = i1 + i * istride;
        if (ipos % jstride == j1) {
#pragma omp parallel for
            for (int k = 0; k < b; ++k)
                dx[b * i + k] = done + ab * (b * ipos + k);
        }
    }
}

template<typename F, typename FPanel>
void panel_colvgen(Matgen<F> const& mg, Panels<FPanel>const& p, double* dx) {
    int nprow = p.nprow;
    int b = p.b;
    int i1 = p.i1;
    int j1 = p.j1;
    int istride = p.istride;
    int jstride = p.jstride;

#pragma omp parallel for
    for (int i = 0; i < nprow; ++i) {
        int ipos = i1 + i * istride;

        if (ipos % jstride == j1) {
            panel_fill_one_with_rand(mg.n, b * ipos, mg.n, b, 1, dx + b * i, 1, mg.seed, false);
        }
    }
}

template<typename F, typename FPanel>
void panel_diaggen(Matgen<F> const& mg, Panels<FPanel>const& p, double* dx) {
    int nprow = p.nprow;
    int b = p.b;
    int i1 = p.i1;
    int j1 = p.j1;
    int istride = p.istride;
    int jstride = p.jstride;

//  PrintLogMsg( "nprow ... %d n ... %d\n", nprow, mg.n );

#pragma omp parallel for
    for (int i = 0; i < nprow; ++i) {
        int ipos = i1 + i * istride;

        if (ipos % jstride == j1) {
            for (int k = 0; k < b; ++k) {
                //dx[b * i + k] = calc_diag(b * ipos + k, mg.n, mg.seed);
                dx[b * i + k] = 0.25 * mg.n;
            }
        }
    }
}

#endif
