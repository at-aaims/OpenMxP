#ifndef PANEL_HPP
#define PANEL_HPP
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

//#include "fp16_gpu_kernels.h"

#include "device_macros.h"

#include "log.hpp"

extern char hostRank[ 1024 ];
extern int half_version;
extern int my_gdirect;
extern int my_init;

#define ALIGN 1023ull

#define myB_SUCCESS 0
#define myB_KEEPTEST 1
#define RB_1R 2
#define RB_1RM 3
#define RB_2RM 4
#define RB_1R_IR 5
#define RB_1RM_IR 6
#define RB_2RM_IR 7

inline void print_matrix(const float *A, int lda, int size) {
    int numcolum = size / lda;
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < numcolum; j++)
            printf("%.6f ", A[i + j * lda]);
        printf("\n");
    }
    printf("\n");
}

inline void print_matrix(float *A, int lda, int size) {
    int numcolum = size / lda;
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < numcolum; j++)
            printf("%.6f ", A[i + j * lda]);
        printf("\n");
    }
    printf("\n");
}

inline void print_matrix(double *A, int lda, int size) {
    int numcolum = size / lda;
    //if(grank == 0 || grank == 4)
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < numcolum; j++)
            printf("%.6lf ", A[i + j * lda]);
        printf("\n");
    }
    printf("\n");
    fflush(stdout);
}

inline void print_diag(double *A, int lda, int size) {
    int numcolum = size / lda;
    for (int i = 0; i < lda; i++) {
        int j = i;
        printf("%.6lf ", A[i + j * lda]);
    }
    printf("\n");
}

inline void print_diag(float *A, int lda, int size) {
    int numcolum = size / lda;
    for (int i = 0; i < lda; i++) {
        int j = i;
        printf("%.6f ", A[i + j * lda]);
    }
    printf("\n");
}

inline void print_last_3diag(float *A, int lda, int size) {
    int numcolum = size / lda;
    for (int i = lda - 3; i < lda; i++) {
        int j = i;
        printf("%f ", A[i + j * lda]);
    }
    printf("\n");
}

inline void print_matrix(__half *A, int lda, int size) {
    int numcolum = size / lda;
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < numcolum; j++)
            printf("%.6f ", __half2float(A[i + j * lda]));
        printf("\n");
    }
    printf("\n");
}

inline void print_linear(__half *A, int size) {
    // int numcolum = size/lda;
    for (int i = 0; i < size; i++) {
        printf("%.3f ", __half2float(A[i]));
    }
    printf("\n");
}

// Convenience function for checking CUDA runtime API results
// // can be wrapped around any runtime API call. No-op in release builds.
inline const char *gpublasGetErrorString(GPUBLAS_STATUS_T status) {
    switch (status) {
    case GPUBLAS_STATUS_SUCCESS:
        return "GPUBLAS_STATUS_SUCCESS";
    case GPUBLAS_STATUS_NOT_INITIALIZED:
        return "GPUBLAS_STATUS_NOT_INITIALIZED";
    case GPUBLAS_STATUS_ALLOC_FAILED:
        return "GPUBLAS_STATUS_ALLOC_FAILED";
    case GPUBLAS_STATUS_INVALID_VALUE:
        return "GPUBLAS_STATUS_INVALID_VALUE";
    case GPUBLAS_STATUS_ARCH_MISMATCH:
        return "GPUBLAS_STATUS_ARCH_MISMATCH";
    case GPUBLAS_STATUS_MAPPING_ERROR:
        return "GPUBLAS_STATUS_MAPPING_ERROR";
    case GPUBLAS_STATUS_EXECUTION_FAILED:
        return "GPUBLAS_STATUS_EXECUTION_FAILED";
    case GPUBLAS_STATUS_INTERNAL_ERROR:
        return "GPUBLAS_STATUS_INTERNAL_ERROR";
    default:
		return "IHAVENOCLUE";
	}
    return "unknown error";
}

inline GPU_ERROR_T checkGPU(GPU_ERROR_T result, const char * fmt, ... ) {
    if (result != GPU_SUCCESS) {
        va_list argPtr;
        va_start( argPtr, fmt );
        vfprintf( stderr, fmt, argPtr );
        va_end( argPtr );
        fprintf( stderr, "GPU Runtime Error: %s\n", GPU_GET_ERROR_STRING(result));
	fflush( stderr );
        assert( (result == GPU_SUCCESS) );
    }
    return result;
}

#ifndef SKIP
inline GPUBLAS_STATUS_T checkGPUblas(GPUBLAS_STATUS_T result) {
    if (result != GPUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "GPU Runtime Error: %s\n",
                gpublasGetErrorString(result));
        assert(result == GPUBLAS_STATUS_SUCCESS);
    }
    return result;
}
#endif

static inline size_t calc_lda_c(size_t n) {
    // XXX This is public version.
    // XXX FOR FUGAKU, CONTACT WITH FUJITSU AND RIKEN FOR SETTINGS WE USED IN
    // THE BENCHMARK.
    size_t const cl_size = 256ull;
    size_t const page_size = 4096ull;
    n = (n + cl_size - 1) / cl_size * cl_size;
    if (n % page_size)
        n += cl_size;
    return n;
}

template <typename FHigh, typename FLow> struct Workspace
{
        FHigh* Imatrix_high;
        FLow*  Imatrix_low;
        size_t Imatrix_ld;
};

template <typename FHigh, typename FLow>
void build_workspace( Workspace<FHigh,FLow> &wks, int b )
{
   size_t high_size = ( size_t )b * ( size_t )b * sizeof( FHigh );
   size_t low_size  = ( size_t )b * ( size_t )b * sizeof( FLow );

   checkGPU( GPU_MALLOC( ( void ** )&(wks.Imatrix_high), high_size  ),
           "<FATAL:%s> %s\n", hostRank, "Allocating WORKSPACE" );
   checkGPU( GPU_MALLOC( ( void ** )&(wks.Imatrix_low), low_size  ),
           "<FATAL:%s> %s\n", hostRank, "Allocating WORKSPACE" );
   wks.Imatrix_ld = b;
}

template <typename F> struct Panels {
    // the panel descriptor.
    // We use 2D block-cyclic layout. This is little simpler than the PBLAS's
    // one. It is like m == n, mb == nb, m%mb == n%nb == 0, and i0==j0==0 in
    // PBLAS's descriptor. Let P is the local sub-matrices and A is the global
    // matrix:
    char *p; // &(P_{i, j}) == p+(i-1)*ldp+(j-1)*ldpp

    F *d_p; // Device pointer
    F *p_init; // Permant storage for initial entries
    int * d_infoArray;

    size_t alloc_size; // allocated memory size in bytes
    size_t lda;
    size_t ldp;
    size_t ldpp;

    size_t d_lda;
    size_t d_ldp;
    size_t d_ldpp;

    int nblocks;      // the number blocks of row and column of A. A_{nblocks,
                      // nblocks} is the right bottom corner
    int b;            // nrow(P_{*,*}) == ncol(P_{*,*})
    int nprow, npcol; // P_{nprow,npcol} is at the right bottom corner
    int i1, j1;       // P_{1, 1} == A_{i1, j1}
    int istride, jstride; // P_{i, j} == A_{i1+(i-1)*istride, j1+(j-1)*jstride}

    int epoch_size;
    bool is_tile;

    int solv;
    int sync;
    int comm;
    int dcomm;
    int skip;
    int stream_flag;
    int alt;
    int progress;

    GPUBLAS_HANDLE_T cuHandle;
    GPUSOLVER_HANDLE_T solvHandle;
    GPU_STREAM_T cuStream;

    F *panel_address(int i, int j) {
        return reinterpret_cast<F *>(p) + i * ldp + j * ldpp;
    }
    F const *panel_address(int i, int j) const {
        return reinterpret_cast<F const *>(p) + i * ldp + j * ldpp;
    }
    F *operator()(int i, int j) { return panel_address(i, j); }
    F const *operator()(int i, int j) const { return panel_address(i, j); }

    F *d_panel_address(int i, int j) {
        return reinterpret_cast<F *>(d_p) + i * ldp + j * ldpp;
    }
    F const *d_panel_address(int i, int j) const {
        return reinterpret_cast<F const *>(d_p) + i * ldp + j * ldpp;
    }
    F *operator()(int i, int j, char device) { return d_panel_address(i, j); }
    F const *operator()(int i, int j, char device) const {
        return d_panel_address(i, j);
    }

    // double-decker placement.
    // lower-precision data are placed on the lower half of the high-precisino
    // matrix.
    template <typename FLow> FLow *lower_deck(int i, int j) {
        // sizeof(FLow)*2 == sizeof(F)
        if (is_tile)
            return reinterpret_cast<FLow *>(
                       p + sizeof(F) * (i * ldp + j * ldpp + lda)) -
                   lda;
        else
            return reinterpret_cast<FLow *>(p + sizeof(F) * (j * ldpp) +
                                            sizeof(FLow) * b * nprow) +
                   i * ldp;
    }
    template <typename FLow> FLow const *lower_deck(int i, int j) const {
        if (is_tile)
            return reinterpret_cast<FLow const *>(
                       p + sizeof(F) * (i * ldp + j * ldpp + lda)) -
                   lda;
        else
            return reinterpret_cast<FLow const *>(p + sizeof(F) * (j * ldpp) +
                                                  sizeof(FLow) * b * nprow) +
                   i * ldp;
    }

    template <typename FLow> FLow *higher_deck(int i, int j) {
        // sizeof(FLow)*2 == sizeof(F)
        if (is_tile)
            return reinterpret_cast<FLow *>(p +
                                            sizeof(F) * (i * ldp + j * ldpp));
        else
            return reinterpret_cast<FLow *>(p + sizeof(F) * (j * ldpp)) +
                   i * ldp;
    }
    template <typename FLow> FLow const *higher_deck(int i, int j) const {
        if (is_tile)
            return reinterpret_cast<FLow const *>(p + sizeof(F) *
                                                          (i * ldp + j * ldpp));
        else
            return reinterpret_cast<FLow const *>(p + sizeof(F) * (j * ldpp)) +
                   i * ldp;
    }

    template <typename FLow> size_t lower_deck_lda() const {
        return sizeof(F) / sizeof(FLow) * lda;
    }
    template <typename FLow> size_t higher_deck_lda() const {
        return lower_deck_lda();
    }
};

// An wrapper to get the panel address for each matrix layout.
template <typename F, typename FLow, bool t> struct DDAdaptor {};
template <typename F, typename FLow> struct DDAdaptor<F, FLow, true> {
    typedef FLow FDeck;
    static FLow *get_deck(Panels<F> &p, int row, int col) {
        return p.template lower_deck<FLow>(row, col);
    }
    static size_t get_ldl(Panels<F> const &p) {
        return p.template lower_deck_lda<FLow>();
    }
};
template <typename F, typename FLow> struct DDAdaptor<F, FLow, false> {
    typedef F FDeck;
    static F *get_deck(Panels<F> &p, int row, int col) { return p(row, col); }
    static size_t get_ldl(Panels<F> const &p) { return p.lda; }
};

template <typename F> struct LRPanels {
    // ther working space for receiving left and right panels
    char *p; // maybe different type ie fp16.

    char *d_p;
    size_t offset;
    // offset from the alloc head
    // in case allocating all panels in a single region to have better control
    // over the alignement.
    size_t lda;
    size_t d_lda;

    size_t ldp;
    int istart;
    bool is_tile;
    bool is_pack;

    void set_start(int i) { istart = i; }
    size_t get_lda() const { return is_tile ? lda : (lda - ldp * istart); }
    F *data() { return reinterpret_cast<F *>(p); }
    F const *data() const { return reinterpret_cast<F const *>(p); }
    F *address(int i) { return reinterpret_cast<F *>(p) + (i - istart) * ldp; }
    F const *address(int i) const {
        return reinterpret_cast<F const *>(p) + (i - istart) * ldp;
    }
    F *operator()(int i) { return address(i); }
    F const *operator()(int i) const { return address(i); }

    F *d_data() { return reinterpret_cast<F *>(d_p); }
    F const *d_data() const { return reinterpret_cast<F const *>(d_p); }
    F *d_address(int i) {
        return reinterpret_cast<F *>(d_p) + (i - istart) * ldp;
    }
    F const *d_address(int i) const {
        return reinterpret_cast<F const *>(d_p) + (i - istart) * ldp;
    }
    F *operator()(int i, char device) { return d_address(i); }
    F const *operator()(int i, char device) const { return d_address(i); }
};

template <typename FHigh, typename FLow>
int build_panels_wGPU(int n, int b, bool tile_layout, bool pack,
                      Panels<FHigh> &p, LRPanels<FLow> *lr, int row, int col,
                      int nrow, int ncol, int nbuf = 2) {
    assert(n >= 0);
    assert(b >= 1);
    assert(n % b == 0);
    assert(n / b > 0);
    assert(nbuf >= 2);
    int nb = n / b;
    p.i1 = row;
    p.j1 = col;
    p.nprow = (nb - p.i1 + nrow - 1) / nrow;
    p.npcol = (nb - p.j1 + ncol - 1) / ncol;
    p.istride = nrow;
    p.jstride = ncol;
    p.nblocks = nb;
    p.b = b;

    GPUSOLVER_STATUS_T solvStat;
	checkGPU(GPU_STREAM_CREATE_WITH_FLAGS(&(p.cuStream), GPU_STREAM_NON_BLOCKING),
	         "<FATAL:%s> %s\n", hostRank, "Stream Create" );

    checkGPUblas(GPUBLAS_CREATE(&p.cuHandle));
	solvStat = GPUSOLVER_CREATE(&p.solvHandle);
	assert(solvStat == GPUSOLVER_STATUS_SUCCESS);

	if(p.stream_flag != 0){
		solvStat = 	GPUSOLVER_SET_STREAM(p.solvHandle, p.cuStream);
		assert(solvStat == GPUSOLVER_STATUS_SUCCESS);
		checkGPUblas(GPUBLAS_SET_STREAM(p.cuHandle,p.cuStream));
	}

    p.lda = b * p.nprow; // calc_lda_c(b*p.nprow * sizeof(FHigh)) / sizeof(FHigh);
    p.d_lda = b * p.nprow;
    p.ldp = b;
    p.ldpp = p.lda * b;
    p.is_tile = false;
    for (int i = 0; i < nbuf; ++i) {
        lr[2 * i].lda = b * p.nprow;
        lr[2 * i].ldp = b;
        lr[2 * i].istart = 0;
        lr[2 * i].is_tile = false;
        lr[2 * i].is_pack = false;
        lr[2 * i + 1].lda = b * p.npcol; // b;
        lr[2 * i + 1].ldp = b;           // b * b;
        lr[2 * i + 1].istart = 0;
        lr[2 * i + 1].is_tile = false; // Transpose version  //true;
        lr[2 * i + 1].is_pack = false;
	} 
    
    size_t sz_p = sizeof(FHigh) * b * b * p.nprow * p.npcol;
    sz_p = (sz_p + ALIGN ) & ~ALIGN;
    size_t sz_l = sizeof(FLow) * b * b * p.nprow;
    sz_l = (sz_l + ALIGN ) & ~ALIGN;
    size_t sz_r = sizeof(FLow) * b * b * p.npcol;
    sz_r = (sz_r + ALIGN ) & ~ALIGN;

    p.alloc_size = sz_p + nbuf * sz_l + nbuf * sz_r + 0x100ull;
    char *addr = NULL;
    
    if(my_init != 1)
    {
        checkGPU( GPU_MALLOC_HOST( (void **)&addr, p.alloc_size ),
            "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] array space" );
        p.p = addr;
        memset(p.p, 0, p.alloc_size);
    }
    size_t tot_gpu = 0;
    tot_gpu += sz_p;
    tot_gpu += (size_t(nbuf) * (sz_l + sz_r));
    PrintLogMsg( "\tgpu p[%ld] l[%ld] r[%ld] ... tot[%ld]\n", sz_p,
		size_t(nbuf) * sz_l, size_t(nbuf) * sz_r, tot_gpu );

    checkGPU( GPU_MALLOC( (void **)&(p.d_p), sz_p ),
            "<FATAL:%s> %s\n", hostRank, "Allocating device[gpu] array space" );

    checkGPU( GPU_MALLOC( ( void ** )&(p.d_infoArray), sizeof( int ) ),
	        "<FATAL:%s> %s\n", hostRank, "Allocating device[gpu] infoArray" );

    for (int i = 0; i < nbuf; ++i) {
        checkGPU( GPU_MALLOC( (void **)&(lr[2 * i].d_p), sz_l ),
	            "<FATAL:%s> %s\n", hostRank, "Allocating device[gpu] left panels" );
        lr[2 * i].offset = sz_p + sz_l * i;
        
        checkGPU( GPU_MALLOC((void **)&(lr[2 * i + 1].d_p), sz_r ),
	            "<FATAL:%s> %s\n", hostRank, "Allocating device[gpu] right panels" );
        
        lr[2 * i + 1].offset = sz_p + nbuf * sz_l + sz_r * i;
        lr[2 * i].p = addr + sz_p + sz_l * i;
        lr[2 * i + 1].p = addr + sz_p + nbuf * sz_l + sz_r * i;
    }
    return nb;
}

template <typename FHigh, typename FLow>
void destruct_panels(Panels<FHigh> &p, LRPanels<FLow> * lr, int nbuf ) {

    GPU_FREE_HOST( p.p );
    GPU_FREE( p.d_p );
    for ( int i = 0; i < nbuf; ++i ) {
       GPU_FREE( lr[ 2 * i     ].d_p );
       GPU_FREE( lr[ 2 * i + 1 ].d_p );
    }

}

// row and column vectors are hold on the node which has the diagoanl block of
// same index. Thus, there is no duplication over nodes.
template <typename F, typename Fv>
void colv2rowv(Panels<F> const &p, Fv const *cx, Fv *rx) {
    int b = p.b;
    for (int i = 0; i < p.nprow; ++i) {
        int ipos = p.i1 + i * p.istride;
        if (ipos % p.jstride == p.j1) {
            int j = (ipos - p.j1) / p.jstride;
#pragma omp parallel for simd
            for (int k = 0; k < b; ++k)
                rx[b * j + k] = cx[b * i + k];
        }
    }
}

template <typename F, typename Fx, typename Fy>
void copycolv(Panels<F> const &p, Fx const *x, Fy *y) {
    int b = p.b;
    for (int i = 0; i < p.nprow; ++i) {
        int ipos = p.i1 + i * p.istride;
        if (ipos % p.jstride == p.j1) {
#pragma omp parallel for simd
            for (int k = 0; k < b; ++k)
                y[b * i + k] = static_cast<Fy>(x[b * i + k]);
        }
    }
}

template <typename F, typename Fv>
void addcolv(Panels<F> const &p, Fv const *x, Fv *y) {
    int b = p.b;
    for (int i = 0; i < p.nprow; ++i) {
        int ipos = p.i1 + i * p.istride;
        if ((ipos % p.jstride) == p.j1) {
#pragma omp parallel for simd
            for (int k = 0; k < b; ++k)
                y[b * i + k] += x[b * i + k];
        }
    }
}

template <typename F, typename Fv>
void divcolv(Panels<F> const &p, Fv const *x, Fv *y) {
    int b = p.b;
    for (int i = 0; i < p.nprow; ++i) {
        int ipos = p.i1 + i * p.istride;
        if ((ipos % p.jstride) == p.j1) {
#pragma omp parallel for simd
            for (int k = 0; k < b; ++k)
                y[b * i + k] /= x[b * i + k];
        }
    }
}


// ORNL VECTOR OPS
 /*
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
            template addcolv_d<<< 1, VECTOR_OP_THREADS>>> (b, x+ii*b, y+ii*b);
        }
    }
}
*/
#endif
