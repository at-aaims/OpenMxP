#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <utility>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include <pthread.h>

#include "panel.hpp"
#include "grid.hpp"
#include "iterative_refinement.hpp"
#include "panel_norm.hpp"
#include "panel_check.hpp"
#include "hpl_rand.hpp"
#include "panel_trf.hpp"
#include "matgen.hpp"
#include "timer.hpp"
#include "highammgen.hpp"
#include "gpu_init.hpp"

#include "panel_trf_gdirect.hpp"
#include "panel_trf_block.hpp"

#include "device_macros.h"
#include "log.hpp"

int   grank, gsize;
int   glog = 1;
float power0;

FILE * tracePtr;
pthread_t threadID;

int gcdOrder[ ] = { 4, 5, 2, 3, 6, 7, 0, 1 };

static int down_clock();

// FHIGH: the floating-point type for panel decompositions
// FLOW : that for GEMM

#define DUMP 0
//#define FHIGH double
#define FHIGH float
#define FLOW __half

#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "NA"
#endif

#ifndef GIT_BRANCH
#define GIT_BRANCH "NA"
#endif


std::string curbranch = GIT_BRANCH;
std::string curhash = GIT_COMMIT_HASH;

char hostRank[ 1024 ];
char sysName[ 1024 ];

int my_reorder = 0;
int reorder[ 8 ] = { 0, 1, 2, 3, 4, 5, 6, 7 }; // rank reorder

int my_skip = 0;  // bit-mask to skip functions ( rocm only )
                  // bit 0 is ir
		  // bit 1 is getrf, bit 2 is trsm, bit 3 is gemm (gdirect only )
int my_solv  =  1;  // cusolver/rocsolver default
int my_init  =  1;  // optimized gpu init default
int my_comm  =  2;  // ring default
int my_dcomm =  2;  // diagonal comm ( same as my_comm )
int my_sync  =  0;
int my_stream_flag = 0;
int my_gdirect     = 1;
int half_version   = 0;
int my_alt         = 1;
int my_progress    = 10;
int my_cpuir       = 0;
int64_t bgn_time;
int64_t end_time;

static void get_flags(int argc,
                      char **argv,
                      bool &tile,
                      int &dd_layout,
                      bool &manual,
                      bool &rdma,
                      bool &warmup,
                      bool &pack,
                      bool &checksum,
                      int &numasize,
                      NumaMap &numamap,
                      int &nbuf,
                      int &jobid,
                      bool &l1est,
                      bool &higham,
                      bool &pwrtune) {
    tile = false;
    dd_layout = 0;
    manual = false;
    rdma = false;
    warmup = false;
    pack = false;
    numasize = 5;
    numamap = NumaMap::ROWDIST;
    nbuf = 2;
    jobid = 0;
    l1est = false;
    higham = false;
    pwrtune = false;
    strcpy( sysName, "Unknown" );
    for(int i = 0; i < argc; ++i) {
        if(!std::strcmp(argv[i], "-not") || !std::strcmp(argv[i], "-nt") || !std::strcmp(argv[i], "--notile"))
            tile = false;
        else if(!std::strcmp(argv[i], "-d") || !std::strcmp(argv[i], "--delayed"))
            dd_layout = 1;
        else if(!std::strcmp(argv[i], "-l") || !std::strcmp(argv[i], "--lazy-init"))
            dd_layout = 1;
        else if(!std::strcmp(argv[i], "--back_buffer"))
            dd_layout = 2;
        else if(!std::strcmp(argv[i], "-m") || !std::strcmp(argv[i], "--manual-progress"))
            manual = true;
        else if(!std::strcmp(argv[i], "-r") || !std::strcmp(argv[i], "--rdma"))
            rdma = true;
        else if(!std::strcmp(argv[i], "-w") || !std::strcmp(argv[i], "--warmup"))
            warmup = true;
        else if(!std::strcmp(argv[i], "--pack"))
            pack = true;
        else if(!std::strcmp(argv[i], "--checksum"))
            checksum = true;
        else if(!std::strcmp(argv[i], "--numa")) {
            if(i == argc - 1)
                break;
            numasize = std::atoi(argv[i + 1]);
            ++i;
        } else if(!std::strcmp(argv[i], "--numa-map")) {
            if(i == argc - 1)
                break;
            if(!std::strcmp(argv[i + 1], "ROWCONT"))
                numamap = NumaMap::ROWCONT;
            else if(!std::strcmp(argv[i + 1], "COLCONT"))
                numamap = NumaMap::COLCONT;
            else if(!std::strcmp(argv[i + 1], "ROWDIST"))
                numamap = NumaMap::ROWDIST;
            else if(!std::strcmp(argv[i + 1], "COLDIST"))
                numamap = NumaMap::COLDIST;
            else if(!std::strcmp(argv[i + 1], "CONT2D"))
                numamap = NumaMap::CONT2D;
            ++i;
        }
        // ORNL modification
        else if(!std::strcmp(argv[i], "-solv")) {
            if(i == argc - 1)
                break;
            my_solv = std::atoi(argv[i + 1]);
            ++i;
        } else if(!std::strcmp(argv[i], "-log")) {
            if(i == argc - 1)
                break;
            glog = std::atoi(argv[i + 1]);
            ++i;
        } else if(!std::strcmp(argv[i], "-init")) {
            if(i == argc - 1)
                break;
            my_init = std::atoi(argv[i + 1]);
            ++i;
        } else if(!std::strcmp(argv[i], "-alt")) {
            if(i == argc - 1)
                break;
            my_alt = std::atoi(argv[i + 1]);
            ++i;
        } else if(!std::strcmp(argv[i], "-progress")) {
            if(i == argc - 1)
                break;
            my_progress = std::atoi(argv[i + 1]);
            ++i;
        } else if(!std::strcmp(argv[i], "-cpuir")) {
            if(i == argc - 1)
                break;
            my_cpuir = std::atoi(argv[i + 1]);
            ++i;
        } else if(!std::strcmp(argv[i], "-comm")) {
            if(i == argc - 1)
                break;
            my_comm = std::atoi(argv[i + 1]);
            ++i;
        } else if(!std::strcmp(argv[i], "-dcomm")) {
            if(i == argc - 1)
                break;
            my_dcomm = std::atoi(argv[i + 1]);
            ++i;
        } else if(!std::strcmp(argv[i], "-sync")) {
            my_sync = 1;
        } else if(!std::strcmp(argv[i], "-skip")) {
	    if ( i == argc - 1 )
	         break;
            my_skip = std::atoi( argv[ i + 1 ] );
	    ++i;
        } else if(!std::strcmp(argv[i], "-gdirect")){
            my_gdirect = 1;
        } else if(!std::strcmp(argv[i], "-no_gdirect")){
            my_gdirect = 0;
	} else if ( !std::strcmp( argv[ i ], "-reorder" ) ) {
            if ( i == argc - 1 ) break;
	    my_reorder = 1;
	    ++i; reorder[ 0 ] = std::atoi( argv[ i ] );
	    ++i; reorder[ 1 ] = std::atoi( argv[ i ] );
	    ++i; reorder[ 2 ] = std::atoi( argv[ i ] );
	    ++i; reorder[ 3 ] = std::atoi( argv[ i ] );
	    ++i; reorder[ 4 ] = std::atoi( argv[ i ] );
	    ++i; reorder[ 5 ] = std::atoi( argv[ i ] );
	    ++i; reorder[ 6 ] = std::atoi( argv[ i ] );
	    ++i; reorder[ 7 ] = std::atoi( argv[ i ] );
        } else if(!std::strcmp(argv[i], "-AHALF")){
            half_version  = 1;
        } else if ( !std::strcmp( argv[ i ], "-sys" ) ) {
	    if ( i == argc - 1 ) break;
	    std::strcpy( sysName, argv[ i + 1 ] );
	    ++i;
	    }
	    // end modification
        else if(!std::strcmp(argv[i], "--nbuf")) {
            if(i == argc - 1)
                break;
            nbuf = std::atoi(argv[i + 1]);
            ++i;
        } else if(!std::strcmp(argv[i], "--job-id")) {
            if(i == argc - 1)
                break;
            jobid = std::atoi(argv[i + 1]);
            ++i;
        } else if(!std::strcmp(argv[i], "-stream")) {
            my_stream_flag = 1;
        } else if(!std::strcmp(argv[i], "--l1est"))
            l1est = true;
        else if(!std::strcmp(argv[i], "--highammat"))
            higham = true;
        else if(!std::strcmp(argv[i], "--pwrtune"))
            pwrtune = true;
        else if(!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            const char *helpmessage = "Coming soon";
            printf("%s", helpmessage);
        } else {
            if(i > 0) {
                printf("Ignoring unknown option: %s\n", argv[i]);
            }
        }
    }
}

static void print_ctime(const char *msg, std::FILE *fp=stdout){
    std::time_t t;
    std::time(&t);
    std::fprintf(fp, "%s%s", msg, std::ctime(&t));
}

static char* curtime() {
    std::time_t t;
    std::time(&t);
    return std::ctime(&t);
}

static int gcd(int a, int b){
    if(a<b) std::swap(a, b);
    return b ? gcd(b, a%b) : a;
}

static int my_lcm(int a, int b) { return a * b / gcd(a, b); }

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &grank );
        MPI_Comm_size( MPI_COMM_WORLD, &gsize );

        if( grank == 0 ) {
            printf("===============================================================================\n");
            printf("OpenMxP - High Performance Mixed Precision Benchmark - NCCS/OLCF\n");
            printf("===============================================================================\n");
            printf("Build Info: git branch: [%s], hash-id: [%s]\n", curbranch.c_str( ), curhash.substr(0,8).c_str( ));
            printf("Running configuration:\n");
            printf("\tRanks = %d\n", gsize );
        
        #pragma omp parallel
	    {
	       if ( omp_get_thread_num( ) == 0 )
         	    printf("\tOpenMP threads = %d\n", omp_get_num_threads( ) );
	    }
	    if ( getenv( "CMD"  ) != NULL ) printf( "\t%s\n", getenv( "CMD" ) );
	    if ( getenv( "NOTE" ) != NULL ) printf( "\tNOTE: %s\n", getenv( "NOTE" ) );
	        fflush( stdout );
        }
    }
    GPU_SET_DEVICE( gcdOrder[ grank % 8 ] );
    GPU_DEVICE_RESET( );
    GPU_DEVICE_SYNCHRONIZE( );

#ifndef CUDA_OLCF_PLATFORM
    rocblas_initialize();
#endif 

    int n = 120;
    int b = 3;
    int nrow = 2;
    int my_ncol = -1;
    bool tile_layout = false;
    int dd_layout = 0;
    bool manual_progress = false;
    bool rdma = false;
    bool warmup = false;
    bool pack = false;
    bool checksum = false;
    int numasize = 0;
    int nbuf = 2;
    int jobid = 0;
    bool l1est = false;
    bool higham = false;
    bool pwrtune = false;
    NumaMap numamap = NumaMap::ROWDIST;
    int seed = 42;

    int  nameLen;
    char hostName[ 1024 ];
    char timeDate[ 1024 ];

    MPI_Get_processor_name( hostName, &nameLen );
    sprintf( hostRank, "%s[%d]", hostName, grank );

    if(argc > 1) n       = std::atoi(argv[1]);
    if(argc > 2) b       = std::atoi(argv[2]);
    if(argc > 3) nrow    = std::atoi(argv[3]);
    if(argc > 4) my_ncol = std::atoi(argv[4]);
    get_flags(argc-4, argv+4, tile_layout, dd_layout, manual_progress,
        rdma, warmup, pack, checksum, numasize, numamap, nbuf, jobid, l1est, higham, pwrtune);

    int my_epoch;
    if ( my_ncol == -1 ) my_ncol  = gsize / nrow;
    my_epoch = b * my_lcm( nrow, my_ncol );

    if( n % my_epoch != 0 ) {
        int nf;
        float f = (float)n / (float)my_epoch;
        nf = (int)(f + 0.5f);
        if ( nf == 0 ) nf = 1;
        n = nf * my_epoch;
    }
    PrintLogMsg( "\tPxQ %dx%d\n", nrow, my_ncol );

    // current look ahead max
    if ( nbuf > 2 ) nbuf = 2;
    dd_layout=0;
    assert(!pack || (pack&&rdma));
    assert(!rdma || (rdma&&!tile_layout));
        
    {
        // memory allocations and set-up of pandel descriptions
        Grid g(MPI_COMM_WORLD, nrow, numasize, numamap);
        Panels<FHIGH> p;
        LRPanels<FLOW> lr[nbuf*2];
        p.solv  = my_solv;
        p.sync  = my_sync;
        p.comm  = my_comm;
	    if ( my_dcomm != -1 )
	    {
	        p.dcomm = my_dcomm;
	    }
	    else
	    {
           p.dcomm = my_comm;
	    }

	    p.skip        = my_skip;
        p.stream_flag = my_stream_flag;
    
        // Print Sovler
        const char * s_Solvers[ ] = { "cublas Sgetrf", "cusolver/rocsolver Sgetrf" };
        assert( ( my_solv >= 0  &&  my_solv <= 1 ) ); 
        PrintLogMsg( "\t%s\n", s_Solvers[ p.solv ] );

        // Print Comm
	    const char * s_Comm[ ] = { "ibcast", "bcast", "1ring", "1ringM",
		                   "2ringM", "1ring_ir", "1ringM_ir","2ringM_ir" };
        assert( ( my_comm >= 0  &&  my_comm <= 7 ) );
        PrintLogMsg( "\t%s comm\n", s_Comm[ my_comm ] );
        if ( my_comm == 3 ) assert( ( nrow > 3 ) && ( my_ncol > 3 ) );
        if ( my_comm == 4 ) assert( ( nrow > 4 ) && ( my_ncol > 4 ) );
        if ( my_comm == 6 ) assert( ( nrow > 3 ) && ( my_ncol > 3 ) );
        if ( my_comm == 7 ) assert( ( nrow > 4 ) && ( my_ncol > 4 ) );
        PrintLogMsg( "\t%s dcomm\n", s_Comm[ my_dcomm ] );

        // Print Grid setting
	    assert( ( numasize >= 0  &&  numasize <= 10 ) );
	    const char * s_Grid[ ] = { "Global Column Major", "Node Grid - 2x3C",
	                           "Node Grid - 3x2C", "Global Row Major",
		                   "Node Grid - 2x4R", "Node Grid - 2x4C",
    				   "Node Grid - 4x2R", "Node Grid - 4x2C",
				   "Node Grid - 1x8" , "Node Grid - 4x4" ,
	                           "Node Grid - Reorder 2x4C" };
	    PrintLogMsg( "\t%s pxq grid\n", s_Grid[ numasize ] );
        int tot_pq = nrow * my_ncol;
        if ( ( numasize == 1  ||  numasize == 2 )  &&  tot_pq < 6 )
        {
        PrintLogMsg( "\tCan't run this grid configuration\n" );
            exit( 1 );
        }
        if ( ( numasize >= 4  &&  numasize <= 8 )  &&  tot_pq < 8 )
        {
        PrintLogMsg( "\tCan't run this grid configuration\n" );
            exit( 1 );
        }

        // Print sync
        const char * s_Sync[ ] = { "no sync", "sync" };
        assert( ( my_sync >= 0  &&  my_sync <= 1 ) );
        PrintLogMsg( "\t%s on sgemm\n", s_Sync[ my_sync ] );

        // Print stream
        const char * s_Stream[ ] = { "default", "non-blocking" };
        assert( ( my_stream_flag >= 0  &&  my_stream_flag <= 1 ) );
        PrintLogMsg( "\t%s stream sgemm\n", s_Stream[ my_stream_flag ] );

        // Print gdirect
        const char * s_Gdirect[ ] = { "copy and send", "gpu direct" };
        assert( ( my_gdirect >= 0  &&  my_gdirect <= 1 ) );
        PrintLogMsg( "\t%s mpi\n", s_Gdirect[ my_gdirect ] );

        // Print reorder
        if ( my_reorder == 1  &&  numasize == 10 )
        PrintLogMsg( "\tRecorder=%d %d %d %d %d %d %d %d\n",
                reorder[ 0 ], reorder[ 1 ], reorder[ 2 ],
                reorder[ 3 ], reorder[ 4 ], reorder[ 5 ],
                reorder[ 6 ], reorder[ 7 ] );

        // Skipping
        if ( my_skip > 0 )
        {
            if ( CHECK_BIT( my_skip, 0 ) ) PrintLogMsg( "\tSkipping IR\n"    );
            if ( CHECK_BIT( my_skip, 1 ) ) PrintLogMsg( "\tSkipping GETRF\n" );
            if ( CHECK_BIT( my_skip, 2 ) ) PrintLogMsg( "\tSkipping TRSM\n"  );
            if ( CHECK_BIT( my_skip, 3 ) ) PrintLogMsg( "\tSkipping GEMM\n"  );
        }
		
        // Epochs
	    auto lcm = [](int a, int b){ return a * b / gcd(a, b); };
        int epoch_size = b * lcm(g.nrow, g.ncol);
        p.epoch_size = epoch_size;

        // Initialization setting, result in different IR
        const char * s_Init[ ] = { "Default HPL_rand", "GPU", "Optimized HPL_rand"};
	    assert( ( my_init >= 0  &&  my_init <= 3 ) );
        PrintLogMsg( "\t%s Initialization\n", s_Init[ my_init ] );

        // Alternative trsm using trsm + fp16gemm
	    p.alt = my_alt;
	    const char * s_Alt[ ] = { "Default Trsm", "Alt-Trsm","Alt2" };
	    assert( ( my_alt >= 0  &&  my_alt <= 2 ) );
	    PrintLogMsg( "\t%s Method\n", s_Alt[ my_alt ] );
	    p.progress = my_progress;

        // Panels building
        PrintLogMsg( "\tBuilding panels\n" );
	    build_panels_wGPU(n, b, tile_layout, pack, p, lr, g.row, g.col, g.nrow, g.ncol, nbuf);
    	PrintLogMsg( "\tFinished building panels\n" );

        int alternative = p.alt;
        Workspace<FHIGH,FLOW> wks;
        Workspace<FHIGH,FLOW> wks2;
        if ( alternative )
        {
            if(grank == 0){printf("BUILD WORK SPACE\n");}
            build_workspace( wks, b );
            build_workspace( wks2, b );
            gen_identity_mat(wks.Imatrix_low, (long)p.b,(long)p.b);
            gen_identity_mat(wks2.Imatrix_low, (long)p.b,(long)p.b);
        }

        // allocating for pivot
        FHIGH * piv;
        FHIGH * d_piv;
        checkGPU( GPU_MALLOC( (void**)&(d_piv), b * b * sizeof(FHIGH) ),
	         "<FATAL:%s> %s\n", hostRank, "Allocating device[gpu] piv" );
        PrintLogMsg( "\tgpu d_piv[%ld]\n", b * b * sizeof( FHIGH ) );
        checkGPU( GPU_MALLOC_HOST( ( void ** )&piv, b * b * sizeof( FHIGH ) ),
	         "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );

        // Allocating meta data
        size_t ldv = b * (p.npcol + p.nprow);
        double *d = static_cast<double*>(std::malloc(sizeof(double) * ldv));
        FHIGH *dhigh = static_cast<FHIGH*>(std::malloc(sizeof(FHIGH) * ldv));
        double *rhs = static_cast<double*>(std::malloc(sizeof(double) * ldv));
        double *x = static_cast<double*>(std::malloc(sizeof(double) * ldv));
        bool wallocated = false;
        double *w =  static_cast<double*>(malloc(sizeof(double)*(ldv*10+2*b*b)));
	    if ( w == NULL || rhs == NULL || dhigh == NULL || d == NULL || x == NULL )
	    {
	        printf( "Failure to allocate memory for meta data\n" );
	        exit( 10 );
	    }
        wallocated = true;
        memset(d, 0, sizeof(double) * ldv);
        
        // Allocate GPU meta data
        double *d_dev;
        double *rhs_dev;
        double *x_dev;
        double *w_dev;
        if(my_init == 1){
            checkGPU( GPU_MALLOC( (void **)&(d_dev), sizeof(double) * ldv ),
                    "<FATAL:%s> %s\n", hostRank, "Allocating device[gpu] array space" );
            checkGPU( GPU_MALLOC( (void **)&(rhs_dev), sizeof(double) * ldv ),
                    "<FATAL:%s> %s\n", hostRank, "Allocating device[gpu] array space" );
            checkGPU( GPU_MALLOC( (void **)&(x_dev), sizeof(double) * ldv ),
                    "<FATAL:%s> %s\n", hostRank, "Allocating device[gpu] array space" );
            checkGPU( GPU_MALLOC( (void **)&(w_dev), sizeof(double) * (ldv*5+2*b*b) ),
                    "<FATAL:%s> %s\n", hostRank, "Allocating device[gpu] array space" );
        }

        // Initialize matrice generator.
        Matgen<FHIGH> mg(seed, n, p.b*(p.istride-1), p.b*(p.jstride-1), dhigh);
        Matgen<double> mgir(seed, n, p.b*(p.istride-1), p.b*(p.jstride-1), d);
	
      
        MPI_Barrier(MPI_COMM_WORLD);  // to callibrate the epoc time
        bgn_time = get_utime( );
   
        if ( my_init == 0 ) {
            pmatgen(mg, p);
            pcolvgen(mgir, p, rhs);
            pdiaggen(mgir, p, d);
        } else if( my_init == 1) {
            panel_matgen_dev(mg, p, d_dev);
            panel_colvgen( mgir, p, rhs );
            GPU_MEMCPY(d, d_dev,ldv*sizeof(double), GPU_MEMCPY_DEVICE_TO_HOST);
            mgir.diag_dev = d_dev;
        } else if( my_init == 2 ){
            double* localSum = (double*)malloc(sizeof(double)*n);
            if ( localSum == NULL ){
                printf( "Failure to allocate localSum\n" );
                exit( 10 );
            }
            memset(localSum, 0, n*sizeof(double)); 
            pmatgen2(mg, p, localSum);
            pcolvgen(mgir, p, rhs);
            pdiaggen2(mg, p, d, localSum);
            free(localSum);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        end_time = get_utime( );
        PrintLogMsg( "\tInitialization %d = %.4f sec\n", my_init, (end_time - bgn_time) * 1.0e-09 );
       
      
        // Output memory usage
        size_t freeMem;
        size_t totMem;
        GPU_MEM_GET_INFO(&freeMem, &totMem);
        PrintLogMsg( "\tGpu memory (free/total) %ld/%ld", freeMem, totMem );

        // Prepare IR
        copycolv(p, rhs, x);
        copycolv(p, d, dhigh);
        double normb;
        normb = colv_infnorm(p, rhs, g);
        double norma;
        norma = hpl_infnorm(p, d, g);
	    size_t bytes = (size_t)(p.b * p.nprow) * (size_t)(p.b * p.npcol) * (size_t)sizeof( FHIGH );
        
        if(my_init == 1){
            GPU_MEMCPY(d_dev, d, sizeof(double) * ldv , GPU_MEMCPY_HOST_TO_DEVICE);
            GPU_MEMCPY(x_dev, x, sizeof(double) * ldv , GPU_MEMCPY_HOST_TO_DEVICE);
            GPU_MEMCPY(rhs_dev, rhs, sizeof(double) * ldv , GPU_MEMCPY_HOST_TO_DEVICE);
            // change mem placement 
            mgir.diag_dev = d_dev;
        }else{
             GPU_MEMCPY(p(0, 0,'d'), p(0, 0), bytes, GPU_MEMCPY_HOST_TO_DEVICE);
        }

        bgn_time = get_utime( );
       
        
        // needed to prevent timer dump overwrites
        if ( std::getenv( "LSB_JOBID" ) != NULL )
        {
	        jobid = atoi( std::getenv( "LSB_JOBID" ) );
        }
	    else if ( std::getenv( "SLURM_JOB_ID" ) != NULL )
        {
	        jobid = atoi( std::getenv( "SLURM_JOB_ID" ) );
        }

        if ( std::getenv( "SLURM_CLUSTER_NAME" ) != NULL  &&
	        std::strcmp( sysName, "Unknown" ) == 0 )
	    {
            std::strcpy( sysName, std::getenv( "SLURM_CLUSTER_NAME" ) );
	    }

        // print-out settings
        if(g.row == 0 && g.col == 0){
            printf( "\tjobid=%d\n", jobid );
            printf( "\tn=%d ln=%d b=%d r=%d c=%d epoch_size=%d\n", 
                    n, n / g.nrow, b, g.nrow, g.ncol, epoch_size );
            if( n % epoch_size ) printf( "\tepoch_size warning=%d\n", n%epoch_size );
            strcpy( timeDate, curtime( ) );
	        char * cPtr = strchr( timeDate, '\n' );
	        if ( cPtr != NULL ) *cPtr = '\0';
	        printf( "#BEGIN_: %s\n", timeDate );
	        fflush( stdout );
        }

        double start_time, stop_time;
        IRErrors er;
        
        Timer::initialize();
        Timer::beg(Timer::TOTAL);
        // start
        MPI_Barrier(MPI_COMM_WORLD);  // ensure all the procs start after
        start_time = MPI_Wtime();



	    if ( my_gdirect  &&  my_comm != 0 ){
            panel_lu_async_wGPU_gdirect_alt2<FHIGH, FLOW, Matgen, 0, false>(p, lr, mg, piv, d_piv, b, g, false,wks,wks2);
            //panel_lu_async_wGPU_gdirect_alt1<FHIGH, FLOW, Matgen, 0, false>(p, lr, mg, piv, d_piv, b, g, false,wks,wks2);
        }
        else if ( my_comm == 0 ) {
            panel_lu_async_wGPU<FHIGH, FLOW, Matgen, 0, false>(p, lr, mg, piv, d_piv, b, g, warmup);
        } else {
			panel_lu_async_wGPU_blockBcast<FHIGH, FLOW, Matgen, 0, false>(p, lr, mg, piv, d_piv, b, g, warmup);
        }
        MPI_Barrier(MPI_COMM_WORLD);  // for extra safety

	    double start_ir = MPI_Wtime( );
	    if ( !CHECK_BIT( my_skip, 0 ) )
	    {
 	       if ( grank == 0 ) printf( "\nIR Start\n" );

	       if ( my_init != 1 ){
                GPU_MEMCPY(p(0, 0), p(0, 0, 'd'), bytes, GPU_MEMCPY_DEVICE_TO_HOST);
                MPI_Barrier(MPI_COMM_WORLD);  // ensure all the procs start after 
                PrintLogMsg( "\nIR Start with CPU\n" );
                er = iterative_refinement2(my_init, p, mgir, x, w, ldv, rhs, 
                                    norma, normb, (warmup ? 1 : 50), g);
            }else{ 
                //GPU_MEMCPY(p(0, 0), p(0, 0, 'd'), bytes, GPU_MEMCPY_DEVICE_TO_HOST);
                PrintLogMsg( "\nIR Start with GPU\n" );
                er = iterative_refinement2(my_init, p, mgir, x_dev, w_dev, ldv, rhs_dev,
                                    norma, normb, (warmup ? 1 : 50), g);
            }
            MPI_Barrier(MPI_COMM_WORLD);  // ensure all the procs stop before
	    }
            
        // stop
        stop_time = MPI_Wtime( );
	    if ( grank == 0 ) printf( "\t IR Time %.2f sec\n", stop_time - start_ir );
        Timer::end(Timer::TOTAL);

        if(g.row == 0 && g.col == 0) {
            double time_duration = stop_time - start_time;
            double hplflops = 2. / 3 * n * n * n + 3. / 2 * n * n;
            if(warmup) {
                int nn = n - (g.nrow + 2) * b;
                double skip = 2. / 3 * nn * nn * nn + 3. / 2 * nn * nn;
                hplflops -= skip;
            }

            printf( "#END___: %s\n", curtime());
            printf( "%.3f sec. %.3f GFlop/s resid = %20.15e hpl-harness = %.9f TFlop/s per GPU = %.3f\n",
                        time_duration,
                        hplflops / time_duration * 1e-9,
                        er.residual,
                        er.hpl_harness,
			            hplflops / time_duration * 1e-12 / ( float )gsize );

            // Submission Output
            printf( "===================================\n"         );
            printf( "NCCS/OLCF Mixed Precision Benchmark\n"         );
            printf( "Code   : OpenMxP\n"                            );
            printf( "Build  : [%s]\n", curbranch.c_str( )           );
            printf( "Hash-id: [%s]\n", curhash.substr(0,8).c_str( ) );
            printf( "BldDate: [%s]\n", __DATE__                     );
            printf( "===================================\n"         );
            printf( "System : %s\n", sysName                        );
            printf( "Date   : %s\n", timeDate                       );
            printf( "JobID  : %d\n", jobid                          );
            printf( "===================================\n"         );
            printf( "N      : %d\n", n                              ); 
            printf( "NB     : %d\n", b                              ); 
            printf( "PMAP   : %s\n", s_Grid[ numasize ]             );
            printf( "P      : %d\n", nrow                           ); 
            printf( "Q      : %d\n", my_ncol                        ); 
            printf( "PFACT  : %s\n", "Right"                        );
            printf( "COMM   : %s\n", s_Comm[ my_comm ]              );
            printf( "DEPTH  : %d\n", 1                              ); 
            printf( "U      : %s\n", "transposed form"              ); 
            printf( "L1     : %s\n", "no-transposed form"           ); 
    	    printf( "TIME(s): %.3f\n", time_duration                ); 
	        printf( "GFLOPS : %.1f\n", hplflops /
                             time_duration * 1.0e-09               ); 
	        printf( "RESID  : %20.15e\n", er.residual               );
            printf( "HARNESS: %.9f\n", er.hpl_harness               );
        }

#if DUMP
        {
            static char filename[1024];
            std::sprintf(filename,
                         (warmup ? "Warmup.%06d.%09d.%04d.%05d.%05d" : "Timerdump.%06d.%09d.%04d.%05d.%05d"),
                         jobid,
                         n,
                         b,
                         gsize,
                         grank);
            if( gsize < 10 || grank == 0 || grank == gsize - 1 ||
	        grank == g.nrow || grank == (g.nrow + 1) * (g.ncol / 2) ||
               grank == gsize - g.nrow )
                Timer::dump_mp( gsize, grank, g.row, g.col, filename );
        }
#endif

        destruct_panels( p, lr, nbuf );
        GPU_FREE_HOST( piv );

        std::free( rhs );
        std::free( x );
        if ( wallocated ) std::free( w );
        std::free( d );
        std::free( dhigh );
    }

    MPI_Finalize();
    return 0;
}

