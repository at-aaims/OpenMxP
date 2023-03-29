#ifndef GRID_HPP
#define GRID_HPP
#include "fp16sim.hpp"
#include <cassert>
#include <cstring>
#include <mpi.h>
#include "log.hpp"

#ifdef __APPLE__
#define aligned_alloc(alignment, size) malloc(size)
#endif

extern int grank, gsize; 
extern int reorder[ 8 ];

enum NumaMap {
    // How to destribute NUMA processes to the process grid.
    ROWCONT, // continuous in row
    COLCONT, // continuous in column
    ROWDIST, // distributed (cyclic) over row
    COLDIST, // distributed (cyclic) over column
    CONT2D   // continuous in 2x2. this is only for nnuma==4
};

struct Grid {
    // vcomm is a communicator for vertical communication (inside a column)
    // row = id(vcomm), nrow = sz(vcomm)
    // hcomm is a communicator for horizontal communication (inside a row)
    // col = id(hcomm), ncol = sz(hcomm)
    int row, col;
    int nrow, ncol;
    int idnuma, nnuma;
    MPI_Comm vcomm, hcomm, commworld;
    Grid(MPI_Comm comm, int nrow, int numasize = 0,
         NumaMap map = NumaMap::ROWCONT)
        : commworld(comm) {
        assert(numasize >= 0);
        assert(numasize != 0 || map != NumaMap::ROWCONT);

        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        if ( gsize % nrow ) MPI_Abort(MPI_COMM_WORLD, 4);
        int ncol = gsize / nrow;
        int myrow, mycol;
        if ( numasize == 1 ) {
     	    PrintMsg("\tNode Grid - 2x3C");
	    idnuma = 0;
            nnuma = 1;
            int myNode = grank / 6;
            int myLocalID = grank % 6;
            int nodeRow = (myNode % (nrow / 2)) * 2;
            int nodeCol = (myNode / (nrow / 2)) * 3;
            myrow = (myLocalID % 2) + nodeRow;
            mycol = (myLocalID / 2) + nodeCol;
        }
	else if ( numasize == 2 ) {
       	    PrintMsg("\tNode Grid - 3x2C");
	    idnuma = 0;
            nnuma = 1;
            int myNode = grank / 6;
            int myLocalID = grank % 6;
            int nodeRow = (myNode % (nrow / 3)) * 2;
            int nodeCol = (myNode / (nrow / 3)) * 2;
            myrow = (myLocalID % 3) + nodeRow;
            mycol = (myLocalID / 3) + nodeCol;
        }
        else if ( numasize == 0 )
	{
            PrintMsg("\tGlobal Column Major");
	    idnuma = 0;
            nnuma = 1;
            myrow = grank % nrow;
            mycol = grank / nrow;
        }
        else if ( numasize == 3 )
	{
            PrintMsg("\tGlobal Row Major");
	    idnuma = 0;
            nnuma = 1;
            myrow = grank / ncol;
            mycol = grank % ncol;
        }
	else if ( numasize == 4 )
	{
            PrintMsg("\tNode Grid - 2x4R");
	    idnuma = 0;
            nnuma = 1;
	    int myNode    = grank / 8;
            int myLocalID = grank % 8;
            int nodeRow = (myNode % (nrow / 2)) * 2;
            int nodeCol = (myNode / (nrow / 2)) * 4;
	    myrow       = myLocalID / 4;
	    mycol       = myLocalID % 4;
	    myrow      += nodeRow;
            mycol      += nodeCol;
	}
	else if ( numasize == 5 )
	{
            PrintMsg("\tNode Grid - 2x4C");
	    idnuma = 0;
            nnuma = 1;
	    int myNode    = grank / 8;
            int myLocalID = grank % 8;
            int nodeRow = (myNode % (nrow / 2)) * 2;
            int nodeCol = (myNode / (nrow / 2)) * 4;
	    myrow       = myLocalID % 2;
	    mycol       = myLocalID / 2;
	    myrow      += nodeRow;
            mycol      += nodeCol;
	}
	else if ( numasize == 6 )
        {
       	    PrintMsg("\tNode Grid - 4x2R");
	    idnuma = 0;
            nnuma = 1;
            int myNode = grank / 8;
            int myLocalID = grank % 8;
            int nodeRow = (myNode % (nrow / 4)) * 4;
            int nodeCol = (myNode / (nrow / 4)) * 2;
	    myrow       = myLocalID / 2;
	    mycol       = myLocalID % 2;
	    myrow      += nodeRow;
            mycol      += nodeCol;
        }
	else if ( numasize == 7 )
        {
       	    PrintMsg("\tNode Grid - 4x2C");
	    idnuma = 0;
            nnuma = 1;
            int myNode = grank / 8;
            int myLocalID = grank % 8;
            int nodeRow = (myNode % (nrow / 4)) * 4;
            int nodeCol = (myNode / (nrow / 4)) * 2;
	    myrow       = myLocalID % 4;
	    mycol       = myLocalID / 4;
	    myrow      += nodeRow;
            mycol      += nodeCol;
        }
	else if ( numasize == 8 )
        {
       	    PrintMsg("\tNode Grid - 1x8C");
	    idnuma = 0;
            nnuma = 1;
	    int myNode    = grank / 8;
            int myLocalID = grank % 8;
            int nodeRow = ( myNode % nrow );
            int nodeCol = ( myNode / nrow ) * 8;
	    myrow       = myLocalID / 8;
	    mycol       = myLocalID % 8;
	    myrow      += nodeRow;
            mycol      += nodeCol;
        }
	else if ( numasize == 9 )
        {
            PrintMsg("\tNode Grid - 4x4");
            idnuma = 0;
            nnuma = 1;
            int myNode = grank / 16;
            int myLocalID = grank % 16;
            int nodeRow = (myNode % (nrow / 4)) * 4;
            int nodeCol = (myNode / (nrow / 4)) * 4;
            switch(myLocalID)
            {
                case 0:
                    myrow= 0;
                    mycol= 0;
                    break;
                case 1:
                    myrow= 1;
                    mycol= 1;
                    break;
                case 2:
                    myrow= 2;
                    mycol= 2;
                    break;
                case 3:
                    myrow= 3;
                    mycol= 3;
                    break;
                case 4:
                    myrow= 0;
                    mycol= 1;
                    break;
                case 5:
                    myrow= 1;
                    mycol= 2;
                    break;
                case 6:
                    myrow= 2;
                    mycol= 3;
                    break;
                case 7:
                    myrow= 3;
                    mycol= 0;
                    break;
                case 8:
                    myrow= 0;
                    mycol= 2;
                    break;
                case 9:
                    myrow= 1;
                    mycol= 3;
                    break;
                case 10:
                    myrow= 2;
                    mycol= 0;
                    break;
                case 11:
                    myrow= 3;
                    mycol= 1;
                    break;
                case 12:
                    myrow= 0;
                    mycol= 3;
                    break;
                case 13:
                    myrow= 1;
                    mycol= 0;
                    break;
                case 14:
                    myrow= 2;
                    mycol= 1;
                    break;
                case 15:
                    myrow= 3;
                    mycol= 2;
                    break;
	    }
        }
	else if ( numasize == 10 )
	{
            PrintMsg("\tNode Grid - Reorder 2x4C");
	    idnuma = 0;
            nnuma = 1;
	    int myNode    = grank / 8;
            int myLocalID = grank % 8;
            int nodeRow = (myNode % (nrow / 2)) * 2;
            int nodeCol = (myNode / (nrow / 2)) * 4;
	    int reorderID;
	    for ( int ii = 0; ii < 8; ii++ )
	    {
	       if ( myLocalID == reorder[ ii ] )
	       {
		  reorderID = ii;
		  break;
	       }
	    }
	    myrow       = reorderID % 2;
	    mycol       = reorderID / 2;
	    myrow      += nodeRow;
            mycol      += nodeCol;
	}
	else {
#if 0	//Future possible usage, dont delete
            assert(size % numasize == 0);
            idnuma = rank % numasize;
            nnuma = numasize;
            switch (map) {
            case NumaMap::ROWCONT: {
                assert(nrow % nnuma == 0);
                myrow = rank % nrow;
                mycol = rank / nrow;
            } break;

            case NumaMap::COLCONT: {
                assert((size / nrow) % nnuma == 0);
                int t = rank / nnuma;
                myrow = t % nrow;
                mycol = (t / nrow) * nnuma + idnuma;
            } break;

            case NumaMap::ROWDIST: {
                assert(nrow % nnuma == 0);
                int rs = nrow / nnuma;
                int t = rank / nnuma;
                myrow = (t % rs) + idnuma * rs;
                mycol = rank / nrow;
            } break;

            case NumaMap::COLDIST: {
                assert((size / nrow) % nnuma == 0);
                int t = rank / nnuma + (size / nnuma) * idnuma;
                myrow = t % nrow;
                mycol = t / nrow;
            } break;

            case NumaMap::CONT2D: {
                assert(nnuma % 2 == 0); // others are not implemented yet
                assert(nrow % 2 == 0);
                assert((size / nrow) % (nnuma / 2) == 0);
                int t = rank / nnuma;
                int grow = t % (nrow / 2);
                int gcol = t / (nrow / 2);
                myrow = grow * 2 + idnuma % 2;
                mycol = gcol * (nnuma / 2) + idnuma / 2;
            } break;
            default:
                std::abort();
            }
#endif
        }

        MPI_Comm_split(comm, mycol, myrow, &vcomm);
        MPI_Comm_split(comm, myrow, mycol, &hcomm);
        this->row = myrow;
        this->col = mycol;
        this->nrow = nrow;
        this->ncol = ncol;
    }
    ~Grid() {
        MPI_Comm_free(&vcomm);
        MPI_Comm_free(&hcomm);
    }
};

template <typename T> struct Mpi_type_wrappe {};

template <> struct Mpi_type_wrappe<fp16> {
    operator MPI_Datatype() { return MPI_SHORT; }
};

template <> struct Mpi_type_wrappe<__half> {
    operator MPI_Datatype() { return MPI_SHORT; }
};

template <> struct Mpi_type_wrappe<float> {
    operator MPI_Datatype() { return MPI_FLOAT; }
};

template <> struct Mpi_type_wrappe<double> {
    operator MPI_Datatype() { return MPI_DOUBLE; }
};

template <typename F> struct T2MPI { static Mpi_type_wrappe<F> type; };

template <typename F> Mpi_type_wrappe<F> T2MPI<F>::type;

#endif
