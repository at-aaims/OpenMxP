# OpenMxP: Open-Source Mixed Precision Computing
- [OpenMxP: Open-Source Mixed Precision Computing](#openmxp-open-source-mixed-precision-computing)
  - [Build instructions ( Frontier/Crusher )](#build-instructions--frontiercrusher-)
  - [Running instructions ( Frontier/Crusher )](#running-instructions--frontiercrusher-)
    - [Comments](#comments)
  - [Build instruction (Summit)](#build-instruction-summit)
  - [Tuning Parameters](#tuning-parameters)
  - [Citation](#citation)
    - [Code Repo](#code-repo)
    - [SC22 Paper](#sc22-paper)
  - [Developers](#developers)
  - [Contributors](#contributors)




## Build instructions ( Frontier/Crusher )

```sh
cd OpenMxP
mkdir build
cd build
cp ../doc/build_OpenMxP_frontier.sh .
```
That script runs `../doc/load_modules_frontier.sh` which may need to be modified for different rocm versions.

```sh
./build_OpenMxP_frontier.sh
```
You should now have a OpenMxP.x86_64 binary.


## Running instructions ( Frontier/Crusher )

```sh
mkdir jobs
cd jobs
cp ../doc/OpenMxP.slurm
```
Change this script to meet your needs.

```sh
sbatch OpenMxP.slurm
```
The output from crusher is in `doc/crusher_example_32x32.out`.

Constraints are PxQ=#GPUs, PxLN=QxLN, B need to be divisiable by TILE size.
Must have at least 3 omp threads.

### Comments

OpenMxP is designed to run at scale.   When it is run at a few number of nodes,
the performance will suffer due to the Iterative Refinement (IR).
At larger scales, this time becomes insignificant in the run.

There are requirements between N, B, PxQ ( process grid ), and the local grid.
Some are enforced while others are not.  It is usually easier to run square
( PxQ ) that are multiples of 8.  The best B tends to be 2560 and the best
performing local N (LN) tends to be 125440.   So this will give a N of P*LN.


## Build instruction (Summit)

```sh
module load cmake gcc/7.4.0 cuda/11.2.0 openblas
git clone git@github.com:at-aaims/OpenMxP
cd hpl-ai && make build && cd build 
```

For release build:

```sh
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

The default optimization level is `-O3`.

For debug build:

```sh
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```
This will have debug info built in.


## Tuning Parameters
```
-log 1 ( print rank 0 messages )

-solv 0 ( use blas )
      1 ( use solver ) # default (fastest)

-comm 0 ( use ibcast )
      1 ( use bcast )    
      2 ( use 1ring )       # default
      3 ( use 1ringM )
      4 ( use 2ringM )

--numa 0 (Global Column Major)   # default
       1 ( Node Grid - 2x3C )    
       2 ( Node Grid - 3x2C )       
       3 ( Global Row Major )    
       4 ( Node Grid - 2x4R )
       5 ( Node Grid - 2x4C )

-alt 0 (TRSM L/U panel)
     1 (TRSM for Diagonal inverse)
     2 (TRTRI for Diagonal inverse)


-sync ( enable cuda device sync after sgemm - currently only for bcast )
```

## Citation


### Code Repo

```
@misc{doecode_102701,
  title = {OpenMxP - Open Source Mixed Precision Computing},
  author = {Lu, Hao and Matheson, Michael and Wang, Feiyi and Joubert, Wayne and Ellis, Austin and Oles, Vladyslav},
  doi = {10.11578/dc.20230315.3},
  url = {https://doi.org/10.11578/dc.20230315.3},
  howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20230315.3}},
  year = {2023},
  month = {mar}
}
```

### SC22 Paper 

```bibtex
@inproceedings{10.5555/3571885.3571988,
    author = {Lu, Hao and Matheson, Michael and Oles, Vladyslav and Ellis, Austin and Joubert, Wayne and Wang, Feiyi},
    title = {Climbing the Summit and Pushing the Frontier of Mixed Precision Benchmarks at Extreme Scale},
    year = {2022},
    isbn = {9784665454445},
    publisher = {IEEE Press},
    booktitle = {Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis},
    articleno = {78},
    numpages = {15},
    doi = {10.1109/SC41404.2022.00083},
    keywords = {linear algebra, parallel programming, exascale computing, high performance computing},
    location = {Dallas, Texas},
    series = {SC '22}
}
```


## Developers
* Hao Lu, <luh1@ornl.gov>
* Michael Matheson, <mathesonma@ornl.gov> (Main Contact)
* Wayne Joubert, <joubert@ornl.gov>
* Feiyi Wang, <fwang2@ornl.gov>
* Vlad Oles, <olesv@ornl.gov> (Past)
* Austin Ellis, <ellisja@ornl.gov> (Past)

## Contributors
* Jakub Kurzak <Jakub.Kurzak@amd.com>
* Alessandro Fanfarillo <Alessandro.Famfarillo@amd.com>
* Noel Chalmers <Noel.Chalmers@amd.com>
* Nicolas Malaya Nicholas <Nicolas.Malaya@amd.com>
* Pak Niu Lui <Pak.Lui@amd.com>
* Hui Liu <Hui.Lui1@amd.com>
* Mazda Sabony <Mazada.Sabony@amd.com>
