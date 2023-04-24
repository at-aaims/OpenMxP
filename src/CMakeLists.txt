EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message( STATUS "Building HPL-AI for Architecture: ${ARCHITECTURE}" )

# FRONTIER / SPOCK language and definitions
if( ${ARCHITECTURE} STREQUAL "x86_64" )

    # Requires atleast `module load cmake/3.21.2-dev``
    cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
    set(ACCEL_COMPILER HIP)
    add_compile_definitions(ROCM_OLCF_PLATFORM)

# SUMMIT language and definitions
elseif( ${ARCHITECTURE} STREQUAL "ppc64le")
    cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
    set(ACCEL_COMPILER CUDA)
    add_compile_definitions(CUDA_OLCF_PLATFORM)

endif()

# enable CUDA or HIP
enable_language(${ACCEL_COMPILER})
project(openMxP LANGUAGES CXX)

# FRONTIER / SPOCK packages
if( ${ARCHITECTURE} STREQUAL "x86_64" )
    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "PATH to which HIP has been installed")
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "PATH to which ROCM has been installed")
    set(CRAY_MPICH_DIR $ENV{CRAY_MPICH_DIR} CACHE PATH "PATH to Cray MPICH ROOT")

    set( GPU_TARGETS "gfx90a" CACHE STRING "GPU TARGETS" )
    set( AMPGPU_TARGETS "gfx90a" CACHE STRING "GPU TARGETS" )
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    find_package(HIP REQUIRED)
    #    find_package(MPI REQUIRED)
    find_package(rocblas REQUIRED)
    find_package(rocsolver REQUIRED)
    find_package(rocrand REQUIRED)
    find_package(hiprand REQUIRED)
    find_package(OpenMP REQUIRED)

    # For GPU Direct on AMD must export these variables:
    # export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
    # export MPICH_GPU_SUPPORT_ENABLED=1
    find_library(GTL mpi_gtl_hsa /opt/cray/pe/mpich/default/gtl/lib)

    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
        message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm-4.x.x/hip or the variable HIP_PATH is set to point to the right location.")
    endif()

    #    find_library(GTL
    #        NAMES mpi_gtl_hsa
    #        PATHS ${CRAY_MPICH_ROOTDIR}/gtl/lib
    #        REQUIRED)
    #file(GLOB HIP_CLANGRT_LIB_SEARCH_PATHS "${CMAKE_HIP_COMPILER}/../lib/clang/*/lib/*")
#   file(GLOB HIP_CLANGRT_LIB_SEARCH_PATHS "/opt/rocm-5.0.2/llvm/lib/clang/14.0.0/lib/linux/*" )
    find_library(CLANGRT_BUILTINS clang_rt.builtins-x86_64 /opt/rocm-5.0.2/llvm/lib/clang/14.0.0/lib/linux )

# SUMMIT packages and compiling
elseif( ${ARCHITECTURE} STREQUAL "ppc64le")
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70)
    endif()

    # ensure atomicAdd() is implemented for FP64
    if(CMAKE_CUDA_ARCHITECTURES LESS 60)
        set(CMAKE_CUDA_ARCHITECTURES 60)
    endif()

    find_package(CUDA 11.4 REQUIRED)
    find_package(MPI)

    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)

endif()


# NON-ACCELERATOR SPECIFIC

# c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# gcc version check
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.4.0")
         message(FATAL_ERROR "Insufficient gcc version, require 6.4.0+")
    endif()
endif()


# Get the current working branch
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get the latest commit hash
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)


# disable in source build
if ( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR}  )
        file(REMOVE test.txt)
        message( FATAL_ERROR "In-source builds not allowed. Please make a new
        directory (called a build directory) and run CMake from there. You may
        need to remove CMakeCache.txt."  )
endif()


# add other sources
#add_subdirectory(externals)

set(HPL_AI_SRCS
      main.cpp
      higham_mat_impl.cpp
      otf_gemv.cpp
      sgetrf_nopiv.cpp
    )

add_definitions(-DEXTERNAL_CONV -DOTF_GEMV_OPTIMIZED -DNO_WARN_X86_INTRINSICS)

# FRONTIER / SPOCK build and linking
if( ${ARCHITECTURE} STREQUAL "x86_64" )

    set(GPU_SRCS
        gpu_init_kernels.cpp
        fp16_gpu_kernels.cpp
    )

    set_source_files_properties(${HPL_AI_SRCS} ${GPU_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(OpenMxP ${HPL_AI_SRCS} ${GPU_SRCS})

    set(CRAY_LIBSCI_PREFIX_DIR $ENV{CRAY_LIBSCI_PREFIX_DIR} CACHE PATH "Path to the libsci libraries")

    message(STATUS "Using libsci BLAS libraries in ${CRAY_LIBSCI_PREFIX_DIR}/lib ")
    target_include_directories(OpenMxP PUBLIC
        #        ${MPI_CXX_INCLUDE_DIRS}
        ${CRAY_MPICH_DIR}/include
        ${ROCM_PATH}/include
        ${ROCM_PATH}/hiprand/include
        ${ROCM_PATH}/rocrand/include
	${ROCM_PATH}/rocthrust/include
        )


    target_link_directories(OpenMxP PUBLIC
        ${ROCM_PATH}/lib
#       ${CRAY_LIBSCI_PREFIX_DIR}/lib
        ${CRAY_MPICH_DIR}/lib
        )

    target_link_libraries(OpenMxP
        #        ${MPI_CXX_LIBRARIES}
#CCE
        #mpi_cray
#GCC
        mpi
        ${GTL}
        amdhip64
        rocblas
        rocsolver
        hiprand
        hsa-runtime64
#       sci_cray_mp hugetlbfs
        ${OpenMP_CXX_LIBRARIES}
#       spdlog cli11
        )

    # Set HIP target architecture for SPOCK MI100 GPUs (May need to change for MI200s)
    #    set_property(TARGET driver PROPERTY HIP_ARCHITECTURES gfx908)
#   set_property(TARGET driver PROPERTY HIP_ARCHITECTURES gfx908)
    set_property(TARGET OpenMxP PROPERTY HIP_ARCHITECTURES gfx90a)
#   set( AMDGPU_TARGETS gfx90a )
    set_property(TARGET OpenMxP PROPERTY CMAKE_HIP_ARCHITECTURES gfx90a)
    target_compile_options(OpenMxP PUBLIC -std=c++14 -O3 )


# SUMMIT build and linking
elseif( ${ARCHITECTURE} STREQUAL "ppc64le")

    set(GPU_SRCS
        gpu_init_kernels.cu
        fp16_gpu_kernels.cu
    )

    set_source_files_properties(GPU_SRCS PROPERTIES LANGUAGE CUDA)

    add_executable(OpenMxP ${HPL_AI_SRCS} ${GPU_SRCS})

    target_include_directories(OpenMxP PUBLIC
        ${MPI_INCLUDE_PATH})

    target_link_libraries(OpenMxP
      ${MPI_LIBRARIES}
      ${CUDA_LIBRARIES}
      ${CUDA_CUBLAS_LIBRARIES}
      ${CUDA_cusolver_LIBRARY}
      ${BLAS_LIBRARIES}
      ${CUDA_curand_LIBRARY}
      #spdlog
      #cli11
      )

    target_link_options(OpenMxP PUBLIC "-fopenmp")
    target_link_options(OpenMxP PUBLIC "-mcpu=native")
    target_compile_options(OpenMxP PUBLIC  -W -fopenmp -mcpu=native -std=c++14 -O3)

endif()


target_compile_definitions(OpenMxP PRIVATE
    "-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")

target_compile_definitions(OpenMxP PRIVATE
    "-DGIT_BRANCH=\"${GIT_BRANCH}\"")

set_target_properties(OpenMxP
  PROPERTIES
    OUTPUT_NAME "OpenMxP."
    SUFFIX ${CMAKE_HOST_SYSTEM_PROCESSOR}
)
