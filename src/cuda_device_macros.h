
#ifndef __HPLAI_CUDA_DEVICE_MACROS__
#define __HPLAI_CUDA_DEVICE_MACROS__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

// *** BASIC CUDA MACROS ***
// Kernel Macros
#define GPU_BLOCKIDX_X \
    blockIdx.x

#define GPU_BLOCKIDX_Y \
    blockIdx.y

#define GPU_BLOCKIDX_Z \
    blockIdx.z


#define GPU_THREADIDX_X \
    threadIdx.x

#define GPU_THREADIDX_Y \
    threadIdx.y

#define GPU_THREADIDX_Z \
    threadIdx.z


#define GPU_BLOCKDIM_X \
    blockDim.x

#define GPU_BLOCKDIM_Y \
    blockDim.y

#define GPU_BLOCKDIM_Z \
    blockDim.z


#define GPU_GRIDDIM_X \
    gridDim.x

#define GPU_GRIDDIM_Y \
    gridDim.y

#define GPU_GRIDDIM_Z \
    gridDim.z


// Types
#define GPU_ERROR_T \
    cudaError_t

#define GPU_STREAM_T \
    cudaStream_t


// Enums
#define GPU_SUCCESS \
    cudaSuccess

#define GPU_STREAM_NON_BLOCKING \
    cudaStreamNonBlocking

#define GPU_R_16F \
    CUDA_R_16F

#define GPU_R_32F \
    CUDA_R_32F

#define GPU_MEMCPY_DEVICE_TO_HOST \
    cudaMemcpyDeviceToHost

#define GPU_MEMCPY_HOST_TO_DEVICE \
    cudaMemcpyHostToDevice

// Kernels
#define GPU_DEVICE_RESET() \
    cudaDeviceReset()

#define GPU_SET_DEVICE(deviceID) \
    cudaSetDevice(deviceID)

#define GPU_DEVICE_SYNCHRONIZE() \
    cudaDeviceSynchronize()

#define GPU_FREE(memPointer) \
    cudaFree(memPointer)

#define GPU_FREE_HOST(memPointer) \
    cudaFreeHost(memPointer)

#define GPU_GET_ERROR_STRING(cudaError) \
    cudaGetErrorString(cudaError)

#define GPU_GET_LAST_ERROR() \
    cudaGetLastError()

#define GPU_MALLOC(memAddress, numBytes) \
    cudaMalloc(memAddress, numBytes)

#define GPU_MALLOC_HOST(memAddress, numBytes) \
    cudaMallocHost(memAddress, numBytes)

#define GPU_MALLOC_MANAGED(memAddress, numBytes) \
    cudaMallocManaged(memAddress, numBytes)

#define GPU_MEMCPY(memPointer_to, memPointer_from, numBytes, directionEnum) \
    cudaMemcpy(memPointer_to, memPointer_from, numBytes, directionEnum)

#define GPU_MEMCPY_2D(memPointer_to, pitchBytes_to, memPointer_from, pitchBytes_from, numBytes_W, numBytes_H, directionEnum) \
    cudaMemcpy2D(memPointer_to, pitchBytes_to, memPointer_from, pitchBytes_from, numBytes_W, numBytes_H, directionEnum)

#define GPU_MEMCPY_DEVICE_TO_DEVICE \
    cudaMemcpyDeviceToDevice

#define GPU_MEM_GET_INFO(freeMem, totalMem) \
    cudaMemGetInfo(freeMem, totalMem)

#define GPU_STREAM_CREATE_WITH_FLAGS(cudaStream, streamTypeEnum) \
    cudaStreamCreateWithFlags(cudaStream, streamTypeEnum)




// *** CUBLAS MACROS ***
// Types
#define GPUBLAS_HANDLE_T \
    cublasHandle_t

#define GPUBLAS_STATUS_T \
    cublasStatus_t

// Enums
#define GPUBLAS_STATUS_SUCCESS \
    CUBLAS_STATUS_SUCCESS

#define GPUBLAS_STATUS_NOT_INITIALIZED \
    CUBLAS_STATUS_NOT_INITIALIZED

#define GPUBLAS_STATUS_ALLOC_FAILED \
    CUBLAS_STATUS_ALLOC_FAILED

#define GPUBLAS_STATUS_INVALID_VALUE \
    CUBLAS_STATUS_INVALID_VALUE

#define GPUBLAS_STATUS_ARCH_MISMATCH \
    CUBLAS_STATUS_ARCH_MISMATCH

#define GPUBLAS_STATUS_MAPPING_ERROR \
    CUBLAS_STATUS_MAPPING_ERROR

#define GPUBLAS_STATUS_EXECUTION_FAILED \
    CUBLAS_STATUS_EXECUTION_FAILED

#define GPUBLAS_STATUS_INTERNAL_ERROR \
    CUBLAS_STATUS_INTERNAL_ERROR

#define GPUBLAS_OP_N \
    CUBLAS_OP_N

#define GPUBLAS_OP_T \
    CUBLAS_OP_T

#define GPUBLAS_SIDE_RIGHT \
    CUBLAS_SIDE_RIGHT

#define GPUBLAS_SIDE_LEFT \
    CUBLAS_SIDE_LEFT

#define GPUBLAS_FILL_MODE_UPPER \
    CUBLAS_FILL_MODE_UPPER

#define GPUBLAS_FILL_MODE_LOWER \
    CUBLAS_FILL_MODE_LOWER

#define GPUBLAS_DIAG_UNIT \
    CUBLAS_DIAG_UNIT

#define GPUBLAS_DIAG_NON_UNIT \
    CUBLAS_DIAG_NON_UNIT

// Kernels
#define GPUBLAS_CREATE(cublasHandle) \
    cublasCreate_v2(cublasHandle)

#define GPUBLAS_SET_STREAM(cublasHandle, cudaStream) \
    cublasSetStream(cublasHandle, cudaStream)

#define GPUBLAS_SGETRF_BATCHED(cublasHandle, N_dim, memPointer_A, lda, memPointer_Pivot, memPointer_Info, batchSize) \
    cublasSgetrfBatched(cublasHandle, N_dim, memPointer_A, lda, memPointer_Pivot, memPointer_Info, batchSize)

#define GPUBLAS_STRSM(cublasHandle, cuSide, cuFill, cuOp, cuDiag, M_dim, N_dim, alpha, memPointer_A, lda, memPointer_B, ldb) \
    cublasStrsm(cublasHandle, cuSide, cuFill, cuOp, cuDiag, M_dim, N_dim, alpha, memPointer_A, lda, memPointer_B, ldb)

// Non-Simple Kernels
//#define GPUBLAS_GET_ERROR_STRING(cublasStatus) \
//    cublasGetErrorString(cublasStatus)

#define GPUBLAS_SGEMM_EX(cublasHandle, cuOp_A, cuOp_B, M_dim, N_dim, k_dim, alpha, memPointer_A, datatype_A, lda, memPointer_B, datatype_B, ldb, beta, memPointer_C, datatype_C, ldc) \
    cublasSgemmEx(cublasHandle, cuOp_A, cuOp_B, M_dim, N_dim, k_dim, alpha, memPointer_A, datatype_A, lda, memPointer_B, datatype_B, ldb, beta, memPointer_C, datatype_C, ldc)



// *** CUSOLVER MACROS ***
// Types
#define GPUSOLVER_HANDLE_T \
    cusolverDnHandle_t

#define GPUSOLVER_STATUS_T \
    cusolverStatus_t


// Enums
#define GPUSOLVER_STATUS_SUCCESS \
    CUSOLVER_STATUS_SUCCESS


// Kernels
#define GPUSOLVER_CREATE(cusolverDnHandle) \
    cusolverDnCreate(cusolverDnHandle)

#define GPUSOLVER_SGETRF(cusolverDnHandle, M_dim, N_dim, memPointer, lda, memAddress_Buffer, memPointer_Pivot, memPointer_Info) \
    cusolverDnSgetrf(cusolverDnHandle, M_dim, N_dim, memPointer, lda, memAddress_Buffer, memPointer_Pivot, memPointer_Info)

//#define GPUSOLVER_SGETRF_BUFFERSIZE(cusolverDnHandle, M_dim, N_dim, memPointer, lda, memAddress_Buffer, memPointer_Pivot, memPointer_Info) \
//    cusolverDnSgetrf_bufferSize(cusolverDnHandle, M_dim, N_dim, memPointer, lda, memAddress_Buffer, memPointer_Pivot, memPointer_Info)

#define GPUSOLVER_SGETRF_BUFFERSIZE(cusolverDnHandle, M_dim, N_dim, memPointer, lda, memAddress_Buffer) \
    cusolverDnSgetrf_bufferSize(cusolverDnHandle, M_dim, N_dim, memPointer, lda, memAddress_Buffer)

#define GPUSOLVER_SET_STREAM(cusolverDnHandle, cudaStream) \
    cusolverDnSetStream(cusolverDnHandle, cudaStream)

// *** CURAND MACROS ***
// Types
#define GPURAND_GENERATOR_T \
    curandGenerator_t


// Enums
#define GPURAND_RNG_PSEUDO_DEFAULT \
    CURAND_RNG_PSEUDO_DEFAULT


// Kernels
#define GPURAND_CREATE_GENERATOR(curandGenerator, curandRngType) \
    curandCreateGenerator(curandGenerator, curandRngType)

#define GPURAND_DESTROY_GENERATOR(curandGenerator) \
    curandDestroyGenerator(curandGenerator)

#define GPURAND_GENERATE_UNIFORM(curandGenerator, memPointer, numBytes) \
    curandGenerateUniform(curandGenerator, memPointer, numBytes)

#define GPURAND_GENERATE_UNIFORM_DOUBLE(curandGenerator, memPointer, numBytes) \
    curandGenerateUniformDouble(curandGenerator, memPointer, numBytes)

#define GPURAND_SET_PSEUDO_RANDOM_GENERATOR_SEED(curandGenerator, seed) \
    curandSetPseudoRandomGeneratorSeed(curandGenerator, seed)



#endif // __HPLAI_CUDA_DEVICE_MACROS__
