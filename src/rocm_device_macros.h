
#ifndef __HPLAI_HIP_DEVICE_MACROS__
#define __HPLAI_HIP_DEVICE_MACROS__

//#include <hip/hip.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hip/hip_fp16.h>
#include <hipblas.h>
#include <rocblas.h>
#include <rocsolver.h>
//#include <hiprand.h>

#define CHECK_BIT(var,pos) ( (var>>pos) & 1 ) 

// #define GPU_EVENT

/*#ifdef GPU_EVENT
#define GPU_EVENTCREATE(a)       hipEventCreate(a)
#define GPU_EVENTRECORD(a,b)     hipEventRecord(a,b)
#define GPU_EVENTSYNC(a)         hipEventSynchronize(a)
#define GPU_EVENTELAPSED(a,b,c)  hipEventElapsedTime(a,b,c)
#define GPU_EVENTDESTROY(a)      hipEventDestroy(a)
#else
#define GPU_EVENTCREATE(a)       { }
#define GPU_EVENTRECORD(a,b)     { }
#define GPU_EVENTSYNC(a)         { }
#define GPU_EVENTELAPSED(a,b,c)  { }
#define GPU_EVENTDESTROY(a)      { }
#endif*/

//#define SKIP       
#ifdef SKIP
#define checkGPUblas(a) { }
#endif

#define GPU_EVENTCREATE(a)       hipEventCreate(a)
#define GPU_EVENTRECORD(a,b)     hipEventRecord(a,b)
#define GPU_EVENTSYNC(a)         hipEventSynchronize(a)
#define GPU_EVENTELAPSED(a,b,c)  hipEventElapsedTime(a,b,c)
#define GPU_EVENTDESTROY(a)      hipEventDestroy(a)
#define GPU_EVENT_T hipEvent_t


// *** BASIC HIP MACROS ***
// Kernel Macros
#define GPU_BLOCKIDX_X \
    hipBlockIdx_x

#define GPU_BLOCKIDX_Y \
    hipBlockIdx_y

#define GPU_BLOCKIDX_Z \
    hipBlockIdx_z


#define GPU_THREADIDX_X \
    hipThreadIdx_x

#define GPU_THREADIDX_Y \
    hipThreadIdx_y

#define GPU_THREADIDX_Z \
    hipThreadIdx_z


#define GPU_BLOCKDIM_X \
    hipBlockDim_x

#define GPU_BLOCKDIM_Y \
    hipBlockDim_y

#define GPU_BLOCKDIM_Z \
    hipBlockDim_z


#define GPU_GRIDDIM_X \
    hipGridDim_x

#define GPU_GRIDDIM_Y \
    hipGridDim_y

#define GPU_GRIDDIM_Z \
    hipGridDim_z


// Types
#define GPU_ERROR_T \
    hipError_t

#define GPU_STREAM_T \
    hipStream_t


// Enums
#define GPU_SUCCESS \
    hipSuccess

#define GPU_STREAM_NON_BLOCKING \
    hipStreamNonBlocking

#ifdef DOUBLE 
#define GPU_R_16F \
    rocblas_datatype_f32_r
//  rocblas_datatype_f64_r

#define GPU_R_32F \
    rocblas_datatype_f32_r
//  rocblas_datatype_f64_r

#else

#define GPU_R_16F \
    rocblas_datatype_f16_r

#define GPU_R_32F \
    rocblas_datatype_f32_r
#endif

#define GPU_R_64F \
    rocblas_datatype_f64_r

#define GPU_MEMCPY_DEVICE_TO_HOST \
    hipMemcpyDeviceToHost

#define GPU_MEMCPY_HOST_TO_DEVICE \
    hipMemcpyHostToDevice

// Kernels
#define GPU_DEVICE_RESET() \
    hipDeviceReset()

#define GPU_SET_DEVICE(deviceID) \
    hipSetDevice(deviceID)

#define GPU_DEVICE_SYNCHRONIZE() \
    hipDeviceSynchronize()

#define GPU_THREAD_SYNCHRONIZE(threadID) \
    hipStreamSynchronize(threadID); 

#define GPU_FREE(memPointer) \
    hipFree(memPointer)

#define GPU_FREE_HOST(memPointer) \
    hipHostFree(memPointer)

#define GPU_GET_ERROR_STRING(hipError) \
    hipGetErrorString(hipError)

#define GPU_GET_LAST_ERROR() \
    hipGetLastError()

#define GPU_MALLOC(memAddress, numBytes) \
    hipMalloc(memAddress, numBytes)

#define GPU_MALLOC_HOST(memAddress, numBytes) \
    hipHostMalloc(memAddress, numBytes)

#define GPU_MALLOC_MANAGED(memAddress, numBytes) \
    hipMallocManaged(memAddress, numBytes)

#define GPU_MEMCPY(memPointer_to, memPointer_from, numBytes, directionEnum) \
    hipMemcpy(memPointer_to, memPointer_from, numBytes, directionEnum)

#define GPU_MEMCPY_2D(memPointer_to, pitchBytes_to, memPointer_from, pitchBytes_from, numBytes_W, numBytes_H, directionEnum) \
    hipMemcpy2D(memPointer_to, pitchBytes_to, memPointer_from, pitchBytes_from, numBytes_W, numBytes_H, directionEnum)

#define GPU_MEMCPY_DEVICE_TO_DEVICE \
    hipMemcpyDeviceToDevice

#define GPU_MEM_GET_INFO(freeMem, totalMem) \
    hipMemGetInfo(freeMem, totalMem)

#define GPU_STREAM_CREATE_WITH_FLAGS(hipStream, streamTypeEnum) \
    hipStreamCreateWithFlags(hipStream, streamTypeEnum)

#define GPU_DAXPY(blasHandle, size, alpha, A, Ainc, B, Binc) \
    rocblas_daxpy(blasHandle, size, alpha, A, Ainc, B, Binc)

#define GPU_DTRSV(blasHandle, rocFill, rocOp, rocDiag, M_dim, memPointer_A, lda, memPointer_B, Binc ) \
    rocblas_dtrsv(blasHandle, rocFill, rocOp, rocDiag, M_dim, memPointer_A, lda, memPointer_B, Binc)



// *** HIPBLAS MACROS ***
// Types
#define GPUBLAS_HANDLE_T \
    rocblas_handle

#define GPUBLAS_STATUS_T \
    rocblas_status

// Enums
#define GPUBLAS_STATUS_SUCCESS \
    rocblas_status_success

#define GPUBLAS_STATUS_NOT_INITIALIZED \
    rocblas_status_invalid_handle

#define GPUBLAS_STATUS_ALLOC_FAILED \
    rocblas_status_memory_error

#define GPUBLAS_STATUS_INVALID_VALUE \
    rocblas_status_invalid_value

// No good option for rocblas_status
#define GPUBLAS_STATUS_ARCH_MISMATCH \
    rocblas_status_perf_degraded

#define GPUBLAS_STATUS_MAPPING_ERROR \
    rocblas_status_invalid_pointer

#define GPUBLAS_STATUS_EXECUTION_FAILED \
    rocblas_status_invalid_size

#define GPUBLAS_STATUS_INTERNAL_ERROR \
    rocblas_status_internal_error

#define GPUBLAS_OP_N \
    rocblas_operation_none

#define GPUBLAS_OP_T \
    rocblas_operation_transpose

#define GPUBLAS_SIDE_RIGHT \
    rocblas_side_right

#define GPUBLAS_SIDE_LEFT \
    rocblas_side_left

#define GPUBLAS_FILL_MODE_UPPER \
    rocblas_fill_upper

#define GPUBLAS_FILL_MODE_LOWER \
    rocblas_fill_lower

#define GPUBLAS_DIAG_UNIT \
    rocblas_diagonal_unit

#define GPUBLAS_DIAG_NON_UNIT \
    rocblas_diagonal_non_unit

// Kernels
#define GPUBLAS_CREATE(rocblasHandle) \
    rocblas_create_handle(rocblasHandle)

#define GPUBLAS_SET_STREAM(rocblasHandle, hipStream) \
    rocblas_set_stream(rocblasHandle, hipStream)

//#define GPUBLAS_SGETRF_BATCHED(hipblasHandle, N_dim, memPointer_A, lda, memPointer_Pivot, memPointer_Info, batchSize) \
//    hipblasSgetrfBatched(hipblasHandle, N_dim, memPointer_A, lda, memPointer_Pivot, memPointer_Info, batchSize)

#define GPUBLAS_STRSM(rocblasHandle, rocSide, rocFill, rocOp, rocDiag, M_dim, N_dim, alpha, memPointer_A, lda, memPointer_B, ldb) \
    rocblas_strsm(rocblasHandle, rocSide, rocFill, rocOp, rocDiag, M_dim, N_dim, alpha, memPointer_A, lda, memPointer_B, ldb)
//dbl  rocblas_dtrsm(rocblasHandle, rocSide, rocFill, rocOp, rocDiag, M_dim, N_dim, alpha, memPointer_A, lda, memPointer_B, ldb)

#define GPU_daxpy(handle, n, alpha, x, incx, y, incy) \
    rocblas_daxpy(handle, n, alpha, x, incx, y, incy)

#define GPU_dscal(handle, n, alpha, x, incx) \
    rocblas_dscal(handle, n, alpha, x, incx)

#define GPU_setValue(x, value, size) \
    hipMemset(x, value, size)


// Non-Simple Kernels
//#define GPUBLAS_GET_ERROR_STRING(hipblasStatus) \
//    hipblasGetErrorString(hipblasStatus)


#define GPUBLAS_SGEMM_EX(hipblasHandle, hipOp_A, hipOp_B, M_dim, N_dim, k_dim, alpha, memPointer_A, datatype_A, lda, memPointer_B, datatype_B, ldb, beta, memPointer_C, datatype_C, ldc) \
    rocblas_gemm_ex(hipblasHandle, hipOp_A, hipOp_B, M_dim, N_dim, k_dim, alpha, memPointer_A, datatype_A, lda, memPointer_B, datatype_B, ldb, beta, memPointer_C, datatype_C, ldc, memPointer_C, datatype_C, ldc, datatype_C, rocblas_gemm_algo_standard, 0, 0)



// *** HIPSOLVER MACROS ***
// Types
#define GPUSOLVER_HANDLE_T \
    rocblas_handle

#define GPUSOLVER_STATUS_T \
    rocblas_status


// Enums
#define GPUSOLVER_STATUS_SUCCESS \
    rocblas_status_success


// Kernels
#define GPUSOLVER_CREATE(rocsolverHandle) \
    rocblas_create_handle(rocsolverHandle)

#define GPUSOLVER_SGETRF(rocsolverHandle, M_dim, N_dim, memPointer, lda, memAddress_Buffer, memPointer_Pivot, memPointer_Info) \
      rocsolver_sgetrf_npvt(rocsolverHandle, M_dim, N_dim, memPointer, lda, memPointer_Info)
//dbl rocsolver_dgetrf_npvt(rocsolverHandle, M_dim, N_dim, memPointer, lda, memPointer_Info)

//#define GPUSOLVER_SGETRF_BUFFERSIZE(hipsolverDnHandle, M_dim, N_dim, memPointer, lda, memAddress_Buffer, memPointer_Pivot, memPointer_Info) \
//    hipsolverDnSgetrf_bufferSize(hipsolverDnHandle, M_dim, N_dim, memPointer, lda, memAddress_Buffer, memPointer_Pivot, memPointer_Info)

//#define GPUSOLVER_SGETRF_BUFFERSIZE(hipsolverDnHandle, M_dim, N_dim, memPointer, lda, memAddress_Buffer) \
//    hipsolverDnSgetrf_bufferSize(hipsolverDnHandle, M_dim, N_dim, memPointer, lda, memAddress_Buffer)

#define GPUSOLVER_SET_STREAM(rocsolverHandle, hipStream) \
    rocblas_set_stream(rocsolverHandle, hipStream)



#if 0
// *** HIPRAND MACROS ***
// Types
#define GPURAND_GENERATOR_T \
    hiprandGenerator_t


// Enums
#define GPURAND_RNG_PSEUDO_DEFAULT \
    HIPRAND_RNG_PSEUDO_DEFAULT


// Kernels
#define GPURAND_CREATE_GENERATOR(hiprandGenerator, hiprandRngType) \
    hiprandCreateGenerator(hiprandGenerator, hiprandRngType)

#define GPURAND_DESTROY_GENERATOR(hiprandGenerator) \
    hiprandDestroyGenerator(hiprandGenerator)

#define GPURAND_GENERATE_UNIFORM(hiprandGenerator, memPointer, numBytes) \
    hiprandGenerateUniform(hiprandGenerator, memPointer, numBytes)

#define GPURAND_GENERATE_UNIFORM_DOUBLE(hiprandGenerator, memPointer, numBytes) \
    hiprandGenerateUniformDouble(hiprandGenerator, memPointer, numBytes)

#define GPURAND_SET_PSEUDO_RANDOM_GENERATOR_SEED(hiprandGenerator, seed) \
    hiprandSetPseudoRandomGeneratorSeed(hiprandGenerator, seed)

#endif

#endif // __HPLAI_HIP_DEVICE_MACROS__
