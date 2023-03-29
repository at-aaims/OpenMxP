
#ifndef __HPLAI_DEVICE_MACROS__
#define __HPLAI_DEVICE_MACROS__

// Temp
/*
#if 0
    #define CUDA_OLCF_PLATFORM 1

#elif
    #define ROCM_OLCF_PLATFORM 1

#else

#endif
*/

#define ROCM_OLCF_PLATFORM 1

// SUMMIT
#ifdef CUDA_OLCF_PLATFORM

#include "cuda_device_macros.h"


// FRONTIER (& SPOCK)
#elif defined(ROCM_OLCF_PLATFORM)

#include "rocm_device_macros.h"

#endif


#endif // __HPLAI_DEVICE_MACROS__
