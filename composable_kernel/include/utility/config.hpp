#ifndef CK_CONFIG_AMD_HPP
#define CK_CONFIG_AMD_HPP

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#endif

// "Constant" address space for kernel parameter
#define CONSTANT __attribute__((address_space(4)))

// GPU target
// should enable one and only one GPU target
#if !(defined(CK_AMD_GPU_GFX803) || defined(CK_AMD_GPU_GFX900) || defined(CK_AMD_GPU_GFX906) || \
      defined(CK_AMD_GPU_GFX908) || defined(CK_AMD_GPU_GFX90A) || defined(CK_AMD_GPU_GFX1030))
#error Need to define (only) one GPU target
#endif

// launch bounds
#define CK_USE_LAUNCH_BOUNDS 1

#ifdef CK_USE_LAUNCH_BOUNDS
#define CK_MAX_THREAD_PER_BLOCK 256
#define CK_MIN_BLOCK_PER_CU 2
#endif

// GPU-specific parameters
#if defined(CK_AMD_GPU_GFX803) || defined(CK_AMD_GPU_GFX900) || defined(CK_AMD_GPU_GFX906) || \
    defined(CK_AMD_GPU_GFX908) || defined(CK_AMD_GPU_GFX90A)
// buffer resourse
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x00020000
// wave size
#define CK_GPU_WAVE_SIZE 64
#elif defined(CK_AMD_GPU_GFX1030)
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31014000
#define CK_GPU_WAVE_SIZE 32
#endif

// FMA instruction
#if defined(CK_AMD_GPU_GFX803) || defined(CK_AMD_GPU_GFX900)
#define CK_USE_AMD_V_MAC_F32
#elif defined(CK_AMD_GPU_GFX906) || defined(CK_AMD_GPU_GFX908) || defined(CK_AMD_GPU_GFX90a) || \
    defined(CK_AMD_GPU_GFX1030)
#define CK_USE_AMD_V_FMAC_F32
#define CK_USE_AMD_V_DOT2_F32_F16
#define CK_USE_AMD_V_DOT4_I32_I8
#endif

// multi index
#define CK_USE_DYNAMICALLY_INDEXED_MULTI_INDEX 0

// AMD inline asm
#ifndef CK_USE_AMD_INLINE_ASM
#define CK_USE_AMD_INLINE_ASM 1
#endif

// AMD inner product (DLOP)
#ifndef CK_USE_AMD_INNER_PRODUCT_INLINE_ASM
#define CK_USE_AMD_INNER_PRODUCT_INLINE_ASM 1
#endif

// AMD buffer addressing
#ifndef CK_USE_AMD_BUFFER_ADDRESSING
#define CK_USE_AMD_BUFFER_ADDRESSING 1
#endif

// only gfx908 support native floating point atomic add
#ifndef CK_USE_AMD_BUFFER_ATOMIC_FADD
#define CK_USE_AMD_BUFFER_ATOMIC_FADD 0
#endif

// AMD XDLOPS
#ifndef CK_USE_AMD_XDLOPS
#define CK_USE_AMD_XDLOPS 0
#endif

// block synchronization only s_wait lgkmcnt(0), not vmcnt(0)
#ifndef CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#define CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM 1
#endif

// experimental implementation for buffer load/store/atomic
#ifndef CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
#define CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
#define CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK
#define CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK 1
#endif

// experimental implementation for in-regsiter sub-dword transpose
#ifndef CK_EXPERIMENTAL_USE_IN_REGISTER_SUB_DWORD_TRANSPOSE
#define CK_EXPERIMENTAL_USE_IN_REGISTER_SUB_DWORD_TRANSPOSE 1
#endif

// pass tensor descriptor by value or void*
#define CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE 0
#define CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER 0
#define CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR 1

// merge transformation use magic number division
#ifndef CK_EXPERIMENTAL_MERGE_USE_MAGIC_DIVISION
#define CK_EXPERIMENTAL_MERGE_USE_MAGIC_DIVISION 1
#endif

// use __builtin_memcpy instead of pointer cast to access a vector from pointer of scalar
#ifndef CK_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
#define CK_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS 0
#endif

// use __builtin_memcpy instead of union to do bit_cast
#ifndef CK_EXPERIMENTAL_USE_MEMCPY_FOR_BIT_CAST
#define CK_EXPERIMENTAL_USE_MEMCPY_FOR_BIT_CAST 1
#endif

// hack: have underlying assumption that need to be satsified, otherwise it's a bug
// hack for forcing register to keep idx_diff_low_const in SGPR. idx_diff_low_const must be
// thread-invariant, otherwise it's a bug
// TODO: separate index calculation into "compile-time", "global", "block", "wave", "thread"
#ifndef CK_HACK_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE
#define CK_HACK_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE 0
#endif

// workaround for compiler crash when compiling recursive lambda
#ifndef CK_WORKAROUND_SWDEV_275126
#define CK_WORKAROUND_SWDEV_275126 1
#endif

// workaround for compiler crash when using buffer load/store for i8
#ifndef CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE
#define CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE 1
#endif

// workaround for compiler gnerating inefficient ds_write instructions
#ifndef CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
#define CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE 1
#endif

// workaround for register spill due to compiler issue, when casting type between fp32 and fp16
#ifndef CK_WORKAROUND_SWDEV_XXXXXX_THREAD_WISE_COPY_V1R4_TYPE_CONVERT_ISSUE
#define CK_WORKAROUND_SWDEV_XXXXXX_THREAD_WISE_COPY_V1R4_TYPE_CONVERT_ISSUE 1
#endif

#ifndef CK_WORKAROUND_SWDEV_XXXXXX_THREAD_WISE_COPY_V1R5_TYPE_CONVERT_ISSUE
#define CK_WORKAROUND_SWDEV_XXXXXX_THREAD_WISE_COPY_V1R5_TYPE_CONVERT_ISSUE 1
#endif

#define CK_USE_DOT4_I32_I8 1

namespace ck {

enum InMemoryDataOperationEnum_t
{
    Set,
    AtomicAdd,
    Add
};

enum ActivTypeEnum_t
{
    None,
    LeakyRelu,
    Sigmoid
};

// index type
using index_t = int32_t;

} // namespace ck
#endif
