#pragma once
#if defined(__NVCC__)
    #include <cuComplex.h>
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
#endif
#if defined(__HIPCC__)
    #include <hip/hip_complex.h>
    #include <hip/hip_runtime.h>
    #include <hipblas/hipblas.h>
#endif
#include <cstdint>
#include <vector>

namespace gemmul8 {

/***
 * workSize returns the required workspace size in bytes.
 */
template <bool is_Complex = false, bool UseExtraWorkspace = true>
size_t workSize(
    size_t m,                         // Number of rows of C
    size_t n,                         // Number of columns of C
    size_t k,                         // Inner dimension <= 2^17
    unsigned num_moduli,              // #moduli, 2 <= num_moduli <= 20
    bool enable_skip_scalA = false,   // [option] Reserve extra space for A to allow skip_scalA
    bool enable_skip_scalB = false,   // [option] Reserve extra space for B to allow skip_scalB
    size_t *workSizeA      = nullptr, // [option] Output: workspace size used for A8i and sftA
    size_t *workSizeB      = nullptr  // [option] Output: workspace size used for B8i and sftB
);

#if defined(__NVCC__)
/***
 * workSize_i32 returns the required workspace size in bytes for INT32 input path.
 * Preconditions:
 * - 9 <= num_moduli <= 20
 * - k <= 2^17
 */
template <bool UseExtraWorkspace = true>
size_t workSize_i32(
    size_t m,                         // Number of rows of C
    size_t n,                         // Number of columns of C
    size_t k,                         // Inner dimension <= 2^17
    unsigned num_moduli,              // #moduli, 9 <= num_moduli <= 20
    size_t *workSizeA      = nullptr, // [option] Output: workspace size used for A8i
    size_t *workSizeB      = nullptr  // [option] Output: workspace size used for B8i
);
#endif

/***
 * GEMM emulation using INT8 Tensor Cores
 */
#if defined(__NVCC__)
template <typename T, bool UseExtraWorkspace = true>
std::vector<double> gemm(
    cublasHandle_t handle,            // Handle to the cuBLAS library context
    cublasOperation_t op_A,           // CUBLAS_OP_N or CUBLAS_OP_T
    cublasOperation_t op_B,           // CUBLAS_OP_N or CUBLAS_OP_T
    size_t m,                         // Number of rows of C
    size_t n,                         // Number of columns of C
    size_t k,                         // Inner dimension <= 2^17
    const T *alpha,                   // Scaling factor for op(A)*op(B)
    const T *const A,                 // 1-D device array of dimensions lda*k (CUBLAS_OP_N) or lda*m (CUBLAS_OP_T)
    size_t lda,                       // Leading dimension of A
    const T *const B,                 // 1-D device array of dimensions ldb*n (CUBLAS_OP_N) or ldb*k (CUBLAS_OP_T)
    size_t ldb,                       // Leading dimension of B
    const T *beta,                    // Scaling factor for C
    T *const C,                       // 1-D device array of dimensions ldc*n
    size_t ldc,                       // Leading dimension of C
    unsigned num_moduli,              // #moduli, 2 <= num_moduli <= 20
    bool fastmode,                    // false (accurate mode) or true (fast mode)
    void *const work,                 // Preallocated workspace
    void *const workA      = nullptr, // [optional] Separate workspace for A (if nullptr, uses work)
    void *const workB      = nullptr, // [optional] Separate workspace for B (if nullptr, uses work)
    bool enable_skip_scalA = false,   // [optional] Enables scaling-skip mechanism for A
    bool enable_skip_scalB = false,   // [optional] Enables scaling-skip mechanism for B
    bool skip_scalA        = false,   // [optional] If true, skip preprocessing for A
    bool skip_scalB        = false    // [optional] If true, skip preprocessing for B
);

/***
 * GEMM emulation using INT8 Tensor Cores (INT32 input path).
 * Computes: C = alpha*op(A)*op(B) + beta*C, where A/B are INT32 and C is INT64.
 * Preconditions:
 * - 9 <= num_moduli <= 20
 * - k <= 2^17
 * - op_A/op_B in {CUBLAS_OP_N, CUBLAS_OP_T}
 * - Result range must fit int64_t (caller responsibility).
 */
template <bool UseExtraWorkspace = true>
std::vector<double> gemm_i32(
    cublasHandle_t handle,            // Handle to the cuBLAS library context
    cublasOperation_t op_A,           // CUBLAS_OP_N or CUBLAS_OP_T
    cublasOperation_t op_B,           // CUBLAS_OP_N or CUBLAS_OP_T
    size_t m,                         // Number of rows of C
    size_t n,                         // Number of columns of C
    size_t k,                         // Inner dimension <= 2^17
    const int64_t *alpha,             // Scaling factor for op(A)*op(B)
    const int32_t *const A,           // 1-D device array of dimensions lda*k (CUBLAS_OP_N) or lda*m (CUBLAS_OP_T)
    size_t lda,                       // Leading dimension of A
    const int32_t *const B,           // 1-D device array of dimensions ldb*n (CUBLAS_OP_N) or ldb*k (CUBLAS_OP_T)
    size_t ldb,                       // Leading dimension of B
    const int64_t *beta,              // Scaling factor for C
    int64_t *const C,                 // 1-D device array of dimensions ldc*n
    size_t ldc,                       // Leading dimension of C
    unsigned num_moduli,              // #moduli, 9 <= num_moduli <= 20
    void *const work,                 // Preallocated workspace
    void *const workA      = nullptr, // [optional] Separate workspace for A (if nullptr, uses work)
    void *const workB      = nullptr  // [optional] Separate workspace for B (if nullptr, uses work)
);

/***
 * Backward-compatible overload:
 * Computes C = op(A)*op(B) (equivalent to alpha=1, beta=0).
 */
template <bool UseExtraWorkspace = true>
std::vector<double> gemm_i32(
    cublasHandle_t handle,            // Handle to the cuBLAS library context
    cublasOperation_t op_A,           // CUBLAS_OP_N or CUBLAS_OP_T
    cublasOperation_t op_B,           // CUBLAS_OP_N or CUBLAS_OP_T
    size_t m,                         // Number of rows of C
    size_t n,                         // Number of columns of C
    size_t k,                         // Inner dimension <= 2^17
    const int32_t *const A,           // 1-D device array of dimensions lda*k (CUBLAS_OP_N) or lda*m (CUBLAS_OP_T)
    size_t lda,                       // Leading dimension of A
    const int32_t *const B,           // 1-D device array of dimensions ldb*n (CUBLAS_OP_N) or ldb*k (CUBLAS_OP_T)
    size_t ldb,                       // Leading dimension of B
    int64_t *const C,                 // 1-D device array of dimensions ldc*n
    size_t ldc,                       // Leading dimension of C
    unsigned num_moduli,              // #moduli, 9 <= num_moduli <= 20
    void *const work,                 // Preallocated workspace
    void *const workA      = nullptr, // [optional] Separate workspace for A (if nullptr, uses work)
    void *const workB      = nullptr  // [optional] Separate workspace for B (if nullptr, uses work)
);
#endif

#if defined(__HIPCC__)
template <typename T, bool UseExtraWorkspace = true>
std::vector<double> gemm(
    hipblasHandle_t handle,           // Handle to the hipBLAS library context
    hipblasOperation_t op_A,          // HIPBLAS_OP_N or HIPBLAS_OP_T
    hipblasOperation_t op_B,          // HIPBLAS_OP_N or HIPBLAS_OP_T
    size_t m,                         // Number of rows of C
    size_t n,                         // Number of columns of C
    size_t k,                         // Inner dimension <= 2^17
    const T *alpha,                   // Scaling factor for op(A)*op(B)
    const T *const A,                 // 1-D device array of dimensions lda*k (HIPBLAS_OP_N) or lda*m (HIPBLAS_OP_T)
    size_t lda,                       // Leading dimension of A
    const T *const B,                 // 1-D device array of dimensions ldb*n (HIPBLAS_OP_N) or ldb*k (HIPBLAS_OP_T)
    size_t ldb,                       // Leading dimension of B
    const T *beta,                    // Scaling factor for C
    T *const C,                       // 1-D device array of dimensions ldc*n
    size_t ldc,                       // Leading dimension of C
    unsigned num_moduli,              // #moduli, 2 <= num_moduli <= 20
    bool fastmode,                    // false (accurate mode) or true (fast mode)
    void *const work,                 // Preallocated workspace
    void *const workA      = nullptr, // [optional] Separate workspace for A (if nullptr, uses work)
    void *const workB      = nullptr, // [optional] Separate workspace for B (if nullptr, uses work)
    bool enable_skip_scalA = false,   // [optional] Enables scaling-skip mechanism for A
    bool enable_skip_scalB = false,   // [optional] Enables scaling-skip mechanism for B
    bool skip_scalA        = false,   // [optional] If true, skip preprocessing for A
    bool skip_scalB        = false    // [optional] If true, skip preprocessing for B
);
#endif

} // namespace gemmul8
