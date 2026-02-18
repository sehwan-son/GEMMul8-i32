#include "../include/gemmul8.hpp"
#include "gemmul8_complex.hpp"
#include "gemmul8_real.hpp"
#if defined(__NVCC__)
    #include "gemmul8_i32.hpp"
#endif

#if !defined(GEMM_ARGS)
    #define GEMM_ARGS(T) cublasHandle_t handle,                                  \
                         cublasOperation_t op_A, cublasOperation_t op_B,         \
                         size_t m, size_t n, size_t k,                           \
                         const T *alpha, const T *const A, size_t lda,           \
                         const T *const B, size_t ldb,                           \
                         const T *beta, T *const C, size_t ldc,                  \
                         unsigned num_moduli, bool fastmode,                     \
                         void *const work, void *const workA, void *const workB, \
                         bool enable_skip_scalA, bool enable_skip_scalB,         \
                         bool skip_scalA, bool skip_scalB
#endif

#if !defined(GEMM_CALL_ARGS)
    #define GEMM_CALL_ARGS handle, op_A, op_B, m, n, k,              \
                           alpha, A, lda, B, ldb, beta, C, ldc,      \
                           num_moduli, fastmode, work, workA, workB, \
                           enable_skip_scalA, enable_skip_scalB,     \
                           skip_scalA, skip_scalB
#endif

namespace gemmul8 {

//------------------------------
// Calculate required work size
//------------------------------
template <> size_t workSize<true, true>(
    size_t m, size_t n, size_t k,
    unsigned num_moduli,
    bool enable_skip_scalA, bool enable_skip_scalB,
    size_t *workSizeA, size_t *workSizeB //
) {
    return oz2::complex::workSize<true>(m, n, k, num_moduli, enable_skip_scalA, enable_skip_scalB, workSizeA, workSizeB);
}
template <> size_t workSize<true, false>(
    size_t m, size_t n, size_t k,
    unsigned num_moduli,
    bool enable_skip_scalA, bool enable_skip_scalB,
    size_t *workSizeA, size_t *workSizeB //
) {
    return oz2::complex::workSize<false>(m, n, k, num_moduli, enable_skip_scalA, enable_skip_scalB, workSizeA, workSizeB);
}

template <> size_t workSize<false, true>(
    size_t m, size_t n, size_t k,
    unsigned num_moduli,
    bool enable_skip_scalA, bool enable_skip_scalB,
    size_t *workSizeA, size_t *workSizeB //
) {
    return oz2::real::workSize<true>(m, n, k, num_moduli, enable_skip_scalA, enable_skip_scalB, workSizeA, workSizeB);
}

template <> size_t workSize<false, false>(
    size_t m, size_t n, size_t k,
    unsigned num_moduli,
    bool enable_skip_scalA, bool enable_skip_scalB,
    size_t *workSizeA, size_t *workSizeB //
) {
    return oz2::real::workSize<false>(m, n, k, num_moduli, enable_skip_scalA, enable_skip_scalB, workSizeA, workSizeB);
}

#if defined(__NVCC__)
template <> size_t workSize_i32<true>(
    size_t m, size_t n, size_t k,
    unsigned num_moduli,
    size_t *workSizeA, size_t *workSizeB //
) {
    return oz2::i32::workSize<true>(m, n, k, num_moduli, workSizeA, workSizeB);
}

template <> size_t workSize_i32<false>(
    size_t m, size_t n, size_t k,
    unsigned num_moduli,
    size_t *workSizeA, size_t *workSizeB //
) {
    return oz2::i32::workSize<false>(m, n, k, num_moduli, workSizeA, workSizeB);
}
#endif

//------------------------------
// GEMM emulation using INT8 Tensor Cores
//------------------------------
template <> std::vector<double> gemm<double, true>(GEMM_ARGS(double)) { return oz2::real::gemm<double, true>(GEMM_CALL_ARGS); }
template <> std::vector<double> gemm<double, false>(GEMM_ARGS(double)) { return oz2::real::gemm<double, false>(GEMM_CALL_ARGS); }
template <> std::vector<double> gemm<float, true>(GEMM_ARGS(float)) { return oz2::real::gemm<float, true>(GEMM_CALL_ARGS); }
template <> std::vector<double> gemm<float, false>(GEMM_ARGS(float)) { return oz2::real::gemm<float, false>(GEMM_CALL_ARGS); }
template <> std::vector<double> gemm<cuFloatComplex, true>(GEMM_ARGS(cuFloatComplex)) { return oz2::complex::gemm<cuFloatComplex, true>(GEMM_CALL_ARGS); }
template <> std::vector<double> gemm<cuFloatComplex, false>(GEMM_ARGS(cuFloatComplex)) { return oz2::complex::gemm<cuFloatComplex, false>(GEMM_CALL_ARGS); }
template <> std::vector<double> gemm<cuDoubleComplex, true>(GEMM_ARGS(cuDoubleComplex)) { return oz2::complex::gemm<cuDoubleComplex, true>(GEMM_CALL_ARGS); }
template <> std::vector<double> gemm<cuDoubleComplex, false>(GEMM_ARGS(cuDoubleComplex)) { return oz2::complex::gemm<cuDoubleComplex, false>(GEMM_CALL_ARGS); }

#if defined(__NVCC__)
template <> std::vector<double> gemm_i32<true>(
    cublasHandle_t handle,
    cublasOperation_t op_A, cublasOperation_t op_B,
    size_t m, size_t n, size_t k,
    const int64_t *alpha,
    const int32_t *const A, size_t lda,
    const int32_t *const B, size_t ldb,
    const int64_t *beta,
    int64_t *const C, size_t ldc,
    unsigned num_moduli,
    void *const work,
    void *const workA,
    void *const workB //
) {
    return oz2::i32::gemm<true>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, work, workA, workB);
}

template <> std::vector<double> gemm_i32<false>(
    cublasHandle_t handle,
    cublasOperation_t op_A, cublasOperation_t op_B,
    size_t m, size_t n, size_t k,
    const int64_t *alpha,
    const int32_t *const A, size_t lda,
    const int32_t *const B, size_t ldb,
    const int64_t *beta,
    int64_t *const C, size_t ldc,
    unsigned num_moduli,
    void *const work,
    void *const workA,
    void *const workB //
) {
    return oz2::i32::gemm<false>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, work, workA, workB);
}

template <> std::vector<double> gemm_i32<true>(
    cublasHandle_t handle,
    cublasOperation_t op_A, cublasOperation_t op_B,
    size_t m, size_t n, size_t k,
    const int32_t *const A, size_t lda,
    const int32_t *const B, size_t ldb,
    int64_t *const C, size_t ldc,
    unsigned num_moduli,
    void *const work,
    void *const workA,
    void *const workB //
) {
    constexpr int64_t alpha = 1;
    constexpr int64_t beta  = 0;
    return oz2::i32::gemm<true>(handle, op_A, op_B, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, num_moduli, work, workA, workB);
}

template <> std::vector<double> gemm_i32<false>(
    cublasHandle_t handle,
    cublasOperation_t op_A, cublasOperation_t op_B,
    size_t m, size_t n, size_t k,
    const int32_t *const A, size_t lda,
    const int32_t *const B, size_t ldb,
    int64_t *const C, size_t ldc,
    unsigned num_moduli,
    void *const work,
    void *const workA,
    void *const workB //
) {
    constexpr int64_t alpha = 1;
    constexpr int64_t beta  = 0;
    return oz2::i32::gemm<false>(handle, op_A, op_B, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, num_moduli, work, workA, workB);
}
#endif

} // namespace gemmul8
