#include "common.hpp"
#include "conv_32i_2_8i_real.hpp"
#include "inverse_scaling_real.hpp"
#include "scaling_accu_real.hpp"
#include "scaling_fast_real.hpp"

namespace oz2 {
namespace real {

//------------------------------
// Calculate required work size
//------------------------------
template <bool UseExtraWorkspace>
__inline__ size_t workSize(
    size_t m, size_t n, size_t k,
    unsigned num_moduli,
    bool enable_skip_scalA,
    bool enable_skip_scalB,
    size_t *workSizeA,
    size_t *workSizeB //
) {
    const size_t lda8i     = padding(k);
    const size_t cola8i    = padding(m);
    const size_t sizeA     = lda8i * cola8i;
    const size_t ldb8i     = lda8i;
    const size_t colb8i    = n;
    const size_t sizeB     = ldb8i * colb8i;
    const size_t ldc32i    = cola8i;
    const size_t colc32i   = colb8i;
    const size_t sizeC     = ldc32i * colc32i;
    const size_t size_vecA = cola8i;
    const size_t size_vecB = padding(n);

    const unsigned num_A8i = num_moduli + ((enable_skip_scalA) ? 1 : 0); // +1 for skip_scalA in accurate mode
    const unsigned num_B8i = num_moduli + ((enable_skip_scalB) ? 1 : 0); // +1 for skip_scalB in accurate mode
    const unsigned num_C8i = num_moduli;

    size_t total_size_A = 0;
    size_t total_size_B = 0;
    size_t total_size_C = 0;
    total_size_A += sizeof(int8_t) * sizeA * num_A8i;
    total_size_A += sizeof(int16_t) * size_vecA;
    total_size_B += sizeof(int8_t) * sizeB * num_B8i;
    total_size_B += sizeof(int16_t) * size_vecB;
    if constexpr (UseExtraWorkspace) {
        total_size_C += sizeof(int32_t) * sizeC * num_moduli;
    } else {
        total_size_C += sizeof(int8_t) * sizeC * num_C8i;
        total_size_C += sizeof(int32_t) * sizeC;
    }

    if (workSizeA != nullptr) *workSizeA = total_size_A;
    if (workSizeB != nullptr) *workSizeB = total_size_B;
    return total_size_A + total_size_B + total_size_C;
}

//------------------------------
// GEMM emulation using INT8 Tensor Cores
//------------------------------
template <typename T, bool UseExtraWorkspace>
__inline__ std::vector<double> gemm(
    cublasHandle_t handle,
    cublasOperation_t op_A, cublasOperation_t op_B,
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *const A, size_t lda,
    const T *const B, size_t ldb,
    const T *beta,
    T *const C, size_t ldc,
    unsigned num_moduli,
    bool fastmode,
    void *const work,
    void *const workA,
    void *const workB,
    bool enable_skip_scalA,
    bool enable_skip_scalB,
    bool skip_scalA,
    bool skip_scalB //
) {
    //------------------------------
    // Timer
    //------------------------------
    std::chrono::system_clock::time_point time_stamp;
    std::vector<double> timer(4, 0.0);

    //------------------------------
    // Set constants
    //------------------------------
    const size_t lda8i       = padding(k);
    const size_t ldb8i       = lda8i;
    const size_t ldc32i      = padding(m);
    const size_t sizeA       = lda8i * ldc32i;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = ldc32i * n;
    const size_t sizeC_4     = sizeC >> 2;
    const size_t size_vecA   = ldc32i;
    const size_t size_vecB   = padding(n);
    const size_t offsetA     = sizeA * num_moduli;
    const size_t offsetB     = sizeB * num_moduli;
    const unsigned table_idx = num_moduli - 2;
    const bool skipA         = skip_scalA && enable_skip_scalA;
    const bool skipB         = skip_scalB && enable_skip_scalB;
    constexpr int32_t one    = 1;
    constexpr int32_t zero   = 0;
    grid_invscal             = (m * n + threads_invscal - 1) / threads_invscal;
    if constexpr (!UseExtraWorkspace) {
        grid_conv32i8i = (sizeC_4 + threads_conv32i8i - 1) / threads_conv32i8i;
    }
    bool P_is_double;

    //------------------------------
    // Set constant memory
    //------------------------------
    if constexpr (std::is_same_v<underlying_t<T>, double>) {
        P_is_double = (num_moduli <= 7);
        if (P_is_double) {
            cudaMemcpyToSymbol(table::qPi, &table::qPi_1[table_idx][0], num_moduli * sizeof(double));
        } else {
            cudaMemcpyToSymbol(table::qPi, &table::qPi_2[num_moduli - 8][0][0], 2 * num_moduli * sizeof(double));
        }
        cudaMemcpyToSymbol(table::MODULI_D, table::moduli_d, (num_moduli - 1) * sizeof(double2));
    } else {
        P_is_double = true;
        cudaMemcpyToSymbol(table::qPi, &table::qPi_1[table_idx][0], num_moduli * sizeof(double));
    }
    cudaMemcpyToSymbol(table::MODULI_F, table::moduli_f, (num_moduli - 1) * sizeof(float2));
    cudaMemcpyToSymbol(table::MODULI_I, table::moduli_i, (num_moduli - 1) * sizeof(int2));

    //------------------------------
    // Set workspace
    //------------------------------
    int8_t *const A8i   = reinterpret_cast<int8_t *>((workA) ? workA : work);
    int16_t *const sftA = reinterpret_cast<int16_t *>(A8i + offsetA + ((enable_skip_scalA) ? sizeA : 0));
    int8_t *const B8i   = reinterpret_cast<int8_t *>((workB) ? workB : ((workA) ? work : (sftA + size_vecA)));
    int16_t *const sftB = reinterpret_cast<int16_t *>(B8i + offsetB + ((enable_skip_scalB) ? sizeB : 0));
    int8_t *C8i_tmp;
    int32_t *C32i_tmp;
    if constexpr (UseExtraWorkspace) {
        C8i_tmp  = nullptr;
        C32i_tmp = reinterpret_cast<int32_t *>((workB) ? ((workA) ? work : (sftA + size_vecA)) : (sftB + size_vecB));
    } else {
        C8i_tmp  = reinterpret_cast<int8_t *>((workB) ? ((workA) ? work : (sftA + size_vecA)) : (sftB + size_vecB));
        C32i_tmp = reinterpret_cast<int32_t *>(C8i_tmp + sizeC * num_moduli);
    }
    int8_t *const C8i   = C8i_tmp;
    int32_t *const C32i = C32i_tmp;

    //------------------------------
    // Scaling
    // A =: diag(2^sftA) * A', A' is integer
    // B =: B' * diag(2^sftB), B' is integer
    // Then, calculating mod for all moduli
    // A8i := A' - p[i]*round(A'/p[i])  (-128 <= A8i <= 127)
    // B8i := B' - p[i]*round(B'/p[i])  (-128 <= A8i <= 127)
    //------------------------------
    timing(time_stamp);
    if (!(skipA && skipB)) {
        // When both scalingA & scalingB are skipped, this is skiped.
        if (fastmode) {
            fast::scaling<T>(op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, lda8i, sizeA, sftA,
                             B, ldb, B8i, ldb8i, sizeB, sftB,
                             table_idx, skipA, skipB);
        } else {
            int8_t *const A8i_high = A8i + ((enable_skip_scalA) ? offsetA : 0);
            int8_t *const B8i_high = B8i + ((enable_skip_scalB) ? offsetB : 0);
            accu::scaling<T>(handle, op_A, op_B, m, n, k, num_moduli,
                             A, lda, A8i, A8i_high, lda8i, sizeA, sftA,
                             B, ldb, B8i, B8i_high, ldb8i, sizeB, sftB,
                             C32i, ldc32i,
                             table_idx, skipA, skipB);
        }
    }
    timing(time_stamp, timer[0]);

    if constexpr (UseExtraWorkspace) {

        for (unsigned i = 0; i < num_moduli; ++i) {
            //-----------------------------
            // Error-free matrix multiplication
            // C32i := A8i*B8i
            //------------------------------

            int blk    = 8192;
            int rem    = n;
            int offset = 0;

            while (rem > 0) {
                size_t nn = (rem <= 12288) ? rem : blk;

                cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             ldc32i, nn, lda8i,
                             &one,
                             A8i + i * sizeA, CUDA_R_8I, lda8i,
                             B8i + i * sizeB + offset * ldb8i, CUDA_R_8I, ldb8i,
                             &zero,
                             C32i + i * sizeC + offset * ldc32i, CUDA_R_32I, ldc32i,
                             CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

                offset += nn;
                rem -= nn;
            }
            timing(time_stamp, timer[1]);
        }

        //------------------------------
        // Accumulation and Inverse scaling
        // C64f = sum(qi*Pi*C8i[i]),
        //  where
        //      Pi := P/p[i], --> Mi
        //      P  := prod(p[all]), --> M
        //      mod(qi*Pi, p[i]) \equiv 1. --> Miyi = 1 (mod m_i)
        // C := C64f - round(C64f/P)*P
        // C := diag(2^sftA) * C * diag(2^sftB)
        //------------------------------
        inverse_scaling<T, int32_t>(P_is_double, num_moduli, m, n, C32i, ldc32i, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
        timing(time_stamp, timer[3]);

    } else {

        for (unsigned i = 0; i < num_moduli; ++i) {
            //-----------------------------
            // Error-free matrix multiplication
            // C32i := A8i*B8i
            //------------------------------

            int blk    = 8192;
            int rem    = n;
            int offset = 0;

            while (rem > 0) {
                size_t nn = (rem <= 12288) ? rem : blk;

                cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             ldc32i, nn, lda8i,
                             &one,
                             A8i + i * sizeA, CUDA_R_8I, lda8i,
                             B8i + i * sizeB + offset * ldb8i, CUDA_R_8I, ldb8i,
                             &zero,
                             C32i + offset * ldc32i, CUDA_R_32I, ldc32i,
                             CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

                offset += nn;
                rem -= nn;
            }
            timing(time_stamp, timer[1]);

            //------------------------------
            // Calculating mod
            // C8i[i] := mod(C32i, p[i]) >= 0
            //------------------------------
            conv_32i_2_8i(i, sizeC_4, C32i, C8i + i * sizeC);
            timing(time_stamp, timer[2]);
        }

        //------------------------------
        // Accumulation and Inverse scaling
        // C64f = sum(qi*Pi*C8i[i]),
        //  where
        //      Pi := P/p[i],
        //      P  := prod(p[all]),
        //      mod(qi*Pi, p[i]) \equiv 1.
        // C := C64f - round(C64f/P)*P
        // C := diag(2^sftA) * C * diag(2^sftB)
        //------------------------------
        inverse_scaling<T>(P_is_double, num_moduli, m, n, C8i, ldc32i, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
        timing(time_stamp, timer[3]);
    }

    return timer;
}

} // namespace real
} // namespace oz2
