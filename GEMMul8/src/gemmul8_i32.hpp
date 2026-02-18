#pragma once
#include "common.hpp"
#include "conv_32i_2_8i_real.hpp"
#include "encoding_i32_real.hpp"
#include "reconstruct_i32_real.hpp"
#include <stdexcept>
#include <string>

namespace oz2 {
namespace i32 {

//------------------------------
// Calculate required work size
//------------------------------
template <bool UseExtraWorkspace>
__inline__ size_t workSize(
    size_t m, size_t n, size_t k,
    unsigned num_moduli,
    size_t *workSizeA,
    size_t *workSizeB //
) {
    if (num_moduli < crt_moduli_used || num_moduli > 20u) {
        throw std::invalid_argument("workSize_i32: num_moduli must satisfy 9 <= num_moduli <= 20.");
    }
    if (k > (1u << 17u)) {
        throw std::invalid_argument("workSize_i32: k must satisfy k <= 2^17.");
    }

    const size_t lda8i   = padding(k);
    const size_t cola8i  = padding(m);
    const size_t sizeA   = lda8i * cola8i;
    const size_t ldb8i   = lda8i;
    const size_t sizeB   = ldb8i * n;
    const size_t ldc32i  = cola8i;
    const size_t sizeC   = ldc32i * n;
    const size_t totalA  = sizeof(int8_t) * sizeA * num_moduli;
    const size_t totalB  = sizeof(int8_t) * sizeB * num_moduli;
    size_t totalC        = 0;
    if constexpr (UseExtraWorkspace) {
        totalC += sizeof(int32_t) * sizeC * num_moduli;
    } else {
        totalC += sizeof(int8_t) * sizeC * num_moduli;
        totalC += sizeof(int32_t) * sizeC;
    }

    if (workSizeA != nullptr) *workSizeA = totalA;
    if (workSizeB != nullptr) *workSizeB = totalB;
    return totalA + totalB + totalC;
}

//------------------------------
// GEMM emulation using INT8 Tensor Cores (INT32 input)
//------------------------------
template <bool UseExtraWorkspace>
__inline__ std::vector<double> gemm(
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
    if (num_moduli < crt_moduli_used || num_moduli > 20u) {
        throw std::invalid_argument("gemm_i32: num_moduli must satisfy 9 <= num_moduli <= 20.");
    }
    if (k > (1u << 17u)) {
        throw std::invalid_argument("gemm_i32: k must satisfy k <= 2^17.");
    }
    if ((op_A != CUBLAS_OP_N && op_A != CUBLAS_OP_T) || (op_B != CUBLAS_OP_N && op_B != CUBLAS_OP_T)) {
        throw std::invalid_argument("gemm_i32: op_A/op_B must be CUBLAS_OP_N or CUBLAS_OP_T.");
    }
    if (alpha == nullptr || beta == nullptr) {
        throw std::invalid_argument("gemm_i32: alpha/beta pointer must not be null.");
    }

    //------------------------------
    // Timer
    //------------------------------
    std::chrono::system_clock::time_point time_stamp;
    std::vector<double> timer(4, 0.0);
    auto check_cublas = [](const cublasStatus_t status, const char *const where) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("gemm_i32: cublas failure at ") + where + ", status=" + std::to_string(static_cast<int>(status)));
        }
    };
    auto check_cuda = [](const char *const where) {
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("gemm_i32: cuda failure at ") + where + ", " + cudaGetErrorString(status));
        }
    };

    //------------------------------
    // Set constants
    //------------------------------
    const size_t lda8i      = padding(k);
    const size_t ldb8i      = lda8i;
    const size_t ldc32i     = padding(m);
    const size_t sizeA      = lda8i * ldc32i;
    const size_t sizeB      = ldb8i * n;
    const size_t sizeC      = ldc32i * n;
    const size_t sizeC_4    = sizeC >> 2;
    const size_t offsetA    = sizeA * num_moduli;
    const size_t offsetB    = sizeB * num_moduli;
    constexpr int32_t one   = 1;
    constexpr int32_t zero  = 0;
    const int blk           = 8192;
    grid_invscal            = (m * n + threads_invscal - 1) / threads_invscal;
    if constexpr (!UseExtraWorkspace) {
        grid_conv32i8i = (sizeC_4 + threads_conv32i8i - 1) / threads_conv32i8i;
    }

    //------------------------------
    // Set constant memory
    //------------------------------
    cudaMemcpyToSymbol(table::MODULI_I, table::moduli_i, (num_moduli - 1u) * sizeof(int2));

    //------------------------------
    // Set workspace
    //------------------------------
    int8_t *const A8i = reinterpret_cast<int8_t *>((workA != nullptr) ? workA : work);
    int8_t *const B8i = reinterpret_cast<int8_t *>((workB != nullptr) ? workB : ((workA != nullptr) ? work : (A8i + offsetA)));

    int8_t *C8i_tmp;
    int32_t *C32i_tmp;
    if constexpr (UseExtraWorkspace) {
        C8i_tmp  = nullptr;
        C32i_tmp = reinterpret_cast<int32_t *>((workB != nullptr) ? ((workA != nullptr) ? work : (A8i + offsetA)) : (B8i + offsetB));
    } else {
        C8i_tmp  = reinterpret_cast<int8_t *>((workB != nullptr) ? ((workA != nullptr) ? work : (A8i + offsetA)) : (B8i + offsetB));
        C32i_tmp = reinterpret_cast<int32_t *>(C8i_tmp + sizeC * num_moduli);
    }
    int8_t *const C8i   = C8i_tmp;
    int32_t *const C32i = C32i_tmp;

    //------------------------------
    // Encode A/B into residues
    //------------------------------
    timing(time_stamp);
    encode(op_A, op_B, m, n, k, num_moduli,
           A, lda, A8i, lda8i, sizeA, ldc32i,
           B, ldb, B8i, ldb8i, sizeB);
    check_cuda("encode");
    timing(time_stamp, timer[0]);

    if constexpr (UseExtraWorkspace) {
        for (unsigned i = 0; i < num_moduli; ++i) {
            int rem    = static_cast<int>(n);
            int offset = 0;
            while (rem > 0) {
                const size_t nn = static_cast<size_t>((rem <= 12288) ? rem : blk);
                check_cublas(
                    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 ldc32i, nn, lda8i,
                                 &one,
                                 A8i + i * sizeA, CUDA_R_8I, lda8i,
                                 B8i + i * sizeB + static_cast<size_t>(offset) * ldb8i, CUDA_R_8I, ldb8i,
                                 &zero,
                                 C32i + i * sizeC + static_cast<size_t>(offset) * ldc32i, CUDA_R_32I, ldc32i,
                                 CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
                    "UseExtraWorkspace=true");
                offset += static_cast<int>(nn);
                rem -= static_cast<int>(nn);
            }
            timing(time_stamp, timer[1]);
        }

        reconstruct<int32_t>(num_moduli, m, n, C32i, ldc32i, sizeC, C, ldc, *alpha, *beta);
        check_cuda("reconstruct<int32_t>");
        timing(time_stamp, timer[3]);
    } else {
        for (unsigned i = 0; i < num_moduli; ++i) {
            int rem    = static_cast<int>(n);
            int offset = 0;
            while (rem > 0) {
                const size_t nn = static_cast<size_t>((rem <= 12288) ? rem : blk);
                check_cublas(
                    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 ldc32i, nn, lda8i,
                                 &one,
                                 A8i + i * sizeA, CUDA_R_8I, lda8i,
                                 B8i + i * sizeB + static_cast<size_t>(offset) * ldb8i, CUDA_R_8I, ldb8i,
                                 &zero,
                                 C32i + static_cast<size_t>(offset) * ldc32i, CUDA_R_32I, ldc32i,
                                 CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
                    "UseExtraWorkspace=false");
                offset += static_cast<int>(nn);
                rem -= static_cast<int>(nn);
            }
            timing(time_stamp, timer[1]);

            oz2::real::conv_32i_2_8i(i, sizeC_4, C32i, C8i + i * sizeC);
            check_cuda("conv_32i_2_8i");
            timing(time_stamp, timer[2]);
        }

        reconstruct<int8_t>(num_moduli, m, n, C8i, ldc32i, sizeC, C, ldc, *alpha, *beta);
        check_cuda("reconstruct<int8_t>");
        timing(time_stamp, timer[3]);
    }

    return timer;
}

} // namespace i32
} // namespace oz2
