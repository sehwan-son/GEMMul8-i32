#pragma once

#include "common.hpp"
#include "conv_32i_2_8i_real.hpp"
#include "encoding_i32_real.hpp"
#include "i32_moduli.hpp"
#include "reconstruct_i32_real.hpp"

#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace oz2 {
namespace i32 {

template <bool UseExtraWorkspace>
__inline__ size_t workSize(
    const size_t m,
    const size_t n,
    const size_t k,
    const unsigned num_moduli,
    size_t *const workSizeA,
    size_t *const workSizeB //
) {
    validate_num_moduli_or_throw(num_moduli, "workSize_i32");
    validate_k_or_throw(k, "workSize_i32");

    const size_t lda8i  = padding(k);
    const size_t cola8i = padding(m);
    const size_t sizeA  = lda8i * cola8i;
    const size_t sizeB  = lda8i * n;
    const size_t sizeC  = cola8i * n;

    const size_t totalA = sizeof(int8_t) * sizeA * num_moduli;
    const size_t totalB = sizeof(int8_t) * sizeB * num_moduli;
    size_t totalC       = 0;
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

template <bool UseExtraWorkspace>
__inline__ std::vector<double> gemm(
    cublasHandle_t handle,
    cublasOperation_t op_A, cublasOperation_t op_B,
    const size_t m, const size_t n, const size_t k,
    const int64_t *const alpha,
    const int32_t *const A, const size_t lda,
    const int32_t *const B, const size_t ldb,
    const int64_t *const beta,
    int64_t *const C, const size_t ldc,
    const unsigned num_moduli,
    void *const work,
    void *const workA,
    void *const workB //
) {
    validate_num_moduli_or_throw(num_moduli, "gemm_i32");
    validate_k_or_throw(k, "gemm_i32");
    if ((op_A != CUBLAS_OP_N && op_A != CUBLAS_OP_T) ||
        (op_B != CUBLAS_OP_N && op_B != CUBLAS_OP_T)) {
        throw std::invalid_argument("gemm_i32: op_A/op_B must be CUBLAS_OP_N or CUBLAS_OP_T.");
    }
    if (alpha == nullptr || beta == nullptr) {
        throw std::invalid_argument("gemm_i32: alpha/beta pointer must not be null.");
    }
    if (A == nullptr || B == nullptr || C == nullptr || work == nullptr) {
        throw std::invalid_argument("gemm_i32: A/B/C/work pointer must not be null.");
    }

    auto check_cublas = [](const cublasStatus_t status, const char *const where) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                std::string("gemm_i32: cublas failure at ") + where +
                ", status=" + std::to_string(static_cast<int>(status)));
        }
    };
    auto check_cuda = [](const char *const where) {
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("gemm_i32: cuda failure at ") + where +
                ", " + cudaGetErrorString(status));
        }
    };
    auto to_int = [](const size_t v, const char *const name) -> int {
        if (v > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(std::string("gemm_i32: ") + name + " is too large for cuBLAS int API.");
        }
        return static_cast<int>(v);
    };

    cudaStream_t stream = 0;
    check_cublas(cublasGetStream(handle, &stream), "cublasGetStream");

    const size_t lda8i   = padding(k);
    const size_t ldb8i   = lda8i;
    const size_t ldc32i  = padding(m);
    const size_t sizeA   = lda8i * ldc32i;
    const size_t sizeB   = ldb8i * n;
    const size_t sizeC   = ldc32i * n;
    const size_t sizeC_4 = sizeC >> 2;
    const size_t offsetA = sizeA * num_moduli;
    const size_t offsetB = sizeB * num_moduli;

    const int lda8i_i  = to_int(lda8i, "lda8i");
    const int ldb8i_i  = to_int(ldb8i, "ldb8i");
    const int ldc32i_i = to_int(ldc32i, "ldc32i");
    const int n_i      = to_int(n, "n");

    int8_t *const A8i = reinterpret_cast<int8_t *>((workA != nullptr) ? workA : work);
    int8_t *const B8i = reinterpret_cast<int8_t *>(
        (workB != nullptr) ? workB : ((workA != nullptr) ? work : (A8i + offsetA)));

    int8_t *C8i  = nullptr;
    int32_t *C32 = nullptr;
    if constexpr (UseExtraWorkspace) {
        C32 = reinterpret_cast<int32_t *>(
            (workB != nullptr)
                ? ((workA != nullptr) ? work : (A8i + offsetA))
                : (B8i + offsetB));
    } else {
        C8i = reinterpret_cast<int8_t *>(
            (workB != nullptr)
                ? ((workA != nullptr) ? work : (A8i + offsetA))
                : (B8i + offsetB));
        C32 = reinterpret_cast<int32_t *>(C8i + sizeC * num_moduli);
    }

    std::vector<double> timer(4, 0.0);
    std::chrono::system_clock::time_point time_stamp;

    timing(stream, time_stamp);
    encode(
        stream,
        op_A, op_B, m, n, k, num_moduli,
        A, lda, A8i, lda8i, sizeA, ldc32i,
        B, ldb, B8i, ldb8i, sizeB);
    check_cuda("encode");
    timing(stream, time_stamp, timer[0]);

    constexpr int32_t one  = 1;
    constexpr int32_t zero = 0;
    constexpr int blk_n    = 12288;
    for (unsigned i = 0; i < num_moduli; ++i) {
        int rem_n    = n_i;
        int col_off  = 0;
        while (rem_n > 0) {
            const int nn = (rem_n <= blk_n) ? rem_n : blk_n;
            if constexpr (UseExtraWorkspace) {
                check_cublas(
                    cublasGemmEx(
                        handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        ldc32i_i, nn, lda8i_i,
                        &one,
                        A8i + i * sizeA, CUDA_R_8I, lda8i_i,
                        B8i + i * sizeB + static_cast<size_t>(col_off) * ldb8i, CUDA_R_8I, ldb8i_i,
                        &zero,
                        C32 + i * sizeC + static_cast<size_t>(col_off) * ldc32i, CUDA_R_32I, ldc32i_i,
                        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
                    "int8_tc_gemm(use_extra=true)");
            } else {
                check_cublas(
                    cublasGemmEx(
                        handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        ldc32i_i, nn, lda8i_i,
                        &one,
                        A8i + i * sizeA, CUDA_R_8I, lda8i_i,
                        B8i + i * sizeB + static_cast<size_t>(col_off) * ldb8i, CUDA_R_8I, ldb8i_i,
                        &zero,
                        C32 + static_cast<size_t>(col_off) * ldc32i, CUDA_R_32I, ldc32i_i,
                        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
                    "int8_tc_gemm(use_extra=false)");
            }
            col_off += nn;
            rem_n -= nn;
        }
        timing(stream, time_stamp, timer[1]);

        if constexpr (!UseExtraWorkspace) {
            oz2::real::conv_32i_2_8i(stream, i, sizeC_4, C32, C8i + i * sizeC);
            check_cuda("conv_32i_2_8i");
            timing(stream, time_stamp, timer[2]);
        }
    }

    if constexpr (UseExtraWorkspace) {
        reconstruct<int32_t>(stream, m, n, C32, ldc32i, sizeC, C, ldc, *alpha, *beta);
        check_cuda("reconstruct<int32_t>");
    } else {
        reconstruct<int8_t>(stream, m, n, C8i, ldc32i, sizeC, C, ldc, *alpha, *beta);
        check_cuda("reconstruct<int8_t>");
    }
    timing(stream, time_stamp, timer[3]);

    return timer;
}

} // namespace i32
} // namespace oz2
