#pragma once

#include "i32_split.hpp"

namespace oz1 {
namespace i32 {

__global__ void encode_A_i32_split_kernel(
    const cublasOperation_t op_A,
    const size_t m, const size_t k,
    const int32_t *const __restrict__ A,
    const size_t lda,
    int8_t *const __restrict__ A8i,
    const size_t lda8i,
    const size_t incA8i,
    const size_t cola8i //
) {
    const auto out_col = static_cast<size_t>(blockIdx.y * blockDim.y + threadIdx.y); // [0, padding(m))
    const auto out_row = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x); // [0, padding(k))
    if (out_col >= cola8i || out_row >= lda8i) return;

    int32_t in = 0;
    if (out_col < m && out_row < k) {
        if (op_A == CUBLAS_OP_N) {
            // A is m x k, and we store op(A)^T = A^T as k x m.
            in = __ldg(A + out_col + out_row * lda);
        } else {
            // A is k x m, and we store op(A)^T = A as k x m.
            in = __ldg(A + out_row + out_col * lda);
        }
    }

    int8_t digits[kNumSplitDigits];
    split_i32_to_i8_digits(in, digits);

    const size_t out_idx = out_col * lda8i + out_row;
#pragma unroll
    for (unsigned d = 0; d < kNumSplitDigits; ++d) {
        A8i[d * incA8i + out_idx] = digits[d];
    }
}

__global__ void encode_B_i32_split_kernel(
    const cublasOperation_t op_B,
    const size_t n, const size_t k,
    const int32_t *const __restrict__ B,
    const size_t ldb,
    int8_t *const __restrict__ B8i,
    const size_t ldb8i,
    const size_t incB8i //
) {
    const auto out_col = static_cast<size_t>(blockIdx.y * blockDim.y + threadIdx.y); // [0, n)
    const auto out_row = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x); // [0, padding(k))
    if (out_col >= n || out_row >= ldb8i) return;

    int32_t in = 0;
    if (out_row < k) {
        if (op_B == CUBLAS_OP_N) {
            // B is k x n, and we store op(B) = B as k x n.
            in = __ldg(B + out_row + out_col * ldb);
        } else {
            // B is n x k, and we store op(B) = B^T as k x n.
            in = __ldg(B + out_col + out_row * ldb);
        }
    }

    int8_t digits[kNumSplitDigits];
    split_i32_to_i8_digits(in, digits);

    const size_t out_idx = out_col * ldb8i + out_row;
#pragma unroll
    for (unsigned d = 0; d < kNumSplitDigits; ++d) {
        B8i[d * incB8i + out_idx] = digits[d];
    }
}

__forceinline__ void encode_split(
    const cudaStream_t stream,
    const cublasOperation_t op_A, const cublasOperation_t op_B,
    const size_t m, const size_t n, const size_t k,
    const int32_t *const A,
    const size_t lda,
    int8_t *const A8i,
    const size_t lda8i,
    const size_t incA8i,
    const size_t cola8i,
    const int32_t *const B,
    const size_t ldb,
    int8_t *const B8i,
    const size_t ldb8i,
    const size_t incB8i //
) {
    constexpr dim3 threads(16, 16);
    const dim3 gridA((lda8i + threads.x - 1) / threads.x, (cola8i + threads.y - 1) / threads.y);
    const dim3 gridB((ldb8i + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    encode_A_i32_split_kernel<<<gridA, threads, 0, stream>>>(op_A, m, k, A, lda, A8i, lda8i, incA8i, cola8i);
    encode_B_i32_split_kernel<<<gridB, threads, 0, stream>>>(op_B, n, k, B, ldb, B8i, ldb8i, incB8i);
}

} // namespace i32
} // namespace oz1
