#pragma once
#include "common.hpp"
#include "template_math.hpp"

namespace oz2 {
namespace i32 {

__forceinline__ __device__ int8_t mod_i32_256(const int32_t in) { return static_cast<int8_t>(in); }

__forceinline__ __device__ int8_t mod_i32_not256(const int32_t in, const int2 p_invp) {
    int32_t tmp = in - __mulhi(in, p_invp.y) * p_invp.x;
    tmp -= (tmp > 127) * p_invp.x;
    tmp += (tmp < -128) * p_invp.x;
    return static_cast<int8_t>(tmp);
}

__forceinline__ __device__ int8_t encode_i32_residue(const int32_t in, const unsigned mod_idx) {
    if (mod_idx == 0u) {
        return mod_i32_256(in);
    }
    const int2 p_invp = table::MODULI_I[mod_idx - 1u];
    return mod_i32_not256(in, p_invp);
}

__global__ void encode_A_i32_kernel(
    const cublasOperation_t op_A,
    const size_t m, const size_t k,
    const unsigned num_moduli,
    const int32_t *const __restrict__ A,
    const size_t lda,
    int8_t *const __restrict__ A8i,
    const size_t lda8i,
    const size_t incA8i,
    const size_t cola8i //
) {
    // Output layout is k x m (column-major indexing: out_col * lda8i + out_row)
    const auto out_col = static_cast<size_t>(blockIdx.y * blockDim.y + threadIdx.y); // [0, m)
    const auto out_row = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x); // [0, k)
    if (out_col >= cola8i || out_row >= lda8i) return;

    int32_t in = 0;
    if (out_col < m && out_row < k) {
        if (op_A == CUBLAS_OP_N) {
            // A is m x k -> store transpose into k x m
            in = __ldg(A + out_col + out_row * lda);
        } else {
            // A is k x m -> store as-is into k x m
            in = __ldg(A + out_row + out_col * lda);
        }
    }

    const size_t out_idx = out_col * lda8i + out_row;
    for (unsigned mod_idx = 0; mod_idx < num_moduli; ++mod_idx) {
        A8i[mod_idx * incA8i + out_idx] = encode_i32_residue(in, mod_idx);
    }
}

__global__ void encode_B_i32_kernel(
    const cublasOperation_t op_B,
    const size_t n, const size_t k,
    const unsigned num_moduli,
    const int32_t *const __restrict__ B,
    const size_t ldb,
    int8_t *const __restrict__ B8i,
    const size_t ldb8i,
    const size_t incB8i //
) {
    // Output layout is k x n (column-major indexing: out_col * ldb8i + out_row)
    const auto out_col = static_cast<size_t>(blockIdx.y * blockDim.y + threadIdx.y); // [0, n)
    const auto out_row = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x); // [0, k)
    if (out_col >= n || out_row >= ldb8i) return;

    int32_t in = 0;
    if (out_row < k) {
        if (op_B == CUBLAS_OP_N) {
            // B is k x n -> store as-is into k x n
            in = __ldg(B + out_row + out_col * ldb);
        } else {
            // B is n x k -> store transpose into k x n
            in = __ldg(B + out_col + out_row * ldb);
        }
    }

    const size_t out_idx = out_col * ldb8i + out_row;
    for (unsigned mod_idx = 0; mod_idx < num_moduli; ++mod_idx) {
        B8i[mod_idx * incB8i + out_idx] = encode_i32_residue(in, mod_idx);
    }
}

__forceinline__ void encode(
    const cublasOperation_t op_A, const cublasOperation_t op_B,
    const size_t m, const size_t n, const size_t k,
    const unsigned num_moduli,
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
    encode_A_i32_kernel<<<gridA, threads>>>(op_A, m, k, num_moduli, A, lda, A8i, lda8i, incA8i, cola8i);
    encode_B_i32_kernel<<<gridB, threads>>>(op_B, n, k, num_moduli, B, ldb, B8i, ldb8i, incB8i);
}

} // namespace i32
} // namespace oz2
