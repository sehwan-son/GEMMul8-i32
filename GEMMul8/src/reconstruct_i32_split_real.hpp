#pragma once

#include "common.hpp"
#include "i32_split.hpp"

namespace oz1 {
namespace real {

__global__ void accumulate_i32_to_i64_kernel(
    const size_t size,
    const int32_t *const __restrict__ in,
    int64_t *const __restrict__ out //
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] += static_cast<int64_t>(in[idx]);
}

__global__ void normalize_coeff_base256_kernel(
    const size_t sizeC,
    const int64_t *const __restrict__ coeff,
    const size_t incCoeff,
    int64_t *const __restrict__ AB //
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;

    __int128 digits[10];
#pragma unroll
    for (unsigned o = 0; o < oz1::i32::kNumOrders; ++o) {
        digits[o] = static_cast<__int128>(coeff[o * incCoeff + idx]);
    }
    digits[oz1::i32::kNumOrders] = 0;

#pragma unroll
    for (unsigned o = 0; o < oz1::i32::kNumOrders; ++o) {
        __int128 v = digits[o];
        __int128 q = v / static_cast<__int128>(oz1::i32::kSplitBase);
        __int128 r = v - q * static_cast<__int128>(oz1::i32::kSplitBase);

        if (r > 127) {
            r -= static_cast<__int128>(oz1::i32::kSplitBase);
            q += 1;
        } else if (r < -128) {
            r += static_cast<__int128>(oz1::i32::kSplitBase);
            q -= 1;
        }

        digits[o] = r;
        digits[o + 1] += q;
    }

    __int128 x = digits[oz1::i32::kNumOrders];
    for (int o = static_cast<int>(oz1::i32::kNumOrders) - 1; o >= 0; --o) {
        x = x * static_cast<__int128>(oz1::i32::kSplitBase) + digits[o];
    }
    AB[idx] = static_cast<int64_t>(x);
}

__global__ void finalize_i64_kernel(
    const size_t m,
    const size_t sizeOut,
    const int64_t *const __restrict__ AB,
    const size_t ldab,
    int64_t *const __restrict__ C,
    const size_t ldc,
    const int64_t alpha,
    const int64_t beta //
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= sizeOut) return;

    const size_t col = idx / m;
    const size_t row = idx - col * m;
    const size_t idx_ab = col * ldab + row;
    const size_t idx_c  = col * ldc + row;

    const __int128 out =
        static_cast<__int128>(alpha) * static_cast<__int128>(AB[idx_ab]) +
        static_cast<__int128>(beta) * static_cast<__int128>(C[idx_c]);
    C[idx_c] = static_cast<int64_t>(out);
}

__forceinline__ void accumulate_order_from_i32(
    const cudaStream_t stream,
    const size_t size,
    const int32_t *const in,
    int64_t *const out //
) {
    const size_t grid = (size + threads_conv_hi2mid - 1) / threads_conv_hi2mid;
    accumulate_i32_to_i64_kernel<<<grid, threads_conv_hi2mid, 0, stream>>>(size, in, out);
}

__forceinline__ void normalize_coeff_to_i64(
    const cudaStream_t stream,
    const size_t sizeC,
    const int64_t *const coeff,
    const size_t incCoeff,
    int64_t *const AB //
) {
    const size_t grid = (sizeC + threads_invscal - 1) / threads_invscal;
    normalize_coeff_base256_kernel<<<grid, threads_invscal, 0, stream>>>(sizeC, coeff, incCoeff, AB);
}

__forceinline__ void finalize_i64(
    const cudaStream_t stream,
    const size_t m, const size_t n,
    const int64_t *const AB,
    const size_t ldab,
    int64_t *const C,
    const size_t ldc,
    const int64_t alpha,
    const int64_t beta //
) {
    const size_t sizeOut = m * n;
    const size_t grid    = (sizeOut + threads_invscal - 1) / threads_invscal;
    finalize_i64_kernel<<<grid, threads_invscal, 0, stream>>>(m, sizeOut, AB, ldab, C, ldc, alpha, beta);
}

} // namespace real
} // namespace oz1
