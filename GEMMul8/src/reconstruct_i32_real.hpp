#pragma once

#include "common.hpp"
#include "i32_moduli.hpp"

namespace oz2 {
namespace i32 {

inline constexpr unsigned crt_moduli_used = kMinNumModuli;

__forceinline__ __device__ int32_t modinv_i32(int32_t a, const int32_t mod) {
    int32_t t     = 0;
    int32_t new_t = 1;
    int32_t r     = mod;
    int32_t new_r = a % mod;

    while (new_r != 0) {
        const int32_t q = r / new_r;

        const int32_t tmp_t = t - q * new_t;
        t                   = new_t;
        new_t               = tmp_t;

        const int32_t tmp_r = r - q * new_r;
        r                   = new_r;
        new_r               = tmp_r;
    }

    while (t < 0) t += mod;
    t %= mod;
    return t;
}

template <typename TCtmp>
__forceinline__ __device__ int32_t get_nonneg_residue(
    const TCtmp *const Ctmp,
    const size_t incCtmp,
    const unsigned i //
);

template <>
__forceinline__ __device__ int32_t get_nonneg_residue<int8_t>(
    const int8_t *const Ctmp,
    const size_t incCtmp,
    const unsigned i //
) {
    const int32_t mod = get_modulus(i);
    int32_t r         = static_cast<int32_t>(Ctmp[i * incCtmp]);
    if (r < 0) r += mod;
    return r;
}

template <>
__forceinline__ __device__ int32_t get_nonneg_residue<int32_t>(
    const int32_t *const Ctmp,
    const size_t incCtmp,
    const unsigned i //
) {
    const int32_t mod = get_modulus(i);
    return mod_nonneg(static_cast<int64_t>(Ctmp[i * incCtmp]), mod);
}

template <typename TCtmp>
__global__ void reconstruct_kernel_general(
    const size_t m,
    const size_t sizeC,
    const size_t incCtmp,
    const TCtmp *const __restrict__ Ctmp,
    const size_t ldctmp,
    int64_t *const __restrict__ C,
    const size_t ldc,
    const int64_t alpha,
    const int64_t beta //
) {
    const auto idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= sizeC) return;

    const auto col = idx / m;
    const auto row = idx - col * m;

    const auto mem_idx       = col * ldctmp + row;
    const TCtmp *const inptr = Ctmp + mem_idx;

    int32_t residues[crt_moduli_used];
#pragma unroll
    for (unsigned i = 0; i < crt_moduli_used; ++i) {
        residues[i] = get_nonneg_residue<TCtmp>(inptr, incCtmp, i);
    }

    __int128 x = residues[0];
    __int128 M = get_modulus(0u);

#pragma unroll
    for (unsigned i = 1; i < crt_moduli_used; ++i) {
        const int32_t mod = get_modulus(i);
        const int32_t xi  = mod_nonneg(static_cast<int64_t>(x % mod), mod);
        int32_t diff      = residues[i] - xi;
        diff %= mod;
        if (diff < 0) diff += mod;

        const int32_t M_mod = mod_nonneg(static_cast<int64_t>(M % mod), mod);
        const int32_t inv   = modinv_i32(M_mod, mod);
        const int32_t t     = static_cast<int32_t>((static_cast<int64_t>(diff) * inv) % mod);

        x += M * static_cast<__int128>(t);
        M *= static_cast<__int128>(mod);
    }

    const __int128 half = M / 2;
    if (x > half) x -= M;

    const auto idxC  = col * ldc + row;
    const int64_t AB = static_cast<int64_t>(x);
    const int64_t C0 = C[idxC];
    const __int128 out =
        static_cast<__int128>(alpha) * static_cast<__int128>(AB) +
        static_cast<__int128>(beta) * static_cast<__int128>(C0);
    C[idxC] = static_cast<int64_t>(out);
}

template <typename TCtmp, int ALPHA, int BETA>
__global__ void reconstruct_kernel_special(
    const size_t m,
    const size_t sizeC,
    const size_t incCtmp,
    const TCtmp *const __restrict__ Ctmp,
    const size_t ldctmp,
    int64_t *const __restrict__ C,
    const size_t ldc //
) {
    const auto idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= sizeC) return;

    const auto col = idx / m;
    const auto row = idx - col * m;

    const auto mem_idx       = col * ldctmp + row;
    const TCtmp *const inptr = Ctmp + mem_idx;

    int32_t residues[crt_moduli_used];
#pragma unroll
    for (unsigned i = 0; i < crt_moduli_used; ++i) {
        residues[i] = get_nonneg_residue<TCtmp>(inptr, incCtmp, i);
    }

    __int128 x = residues[0];
    __int128 M = get_modulus(0u);

#pragma unroll
    for (unsigned i = 1; i < crt_moduli_used; ++i) {
        const int32_t mod = get_modulus(i);
        const int32_t xi  = mod_nonneg(static_cast<int64_t>(x % mod), mod);
        int32_t diff      = residues[i] - xi;
        diff %= mod;
        if (diff < 0) diff += mod;

        const int32_t M_mod = mod_nonneg(static_cast<int64_t>(M % mod), mod);
        const int32_t inv   = modinv_i32(M_mod, mod);
        const int32_t t     = static_cast<int32_t>((static_cast<int64_t>(diff) * inv) % mod);

        x += M * static_cast<__int128>(t);
        M *= static_cast<__int128>(mod);
    }

    const __int128 half = M / 2;
    if (x > half) x -= M;

    const auto idxC  = col * ldc + row;
    const int64_t AB = static_cast<int64_t>(x);
    if constexpr (ALPHA == 1 && BETA == 0) {
        C[idxC] = AB;
    } else if constexpr (ALPHA == 1 && BETA == 1) {
        const __int128 out = static_cast<__int128>(AB) + static_cast<__int128>(C[idxC]);
        C[idxC]            = static_cast<int64_t>(out);
    } else if constexpr (ALPHA == -1 && BETA == 0) {
        const __int128 out = -static_cast<__int128>(AB);
        C[idxC]            = static_cast<int64_t>(out);
    } else if constexpr (ALPHA == -1 && BETA == 1) {
        const __int128 out = static_cast<__int128>(C[idxC]) - static_cast<__int128>(AB);
        C[idxC]            = static_cast<int64_t>(out);
    }
}

template <typename TCtmp>
__inline__ void reconstruct(
    const cudaStream_t stream,
    const size_t m, const size_t n,
    const TCtmp *const Ctmp,
    const size_t ldctmp,
    const size_t incCtmp,
    int64_t *const C,
    const size_t ldc,
    const int64_t alpha,
    const int64_t beta //
) {
    const size_t sizeC = m * n;
    const size_t grid  = (sizeC + threads_invscal - 1) / threads_invscal;

    if (alpha == 1) {
        if (beta == 0) {
            reconstruct_kernel_special<TCtmp, 1, 0><<<grid, threads_invscal, 0, stream>>>(m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc);
            return;
        }
        if (beta == 1) {
            reconstruct_kernel_special<TCtmp, 1, 1><<<grid, threads_invscal, 0, stream>>>(m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc);
            return;
        }
    } else if (alpha == -1) {
        if (beta == 0) {
            reconstruct_kernel_special<TCtmp, -1, 0><<<grid, threads_invscal, 0, stream>>>(m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc);
            return;
        }
        if (beta == 1) {
            reconstruct_kernel_special<TCtmp, -1, 1><<<grid, threads_invscal, 0, stream>>>(m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc);
            return;
        }
    }

    reconstruct_kernel_general<TCtmp><<<grid, threads_invscal, 0, stream>>>(m, sizeC, incCtmp, Ctmp, ldctmp, C, ldc, alpha, beta);
}

} // namespace i32
} // namespace oz2
