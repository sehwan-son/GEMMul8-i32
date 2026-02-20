#pragma once

#include "common.hpp"
#include "i32_moduli.hpp"

namespace oz2 {
namespace real {

__global__ void conv_32i_2_8i_kernel(
    const size_t sizeC_4,
    const int4 *const __restrict__ C32x4,
    char4 *const __restrict__ C8x4,
    const int32_t mod //
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= sizeC_4) return;

    const int4 in = C32x4[idx];
    char4 out;
    out.x = oz2::i32::mod_centered_i8(in.x, mod);
    out.y = oz2::i32::mod_centered_i8(in.y, mod);
    out.z = oz2::i32::mod_centered_i8(in.z, mod);
    out.w = oz2::i32::mod_centered_i8(in.w, mod);
    C8x4[idx] = out;
}

__forceinline__ void conv_32i_2_8i(
    const cudaStream_t stream,
    const unsigned mod_idx,
    const size_t sizeC_4,
    const int32_t *const C32,
    int8_t *const C8 //
) {
    const int32_t mod = oz2::i32::get_modulus(mod_idx);
    const size_t grid = (sizeC_4 + threads_conv_hi2mid - 1) / threads_conv_hi2mid;
    conv_32i_2_8i_kernel<<<grid, threads_conv_hi2mid, 0, stream>>>(
        sizeC_4,
        reinterpret_cast<const int4 *>(C32),
        reinterpret_cast<char4 *>(C8),
        mod);
}

} // namespace real
} // namespace oz2
