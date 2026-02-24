#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace oz1 {
namespace i32 {

inline constexpr unsigned kFixedNumModuli = 5u;
inline constexpr unsigned kNumSplitDigits = 5u;
inline constexpr unsigned kNumOrders      = 2u * kNumSplitDigits - 1u; // 0..8
inline constexpr int64_t kSplitBase       = 256;

inline void validate_num_moduli_or_throw(const unsigned num_moduli, const char *const where) {
    if (num_moduli != kFixedNumModuli) {
        throw std::invalid_argument(
            std::string(where) + ": OZAKI1_SPLIT requires num_moduli == " + std::to_string(kFixedNumModuli) + ".");
    }
}

// Signed base-256 expansion with fixed 5 digits.
// x = sum_{i=0..4} d_i * 256^i, d_i in [-128, 127].
__host__ __device__ __forceinline__ void split_i32_to_i8_digits(
    const int32_t x,
    int8_t out[kNumSplitDigits] //
) {
    int64_t rem = static_cast<int64_t>(x);
#pragma unroll
    for (unsigned i = 0; i + 1u < kNumSplitDigits; ++i) {
        int64_t q = rem / kSplitBase;
        int64_t r = rem - q * kSplitBase;
        if (r > 127) {
            r -= kSplitBase;
            q += 1;
        } else if (r < -128) {
            r += kSplitBase;
            q -= 1;
        }
        out[i] = static_cast<int8_t>(r);
        rem    = q;
    }
    out[kNumSplitDigits - 1u] = static_cast<int8_t>(rem);
}

} // namespace i32
} // namespace oz1
