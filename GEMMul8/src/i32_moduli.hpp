#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace oz2 {
namespace i32 {

inline constexpr unsigned kMinNumModuli = 9u;
inline constexpr unsigned kMaxNumModuli = 20u;
inline constexpr size_t kMaxK           = (1u << 17u);

inline constexpr std::array<int32_t, kMaxNumModuli> kModuli = {
    256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
    223, 217, 211, 199, 197, 193, 191, 181, 179, 173,
};

__host__ __device__ __forceinline__ constexpr int32_t get_modulus(const unsigned idx) {
    switch (idx) {
    case 0: return 256;
    case 1: return 255;
    case 2: return 253;
    case 3: return 251;
    case 4: return 247;
    case 5: return 241;
    case 6: return 239;
    case 7: return 233;
    case 8: return 229;
    case 9: return 227;
    case 10: return 223;
    case 11: return 217;
    case 12: return 211;
    case 13: return 199;
    case 14: return 197;
    case 15: return 193;
    case 16: return 191;
    case 17: return 181;
    case 18: return 179;
    case 19: return 173;
    default: return 1;
    }
}

__host__ __device__ __forceinline__ int32_t mod_nonneg(const int64_t in, const int32_t mod) {
    int32_t r = static_cast<int32_t>(in % static_cast<int64_t>(mod));
    if (r < 0) r += mod;
    return r;
}

__host__ __device__ __forceinline__ int8_t mod_centered_i8(const int64_t in, const int32_t mod) {
    int32_t r = static_cast<int32_t>(in % static_cast<int64_t>(mod));
    if (r > 127) r -= mod;
    if (r < -128) r += mod;
    return static_cast<int8_t>(r);
}

constexpr __int128 prefix_product(const unsigned num_moduli) {
    __int128 out = 1;
    for (unsigned i = 0; i < num_moduli && i < kMaxNumModuli; ++i) {
        out *= static_cast<__int128>(kModuli[i]);
    }
    return out;
}

constexpr unsigned required_num_moduli_for_bounds(const uint64_t max_abs_a, const uint64_t max_abs_b, const size_t k) {
    if (max_abs_a == 0u || max_abs_b == 0u || k == 0u) return 1u;
    const __int128 max_abs = static_cast<__int128>(max_abs_a) * static_cast<__int128>(max_abs_b) * static_cast<__int128>(k);
    const __int128 need    = max_abs * 2 + 1;
    __int128 p             = 1;
    for (unsigned i = 0; i < kMaxNumModuli; ++i) {
        p *= static_cast<__int128>(kModuli[i]);
        if (p >= need) return i + 1u;
    }
    return kMaxNumModuli + 1u;
}

constexpr size_t max_k_for_num_moduli(const unsigned num_moduli, const uint64_t max_abs_a, const uint64_t max_abs_b) {
    if (max_abs_a == 0u || max_abs_b == 0u) return std::numeric_limits<size_t>::max();
    const __int128 denom = static_cast<__int128>(max_abs_a) * static_cast<__int128>(max_abs_b);
    const __int128 range = prefix_product(num_moduli) / 2;
    if (denom <= 0 || range <= 0) return 0;
    const __int128 kmax = (range - 1) / denom;
    if (kmax <= 0) return 0;
    if (kmax >= static_cast<__int128>(std::numeric_limits<size_t>::max())) {
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(kmax);
}

inline std::string int128_to_string(__int128 v) {
    if (v == 0) return "0";
    bool neg = false;
    if (v < 0) {
        neg = true;
        v   = -v;
    }
    std::string out;
    while (v > 0) {
        const int digit = static_cast<int>(v % 10);
        out.push_back(static_cast<char>('0' + digit));
        v /= 10;
    }
    if (neg) out.push_back('-');
    std::reverse(out.begin(), out.end());
    return out;
}

inline void validate_num_moduli_or_throw(const unsigned num_moduli, const char *const where) {
    if (num_moduli < kMinNumModuli || num_moduli > kMaxNumModuli) {
        throw std::invalid_argument(
            std::string(where) + ": num_moduli must satisfy " +
            std::to_string(kMinNumModuli) + " <= num_moduli <= " + std::to_string(kMaxNumModuli) + ".");
    }
}

inline void validate_k_or_throw(const size_t k, const char *const where) {
    if (k > kMaxK) {
        throw std::invalid_argument(
            std::string(where) + ": k must satisfy k <= " + std::to_string(kMaxK) + ".");
    }
}

} // namespace i32
} // namespace oz2
