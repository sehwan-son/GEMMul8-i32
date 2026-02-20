#include "../include/gemmul8.hpp"
#include "../src/i32_moduli.hpp"
#include "getWatt.hpp"
#include "self_hipify.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Shape3 = std::tuple<size_t, size_t, size_t>;
using OpPair = std::pair<cublasOperation_t, cublasOperation_t>;

enum class CpuCheckMode { OFF,
                          SAMPLE,
                          FULL };

void check_cuda(const cudaError_t status, const char *const where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
    }
}

void check_cublas(const cublasStatus_t status, const char *const where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(where) + ": cublas status=" + std::to_string(static_cast<int>(status)));
    }
}

std::string sanitize_token(std::string text) {
    for (char &ch : text) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    return text;
}

std::string make_timestamp() {
    const std::time_t t = std::time(nullptr);
    std::tm tm_value {};
#if defined(_WIN32)
    localtime_s(&tm_value, &t);
#else
    localtime_r(&t, &tm_value);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_value);
    return std::string(buf);
}

char op_to_char(const cublasOperation_t op) {
    return (op == CUBLAS_OP_T) ? 'T' : 'N';
}

std::string op_pair_to_string(const cublasOperation_t op_A, const cublasOperation_t op_B) {
    std::string out;
    out.push_back(op_to_char(op_A));
    out.push_back(op_to_char(op_B));
    return out;
}

double gemm_int_ops(const size_t m, const size_t n, const size_t k) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
}

double compute_gflops(const double ops, const double elapsed_ms) {
    if (elapsed_ms <= 0.0) return 0.0;
    return ops / (elapsed_ms * 1.0e6);
}

std::string trim_copy(const std::string &text) {
    const size_t begin = text.find_first_not_of(" \t");
    if (begin == std::string::npos) return "";
    const size_t end = text.find_last_not_of(" \t");
    return text.substr(begin, end - begin + 1u);
}

int parse_int_arg(const std::string &text, const char *const name) {
    try {
        size_t parsed = 0;
        const int value = std::stoi(text, &parsed);
        if (parsed != text.size()) throw std::invalid_argument("trailing");
        return value;
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string("invalid ") + name + ": '" + text + "'");
    }
}

double parse_double_arg(const std::string &text, const char *const name) {
    try {
        size_t parsed = 0;
        const double value = std::stod(text, &parsed);
        if (parsed != text.size()) throw std::invalid_argument("trailing");
        return value;
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string("invalid ") + name + ": '" + text + "'");
    }
}

size_t parse_positive_size_token(const std::string &token, const char *const name) {
    try {
        size_t parsed = 0;
        const unsigned long long value = std::stoull(token, &parsed);
        if (parsed != token.size() || value == 0ULL) throw std::invalid_argument("bad");
        return static_cast<size_t>(value);
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string("invalid ") + name + " token: '" + token + "'");
    }
}

std::vector<size_t> parse_size_list(const std::string &text, const char *const name = "--sizes") {
    std::vector<size_t> out;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        const size_t end = (comma == std::string::npos) ? text.size() : comma;
        const std::string token = trim_copy(text.substr(begin, end - begin));
        if (token.empty()) {
            throw std::invalid_argument(std::string("invalid ") + name + ": empty token in '" + text + "'");
        }
        out.push_back(parse_positive_size_token(token, name));

        if (comma == std::string::npos) break;
        begin = comma + 1u;
    }
    if (out.empty()) {
        throw std::invalid_argument(std::string("invalid ") + name + ": no values");
    }
    return out;
}

std::vector<unsigned> parse_unsigned_list(const std::string &text, const char *const name) {
    std::vector<unsigned> out;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        const size_t end = (comma == std::string::npos) ? text.size() : comma;
        const std::string token = trim_copy(text.substr(begin, end - begin));
        if (token.empty()) {
            throw std::invalid_argument(std::string("invalid ") + name + ": empty token in '" + text + "'");
        }
        try {
            size_t parsed = 0;
            const unsigned long long value = std::stoull(token, &parsed);
            if (parsed != token.size() || value == 0ULL) throw std::invalid_argument("bad");
            out.push_back(static_cast<unsigned>(value));
        } catch (const std::exception &) {
            throw std::invalid_argument(std::string("invalid ") + name + " token: '" + token + "'");
        }
        if (comma == std::string::npos) break;
        begin = comma + 1u;
    }
    if (out.empty()) {
        throw std::invalid_argument(std::string("invalid ") + name + ": no values");
    }
    return out;
}

std::vector<Shape3> make_square_shapes(const std::vector<size_t> &sizes) {
    std::vector<Shape3> out;
    out.reserve(sizes.size());
    for (const size_t n : sizes) {
        out.emplace_back(n, n, n);
    }
    return out;
}

std::vector<Shape3> parse_shape_list(const std::string &text) {
    std::vector<Shape3> out;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        const size_t end = (comma == std::string::npos) ? text.size() : comma;
        const std::string token = trim_copy(text.substr(begin, end - begin));
        if (token.empty()) {
            throw std::invalid_argument("invalid --shapes: empty token in '" + text + "'");
        }

        const size_t x0 = token.find('x');
        const size_t x1 = token.find('x', x0 == std::string::npos ? x0 : x0 + 1u);
        if (x0 == std::string::npos || x1 == std::string::npos || token.find('x', x1 + 1u) != std::string::npos) {
            throw std::invalid_argument("invalid --shapes token: '" + token + "' (expected mxnxk)");
        }

        const std::string m_txt = trim_copy(token.substr(0, x0));
        const std::string n_txt = trim_copy(token.substr(x0 + 1u, x1 - x0 - 1u));
        const std::string k_txt = trim_copy(token.substr(x1 + 1u));
        if (m_txt.empty() || n_txt.empty() || k_txt.empty()) {
            throw std::invalid_argument("invalid --shapes token: '" + token + "' (empty dim)");
        }

        const size_t m = parse_positive_size_token(m_txt, "--shapes");
        const size_t n = parse_positive_size_token(n_txt, "--shapes");
        const size_t k = parse_positive_size_token(k_txt, "--shapes");
        out.emplace_back(m, n, k);

        if (comma == std::string::npos) break;
        begin = comma + 1u;
    }
    if (out.empty()) {
        throw std::invalid_argument("invalid --shapes: no values");
    }
    return out;
}

OpPair parse_op_pair_token(const std::string &text) {
    if (text.size() != 2u) {
        throw std::invalid_argument("invalid --ops token: '" + text + "' (expected NN/NT/TN/TT)");
    }
    const char a = static_cast<char>(std::toupper(static_cast<unsigned char>(text[0])));
    const char b = static_cast<char>(std::toupper(static_cast<unsigned char>(text[1])));
    if ((a != 'N' && a != 'T') || (b != 'N' && b != 'T')) {
        throw std::invalid_argument("invalid --ops token: '" + text + "' (expected NN/NT/TN/TT)");
    }
    return {a == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N, b == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N};
}

std::vector<OpPair> parse_op_pair_list(const std::string &text) {
    std::vector<OpPair> out;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        const size_t end = (comma == std::string::npos) ? text.size() : comma;
        const std::string token = trim_copy(text.substr(begin, end - begin));
        if (token.empty()) {
            throw std::invalid_argument("invalid --ops: empty token in '" + text + "'");
        }
        out.push_back(parse_op_pair_token(token));
        if (comma == std::string::npos) break;
        begin = comma + 1u;
    }
    if (out.empty()) {
        throw std::invalid_argument("invalid --ops: no values");
    }
    return out;
}

CpuCheckMode parse_cpu_mode(const std::string &text) {
    const std::string t = text;
    if (t == "off") return CpuCheckMode::OFF;
    if (t == "sample") return CpuCheckMode::SAMPLE;
    if (t == "full") return CpuCheckMode::FULL;
    throw std::invalid_argument("invalid --cpu-check: '" + text + "' (expected off|sample|full)");
}

const char *cpu_mode_name(const CpuCheckMode mode) {
    switch (mode) {
    case CpuCheckMode::OFF: return "off";
    case CpuCheckMode::SAMPLE: return "sample";
    case CpuCheckMode::FULL: return "full";
    }
    return "unknown";
}

void validate_moduli_list(const std::vector<unsigned> &moduli) {
    for (const auto value : moduli) {
        if (value < oz2::i32::kMinNumModuli || value > oz2::i32::kMaxNumModuli) {
            throw std::invalid_argument(
                "invalid num_moduli: " + std::to_string(value) +
                " (valid range: " + std::to_string(oz2::i32::kMinNumModuli) +
                ".." + std::to_string(oz2::i32::kMaxNumModuli) + ")");
        }
    }
}

void print_usage(const char *const prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " [iters] [warmup] [sizes_csv] [moduli_csv]\n"
        << "  " << prog << " --iters <int> --warmup <int>\n"
        << "           [--sizes <n1,n2,...> | --shapes <m1xn1xk1,m2xn2xk2,...>]\n"
        << "           [--moduli <9,10,...,20>] [--ops <NN,NT,TN,TT>] [--watt]\n"
        << "           [--with-int8-baseline]\n"
        << "           [--phi <double>] [--scale-exp <int>] [--input-bound <int>]\n"
        << "           [--cpu-check <off|sample|full>] [--cpu-samples <int>]\n"
        << "Examples:\n"
        << "  " << prog << " 5 2\n"
        << "  " << prog << " --iters 5 --warmup 2 --sizes 2048,4096 --ops NN\n"
        << "  " << prog << " --iters 5 --warmup 2 --sizes 2048,4096 --ops NN --with-int8-baseline\n"
        << "  " << prog << " --iters 5 --warmup 2 --shapes 1024x2048x4096 --moduli 9 --phi 0.5 --scale-exp 10\n";
}

struct CliConfig {
    int iters = 5;
    int warmup = 2;
    std::vector<Shape3> shapes = make_square_shapes({256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u});
    std::vector<unsigned> moduli = {9u};
    bool moduli_explicit = false;
    std::vector<OpPair> ops = {
        {CUBLAS_OP_N, CUBLAS_OP_N},
    };
    bool enable_watt = false;
    double phi = 0.5;
    int scale_exp = 10;
    int input_bound = 1024;
    bool with_int8_baseline = false;
    CpuCheckMode cpu_check_mode = CpuCheckMode::SAMPLE;
    int cpu_samples = 64;
};

CliConfig parse_cli(const int argc, char **argv) {
    CliConfig cfg;
    if (argc <= 1) return cfg;

    bool has_flag = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "-h" || arg.rfind("--", 0u) == 0u) {
            has_flag = true;
            break;
        }
    }

    if (!has_flag) {
        if (argc >= 2) cfg.iters = std::max(1, parse_int_arg(argv[1], "iters"));
        if (argc >= 3) cfg.warmup = std::max(0, parse_int_arg(argv[2], "warmup"));
        if (argc >= 4) cfg.shapes = make_square_shapes(parse_size_list(argv[3], "sizes_csv"));
        if (argc >= 5) {
            cfg.moduli = parse_unsigned_list(argv[4], "moduli_csv");
            cfg.moduli_explicit = true;
        }
        if (argc >= 6) {
            throw std::invalid_argument("too many positional args. Use --help for usage.");
        }
        validate_moduli_list(cfg.moduli);
        return cfg;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--iters") {
            if (i + 1 >= argc) throw std::invalid_argument("--iters requires a value");
            cfg.iters = std::max(1, parse_int_arg(argv[++i], "--iters"));
            continue;
        }
        if (arg == "--warmup") {
            if (i + 1 >= argc) throw std::invalid_argument("--warmup requires a value");
            cfg.warmup = std::max(0, parse_int_arg(argv[++i], "--warmup"));
            continue;
        }
        if (arg == "--sizes") {
            if (i + 1 >= argc) throw std::invalid_argument("--sizes requires a value");
            cfg.shapes = make_square_shapes(parse_size_list(argv[++i], "--sizes"));
            continue;
        }
        if (arg == "--shapes") {
            if (i + 1 >= argc) throw std::invalid_argument("--shapes requires a value");
            cfg.shapes = parse_shape_list(argv[++i]);
            continue;
        }
        if (arg == "--moduli") {
            if (i + 1 >= argc) throw std::invalid_argument("--moduli requires a value");
            cfg.moduli = parse_unsigned_list(argv[++i], "--moduli");
            cfg.moduli_explicit = true;
            continue;
        }
        if (arg == "--ops") {
            if (i + 1 >= argc) throw std::invalid_argument("--ops requires a value");
            cfg.ops = parse_op_pair_list(argv[++i]);
            continue;
        }
        if (arg == "--watt") {
            cfg.enable_watt = true;
            continue;
        }
        if (arg == "--with-int8-baseline") {
            cfg.with_int8_baseline = true;
            continue;
        }
        if (arg == "--phi") {
            if (i + 1 >= argc) throw std::invalid_argument("--phi requires a value");
            cfg.phi = parse_double_arg(argv[++i], "--phi");
            continue;
        }
        if (arg == "--scale-exp") {
            if (i + 1 >= argc) throw std::invalid_argument("--scale-exp requires a value");
            cfg.scale_exp = parse_int_arg(argv[++i], "--scale-exp");
            continue;
        }
        if (arg == "--input-bound") {
            if (i + 1 >= argc) throw std::invalid_argument("--input-bound requires a value");
            cfg.input_bound = std::max(1, parse_int_arg(argv[++i], "--input-bound"));
            continue;
        }
        if (arg == "--cpu-check") {
            if (i + 1 >= argc) throw std::invalid_argument("--cpu-check requires a value");
            cfg.cpu_check_mode = parse_cpu_mode(argv[++i]);
            continue;
        }
        if (arg == "--cpu-samples") {
            if (i + 1 >= argc) throw std::invalid_argument("--cpu-samples requires a value");
            cfg.cpu_samples = std::max(1, parse_int_arg(argv[++i], "--cpu-samples"));
            continue;
        }
        throw std::invalid_argument("unknown option: " + arg);
    }

    validate_moduli_list(cfg.moduli);
    return cfg;
}

void fill_matrix_exponent_i32(
    std::vector<int32_t> &mat,
    const size_t rows,
    const size_t cols,
    const size_t ld,
    std::mt19937 &rng,
    const double phi,
    const int scale_exp,
    const int32_t input_bound //
) {
    std::uniform_real_distribution<double> uni(-0.5, 0.5);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const double scale = std::ldexp(1.0, scale_exp);
    const double lim = static_cast<double>(input_bound);

    std::fill(mat.begin(), mat.end(), 0);
    for (size_t col = 0; col < cols; ++col) {
        for (size_t row = 0; row < rows; ++row) {
            const double x = uni(rng) * std::exp(gauss(rng) * phi) * scale;
            const double clamped = std::max(-lim, std::min(lim, x));
            mat[col * ld + row] = static_cast<int32_t>(std::llround(clamped));
        }
    }
}

uint64_t max_abs_i32(const std::vector<int32_t> &x) {
    uint64_t out = 0;
    for (const int32_t v : x) {
        const int64_t vv = static_cast<int64_t>(v);
        const uint64_t a = static_cast<uint64_t>(vv < 0 ? -vv : vv);
        out = std::max(out, a);
    }
    return out;
}

void cast_i32_to_i8(const std::vector<int32_t> &in, std::vector<int8_t> &out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = static_cast<int8_t>(in[i]);
    }
}

cublasStatus_t probe_cublas_int32(cublasHandle_t handle) {
    int32_t *dA = nullptr;
    int32_t *dB = nullptr;
    int32_t *dC = nullptr;
    check_cuda(cudaMalloc(&dA, sizeof(int32_t)), "probe cudaMalloc dA");
    check_cuda(cudaMalloc(&dB, sizeof(int32_t)), "probe cudaMalloc dB");
    check_cuda(cudaMalloc(&dC, sizeof(int32_t)), "probe cudaMalloc dC");

    const int32_t alpha = 1;
    const int32_t beta = 0;
    const cublasStatus_t status = cublasGemmEx(handle,
                                               CUBLAS_OP_N,
                                               CUBLAS_OP_N,
                                               1,
                                               1,
                                               1,
                                               &alpha,
                                               dA,
                                               CUDA_R_32I,
                                               1,
                                               dB,
                                               CUDA_R_32I,
                                               1,
                                               &beta,
                                               dC,
                                               CUDA_R_32I,
                                               1,
                                               CUBLAS_COMPUTE_32I,
                                               CUBLAS_GEMM_DEFAULT);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return status;
}

__global__ void gemm_i32_exact_kernel(
    const int trans_A,
    const int trans_B,
    const size_t m,
    const size_t n,
    const size_t k,
    const int32_t *const __restrict__ A,
    const size_t lda,
    const int32_t *const __restrict__ B,
    const size_t ldb,
    int64_t *const __restrict__ C,
    const size_t ldc //
) {
    const size_t row = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const size_t col = static_cast<size_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= m || col >= n) return;

    int64_t sum = 0;
    for (size_t t = 0; t < k; ++t) {
        const int32_t a = (trans_A == 0) ? A[t * lda + row] : A[row * lda + t];
        const int32_t b = (trans_B == 0) ? B[col * ldb + t] : B[t * ldb + col];
        sum += static_cast<int64_t>(a) * static_cast<int64_t>(b);
    }

    C[col * ldc + row] = sum;
}

double bench_exact_i32_i32_i64_ms(
    const cublasOperation_t op_A,
    const cublasOperation_t op_B,
    const size_t m,
    const size_t n,
    const size_t k,
    const int32_t *const dA,
    const size_t lda,
    const int32_t *const dB,
    const size_t ldb,
    int64_t *const dC64,
    const size_t ldc,
    const int warmup,
    const int iters //
) {
    const dim3 threads(16u, 16u);
    const dim3 grid(
        static_cast<unsigned>((m + threads.x - 1u) / threads.x),
        static_cast<unsigned>((n + threads.y - 1u) / threads.y));

    const int trans_A = (op_A == CUBLAS_OP_T) ? 1 : 0;
    const int trans_B = (op_B == CUBLAS_OP_T) ? 1 : 0;

    for (int i = 0; i < warmup; ++i) {
        gemm_i32_exact_kernel<<<grid, threads>>>(
            trans_A, trans_B, m, n, k,
            dA, lda, dB, ldb, dC64, ldc);
    }
    check_cuda(cudaGetLastError(), "exact_i32_i32_i64 warmup launch");
    check_cuda(cudaDeviceSynchronize(), "exact_i32_i32_i64 warmup sync");

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate exact start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate exact stop");

    check_cuda(cudaEventRecord(start), "cudaEventRecord exact start");
    for (int i = 0; i < iters; ++i) {
        gemm_i32_exact_kernel<<<grid, threads>>>(
            trans_A, trans_B, m, n, k,
            dA, lda, dB, ldb, dC64, ldc);
    }
    check_cuda(cudaGetLastError(), "exact_i32_i32_i64 iter launch");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord exact stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize exact stop");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime exact");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(elapsed_ms) / static_cast<double>(iters);
}

double bench_cublas_i8_ms(
    cublasHandle_t handle,
    const cublasOperation_t op_A,
    const cublasOperation_t op_B,
    const size_t m,
    const size_t n,
    const size_t k,
    const int8_t *dA8,
    const size_t lda,
    const int8_t *dB8,
    const size_t ldb,
    int32_t *dC32,
    const size_t ldc,
    const int warmup,
    const int iters //
) {
    constexpr int32_t alpha = 1;
    constexpr int32_t beta = 0;

    for (int i = 0; i < warmup; ++i) {
        check_cublas(cublasGemmEx(handle,
                                  op_A,
                                  op_B,
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  dA8,
                                  CUDA_R_8I,
                                  lda,
                                  dB8,
                                  CUDA_R_8I,
                                  ldb,
                                  &beta,
                                  dC32,
                                  CUDA_R_32I,
                                  ldc,
                                  CUBLAS_COMPUTE_32I,
                                  CUBLAS_GEMM_DEFAULT),
                    "bench cublas int8 warmup");
    }
    check_cuda(cudaDeviceSynchronize(), "bench cublas int8 warmup sync");

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iters; ++i) {
        check_cublas(cublasGemmEx(handle,
                                  op_A,
                                  op_B,
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  dA8,
                                  CUDA_R_8I,
                                  lda,
                                  dB8,
                                  CUDA_R_8I,
                                  ldb,
                                  &beta,
                                  dC32,
                                  CUDA_R_32I,
                                  ldc,
                                  CUBLAS_COMPUTE_32I,
                                  CUBLAS_GEMM_DEFAULT),
                    "bench cublas int8 iter");
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(elapsed_ms) / static_cast<double>(iters);
}

struct AccuracyStats {
    double max_abs_error = 0.0;
    double med_abs_error = 0.0;
    size_t mismatch_count = 0u;
    bool exact_match = true;
};

std::vector<int64_t> copy_i64_from_device(const int64_t *const d_src, const size_t count) {
    std::vector<int64_t> out(count);
    check_cuda(cudaMemcpy(out.data(), d_src, sizeof(int64_t) * count, cudaMemcpyDeviceToHost), "cudaMemcpy i64 device->host");
    return out;
}

AccuracyStats compare_i64_outputs(
    const std::vector<int64_t> &got,
    const std::vector<int64_t> &ref //
) {
    if (got.size() != ref.size()) {
        throw std::runtime_error("compare_i64_outputs: size mismatch");
    }

    AccuracyStats out;
    uint64_t max_abs_u64 = 0u;
    for (size_t i = 0; i < got.size(); ++i) {
        const int64_t g = got[i];
        const int64_t r = ref[i];
        if (g != r) {
            out.mismatch_count++;
            const uint64_t abs_diff = (g >= r)
                                          ? static_cast<uint64_t>(g - r)
                                          : static_cast<uint64_t>(r - g);
            max_abs_u64 = std::max(max_abs_u64, abs_diff);
        }
    }
    out.exact_match = (out.mismatch_count == 0u);
    out.max_abs_error = static_cast<double>(max_abs_u64);
    out.med_abs_error = out.exact_match ? 0.0 : static_cast<double>(max_abs_u64);
    return out;
}

int64_t cpu_dot(
    const std::vector<int32_t> &A,
    const std::vector<int32_t> &B,
    const cublasOperation_t op_A,
    const cublasOperation_t op_B,
    const size_t lda,
    const size_t ldb,
    const size_t row,
    const size_t col,
    const size_t k //
) {
    int64_t sum = 0;
    for (size_t t = 0; t < k; ++t) {
        const int32_t a = (op_A == CUBLAS_OP_N) ? A[t * lda + row] : A[row * lda + t];
        const int32_t b = (op_B == CUBLAS_OP_N) ? B[col * ldb + t] : B[t * ldb + col];
        sum += static_cast<int64_t>(a) * static_cast<int64_t>(b);
    }
    return sum;
}

struct CpuCheckResult {
    bool ok = true;
    size_t checked = 0;
    size_t mismatch_count = 0;
    double max_abs_error = 0.0;
};

CpuCheckResult validate_exact_baseline_on_cpu(
    const CpuCheckMode mode,
    const int cpu_samples,
    const cublasOperation_t op_A,
    const cublasOperation_t op_B,
    const size_t m,
    const size_t n,
    const size_t k,
    const std::vector<int32_t> &hA,
    const size_t lda,
    const std::vector<int32_t> &hB,
    const size_t ldb,
    const std::vector<int64_t> &h_exact,
    const size_t ldc //
) {
    CpuCheckResult out;
    if (mode == CpuCheckMode::OFF) return out;

    auto check_one = [&](const size_t row, const size_t col) {
        const int64_t ref = cpu_dot(hA, hB, op_A, op_B, lda, ldb, row, col, k);
        const int64_t got = h_exact[col * ldc + row];
        out.checked++;
        if (ref != got) {
            out.ok = false;
            out.mismatch_count++;
            const uint64_t abs_diff = (ref >= got)
                                          ? static_cast<uint64_t>(ref - got)
                                          : static_cast<uint64_t>(got - ref);
            out.max_abs_error = std::max(out.max_abs_error, static_cast<double>(abs_diff));
        }
    };

    if (mode == CpuCheckMode::FULL) {
        for (size_t col = 0; col < n; ++col) {
            for (size_t row = 0; row < m; ++row) {
                check_one(row, col);
            }
        }
    } else {
        const size_t total = m * n;
        const size_t samples = std::min<size_t>(total, static_cast<size_t>(std::max(1, cpu_samples)));
        std::mt19937_64 rng(static_cast<uint64_t>(m) * 1315423911ULL ^
                            static_cast<uint64_t>(n) * 2654435761ULL ^
                            static_cast<uint64_t>(k) * 11400714819323198485ULL ^
                            static_cast<uint64_t>(op_to_char(op_A) << 8 | op_to_char(op_B)));
        std::uniform_int_distribution<size_t> row_dist(0, m - 1);
        std::uniform_int_distribution<size_t> col_dist(0, n - 1);
        for (size_t i = 0; i < samples; ++i) {
            check_one(row_dist(rng), col_dist(rng));
        }
    }

    return out;
}

std::vector<double> bench_exact_i32_i32_i64_watt(
    const cublasOperation_t op_A,
    const cublasOperation_t op_B,
    const size_t m,
    const size_t n,
    const size_t k,
    const int32_t *const dA,
    const size_t lda,
    const int32_t *const dB,
    const size_t ldb,
    int64_t *const dC64,
    const size_t ldc //
) {
    const dim3 threads(16u, 16u);
    const dim3 grid(
        static_cast<unsigned>((m + threads.x - 1u) / threads.x),
        static_cast<unsigned>((n + threads.y - 1u) / threads.y));
    const int trans_A = (op_A == CUBLAS_OP_T) ? 1 : 0;
    const int trans_B = (op_B == CUBLAS_OP_T) ? 1 : 0;

    const std::vector<double> res = getWatt::getWatt(
        [&]() {
            gemm_i32_exact_kernel<<<grid, threads>>>(
                trans_A, trans_B, m, n, k, dA, lda, dB, ldb, dC64, ldc);
        },
        m,
        n,
        k);

    check_cuda(cudaGetLastError(), "exact_i32_i32_i64 watt launch");
    check_cuda(cudaDeviceSynchronize(), "exact_i32_i32_i64 watt sync");
    return res;
}

std::vector<double> bench_cublas_i8_watt(
    cublasHandle_t handle,
    const cublasOperation_t op_A,
    const cublasOperation_t op_B,
    const size_t m,
    const size_t n,
    const size_t k,
    const int8_t *const dA8,
    const size_t lda,
    const int8_t *const dB8,
    const size_t ldb,
    int32_t *const dC32,
    const size_t ldc,
    const unsigned repeat_per_equivalent_result //
) {
    constexpr int32_t alpha = 1;
    constexpr int32_t beta = 0;
    const unsigned repeats = std::max(1u, repeat_per_equivalent_result);

    const std::vector<double> res = getWatt::getWatt(
        [&]() {
            for (unsigned r = 0; r < repeats; ++r) {
                cublasGemmEx(handle,
                             op_A,
                             op_B,
                             m,
                             n,
                             k,
                             &alpha,
                             dA8,
                             CUDA_R_8I,
                             lda,
                             dB8,
                             CUDA_R_8I,
                             ldb,
                             &beta,
                             dC32,
                             CUDA_R_32I,
                             ldc,
                             CUBLAS_COMPUTE_32I,
                             CUBLAS_GEMM_DEFAULT);
            }
        },
        m,
        n,
        k);

    check_cuda(cudaDeviceSynchronize(), "cublas_i8 watt sync");
    return res;
}

struct I32BenchBreakdown {
    double encode_ms = 0.0;
    double tc_gemm_ms = 0.0;
    double conv32to8_ms = 0.0;
    double reconstruct_ms = 0.0;
    double total_ms = 0.0;
};

struct StageRatio {
    double encode_pct = 0.0;
    double tc_pct = 0.0;
    double conv_pct = 0.0;
    double reconstruct_pct = 0.0;
};

StageRatio to_stage_ratio(const I32BenchBreakdown &b) {
    StageRatio out;
    if (b.total_ms <= 0.0) return out;
    out.encode_pct = b.encode_ms * 100.0 / b.total_ms;
    out.tc_pct = b.tc_gemm_ms * 100.0 / b.total_ms;
    out.conv_pct = b.conv32to8_ms * 100.0 / b.total_ms;
    out.reconstruct_pct = b.reconstruct_ms * 100.0 / b.total_ms;
    return out;
}

template <bool UseExtraWorkspace>
I32BenchBreakdown bench_gemmul8_i32_ms(
    cublasHandle_t handle,
    const cublasOperation_t op_A,
    const cublasOperation_t op_B,
    const size_t m,
    const size_t n,
    const size_t k,
    const int32_t *dA,
    const size_t lda,
    const int32_t *dB,
    const size_t ldb,
    int64_t *dC,
    const size_t ldc,
    const unsigned num_moduli,
    const int warmup,
    const int iters //
) {
    constexpr int64_t alpha = 1;
    constexpr int64_t beta = 0;
    void *work = nullptr;
    const size_t work_size = gemmul8::workSize_i32<UseExtraWorkspace>(m, n, k, num_moduli);
    check_cuda(cudaMalloc(&work, work_size), "cudaMalloc work");

    for (int i = 0; i < warmup; ++i) {
        gemmul8::gemm_i32<UseExtraWorkspace>(
            handle,
            op_A,
            op_B,
            m,
            n,
            k,
            &alpha,
            dA,
            lda,
            dB,
            ldb,
            &beta,
            dC,
            ldc,
            num_moduli,
            work);
    }
    check_cuda(cudaDeviceSynchronize(), "gemm_i32 warmup sync");

    std::vector<double> accum_ns(4, 0.0);
    for (int i = 0; i < iters; ++i) {
        const std::vector<double> t_ns = gemmul8::gemm_i32<UseExtraWorkspace>(
            handle,
            op_A,
            op_B,
            m,
            n,
            k,
            &alpha,
            dA,
            lda,
            dB,
            ldb,
            &beta,
            dC,
            ldc,
            num_moduli,
            work);
        for (size_t j = 0; j < 4; ++j) {
            accum_ns[j] += t_ns[j];
        }
    }

    check_cuda(cudaDeviceSynchronize(), "gemm_i32 iter sync");

    I32BenchBreakdown out;
    const double inv_iters = 1.0 / static_cast<double>(iters);
    out.encode_ms = accum_ns[0] * inv_iters * 1e-6;
    out.tc_gemm_ms = accum_ns[1] * inv_iters * 1e-6;
    out.conv32to8_ms = accum_ns[2] * inv_iters * 1e-6;
    out.reconstruct_ms = accum_ns[3] * inv_iters * 1e-6;
    out.total_ms = out.encode_ms + out.tc_gemm_ms + out.conv32to8_ms + out.reconstruct_ms;

    cudaFree(work);
    return out;
}

template <bool UseExtraWorkspace>
std::vector<double> bench_gemmul8_i32_watt(
    cublasHandle_t handle,
    const cublasOperation_t op_A,
    const cublasOperation_t op_B,
    const size_t m,
    const size_t n,
    const size_t k,
    const int32_t *const dA,
    const size_t lda,
    const int32_t *const dB,
    const size_t ldb,
    int64_t *const dC64,
    const size_t ldc,
    const unsigned num_moduli //
) {
    constexpr int64_t alpha = 1;
    constexpr int64_t beta = 0;

    void *work = nullptr;
    const size_t work_size = gemmul8::workSize_i32<UseExtraWorkspace>(m, n, k, num_moduli);
    check_cuda(cudaMalloc(&work, work_size), "cudaMalloc gemmul8 watt work");

    const std::vector<double> res = getWatt::getWatt(
        [&]() {
            gemmul8::gemm_i32<UseExtraWorkspace>(
                handle,
                op_A,
                op_B,
                m,
                n,
                k,
                &alpha,
                dA,
                lda,
                dB,
                ldb,
                &beta,
                dC64,
                ldc,
                num_moduli,
                work);
        },
        m,
        n,
        k);

    check_cuda(cudaDeviceSynchronize(), "gemmul8 watt sync");
    cudaFree(work);
    return res;
}

struct PrecisionReport {
    unsigned required_num_moduli = 0;
    bool exact_guaranteed = false;
    size_t max_k_for_num_moduli = 0;
    __int128 required_dynamic_range = 0;
    __int128 crt_dynamic_range = 0;
};

PrecisionReport evaluate_precision(
    const uint64_t max_abs_a,
    const uint64_t max_abs_b,
    const size_t k,
    const unsigned num_moduli //
) {
    PrecisionReport out;
    out.required_num_moduli = oz2::i32::required_num_moduli_for_bounds(max_abs_a, max_abs_b, k);
    const __int128 max_abs = static_cast<__int128>(max_abs_a) *
                             static_cast<__int128>(max_abs_b) *
                             static_cast<__int128>(k);
    out.required_dynamic_range = 2 * max_abs + 1;
    out.crt_dynamic_range = oz2::i32::prefix_product(num_moduli);
    out.max_k_for_num_moduli = oz2::i32::max_k_for_num_moduli(num_moduli, max_abs_a, max_abs_b);
    out.exact_guaranteed = (out.required_num_moduli <= num_moduli) && (max_abs <= static_cast<__int128>(std::numeric_limits<int64_t>::max()));
    return out;
}

void write_oz2_time_row(
    std::ofstream &out,
    const double phi,
    const size_t m,
    const size_t n,
    const size_t k,
    const std::string &func,
    const AccuracyStats &acc,
    const I32BenchBreakdown &b,
    const double gflops //
) {
    out << phi << ','
        << m << ','
        << n << ','
        << k << ','
        << func << ','
        << acc.max_abs_error << ','
        << acc.med_abs_error << ','
        << (gflops * 1.0e-3) << ','
        << (b.total_ms * 1.0e-3) << ','
        << (b.encode_ms * 1.0e-3) << ','
        << (b.tc_gemm_ms * 1.0e-3) << ','
        << (b.conv32to8_ms * 1.0e-3) << ','
        << (b.reconstruct_ms * 1.0e-3) << ','
        << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        CliConfig cfg = parse_cli(argc, argv);

        if (!cfg.moduli_explicit) {
            size_t max_k_shape = 0;
            for (const auto &[m, n, k] : cfg.shapes) {
                (void)m;
                (void)n;
                max_k_shape = std::max(max_k_shape, k);
            }
            unsigned required = oz2::i32::required_num_moduli_for_bounds(
                static_cast<uint64_t>(cfg.input_bound),
                static_cast<uint64_t>(cfg.input_bound),
                max_k_shape);
            required = std::max(required, oz2::i32::kMinNumModuli);
            if (required > oz2::i32::kMaxNumModuli) {
                throw std::runtime_error(
                    "auto-selected required_num_moduli exceeds supported range. reduce --input-bound or k.");
            }
            cfg.moduli = {required};
        }

        validate_moduli_list(cfg.moduli);

        const int iters = cfg.iters;
        const int warmup = cfg.warmup;

        cublasHandle_t handle;
        check_cublas(cublasCreate(&handle), "cublasCreate");

        int dev = 0;
        check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
        cudaDeviceProp prop;
        check_cuda(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");

        const std::string device_name = sanitize_token(std::string(prop.name));
        const std::string timestamp = make_timestamp();
        const std::string file_i32 = "i32_bench_speedup_" + device_name + "_" + timestamp + ".csv";
        const std::string file_oz2_time = "oz2_results_i32_time_" + device_name + "_" + timestamp + ".csv";

        const cublasStatus_t int32_status = probe_cublas_int32(handle);
        const bool cublas_int32_supported = (int32_status == CUBLAS_STATUS_SUCCESS);

        std::ofstream out(file_i32);
        out << std::scientific;
        out << "m,n,k,opA,opB,use_extra,num_moduli,iters,warmup,"
               "gemmul8_total_ms,gemmul8_encode_ms,gemmul8_tc_ms,gemmul8_conv32to8_ms,gemmul8_reconstruct_ms,"
               "gemmul8_gflops,"
               "gemmul8_max_abs_error,gemmul8_mismatch_count,gemmul8_exact_match,"
               "gemmul8_watt,gemmul8_gflops_per_watt,"
               "exact_i32_i32_i64_ms,speedup_vs_exact_i32_i32_i64,"
               "exact_i32_i32_i64_gflops,"
               "baseline_i32_cuda_core_ms,baseline_i32_cuda_core_gflops,speedup_vs_baseline_i32_cuda_core,"
               "exact_i32_i32_i64_watt,exact_i32_i32_i64_gflops_per_watt,"
               "cublas_i8_single_ms,cublas_i8_x_moduli_ms,speedup_vs_cublas_i8_single,speedup_vs_cublas_i8_x_moduli,"
               "cublas_i8_single_gflops,cublas_i8_x_moduli_gflops,"
               "cublas_i8_single_watt,cublas_i8_single_gflops_per_watt,"
               "cublas_i8_x_moduli_watt,cublas_i8_x_moduli_gflops_per_watt,"
               "cublas_i32_supported,cublas_i32_probe_status,"
               "phi,scale_exp,input_bound,max_abs_a,max_abs_b,required_num_moduli,max_k_for_num_moduli,precision_guaranteed,"
               "encode_pct,tc_pct,conv32to8_pct,reconstruct_pct,"
               "cpu_check_mode,cpu_checked,cpu_mismatch,cpu_max_abs_error\n";

        std::ofstream out_time(file_oz2_time);
        out_time << std::scientific;
        out_time << "phi,m,n,k,function,err_max,err_med,TFLOPS,total_time[sec],quantization,low_prec_gemm,requantization,dequantization,\n";

        std::cout << "[INFO] device=" << prop.name
                  << " compute_cap=" << prop.major << "." << prop.minor << '\n';
        std::cout << "[INFO] cublas int32*int32->int32 support="
                  << (cublas_int32_supported ? "yes" : "no")
                  << " (status=" << static_cast<int>(int32_status) << ")\n";
        std::cout << "[INFO] benchmark settings: iters=" << iters
                  << " warmup=" << warmup
                  << " watt=" << (cfg.enable_watt ? "on" : "off")
                  << " aux_int8_baseline=" << (cfg.with_int8_baseline ? "on" : "off")
                  << " phi=" << cfg.phi
                  << " scale_exp=" << cfg.scale_exp
                  << " input_bound=" << cfg.input_bound
                  << " cpu_check=" << cpu_mode_name(cfg.cpu_check_mode)
                  << " cpu_samples=" << cfg.cpu_samples
                  << '\n';
        std::cout << "[INFO] shapes:";
        for (const auto &[m, n, k] : cfg.shapes) {
            std::cout << " " << m << "x" << n << "x" << k;
        }
        std::cout << '\n';
        std::cout << "[INFO] moduli:";
        for (const auto mod : cfg.moduli) {
            std::cout << " " << mod;
        }
        std::cout << '\n';
        std::cout << "[INFO] ops:";
        for (const auto &op_pair : cfg.ops) {
            std::cout << " " << op_pair_to_string(op_pair.first, op_pair.second);
        }
        std::cout << '\n';
        std::cout << "[INFO] csv=" << file_i32 << '\n';
        std::cout << "[INFO] fp64-style-time-csv=" << file_oz2_time << '\n';

        std::mt19937 rng(123456u);
        auto safe_ratio = [](const double num, const double den) -> double {
            if (den <= 0.0) return 0.0;
            return num / den;
        };

        for (const auto &[m, n, k] : cfg.shapes) {
            for (const auto &op_pair : cfg.ops) {
                const cublasOperation_t op_A = op_pair.first;
                const cublasOperation_t op_B = op_pair.second;

                const size_t lda = (op_A == CUBLAS_OP_N) ? m : k;
                const size_t ldb = (op_B == CUBLAS_OP_N) ? k : n;
                const size_t ldc = m;

                const size_t rowsA = (op_A == CUBLAS_OP_N) ? m : k;
                const size_t colsA = (op_A == CUBLAS_OP_N) ? k : m;
                const size_t rowsB = (op_B == CUBLAS_OP_N) ? k : n;
                const size_t colsB = (op_B == CUBLAS_OP_N) ? n : k;

                std::vector<int32_t> hA(lda * colsA);
                std::vector<int32_t> hB(ldb * colsB);
                fill_matrix_exponent_i32(hA, rowsA, colsA, lda, rng, cfg.phi, cfg.scale_exp, static_cast<int32_t>(cfg.input_bound));
                fill_matrix_exponent_i32(hB, rowsB, colsB, ldb, rng, cfg.phi, cfg.scale_exp, static_cast<int32_t>(cfg.input_bound));

                const uint64_t max_abs_a = max_abs_i32(hA);
                const uint64_t max_abs_b = max_abs_i32(hB);

                std::vector<int8_t> hA8;
                std::vector<int8_t> hB8;
                if (cfg.with_int8_baseline) {
                    cast_i32_to_i8(hA, hA8);
                    cast_i32_to_i8(hB, hB8);
                }

                int32_t *dA = nullptr;
                int32_t *dB = nullptr;
                int64_t *dC64 = nullptr;
                int8_t *dA8 = nullptr;
                int8_t *dB8 = nullptr;
                int32_t *dC32 = nullptr;

                check_cuda(cudaMalloc(&dA, sizeof(int32_t) * hA.size()), "cudaMalloc dA");
                check_cuda(cudaMalloc(&dB, sizeof(int32_t) * hB.size()), "cudaMalloc dB");
                check_cuda(cudaMalloc(&dC64, sizeof(int64_t) * ldc * n), "cudaMalloc dC64");
                if (cfg.with_int8_baseline) {
                    check_cuda(cudaMalloc(&dA8, sizeof(int8_t) * hA8.size()), "cudaMalloc dA8");
                    check_cuda(cudaMalloc(&dB8, sizeof(int8_t) * hB8.size()), "cudaMalloc dB8");
                    check_cuda(cudaMalloc(&dC32, sizeof(int32_t) * ldc * n), "cudaMalloc dC32");
                }

                check_cuda(cudaMemcpy(dA, hA.data(), sizeof(int32_t) * hA.size(), cudaMemcpyHostToDevice), "cudaMemcpy dA");
                check_cuda(cudaMemcpy(dB, hB.data(), sizeof(int32_t) * hB.size(), cudaMemcpyHostToDevice), "cudaMemcpy dB");
                if (cfg.with_int8_baseline) {
                    check_cuda(cudaMemcpy(dA8, hA8.data(), sizeof(int8_t) * hA8.size(), cudaMemcpyHostToDevice), "cudaMemcpy dA8");
                    check_cuda(cudaMemcpy(dB8, hB8.data(), sizeof(int8_t) * hB8.size(), cudaMemcpyHostToDevice), "cudaMemcpy dB8");
                }

                double cublas_i8_single_ms = 0.0;
                if (cfg.with_int8_baseline) {
                    cublas_i8_single_ms = bench_cublas_i8_ms(
                        handle, op_A, op_B, m, n, k,
                        dA8, lda, dB8, ldb, dC32, ldc,
                        warmup, iters);
                }
                const double exact_i32_i32_i64_ms = bench_exact_i32_i32_i64_ms(
                    op_A, op_B, m, n, k,
                    dA, lda, dB, ldb, dC64, ldc,
                    warmup, iters);
                const std::vector<int64_t> h_exact = copy_i64_from_device(dC64, ldc * n);

                const CpuCheckResult cpu_check = validate_exact_baseline_on_cpu(
                    cfg.cpu_check_mode,
                    cfg.cpu_samples,
                    op_A,
                    op_B,
                    m,
                    n,
                    k,
                    hA,
                    lda,
                    hB,
                    ldb,
                    h_exact,
                    ldc);
                if (!cpu_check.ok) {
                    std::cerr << "[WARN] CPU verification mismatch: op="
                              << op_pair_to_string(op_A, op_B)
                              << " m=" << m << " n=" << n << " k=" << k
                              << " mismatches=" << cpu_check.mismatch_count
                              << " max_abs_error=" << cpu_check.max_abs_error
                              << " checked=" << cpu_check.checked << '\n';
                }

                const double ops = gemm_int_ops(m, n, k);
                const double exact_gflops = compute_gflops(ops, exact_i32_i32_i64_ms);
                const double cublas_single_gflops = cfg.with_int8_baseline ? compute_gflops(ops, cublas_i8_single_ms) : 0.0;

                const std::vector<double> exact_watt_res = cfg.enable_watt
                                                               ? bench_exact_i32_i32_i64_watt(
                                                                     op_A, op_B, m, n, k,
                                                                     dA, lda, dB, ldb, dC64, ldc)
                                                               : std::vector<double>{};
                const std::vector<double> cublas_single_watt_res = (cfg.enable_watt && cfg.with_int8_baseline)
                                                                        ? bench_cublas_i8_watt(
                                                                              handle, op_A, op_B, m, n, k,
                                                                              dA8, lda, dB8, ldb, dC32, ldc, 1u)
                                                                        : std::vector<double>{};
                const double exact_watt = cfg.enable_watt ? exact_watt_res[0] : 0.0;
                const double exact_gflops_per_watt = cfg.enable_watt ? (exact_watt_res[1] * 1e-9) : 0.0;
                const double cublas_single_watt = (cfg.enable_watt && cfg.with_int8_baseline) ? cublas_single_watt_res[0] : 0.0;
                const double cublas_single_gflops_per_watt = (cfg.enable_watt && cfg.with_int8_baseline) ? (cublas_single_watt_res[1] * 1e-9) : 0.0;

                I32BenchBreakdown baseline_breakdown {};
                baseline_breakdown.total_ms = exact_i32_i32_i64_ms;
                AccuracyStats baseline_acc {};
                baseline_acc.exact_match = true;

                write_oz2_time_row(
                    out_time,
                    cfg.phi,
                    m,
                    n,
                    k,
                    "I32-exact-" + op_pair_to_string(op_A, op_B),
                    baseline_acc,
                    baseline_breakdown,
                    exact_gflops);

                for (const unsigned num_moduli : cfg.moduli) {
                    const PrecisionReport pr = evaluate_precision(max_abs_a, max_abs_b, k, num_moduli);

                    const I32BenchBreakdown b_true = bench_gemmul8_i32_ms<true>(
                        handle, op_A, op_B, m, n, k, dA, lda, dB, ldb, dC64, ldc, num_moduli, warmup, iters);
                    const AccuracyStats acc_true = compare_i64_outputs(copy_i64_from_device(dC64, ldc * n), h_exact);
                    const std::vector<double> gemmul8_watt_true = cfg.enable_watt
                                                                       ? bench_gemmul8_i32_watt<true>(
                                                                             handle, op_A, op_B, m, n, k,
                                                                             dA, lda, dB, ldb, dC64, ldc, num_moduli)
                                                                       : std::vector<double>{};

                    const I32BenchBreakdown b_false = bench_gemmul8_i32_ms<false>(
                        handle, op_A, op_B, m, n, k, dA, lda, dB, ldb, dC64, ldc, num_moduli, warmup, iters);
                    const AccuracyStats acc_false = compare_i64_outputs(copy_i64_from_device(dC64, ldc * n), h_exact);
                    const std::vector<double> gemmul8_watt_false = cfg.enable_watt
                                                                        ? bench_gemmul8_i32_watt<false>(
                                                                              handle, op_A, op_B, m, n, k,
                                                                              dA, lda, dB, ldb, dC64, ldc, num_moduli)
                                                                        : std::vector<double>{};

                    const double cublas_i8_x_moduli_ms = cfg.with_int8_baseline
                                                              ? (cublas_i8_single_ms * static_cast<double>(num_moduli))
                                                              : 0.0;
                    const double cublas_x_moduli_gflops = cfg.with_int8_baseline
                                                               ? compute_gflops(ops * static_cast<double>(num_moduli), cublas_i8_x_moduli_ms)
                                                               : 0.0;
                    const std::vector<double> cublas_x_moduli_watt_res = (cfg.enable_watt && cfg.with_int8_baseline)
                                                                              ? bench_cublas_i8_watt(
                                                                                    handle, op_A, op_B, m, n, k,
                                                                                    dA8, lda, dB8, ldb, dC32, ldc, num_moduli)
                                                                              : std::vector<double>{};
                    const double cublas_x_moduli_watt = (cfg.enable_watt && cfg.with_int8_baseline) ? cublas_x_moduli_watt_res[0] : 0.0;
                    const double cublas_x_moduli_gflops_per_watt = (cfg.enable_watt && cfg.with_int8_baseline) ? (cublas_x_moduli_watt_res[1] * 1e-9) : 0.0;

                    auto dump_row = [&](const bool use_extra,
                                        const I32BenchBreakdown &b,
                                        const AccuracyStats &acc,
                                        const std::vector<double> &watt_res,
                                        const std::string &func_label) {
                        const double speedup_exact = safe_ratio(exact_i32_i32_i64_ms, b.total_ms);
                        const double speedup_baseline = speedup_exact;
                        const double speedup_single = cfg.with_int8_baseline ? safe_ratio(cublas_i8_single_ms, b.total_ms) : 0.0;
                        const double speedup_x_moduli = cfg.with_int8_baseline ? safe_ratio(cublas_i8_x_moduli_ms, b.total_ms) : 0.0;
                        const double gemmul8_gflops = compute_gflops(ops, b.total_ms);
                        const double gemmul8_watt = cfg.enable_watt ? watt_res[0] : 0.0;
                        const double gemmul8_gflops_per_watt = cfg.enable_watt ? (watt_res[1] * 1e-9) : 0.0;
                        const StageRatio ratio = to_stage_ratio(b);

                        out << m << ','
                            << n << ','
                            << k << ','
                            << op_to_char(op_A) << ','
                            << op_to_char(op_B) << ','
                            << (use_extra ? 1 : 0) << ','
                            << num_moduli << ','
                            << iters << ','
                            << warmup << ','
                            << b.total_ms << ','
                            << b.encode_ms << ','
                            << b.tc_gemm_ms << ','
                            << b.conv32to8_ms << ','
                            << b.reconstruct_ms << ','
                            << gemmul8_gflops << ','
                            << acc.max_abs_error << ','
                            << acc.mismatch_count << ','
                            << (acc.exact_match ? 1 : 0) << ',';

                        if (cfg.enable_watt) out << gemmul8_watt;
                        out << ',';
                        if (cfg.enable_watt) out << gemmul8_gflops_per_watt;
                        out << ',';

                        out << exact_i32_i32_i64_ms << ','
                            << speedup_exact << ','
                            << exact_gflops << ',';
                        out << exact_i32_i32_i64_ms << ','
                            << exact_gflops << ','
                            << speedup_baseline << ',';
                        if (cfg.enable_watt) out << exact_watt;
                        out << ',';
                        if (cfg.enable_watt) out << exact_gflops_per_watt;
                        out << ',';

                        if (cfg.with_int8_baseline) {
                            out << cublas_i8_single_ms << ','
                                << cublas_i8_x_moduli_ms << ','
                                << speedup_single << ','
                                << speedup_x_moduli << ','
                                << cublas_single_gflops << ','
                                << cublas_x_moduli_gflops << ',';
                            if (cfg.enable_watt) out << cublas_single_watt;
                            out << ',';
                            if (cfg.enable_watt) out << cublas_single_gflops_per_watt;
                            out << ',';
                            if (cfg.enable_watt) out << cublas_x_moduli_watt;
                            out << ',';
                            if (cfg.enable_watt) out << cublas_x_moduli_gflops_per_watt;
                            out << ',';
                        } else {
                            out << ",,,,,,,,,,";
                        }

                        out << (cublas_int32_supported ? 1 : 0) << ','
                            << static_cast<int>(int32_status) << ','
                            << cfg.phi << ','
                            << cfg.scale_exp << ','
                            << cfg.input_bound << ','
                            << max_abs_a << ','
                            << max_abs_b << ','
                            << pr.required_num_moduli << ','
                            << pr.max_k_for_num_moduli << ','
                            << (pr.exact_guaranteed ? 1 : 0) << ','
                            << ratio.encode_pct << ','
                            << ratio.tc_pct << ','
                            << ratio.conv_pct << ','
                            << ratio.reconstruct_pct << ','
                            << cpu_mode_name(cfg.cpu_check_mode) << ','
                            << cpu_check.checked << ','
                            << cpu_check.mismatch_count << ','
                            << cpu_check.max_abs_error
                            << '\n';

                        write_oz2_time_row(out_time, cfg.phi, m, n, k, func_label, acc, b, gemmul8_gflops);

                        std::cout << "[BENCH] m=" << m << " n=" << n << " k=" << k
                                  << " op=" << op_pair_to_string(op_A, op_B)
                                  << " mod=" << num_moduli
                                  << " use_extra=" << (use_extra ? "true" : "false")
                                  << " exact_guaranteed=" << (pr.exact_guaranteed ? "yes" : "no")
                                  << " required_mod=" << pr.required_num_moduli
                                  << " max_k@mod=" << pr.max_k_for_num_moduli
                                  << " gemmul8_total_ms=" << b.total_ms
                                  << " stage_pct=(" << ratio.encode_pct << ","
                                  << ratio.tc_pct << ","
                                  << ratio.conv_pct << ","
                                  << ratio.reconstruct_pct << ")"
                                  << " gemmul8_gflops=" << gemmul8_gflops
                                  << " mismatch=" << acc.mismatch_count
                                  << " speedup(exact)=" << speedup_exact;
                        if (cfg.with_int8_baseline) {
                            std::cout << " speedup(single)=" << speedup_single
                                      << " speedup(xmoduli)=" << speedup_x_moduli;
                        } else {
                            std::cout << " speedup(single)=NA"
                                      << " speedup(xmoduli)=NA";
                        }
                        if (cfg.enable_watt) {
                            std::cout << " gemmul8_watt=" << gemmul8_watt
                                      << " gemmul8_gflops_per_watt=" << gemmul8_gflops_per_watt;
                        }
                        std::cout << '\n';
                    };

                    dump_row(true, b_true, acc_true, gemmul8_watt_true,
                             "I32-accu-" + std::to_string(num_moduli) + "-extra");
                    dump_row(false, b_false, acc_false, gemmul8_watt_false,
                             "I32-accu-" + std::to_string(num_moduli) + "-compact");
                }

                cudaFree(dA);
                cudaFree(dB);
                cudaFree(dC64);
                if (dA8 != nullptr) cudaFree(dA8);
                if (dB8 != nullptr) cudaFree(dB8);
                if (dC32 != nullptr) cudaFree(dC32);
            }
        }

        out.close();
        out_time.close();
        check_cublas(cublasDestroy(handle), "cublasDestroy");
        std::cout << "[INFO] benchmark finished\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
