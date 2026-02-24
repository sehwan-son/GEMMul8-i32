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
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Shape3 = std::tuple<size_t, size_t, size_t>;
using OpPair = std::pair<cublasOperation_t, cublasOperation_t>;

enum class CpuCheckMode { OFF,
                          SAMPLE,
                          FULL };

enum class BaselineI32Policy {
    BEST,
    LAST
};

enum class I32SchemeMode {
    OZ2,
    OZ1,
    BOTH
};

struct BaselineI32Key {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    char opA = 'N';
    char opB = 'N';

    bool operator<(const BaselineI32Key &rhs) const {
        if (m != rhs.m) return m < rhs.m;
        if (n != rhs.n) return n < rhs.n;
        if (k != rhs.k) return k < rhs.k;
        if (opA != rhs.opA) return opA < rhs.opA;
        return opB < rhs.opB;
    }
};

struct BaselineI32Value {
    double time_ms = 0.0;
    double gflops = 0.0;
    size_t row_index = 0;
};

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

BaselineI32Policy parse_baseline_i32_policy(const std::string &text) {
    if (text == "best") return BaselineI32Policy::BEST;
    if (text == "last") return BaselineI32Policy::LAST;
    throw std::invalid_argument("invalid --baseline-i32-policy: '" + text + "' (expected best|last)");
}

I32SchemeMode parse_i32_scheme_mode(const std::string &text) {
    const std::string t = trim_copy(text);
    if (t == "oz2") return I32SchemeMode::OZ2;
    if (t == "oz1") return I32SchemeMode::OZ1;
    if (t == "both") return I32SchemeMode::BOTH;
    throw std::invalid_argument("invalid --i32-scheme value: '" + text + "' (expected oz2|oz1|both)");
}

const char *i32_scheme_tag(const gemmul8::I32Scheme scheme) {
    switch (scheme) {
    case gemmul8::I32Scheme::OZAKI1_SPLIT: return "oz1";
    case gemmul8::I32Scheme::OZAKI2_CRT:
    default: return "oz2";
    }
}

const char *i32_scheme_mode_name(const I32SchemeMode mode) {
    switch (mode) {
    case I32SchemeMode::OZ2: return "oz2";
    case I32SchemeMode::OZ1: return "oz1";
    case I32SchemeMode::BOTH:
    default: return "both";
    }
}

const char *cpu_mode_name(const CpuCheckMode mode) {
    switch (mode) {
    case CpuCheckMode::OFF: return "off";
    case CpuCheckMode::SAMPLE: return "sample";
    case CpuCheckMode::FULL: return "full";
    }
    return "unknown";
}

const char *baseline_i32_policy_name(const BaselineI32Policy mode) {
    switch (mode) {
    case BaselineI32Policy::BEST: return "best";
    case BaselineI32Policy::LAST: return "last";
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

std::vector<unsigned> make_auto_moduli_candidates(
    const uint64_t max_abs_a,
    const uint64_t max_abs_b,
    const size_t k,
    const bool tune_moduli,
    const int tune_window //
) {
    unsigned required = oz2::i32::required_num_moduli_for_bounds(max_abs_a, max_abs_b, k);
    required = std::max(required, oz2::i32::kMinNumModuli);
    if (required > oz2::i32::kMaxNumModuli) {
        throw std::runtime_error(
            "auto-selected required_num_moduli exceeds supported range. reduce --input-bound or k.");
    }

    const unsigned upper = tune_moduli
                               ? std::min<unsigned>(
                                     oz2::i32::kMaxNumModuli,
                                     static_cast<unsigned>(required + std::max(0, tune_window)))
                               : required;

    std::vector<unsigned> out;
    out.reserve(static_cast<size_t>(upper - required + 1u));
    for (unsigned mod = required; mod <= upper; ++mod) {
        out.push_back(mod);
    }
    return out;
}

std::vector<std::string> split_csv_line_simple(const std::string &line) {
    std::vector<std::string> out;
    size_t begin = 0;
    while (begin <= line.size()) {
        const size_t comma = line.find(',', begin);
        const size_t end = (comma == std::string::npos) ? line.size() : comma;
        out.emplace_back(line.substr(begin, end - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1u;
    }
    return out;
}

int find_col_index(const std::vector<std::string> &header, const std::vector<std::string> &candidates) {
    for (size_t i = 0; i < header.size(); ++i) {
        for (const auto &name : candidates) {
            if (header[i] == name) return static_cast<int>(i);
        }
    }
    return -1;
}

double parse_csv_double(const std::vector<std::string> &row, const int idx, const char *const field_name) {
    if (idx < 0 || idx >= static_cast<int>(row.size())) {
        throw std::runtime_error(std::string("missing field: ") + field_name);
    }
    return parse_double_arg(trim_copy(row[static_cast<size_t>(idx)]), field_name);
}

uint64_t parse_csv_u64(const std::vector<std::string> &row, const int idx, const char *const field_name) {
    if (idx < 0 || idx >= static_cast<int>(row.size())) {
        throw std::runtime_error(std::string("missing field: ") + field_name);
    }
    const std::string token = trim_copy(row[static_cast<size_t>(idx)]);
    try {
        size_t parsed = 0;
        const unsigned long long value = std::stoull(token, &parsed);
        if (parsed != token.size()) throw std::invalid_argument("trailing");
        return static_cast<uint64_t>(value);
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string("invalid ") + field_name + ": '" + token + "'");
    }
}

size_t parse_csv_size(const std::vector<std::string> &row, const int idx, const char *const field_name) {
    if (idx < 0 || idx >= static_cast<int>(row.size())) {
        throw std::runtime_error(std::string("missing field: ") + field_name);
    }
    return parse_positive_size_token(trim_copy(row[static_cast<size_t>(idx)]), field_name);
}

char parse_csv_op_char(const std::vector<std::string> &row, const int idx, const char default_value) {
    if (idx < 0 || idx >= static_cast<int>(row.size())) return default_value;
    const std::string token = trim_copy(row[static_cast<size_t>(idx)]);
    if (token.empty()) return default_value;
    const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(token[0])));
    return (c == 'T') ? 'T' : 'N';
}

std::map<BaselineI32Key, BaselineI32Value> load_baseline_i32_csv(
    const std::string &path,
    const BaselineI32Policy policy //
) {
    std::map<BaselineI32Key, BaselineI32Value> out;
    if (path.empty()) return out;

    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open --baseline-i32-csv: " + path);
    }

    std::string header_line;
    if (!std::getline(in, header_line)) {
        throw std::runtime_error("empty --baseline-i32-csv: " + path);
    }

    const std::vector<std::string> header = split_csv_line_simple(header_line);
    const int idx_m = find_col_index(header, {"M", "m"});
    const int idx_n = find_col_index(header, {"N", "n"});
    const int idx_k = find_col_index(header, {"K", "k"});
    const int idx_opA = find_col_index(header, {"opA"});
    const int idx_opB = find_col_index(header, {"opB"});
    const int idx_time = find_col_index(header, {"time_ms"});
    const int idx_gflops = find_col_index(header, {"gflops"});
    const int idx_mismatch = find_col_index(header, {"mismatch_count"});

    if (idx_m < 0 || idx_n < 0 || idx_k < 0 || idx_time < 0 || idx_gflops < 0) {
        throw std::runtime_error(
            "invalid --baseline-i32-csv header: require M/N/K(or m/n/k), time_ms, gflops");
    }

    size_t row_idx = 0;
    std::string line;
    while (std::getline(in, line)) {
        row_idx++;
        const std::string trimmed = trim_copy(line);
        if (trimmed.empty()) continue;

        const std::vector<std::string> row = split_csv_line_simple(trimmed);
        BaselineI32Key key {};
        key.m = parse_csv_size(row, idx_m, "M");
        key.n = parse_csv_size(row, idx_n, "N");
        key.k = parse_csv_size(row, idx_k, "K");
        key.opA = parse_csv_op_char(row, idx_opA, 'N');
        key.opB = parse_csv_op_char(row, idx_opB, 'N');

        BaselineI32Value val {};
        val.time_ms = parse_csv_double(row, idx_time, "time_ms");
        val.gflops = parse_csv_double(row, idx_gflops, "gflops");
        val.row_index = row_idx;
        if (!std::isfinite(val.time_ms) || val.time_ms <= 0.0) continue;
        if (!std::isfinite(val.gflops) || val.gflops <= 0.0) continue;
        if (idx_mismatch >= 0) {
            const uint64_t mismatch_count = parse_csv_u64(row, idx_mismatch, "mismatch_count");
            if (mismatch_count != 0u) continue;
        }

        auto it = out.find(key);
        if (it == out.end()) {
            out.emplace(key, val);
            continue;
        }

        if (policy == BaselineI32Policy::LAST) {
            it->second = val;
        } else {
            if (val.time_ms < it->second.time_ms) {
                it->second = val;
            }
        }
    }

    return out;
}

void print_usage(const char *const prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " [iters] [warmup] [sizes_csv] [moduli_csv]\n"
        << "  " << prog << " --iters <int> --warmup <int>\n"
        << "           [--sizes <n1,n2,...> | --shapes <m1xn1xk1,m2xn2xk2,...>]\n"
        << "           [--moduli <5,6,...,20>] [--ops <NN,NT,TN,TT>] [--watt]\n"
        << "           [--i32-scheme <oz2|oz1|both>]\n"
        << "           [--tune-moduli] [--moduli-tune-window <int>]\n"
        << "           [--with-int8-baseline]\n"
        << "           [--baseline-i32-csv <path>] [--baseline-i32-policy <best|last>]\n"
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
    std::vector<unsigned> moduli = {5u};
    bool moduli_explicit = false;
    std::vector<OpPair> ops = {
        {CUBLAS_OP_N, CUBLAS_OP_N},
    };
    bool enable_watt = false;
    double phi = 0.5;
    int scale_exp = 10;
    int input_bound = 1024;
    bool with_int8_baseline = false;
    I32SchemeMode i32_scheme = I32SchemeMode::BOTH;
    bool tune_moduli = false;
    int moduli_tune_window = 2;
    std::string baseline_i32_csv {};
    BaselineI32Policy baseline_i32_policy = BaselineI32Policy::BEST;
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
        if (cfg.moduli_explicit) validate_moduli_list(cfg.moduli);
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
        if (arg == "--tune-moduli") {
            cfg.tune_moduli = true;
            continue;
        }
        if (arg == "--i32-scheme") {
            if (i + 1 >= argc) throw std::invalid_argument("--i32-scheme requires a value");
            cfg.i32_scheme = parse_i32_scheme_mode(argv[++i]);
            continue;
        }
        if (arg == "--moduli-tune-window") {
            if (i + 1 >= argc) throw std::invalid_argument("--moduli-tune-window requires a value");
            cfg.moduli_tune_window = std::max(0, parse_int_arg(argv[++i], "--moduli-tune-window"));
            continue;
        }
        if (arg == "--with-int8-baseline") {
            cfg.with_int8_baseline = true;
            continue;
        }
        if (arg == "--baseline-i32-csv") {
            if (i + 1 >= argc) throw std::invalid_argument("--baseline-i32-csv requires a value");
            cfg.baseline_i32_csv = argv[++i];
            continue;
        }
        if (arg == "--baseline-i32-policy") {
            if (i + 1 >= argc) throw std::invalid_argument("--baseline-i32-policy requires a value");
            cfg.baseline_i32_policy = parse_baseline_i32_policy(argv[++i]);
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

    if (cfg.moduli_explicit) validate_moduli_list(cfg.moduli);
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
    const gemmul8::I32Scheme scheme,
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
    const size_t work_size = gemmul8::workSize_i32<UseExtraWorkspace>(m, n, k, num_moduli, nullptr, nullptr, scheme);
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
            work,
            nullptr,
            nullptr,
            scheme);
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
            work,
            nullptr,
            nullptr,
            scheme);
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
    const gemmul8::I32Scheme scheme,
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
    const size_t work_size = gemmul8::workSize_i32<UseExtraWorkspace>(m, n, k, num_moduli, nullptr, nullptr, scheme);
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
                work,
                nullptr,
                nullptr,
                scheme);
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

PrecisionReport evaluate_precision_oz1(
    const uint64_t max_abs_a,
    const uint64_t max_abs_b,
    const size_t k //
) {
    PrecisionReport out;
    out.required_num_moduli = 5u;
    out.crt_dynamic_range = 0;

    const __int128 per_term =
        static_cast<__int128>(max_abs_a) *
        static_cast<__int128>(max_abs_b);
    const __int128 max_abs =
        per_term * static_cast<__int128>(k);
    out.required_dynamic_range = 2 * max_abs + 1;

    const __int128 i64_max = static_cast<__int128>(std::numeric_limits<int64_t>::max());
    if (per_term <= 0) {
        out.max_k_for_num_moduli = std::numeric_limits<size_t>::max();
    } else {
        const __int128 max_k = i64_max / per_term;
        out.max_k_for_num_moduli =
            (max_k > static_cast<__int128>(std::numeric_limits<size_t>::max()))
                ? std::numeric_limits<size_t>::max()
                : static_cast<size_t>(max_k);
    }

    out.exact_guaranteed = (max_abs <= i64_max);
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
        if (cfg.moduli_explicit) validate_moduli_list(cfg.moduli);
        if (cfg.i32_scheme == I32SchemeMode::OZ1 && cfg.moduli_explicit) {
            for (const unsigned mod : cfg.moduli) {
                if (mod != 5u) {
                    throw std::invalid_argument(
                        "OZ1 requires num_moduli=5 (remove --moduli or set --moduli 5).");
                }
            }
        }
        if (cfg.i32_scheme == I32SchemeMode::OZ1 && cfg.tune_moduli) {
            std::cerr << "[WARN] --tune-moduli is ignored for --i32-scheme oz1 (fixed num_moduli=5)\n";
        }
        const std::map<BaselineI32Key, BaselineI32Value> baseline_i32_rows =
            load_baseline_i32_csv(cfg.baseline_i32_csv, cfg.baseline_i32_policy);

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
        out << "m,n,k,opA,opB,i32_scheme,use_extra,num_moduli,iters,warmup,"
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
                  << " i32_scheme=" << i32_scheme_mode_name(cfg.i32_scheme)
                  << " baseline_i32_csv=" << (cfg.baseline_i32_csv.empty() ? "off" : cfg.baseline_i32_csv)
                  << " baseline_i32_policy=" << baseline_i32_policy_name(cfg.baseline_i32_policy)
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
        std::cout << "[INFO] moduli=";
        if (cfg.moduli_explicit) {
            for (const auto mod : cfg.moduli) {
                std::cout << " " << mod;
            }
            if (cfg.tune_moduli) {
                std::cout << " (explicit sweep + tuned best pick)";
            }
        } else {
            std::cout << " auto(required per shape/op)";
            if (cfg.tune_moduli) {
                std::cout << " + tune_window=" << cfg.moduli_tune_window;
            }
        }
        std::cout << '\n';
        std::cout << "[INFO] ops:";
        for (const auto &op_pair : cfg.ops) {
            std::cout << " " << op_pair_to_string(op_pair.first, op_pair.second);
        }
        std::cout << '\n';
        std::cout << "[INFO] csv=" << file_i32 << '\n';
        std::cout << "[INFO] fp64-style-time-csv=" << file_oz2_time << '\n';
        std::cout << "[INFO] baseline_i32_rows_loaded=" << baseline_i32_rows.size() << '\n';

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

                double baseline_i32_ms = exact_i32_i32_i64_ms;
                double baseline_i32_gflops = exact_gflops;
                bool baseline_i32_from_csv = false;
                if (!baseline_i32_rows.empty()) {
                    BaselineI32Key bkey {};
                    bkey.m = m;
                    bkey.n = n;
                    bkey.k = k;
                    bkey.opA = op_to_char(op_A);
                    bkey.opB = op_to_char(op_B);
                    const auto it_base = baseline_i32_rows.find(bkey);
                    if (it_base != baseline_i32_rows.end()) {
                        baseline_i32_ms = it_base->second.time_ms;
                        baseline_i32_gflops = it_base->second.gflops;
                        baseline_i32_from_csv = true;
                    }
                }

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

                const bool run_oz2 = (cfg.i32_scheme == I32SchemeMode::OZ2 || cfg.i32_scheme == I32SchemeMode::BOTH);
                const bool run_oz1 = (cfg.i32_scheme == I32SchemeMode::OZ1 || cfg.i32_scheme == I32SchemeMode::BOTH);
                std::map<unsigned, std::vector<double>> cublas_x_moduli_watt_cache;
                auto dump_row = [&](const gemmul8::I32Scheme scheme,
                                    const bool use_extra,
                                    const unsigned num_moduli,
                                    const PrecisionReport &pr,
                                    const I32BenchBreakdown &b,
                                    const AccuracyStats &acc,
                                    const std::vector<double> &watt_res,
                                    const std::string &func_label) {
                    const double cublas_i8_x_moduli_ms = cfg.with_int8_baseline
                                                             ? (cublas_i8_single_ms * static_cast<double>(num_moduli))
                                                             : 0.0;
                    const double cublas_x_moduli_gflops = cfg.with_int8_baseline
                                                              ? compute_gflops(
                                                                    ops * static_cast<double>(num_moduli),
                                                                    cublas_i8_x_moduli_ms)
                                                              : 0.0;
                    const std::vector<double> *cublas_x_moduli_watt_res = nullptr;
                    if (cfg.enable_watt && cfg.with_int8_baseline) {
                        auto it = cublas_x_moduli_watt_cache.find(num_moduli);
                        if (it == cublas_x_moduli_watt_cache.end()) {
                            it = cublas_x_moduli_watt_cache.emplace(
                                num_moduli,
                                bench_cublas_i8_watt(
                                    handle,
                                    op_A,
                                    op_B,
                                    m,
                                    n,
                                    k,
                                    dA8,
                                    lda,
                                    dB8,
                                    ldb,
                                    dC32,
                                    ldc,
                                    num_moduli))
                                     .first;
                        }
                        cublas_x_moduli_watt_res = &it->second;
                    }
                    const double cublas_x_moduli_watt = (cfg.enable_watt && cfg.with_int8_baseline)
                                                            ? (*cublas_x_moduli_watt_res)[0]
                                                            : 0.0;
                    const double cublas_x_moduli_gflops_per_watt = (cfg.enable_watt && cfg.with_int8_baseline)
                                                                        ? ((*cublas_x_moduli_watt_res)[1] * 1e-9)
                                                                        : 0.0;

                    const double speedup_exact = safe_ratio(exact_i32_i32_i64_ms, b.total_ms);
                    const double speedup_baseline = safe_ratio(baseline_i32_ms, b.total_ms);
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
                        << i32_scheme_tag(scheme) << ','
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
                    out << baseline_i32_ms << ','
                        << baseline_i32_gflops << ','
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
                              << " scheme=" << i32_scheme_tag(scheme)
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
                              << " speedup(exact)=" << speedup_exact
                              << " speedup(baseline_i32)=" << speedup_baseline
                              << " baseline_src=" << (baseline_i32_from_csv ? "csv" : "exact_fallback");
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
                if (run_oz2) {
                    std::vector<unsigned> moduli_candidates = cfg.moduli_explicit
                                                                  ? cfg.moduli
                                                                  : make_auto_moduli_candidates(
                                                                        max_abs_a,
                                                                        max_abs_b,
                                                                        k,
                                                                        cfg.tune_moduli,
                                                                        cfg.moduli_tune_window);
                    std::sort(moduli_candidates.begin(), moduli_candidates.end());
                    moduli_candidates.erase(std::unique(moduli_candidates.begin(), moduli_candidates.end()),
                                           moduli_candidates.end());
                    if (moduli_candidates.empty()) {
                        throw std::runtime_error("no moduli candidates available");
                    }
                    validate_moduli_list(moduli_candidates);

                    if (cfg.tune_moduli) {
                        std::cout << "[TUNE] m=" << m
                                  << " n=" << n
                                  << " k=" << k
                                  << " op=" << op_pair_to_string(op_A, op_B)
                                  << " scheme=oz2"
                                  << " candidates:";
                        for (const unsigned mod : moduli_candidates) {
                            std::cout << " " << mod;
                        }
                        std::cout << '\n';
                    }

                    struct BenchCandidate {
                        bool use_extra = false;
                        unsigned num_moduli = 0;
                        PrecisionReport pr {};
                        I32BenchBreakdown breakdown {};
                        AccuracyStats acc {};
                        std::vector<double> watt {};
                        std::string func_label {};
                    };

                    std::vector<BenchCandidate> candidates;
                    if (cfg.tune_moduli) {
                        candidates.reserve(moduli_candidates.size() * 2u);
                    }

                    for (const unsigned num_moduli : moduli_candidates) {
                        const PrecisionReport pr = evaluate_precision(max_abs_a, max_abs_b, k, num_moduli);

                        const I32BenchBreakdown b_true = bench_gemmul8_i32_ms<true>(
                            handle, gemmul8::I32Scheme::OZAKI2_CRT, op_A, op_B, m, n, k,
                            dA, lda, dB, ldb, dC64, ldc, num_moduli, warmup, iters);
                        const AccuracyStats acc_true = compare_i64_outputs(copy_i64_from_device(dC64, ldc * n), h_exact);
                        const std::vector<double> gemmul8_watt_true = cfg.enable_watt
                                                                           ? bench_gemmul8_i32_watt<true>(
                                                                                 handle, gemmul8::I32Scheme::OZAKI2_CRT, op_A, op_B, m, n, k,
                                                                                 dA, lda, dB, ldb, dC64, ldc, num_moduli)
                                                                           : std::vector<double> {};

                        const I32BenchBreakdown b_false = bench_gemmul8_i32_ms<false>(
                            handle, gemmul8::I32Scheme::OZAKI2_CRT, op_A, op_B, m, n, k,
                            dA, lda, dB, ldb, dC64, ldc, num_moduli, warmup, iters);
                        const AccuracyStats acc_false = compare_i64_outputs(copy_i64_from_device(dC64, ldc * n), h_exact);
                        const std::vector<double> gemmul8_watt_false = cfg.enable_watt
                                                                            ? bench_gemmul8_i32_watt<false>(
                                                                                  handle, gemmul8::I32Scheme::OZAKI2_CRT, op_A, op_B, m, n, k,
                                                                                  dA, lda, dB, ldb, dC64, ldc, num_moduli)
                                                                            : std::vector<double> {};

                        if (cfg.tune_moduli) {
                            candidates.push_back(BenchCandidate {
                                true,
                                num_moduli,
                                pr,
                                b_true,
                                acc_true,
                                std::move(gemmul8_watt_true),
                                "I32-accu-" + std::to_string(num_moduli) + "-extra",
                            });
                            candidates.push_back(BenchCandidate {
                                false,
                                num_moduli,
                                pr,
                                b_false,
                                acc_false,
                                std::move(gemmul8_watt_false),
                                "I32-accu-" + std::to_string(num_moduli) + "-compact",
                            });
                        } else {
                            dump_row(
                                gemmul8::I32Scheme::OZAKI2_CRT,
                                true,
                                num_moduli,
                                pr,
                                b_true,
                                acc_true,
                                gemmul8_watt_true,
                                "I32-accu-" + std::to_string(num_moduli) + "-extra");
                            dump_row(
                                gemmul8::I32Scheme::OZAKI2_CRT,
                                false,
                                num_moduli,
                                pr,
                                b_false,
                                acc_false,
                                gemmul8_watt_false,
                                "I32-accu-" + std::to_string(num_moduli) + "-compact");
                        }
                    }

                    if (cfg.tune_moduli) {
                        auto candidate_tier = [](const BenchCandidate &c) -> int {
                            if (c.breakdown.total_ms <= 0.0) return 0;
                            if (c.pr.exact_guaranteed && c.acc.exact_match) return 3;
                            if (c.acc.exact_match) return 2;
                            return 1;
                        };
                        auto better_candidate = [&](const BenchCandidate &lhs, const BenchCandidate &rhs) -> bool {
                            const int lhs_tier = candidate_tier(lhs);
                            const int rhs_tier = candidate_tier(rhs);
                            if (lhs_tier != rhs_tier) return lhs_tier > rhs_tier;
                            if (std::abs(lhs.breakdown.total_ms - rhs.breakdown.total_ms) > 1e-12) {
                                return lhs.breakdown.total_ms < rhs.breakdown.total_ms;
                            }
                            return lhs.num_moduli < rhs.num_moduli;
                        };

                        auto choose_best = [&](const bool use_extra) -> const BenchCandidate * {
                            const BenchCandidate *best = nullptr;
                            for (const auto &cand : candidates) {
                                if (cand.use_extra != use_extra) continue;
                                if (best == nullptr || better_candidate(cand, *best)) {
                                    best = &cand;
                                }
                            }
                            return best;
                        };

                        const BenchCandidate *best_true = choose_best(true);
                        const BenchCandidate *best_false = choose_best(false);
                        if (best_true == nullptr || best_false == nullptr) {
                            throw std::runtime_error("failed to select tuned moduli candidate");
                        }

                        auto log_selected = [&](const BenchCandidate &cand) {
                            const int tier = candidate_tier(cand);
                            std::cout << "[TUNE] m=" << m
                                      << " n=" << n
                                      << " k=" << k
                                      << " op=" << op_pair_to_string(op_A, op_B)
                                      << " scheme=oz2"
                                      << " use_extra=" << (cand.use_extra ? "true" : "false")
                                      << " selected_mod=" << cand.num_moduli
                                      << " tier=" << tier
                                      << " exact_guaranteed=" << (cand.pr.exact_guaranteed ? "yes" : "no")
                                      << " exact_match=" << (cand.acc.exact_match ? "yes" : "no")
                                      << " total_ms=" << cand.breakdown.total_ms
                                      << '\n';
                        };

                        log_selected(*best_true);
                        log_selected(*best_false);

                        dump_row(
                            gemmul8::I32Scheme::OZAKI2_CRT,
                            best_true->use_extra,
                            best_true->num_moduli,
                            best_true->pr,
                            best_true->breakdown,
                            best_true->acc,
                            best_true->watt,
                            best_true->func_label);
                        dump_row(
                            gemmul8::I32Scheme::OZAKI2_CRT,
                            best_false->use_extra,
                            best_false->num_moduli,
                            best_false->pr,
                            best_false->breakdown,
                            best_false->acc,
                            best_false->watt,
                            best_false->func_label);
                    }
                }

                if (run_oz1) {
                    constexpr unsigned num_moduli = 5u;
                    const PrecisionReport pr = evaluate_precision_oz1(max_abs_a, max_abs_b, k);

                    const I32BenchBreakdown b_true = bench_gemmul8_i32_ms<true>(
                        handle, gemmul8::I32Scheme::OZAKI1_SPLIT, op_A, op_B, m, n, k,
                        dA, lda, dB, ldb, dC64, ldc, num_moduli, warmup, iters);
                    const AccuracyStats acc_true = compare_i64_outputs(copy_i64_from_device(dC64, ldc * n), h_exact);
                    const std::vector<double> gemmul8_watt_true = cfg.enable_watt
                                                                       ? bench_gemmul8_i32_watt<true>(
                                                                             handle, gemmul8::I32Scheme::OZAKI1_SPLIT, op_A, op_B, m, n, k,
                                                                             dA, lda, dB, ldb, dC64, ldc, num_moduli)
                                                                       : std::vector<double> {};

                    const I32BenchBreakdown b_false = bench_gemmul8_i32_ms<false>(
                        handle, gemmul8::I32Scheme::OZAKI1_SPLIT, op_A, op_B, m, n, k,
                        dA, lda, dB, ldb, dC64, ldc, num_moduli, warmup, iters);
                    const AccuracyStats acc_false = compare_i64_outputs(copy_i64_from_device(dC64, ldc * n), h_exact);
                    const std::vector<double> gemmul8_watt_false = cfg.enable_watt
                                                                        ? bench_gemmul8_i32_watt<false>(
                                                                              handle, gemmul8::I32Scheme::OZAKI1_SPLIT, op_A, op_B, m, n, k,
                                                                              dA, lda, dB, ldb, dC64, ldc, num_moduli)
                                                                        : std::vector<double> {};

                    dump_row(
                        gemmul8::I32Scheme::OZAKI1_SPLIT,
                        true,
                        num_moduli,
                        pr,
                        b_true,
                        acc_true,
                        gemmul8_watt_true,
                        "I32-os1-5-extra");
                    dump_row(
                        gemmul8::I32Scheme::OZAKI1_SPLIT,
                        false,
                        num_moduli,
                        pr,
                        b_false,
                        acc_false,
                        gemmul8_watt_false,
                        "I32-os1-5-compact");
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
