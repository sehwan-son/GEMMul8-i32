#include "../include/gemmul8.hpp"
#include "self_hipify.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

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

void fill_matrix(
    std::vector<int32_t> &mat,
    const size_t rows,
    const size_t cols,
    const size_t ld,
    std::mt19937 &rng //
) {
    std::uniform_int_distribution<int32_t> dist(-1024, 1024);
    std::fill(mat.begin(), mat.end(), 0);
    for (size_t col = 0; col < cols; ++col) {
        for (size_t row = 0; row < rows; ++row) {
            mat[col * ld + row] = dist(rng);
        }
    }
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
    const int32_t beta  = 0;
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
    constexpr int32_t beta  = 0;

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

struct I32BenchBreakdown {
    double encode_ms     = 0.0;
    double tc_gemm_ms    = 0.0;
    double conv32to8_ms  = 0.0;
    double reconstruct_ms = 0.0;
    double total_ms      = 0.0;
};

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
    constexpr int64_t beta  = 0;
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

} // namespace

int main(int argc, char **argv) {
    try {
        int iters = 5;
        int warmup = 2;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }
        if (argc >= 3) {
            warmup = std::max(0, std::atoi(argv[2]));
        }

        cublasHandle_t handle;
        check_cublas(cublasCreate(&handle), "cublasCreate");

        int dev = 0;
        check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
        cudaDeviceProp prop;
        check_cuda(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");

        const std::string device_name = sanitize_token(std::string(prop.name));
        const std::string timestamp = make_timestamp();
        const std::string file_name = "i32_bench_speedup_" + device_name + "_" + timestamp + ".csv";

        const cublasStatus_t int32_status = probe_cublas_int32(handle);
        const bool cublas_int32_supported = (int32_status == CUBLAS_STATUS_SUCCESS);

        std::ofstream out(file_name);
        out << std::scientific;
        out << "m,n,k,opA,opB,use_extra,num_moduli,iters,warmup,"
               "gemmul8_total_ms,gemmul8_encode_ms,gemmul8_tc_ms,gemmul8_conv32to8_ms,gemmul8_reconstruct_ms,"
               "gemmul8_gflops,"
               "exact_i32_i32_i64_ms,speedup_vs_exact_i32_i32_i64,"
               "exact_i32_i32_i64_gflops,"
               "cublas_i8_single_ms,cublas_i8_x_moduli_ms,speedup_vs_cublas_i8_single,speedup_vs_cublas_i8_x_moduli,"
               "cublas_i8_single_gflops,cublas_i8_x_moduli_gflops,"
               "cublas_i32_supported,cublas_i32_probe_status\n";

        std::cout << "[INFO] device=" << prop.name
                  << " compute_cap=" << prop.major << "." << prop.minor << '\n';
        std::cout << "[INFO] cublas int32*int32->int32 support="
                  << (cublas_int32_supported ? "yes" : "no")
                  << " (status=" << static_cast<int>(int32_status) << ")\n";
        std::cout << "[INFO] benchmark settings: iters=" << iters << " warmup=" << warmup << '\n';
        std::cout << "[INFO] csv=" << file_name << '\n';

        std::mt19937 rng(123456u);

        const std::vector<size_t> size_list = {256u, 512u, 1024u, 4096u};
        const std::vector<unsigned> moduli_list = {9u, 20u};
        const std::vector<std::pair<cublasOperation_t, cublasOperation_t>> ops = {
            {CUBLAS_OP_N, CUBLAS_OP_N},
            {CUBLAS_OP_N, CUBLAS_OP_T},
            {CUBLAS_OP_T, CUBLAS_OP_N},
        };

        for (const size_t n_dim : size_list) {
            const size_t m = n_dim;
            const size_t n = n_dim;
            const size_t k = n_dim;

            for (const auto &op_pair : ops) {
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
                fill_matrix(hA, rowsA, colsA, lda, rng);
                fill_matrix(hB, rowsB, colsB, ldb, rng);

                std::vector<int8_t> hA8;
                std::vector<int8_t> hB8;
                cast_i32_to_i8(hA, hA8);
                cast_i32_to_i8(hB, hB8);

                int32_t *dA = nullptr;
                int32_t *dB = nullptr;
                int64_t *dC64 = nullptr;
                int8_t *dA8 = nullptr;
                int8_t *dB8 = nullptr;
                int32_t *dC32 = nullptr;

                check_cuda(cudaMalloc(&dA, sizeof(int32_t) * hA.size()), "cudaMalloc dA");
                check_cuda(cudaMalloc(&dB, sizeof(int32_t) * hB.size()), "cudaMalloc dB");
                check_cuda(cudaMalloc(&dC64, sizeof(int64_t) * ldc * n), "cudaMalloc dC64");
                check_cuda(cudaMalloc(&dA8, sizeof(int8_t) * hA8.size()), "cudaMalloc dA8");
                check_cuda(cudaMalloc(&dB8, sizeof(int8_t) * hB8.size()), "cudaMalloc dB8");
                check_cuda(cudaMalloc(&dC32, sizeof(int32_t) * ldc * n), "cudaMalloc dC32");

                check_cuda(cudaMemcpy(dA, hA.data(), sizeof(int32_t) * hA.size(), cudaMemcpyHostToDevice), "cudaMemcpy dA");
                check_cuda(cudaMemcpy(dB, hB.data(), sizeof(int32_t) * hB.size(), cudaMemcpyHostToDevice), "cudaMemcpy dB");
                check_cuda(cudaMemcpy(dA8, hA8.data(), sizeof(int8_t) * hA8.size(), cudaMemcpyHostToDevice), "cudaMemcpy dA8");
                check_cuda(cudaMemcpy(dB8, hB8.data(), sizeof(int8_t) * hB8.size(), cudaMemcpyHostToDevice), "cudaMemcpy dB8");

                const double cublas_i8_single_ms = bench_cublas_i8_ms(
                    handle, op_A, op_B, m, n, k,
                    dA8, lda, dB8, ldb, dC32, ldc,
                    warmup, iters);
                const double exact_i32_i32_i64_ms = bench_exact_i32_i32_i64_ms(
                    op_A, op_B, m, n, k,
                    dA, lda, dB, ldb, dC64, ldc,
                    warmup, iters);

                for (const unsigned num_moduli : moduli_list) {
                    const I32BenchBreakdown b_true = bench_gemmul8_i32_ms<true>(
                        handle,
                        op_A,
                        op_B,
                        m,
                        n,
                        k,
                        dA,
                        lda,
                        dB,
                        ldb,
                        dC64,
                        ldc,
                        num_moduli,
                        warmup,
                        iters);

                    const I32BenchBreakdown b_false = bench_gemmul8_i32_ms<false>(
                        handle,
                        op_A,
                        op_B,
                        m,
                        n,
                        k,
                        dA,
                        lda,
                        dB,
                        ldb,
                        dC64,
                        ldc,
                        num_moduli,
                        warmup,
                        iters);

                    const double cublas_i8_x_moduli_ms = cublas_i8_single_ms * static_cast<double>(num_moduli);
                    const double ops = gemm_int_ops(m, n, k);

                    auto dump_row = [&](const bool use_extra, const I32BenchBreakdown &b) {
                        const double speedup_exact = exact_i32_i32_i64_ms / b.total_ms;
                        const double speedup_single = cublas_i8_single_ms / b.total_ms;
                        const double speedup_x_moduli = cublas_i8_x_moduli_ms / b.total_ms;
                        const double gemmul8_gflops = compute_gflops(ops, b.total_ms);
                        const double exact_gflops = compute_gflops(ops, exact_i32_i32_i64_ms);
                        const double cublas_single_gflops = compute_gflops(ops, cublas_i8_single_ms);
                        const double cublas_x_moduli_gflops =
                            compute_gflops(ops * static_cast<double>(num_moduli), cublas_i8_x_moduli_ms);

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
                            << exact_i32_i32_i64_ms << ','
                            << speedup_exact << ','
                            << exact_gflops << ','
                            << cublas_i8_single_ms << ','
                            << cublas_i8_x_moduli_ms << ','
                            << speedup_single << ','
                            << speedup_x_moduli << ','
                            << cublas_single_gflops << ','
                            << cublas_x_moduli_gflops << ','
                            << (cublas_int32_supported ? 1 : 0) << ','
                            << static_cast<int>(int32_status)
                            << '\n';

                        std::cout << "[BENCH] n=" << n
                                  << " op=" << op_pair_to_string(op_A, op_B)
                                  << " mod=" << num_moduli
                                  << " use_extra=" << (use_extra ? "true" : "false")
                                  << " gemmul8_total_ms=" << b.total_ms
                                  << " gemmul8_gflops=" << gemmul8_gflops
                                  << " exact_i32_i64_ms=" << exact_i32_i32_i64_ms
                                  << " exact_gflops=" << exact_gflops
                                  << " speedup(exact)=" << speedup_exact
                                  << " cublas_i8_ms=" << cublas_i8_single_ms
                                  << " cublas_i8_gflops=" << cublas_single_gflops
                                  << " speedup(single)=" << speedup_single
                                  << " speedup(xmoduli)=" << speedup_x_moduli
                                  << '\n';
                    };

                    dump_row(true, b_true);
                    dump_row(false, b_false);
                }

                cudaFree(dA);
                cudaFree(dB);
                cudaFree(dC64);
                cudaFree(dA8);
                cudaFree(dB8);
                cudaFree(dC32);
            }
        }

        out.close();
        check_cublas(cublasDestroy(handle), "cublasDestroy");
        std::cout << "[INFO] benchmark finished\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
