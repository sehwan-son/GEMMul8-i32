#include "../include/gemmul8.hpp"
#include "self_hipify.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

double compute_gflops(const size_t m, const size_t n, const size_t k, const double elapsed_ms) {
    if (elapsed_ms <= 0.0) return 0.0;
    const double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    return ops / (elapsed_ms * 1.0e6);
}

struct RunStats {
    double gemmul8_ms     = 0.0;
    double gemmul8_gflops = 0.0;
    double cpu_ref_ms     = 0.0;
    double cpu_ref_gflops = 0.0;
};

void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
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

void fill_matrix_i64(
    std::vector<int64_t> &mat,
    const size_t rows,
    const size_t cols,
    const size_t ld,
    std::mt19937 &rng //
) {
    std::uniform_int_distribution<int64_t> dist(-1000000LL, 1000000LL);
    std::fill(mat.begin(), mat.end(), 0);
    for (size_t col = 0; col < cols; ++col) {
        for (size_t row = 0; row < rows; ++row) {
            mat[col * ld + row] = dist(rng);
        }
    }
}

int64_t ref_dot(
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

void cpu_gemm_ref(
    const std::vector<int32_t> &A,
    const std::vector<int32_t> &B,
    const cublasOperation_t op_A,
    const cublasOperation_t op_B,
    const size_t m,
    const size_t n,
    const size_t k,
    const size_t lda,
    const size_t ldb,
    const size_t ldc,
    const int64_t alpha,
    const int64_t beta,
    const std::vector<int64_t> &C_in,
    std::vector<int64_t> &C_out //
) {
    C_out = C_in;
    for (size_t col = 0; col < n; ++col) {
        for (size_t row = 0; row < m; ++row) {
            const int64_t AB = ref_dot(A, B, op_A, op_B, lda, ldb, row, col, k);
            const int64_t C0 = C_in[col * ldc + row];
            const __int128 out = static_cast<__int128>(alpha) * static_cast<__int128>(AB) +
                                 static_cast<__int128>(beta) * static_cast<__int128>(C0);
            C_out[col * ldc + row] = static_cast<int64_t>(out);
        }
    }
}

template <bool UseExtraWorkspace>
RunStats run_case(
    cublasHandle_t handle,
    const cublasOperation_t op_A,
    const cublasOperation_t op_B,
    const size_t m, const size_t n, const size_t k,
    const unsigned num_moduli,
    std::mt19937 &rng,
    const int64_t alpha,
    const int64_t beta //
) {
    const size_t lda = (op_A == CUBLAS_OP_N) ? m : k;
    const size_t ldb = (op_B == CUBLAS_OP_N) ? k : n;
    const size_t ldc = m;

    const size_t rowsA = (op_A == CUBLAS_OP_N) ? m : k;
    const size_t colsA = (op_A == CUBLAS_OP_N) ? k : m;
    const size_t rowsB = (op_B == CUBLAS_OP_N) ? k : n;
    const size_t colsB = (op_B == CUBLAS_OP_N) ? n : k;

    std::vector<int32_t> hA(lda * colsA);
    std::vector<int32_t> hB(ldb * colsB);
    std::vector<int64_t> hC(ldc * n, 0);
    std::vector<int64_t> hC_init(ldc * n, 0);
    std::vector<int64_t> hRef(ldc * n, 0);

    fill_matrix(hA, rowsA, colsA, lda, rng);
    fill_matrix(hB, rowsB, colsB, ldb, rng);
    fill_matrix_i64(hC_init, m, n, ldc, rng);
    const auto cpu_t0 = std::chrono::high_resolution_clock::now();
    cpu_gemm_ref(hA, hB, op_A, op_B, m, n, k, lda, ldb, ldc, alpha, beta, hC_init, hRef);
    const auto cpu_t1 = std::chrono::high_resolution_clock::now();
    const double cpu_ref_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_t1 - cpu_t0).count()) * 1.0e-6;

    int32_t *dA = nullptr;
    int32_t *dB = nullptr;
    int64_t *dC = nullptr;
    void *work  = nullptr;

    check_cuda(cudaMalloc(&dA, sizeof(int32_t) * hA.size()), "cudaMalloc dA");
    check_cuda(cudaMalloc(&dB, sizeof(int32_t) * hB.size()), "cudaMalloc dB");
    check_cuda(cudaMalloc(&dC, sizeof(int64_t) * hC.size()), "cudaMalloc dC");

    const size_t work_size = gemmul8::workSize_i32<UseExtraWorkspace>(m, n, k, num_moduli);
    check_cuda(cudaMalloc(&work, work_size), "cudaMalloc work");

    check_cuda(cudaMemcpy(dA, hA.data(), sizeof(int32_t) * hA.size(), cudaMemcpyHostToDevice), "cudaMemcpy A");
    check_cuda(cudaMemcpy(dB, hB.data(), sizeof(int32_t) * hB.size(), cudaMemcpyHostToDevice), "cudaMemcpy B");
    check_cuda(cudaMemcpy(dC, hC_init.data(), sizeof(int64_t) * hC.size(), cudaMemcpyHostToDevice), "cudaMemcpy C init");

    const std::vector<double> t_ns = gemmul8::gemm_i32<UseExtraWorkspace>(
        handle,
        op_A, op_B,
        m, n, k,
        &alpha,
        dA, lda,
        dB, ldb,
        &beta,
        dC, ldc,
        num_moduli,
        work);
    const double gemmul8_ms = (t_ns[0] + t_ns[1] + t_ns[2] + t_ns[3]) * 1.0e-6;

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check_cuda(cudaMemcpy(hC.data(), dC, sizeof(int64_t) * hC.size(), cudaMemcpyDeviceToHost), "cudaMemcpy C");

    for (size_t i = 0; i < hC.size(); ++i) {
        if (hC[i] != hRef[i]) {
            const size_t col = i / ldc;
            const size_t row = i - col * ldc;
            std::cerr << "[FAIL] mismatch"
                      << " use_extra=" << (UseExtraWorkspace ? "true" : "false")
                      << " opA=" << ((op_A == CUBLAS_OP_N) ? "N" : "T")
                      << " opB=" << ((op_B == CUBLAS_OP_N) ? "N" : "T")
                      << " m=" << m << " n=" << n << " k=" << k
                      << " alpha=" << alpha << " beta=" << beta
                      << " num_moduli=" << num_moduli
                      << " row=" << row << " col=" << col
                      << " got=" << hC[i] << " ref=" << hRef[i]
                      << std::endl;
            throw std::runtime_error("test_int32 mismatch");
        }
    }

    cudaFree(work);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    RunStats out;
    out.gemmul8_ms     = gemmul8_ms;
    out.gemmul8_gflops = compute_gflops(m, n, k, gemmul8_ms);
    out.cpu_ref_ms     = cpu_ref_ms;
    out.cpu_ref_gflops = compute_gflops(m, n, k, cpu_ref_ms);
    return out;
}

void run_invalid_num_moduli_case(cublasHandle_t handle) {
    const size_t m = 8;
    const size_t n = 8;
    const size_t k = 8;

    int32_t *dA = nullptr;
    int32_t *dB = nullptr;
    int64_t *dC = nullptr;
    void *work  = nullptr;

    check_cuda(cudaMalloc(&dA, sizeof(int32_t) * m * k), "cudaMalloc invalid dA");
    check_cuda(cudaMalloc(&dB, sizeof(int32_t) * k * n), "cudaMalloc invalid dB");
    check_cuda(cudaMalloc(&dC, sizeof(int64_t) * m * n), "cudaMalloc invalid dC");

    const size_t work_size = gemmul8::workSize_i32<true>(m, n, k, 9u);
    check_cuda(cudaMalloc(&work, work_size), "cudaMalloc invalid work");

    bool threw = false;
    try {
        constexpr int64_t alpha = 1;
        constexpr int64_t beta  = 0;
        gemmul8::gemm_i32<true>(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            dA, m,
            dB, k,
            &beta,
            dC, m,
            8u,
            work);
    } catch (const std::invalid_argument &) {
        threw = true;
    }

    cudaFree(work);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    if (!threw) {
        throw std::runtime_error("gemm_i32 must throw on num_moduli=8.");
    }
}

} // namespace

int main() {
    try {
        cublasHandle_t handle;
        cublasCreate(&handle);

        std::mt19937 rng(123456u);
        const std::vector<std::tuple<size_t, size_t, size_t>> shapes = {
            {256u, 256u, 256u},
            {512u, 512u, 512u},
            {1024u, 1024u, 1024u},
            {4096u, 4096u, 4096u},
        };
        const std::vector<std::pair<cublasOperation_t, cublasOperation_t>> ops = {
            {CUBLAS_OP_N, CUBLAS_OP_N},
            {CUBLAS_OP_N, CUBLAS_OP_T},
            {CUBLAS_OP_T, CUBLAS_OP_N},
        };
        const std::vector<unsigned> moduli_list = {9u, 20u};
        const std::vector<std::pair<int64_t, int64_t>> alpha_beta_list = {
            {1, 0},
            {2, 3},
            {-1, 1},
        };

        for (const auto &[alpha, beta] : alpha_beta_list) {
            for (const auto &[m, n, k] : shapes) {
                for (const auto &[op_A, op_B] : ops) {
                    for (const auto num_moduli : moduli_list) {
                        const RunStats stats_true  = run_case<true>(handle, op_A, op_B, m, n, k, num_moduli, rng, alpha, beta);
                        const RunStats stats_false = run_case<false>(handle, op_A, op_B, m, n, k, num_moduli, rng, alpha, beta);
                        std::cout << "[PASS] "
                                  << "m=" << m << " n=" << n << " k=" << k
                                  << " opA=" << ((op_A == CUBLAS_OP_N) ? "N" : "T")
                                  << " opB=" << ((op_B == CUBLAS_OP_N) ? "N" : "T")
                                  << " alpha=" << alpha << " beta=" << beta
                                  << " num_moduli=" << num_moduli
                                  << " use_extra=true"
                                  << " gemmul8_ms=" << stats_true.gemmul8_ms
                                  << " gemmul8_gflops=" << stats_true.gemmul8_gflops
                                  << " cpu_ref_ms=" << stats_true.cpu_ref_ms
                                  << " cpu_ref_gflops=" << stats_true.cpu_ref_gflops
                                  << std::endl;
                        std::cout << "[PASS] "
                                  << "m=" << m << " n=" << n << " k=" << k
                                  << " opA=" << ((op_A == CUBLAS_OP_N) ? "N" : "T")
                                  << " opB=" << ((op_B == CUBLAS_OP_N) ? "N" : "T")
                                  << " alpha=" << alpha << " beta=" << beta
                                  << " num_moduli=" << num_moduli
                                  << " use_extra=false"
                                  << " gemmul8_ms=" << stats_false.gemmul8_ms
                                  << " gemmul8_gflops=" << stats_false.gemmul8_gflops
                                  << " cpu_ref_ms=" << stats_false.cpu_ref_ms
                                  << " cpu_ref_gflops=" << stats_false.cpu_ref_gflops
                                  << std::endl;
                    }
                }
            }
        }

        run_invalid_num_moduli_case(handle);
        std::cout << "[PASS] invalid num_moduli case" << std::endl;

        cublasDestroy(handle);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[FAIL] " << e.what() << std::endl;
        return 1;
    }
}
