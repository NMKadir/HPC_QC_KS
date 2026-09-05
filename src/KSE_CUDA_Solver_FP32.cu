%%writefile KSE_CUDA_Solver_FP32.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>

#define M_PI     3.14159265358979323846f
#define N        65536
#define L        (32.0f*M_PI)
#define DT       0.1f
#define NSTEPS   3000
#define SAVE_EVERY 10

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define CUFFT_CHECK(call) do { \
    cufftResult err = (call); \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %s:%d: code %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while (0)

__device__ cuFloatComplex cexp_device(cuFloatComplex z) {
    float ex = expf(z.x);   
    return make_cuFloatComplex(ex * cosf(z.y), ex * sinf(z.y));
}

static float h_x[N], h_k[N];
static std::complex<float> h_Lk[N];
static std::complex<float> h_E[N], h_E2[N];
static std::complex<float> h_Q[N], h_f1[N], h_f2[N], h_f3[N];
static std::complex<float> h_u_hat_init[N];

void host_init_grid(void) {
    const float dx = L / (float)N;
    const float k_factor = (2.0f * M_PI) / L;
    for (int i = 0; i < N; i++) {
        h_x[i] = i * dx;
        h_k[i] = (i < N / 2) ? i * k_factor : (i - N) * k_factor;
    }
}

void host_compute_linear_operator(void) {
    for (int i = 0; i < N; i++) {
        float k2 = h_k[i] * h_k[i]; 
        h_Lk[i] = k2 - k2 * k2;
    }
}

void host_compute_etdrk4_coefficients(void) {
    const int M = 32;
    const float R = 1.0f;
    for (int i = 0; i < N; i++) {
        std::complex<float> Lk = h_Lk[i];
        h_E[i]  = std::exp(Lk * DT);
        h_E2[i] = std::exp(Lk * (DT / 2.0f));

        std::complex<float> sum_Q(0.0f, 0.0f), sum_f1(0.0f, 0.0f),
                              sum_f2(0.0f, 0.0f), sum_f3(0.0f, 0.0f);
        for (int j = 0; j < M; j++) {
            float theta = (j + 0.5f) * 2.0f * M_PI / M;
            std::complex<float> offset(R * cosf(theta), R * sinf(theta));
            std::complex<float> z = Lk * DT + offset;

            std::complex<float> ez  = std::exp(z);
            std::complex<float> ez2 = std::exp(z / 2.0f);

            sum_Q  += (ez2 - 1.0f) / z;
            sum_f1 += (-4.0f - z + ez * (4.0f - 3.0f * z + z * z)) / (z * z * z);
            sum_f2 += (2.0f + z + ez * (-2.0f + z)) / (z * z * z);
            sum_f3 += (-4.0f - 3.0f * z - z * z + ez * (4.0f - z)) / (z * z * z);
        }
        h_Q[i]  = DT * (sum_Q / (float)M);
        h_f1[i] = DT * (sum_f1 / (float)M);
        h_f2[i] = DT * (sum_f2 / (float)M);
        h_f3[i] = DT * (sum_f3 / (float)M);
    }
}

void host_init_condition(void) {
    for (int i = 0; i < N; i++) {
        h_u_hat_init[i] = cosf(h_x[i] / 16.0f) * (1.0f + sinf(h_x[i] / 16.0f));
    }
}

__global__ void kernel_copy_in(cuFloatComplex *u_phys,
                                 const cuFloatComplex *u_in_hat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u_phys[i] = u_in_hat[i];
    }
}

__global__ void kernel_square_normalize(cuFloatComplex *u_phys, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float u_val = u_phys[i].x / (float)n;
        u_phys[i] = make_cuFloatComplex(u_val * u_val, 0.0f);
    }
}

__global__ void kernel_multiply_ik_dealias(cuFloatComplex *N_out_hat,
                                             const cuFloatComplex *u_phys,
                                             const float *k, int n, int kcut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int dist = (i <= n / 2) ? i : (n - i);
        if (dist > kcut) {
            N_out_hat[i] = make_cuFloatComplex(0.0f, 0.0f);
        } else {
            float w_x = u_phys[i].x;
            float w_y = u_phys[i].y;
            N_out_hat[i] = make_cuFloatComplex(0.5f * k[i] * w_y, -0.5f * k[i] * w_x);
        }
    }
}

__global__ void kernel_stage_a(cuFloatComplex *a, const cuFloatComplex *u_hat,
                                 const cuFloatComplex *E2, const cuFloatComplex *Q,
                                 const cuFloatComplex *Nu, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuFloatComplex term1 = cuCmulf(E2[i], u_hat[i]);
        cuFloatComplex term2 = cuCmulf(Q[i], Nu[i]);
        a[i] = cuCaddf(term1, term2);
    }
}

__global__ void kernel_stage_b(cuFloatComplex *b, const cuFloatComplex *u_hat,
                                 const cuFloatComplex *E2, const cuFloatComplex *Q,
                                 const cuFloatComplex *Na, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuFloatComplex term1 = cuCmulf(E2[i], u_hat[i]);
        cuFloatComplex term2 = cuCmulf(Q[i], Na[i]);
        b[i] = cuCaddf(term1, term2);
    }
}

__global__ void kernel_stage_c(cuFloatComplex *c, const cuFloatComplex *a,
                                 const cuFloatComplex *E2, const cuFloatComplex *Q,
                                 const cuFloatComplex *Nb, const cuFloatComplex *Nu, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuFloatComplex two_Nb = cuCaddf(Nb[i], Nb[i]);
        cuFloatComplex diff = cuCsubf(two_Nb, Nu[i]);
        cuFloatComplex term1 = cuCmulf(E2[i], a[i]);
        cuFloatComplex term2 = cuCmulf(Q[i], diff);
        c[i] = cuCaddf(term1, term2);
    }
}

__global__ void kernel_final_update(cuFloatComplex *u_hat,
                                      const cuFloatComplex *E,
                                      const cuFloatComplex *Nu, const cuFloatComplex *Na,
                                      const cuFloatComplex *Nb, const cuFloatComplex *Nc,
                                      const cuFloatComplex *f1, const cuFloatComplex *f2,
                                      const cuFloatComplex *f3, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuFloatComplex term1 = cuCmulf(E[i], u_hat[i]);
        cuFloatComplex term2 = cuCmulf(Nu[i], f1[i]);
        
        cuFloatComplex Na_plus_Nb = cuCaddf(Na[i], Nb[i]);
        cuFloatComplex two_Na_plus_Nb = cuCaddf(Na_plus_Nb, Na_plus_Nb);
        cuFloatComplex term3 = cuCmulf(two_Na_plus_Nb, f2[i]);
        
        cuFloatComplex term4 = cuCmulf(Nc[i], f3[i]);

        u_hat[i] = cuCaddf(cuCaddf(term1, term2), cuCaddf(term3, term4));
    }
}

static cuFloatComplex *d_u_hat, *d_u_phys;
static cuFloatComplex *d_E, *d_E2, *d_Q, *d_f1, *d_f2, *d_f3;
static cuFloatComplex *d_Nu, *d_Na, *d_Nb, *d_Nc, *d_a, *d_b, *d_c;
static float *d_k;
static cufftHandle plan;

static const int BLOCK = 256;
static const int GRID  = (N + BLOCK - 1) / BLOCK;

void compute_nonlinear(cuFloatComplex *u_in_hat, cuFloatComplex *N_out_hat) {
    kernel_copy_in<<<GRID, BLOCK>>>(d_u_phys, u_in_hat, N);

    CUFFT_CHECK(cufftExecC2C(plan, d_u_phys, d_u_phys, CUFFT_INVERSE));

    kernel_square_normalize<<<GRID, BLOCK>>>(d_u_phys, N);

    CUFFT_CHECK(cufftExecC2C(plan, d_u_phys, d_u_phys, CUFFT_FORWARD));

    int kcut = N / 3;
    kernel_multiply_ik_dealias<<<GRID, BLOCK>>>(N_out_hat, d_u_phys, d_k, N, kcut);
}

void etdrk4_step(void) {
    compute_nonlinear(d_u_hat, d_Nu);
    kernel_stage_a<<<GRID, BLOCK>>>(d_a, d_u_hat, d_E2, d_Q, d_Nu, N);

    compute_nonlinear(d_a, d_Na);
    kernel_stage_b<<<GRID, BLOCK>>>(d_b, d_u_hat, d_E2, d_Q, d_Na, N);

    compute_nonlinear(d_b, d_Nb);
    kernel_stage_c<<<GRID, BLOCK>>>(d_c, d_a, d_E2, d_Q, d_Nb, d_Nu, N);

    compute_nonlinear(d_c, d_Nc);
    kernel_final_update<<<GRID, BLOCK>>>(d_u_hat, d_E, d_Nu, d_Na, d_Nb, d_Nc,
                                          d_f1, d_f2, d_f3, N);
}

void save_snapshot(FILE *fp, int step) {
    static cuFloatComplex *d_tmp = NULL;
    static cuFloatComplex *h_tmp = NULL;
    if (!d_tmp) {
        CUDA_CHECK(cudaMalloc(&d_tmp, N * sizeof(cuFloatComplex)));
        h_tmp = (cuFloatComplex*)malloc(N * sizeof(cuFloatComplex));
    }

    CUDA_CHECK(cudaMemcpy(d_tmp, d_u_hat, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice));
    CUFFT_CHECK(cufftExecC2C(plan, d_tmp, d_tmp, CUFFT_INVERSE));
    CUDA_CHECK(cudaMemcpy(h_tmp, d_tmp, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

    fprintf(fp, "%d", step);
    for (int i = 0; i < N; i++) {
        float val = h_tmp[i].x / (float)N;
        fprintf(fp, ",%f", val);
    }
    fprintf(fp, "\n");
}

int main(void) {
    host_init_grid();
    host_compute_linear_operator();
    host_compute_etdrk4_coefficients();
    host_init_condition();

    CUDA_CHECK(cudaMalloc(&d_u_hat,  N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_u_phys, N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_k,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_E,      N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_E2,     N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_Q,      N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_f1,     N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_f2,     N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_f3,     N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_Nu,     N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_Na,     N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_Nb,     N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_Nc,     N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_a,      N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b,      N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_c,      N * sizeof(cuFloatComplex)));

    CUDA_CHECK(cudaMemcpy(d_k,  h_k,  N * sizeof(float),           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,  h_E,  N * sizeof(cuFloatComplex),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E2, h_E2, N * sizeof(cuFloatComplex),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q,  h_Q,  N * sizeof(cuFloatComplex),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f1, h_f1, N * sizeof(cuFloatComplex),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f2, h_f2, N * sizeof(cuFloatComplex),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f3, h_f3, N * sizeof(cuFloatComplex),  cudaMemcpyHostToDevice));

    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    CUDA_CHECK(cudaMemcpy(d_u_phys, h_u_hat_init, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    CUFFT_CHECK(cufftExecC2C(plan, d_u_phys, d_u_phys, CUFFT_FORWARD));
    CUDA_CHECK(cudaMemcpy(d_u_hat, d_u_phys, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice));

    FILE *fp = fopen("kse_output_cuda_fp32_65536.csv", "w");
    if (!fp) { perror("fopen"); return 1; }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int step = 0; step < NSTEPS; step++) {
        etdrk4_step();
        if (step % SAVE_EVERY == 0) {
            save_snapshot(fp, step);
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("CUDA Time: %f seconds\n", ms / 1000.0);

    fclose(fp);
    cufftDestroy(plan);

    printf("Done. Output written to kse_output_cuda.csv\n");
    return 0;
}
 /*   nvcc -O3 KSE_CUDA_Solver_FP32.cu -o kse_cuda_fp32 -lcufft
   ./kse_cuda_fp32
   */
