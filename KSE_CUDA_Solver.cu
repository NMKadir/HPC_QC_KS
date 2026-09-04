 #include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>

#define M_PI     3.14159265358979323846
#define N        32768
#define L        (32.0*M_PI)
#define DT       0.1
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

__device__ cuDoubleComplex cexp_device(cuDoubleComplex z) {
    double ex = exp(z.x);   
    return make_cuDoubleComplex(ex * cos(z.y), ex * sin(z.y));
}

static double h_x[N], h_k[N];
static std::complex<double> h_Lk[N];
static std::complex<double> h_E[N], h_E2[N];
static std::complex<double> h_Q[N], h_f1[N], h_f2[N], h_f3[N];
static std::complex<double> h_u_hat_init[N];

void host_init_grid(void) {
    const double dx = L / (double)N;
    const double k_factor = (2.0 * M_PI) / L;
    for (int i = 0; i < N; i++) {
        h_x[i] = i * dx;
        h_k[i] = (i < N / 2) ? i * k_factor : (i - N) * k_factor;
    }
}

void host_compute_linear_operator(void) {
    for (int i = 0; i < N; i++) {
        double k2 = h_k[i] * h_k[i]; 
        h_Lk[i] = k2 - k2 * k2;
    }
}

void host_compute_etdrk4_coefficients(void) {
    const int M = 32;
    const double R = 1.0;
    for (int i = 0; i < N; i++) {
        std::complex<double> Lk = h_Lk[i];
        h_E[i]  = std::exp(Lk * DT);
        h_E2[i] = std::exp(Lk * (DT / 2.0));

        std::complex<double> sum_Q(0.0, 0.0), sum_f1(0.0, 0.0),
                              sum_f2(0.0, 0.0), sum_f3(0.0, 0.0);
        for (int j = 0; j < M; j++) {
            double theta = (j + 0.5) * 2.0 * M_PI / M;
            std::complex<double> offset(R * cos(theta), R * sin(theta));
            std::complex<double> z = Lk * DT + offset;

            std::complex<double> ez  = std::exp(z);
            std::complex<double> ez2 = std::exp(z / 2.0);

            sum_Q  += (ez2 - 1.0) / z;
            sum_f1 += (-4.0 - z + ez * (4.0 - 3.0 * z + z * z)) / (z * z * z);
            sum_f2 += (2.0 + z + ez * (-2.0 + z)) / (z * z * z);
            sum_f3 += (-4.0 - 3.0 * z - z * z + ez * (4.0 - z)) / (z * z * z);
        }
        h_Q[i]  = DT * (sum_Q / (double)M);
        h_f1[i] = DT * (sum_f1 / (double)M);
        h_f2[i] = DT * (sum_f2 / (double)M);
        h_f3[i] = DT * (sum_f3 / (double)M);
    }
}

void host_init_condition(void) {
    for (int i = 0; i < N; i++) {
        h_u_hat_init[i] = cos(h_x[i] / 16.0) * (1.0 + sin(h_x[i] / 16.0));
    }
}

__global__ void kernel_copy_in(cuDoubleComplex *u_phys,
                                 const cuDoubleComplex *u_in_hat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u_phys[i] = u_in_hat[i];
    }
}

__global__ void kernel_square_normalize(cuDoubleComplex *u_phys, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double u_val = u_phys[i].x / (double)n;
        u_phys[i] = make_cuDoubleComplex(u_val * u_val, 0.0);
    }
}

__global__ void kernel_multiply_ik_dealias(cuDoubleComplex *N_out_hat,
                                             const cuDoubleComplex *u_phys,
                                             const double *k, int n, int kcut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int dist = (i <= n / 2) ? i : (n - i);
        if (dist > kcut) {
            N_out_hat[i] = make_cuDoubleComplex(0.0, 0.0);
        } else {
            double w_x = u_phys[i].x;
            double w_y = u_phys[i].y;
            N_out_hat[i] = make_cuDoubleComplex(0.5 * k[i] * w_y, -0.5 * k[i] * w_x);
        }
    }
}

__global__ void kernel_stage_a(cuDoubleComplex *a, const cuDoubleComplex *u_hat,
                                 const cuDoubleComplex *E2, const cuDoubleComplex *Q,
                                 const cuDoubleComplex *Nu, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuDoubleComplex term1 = cuCmul(E2[i], u_hat[i]);
        cuDoubleComplex term2 = cuCmul(Q[i], Nu[i]);
        a[i] = cuCadd(term1, term2);
    }
}

__global__ void kernel_stage_b(cuDoubleComplex *b, const cuDoubleComplex *u_hat,
                                 const cuDoubleComplex *E2, const cuDoubleComplex *Q,
                                 const cuDoubleComplex *Na, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuDoubleComplex term1 = cuCmul(E2[i], u_hat[i]);
        cuDoubleComplex term2 = cuCmul(Q[i], Na[i]);
        b[i] = cuCadd(term1, term2);
    }
}

__global__ void kernel_stage_c(cuDoubleComplex *c, const cuDoubleComplex *a,
                                 const cuDoubleComplex *E2, const cuDoubleComplex *Q,
                                 const cuDoubleComplex *Nb, const cuDoubleComplex *Nu, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuDoubleComplex two_Nb = cuCadd(Nb[i], Nb[i]);
        cuDoubleComplex diff = cuCsub(two_Nb, Nu[i]);
        cuDoubleComplex term1 = cuCmul(E2[i], a[i]);
        cuDoubleComplex term2 = cuCmul(Q[i], diff);
        c[i] = cuCadd(term1, term2);
    }
}

__global__ void kernel_final_update(cuDoubleComplex *u_hat,
                                      const cuDoubleComplex *E,
                                      const cuDoubleComplex *Nu, const cuDoubleComplex *Na,
                                      const cuDoubleComplex *Nb, const cuDoubleComplex *Nc,
                                      const cuDoubleComplex *f1, const cuDoubleComplex *f2,
                                      const cuDoubleComplex *f3, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuDoubleComplex term1 = cuCmul(E[i], u_hat[i]);
        cuDoubleComplex term2 = cuCmul(Nu[i], f1[i]);
        
        cuDoubleComplex Na_plus_Nb = cuCadd(Na[i], Nb[i]);
        cuDoubleComplex two_Na_plus_Nb = cuCadd(Na_plus_Nb, Na_plus_Nb);
        cuDoubleComplex term3 = cuCmul(two_Na_plus_Nb, f2[i]);
        
        cuDoubleComplex term4 = cuCmul(Nc[i], f3[i]);

        u_hat[i] = cuCadd(cuCadd(term1, term2), cuCadd(term3, term4));
    }
}

static cuDoubleComplex *d_u_hat, *d_u_phys;
static cuDoubleComplex *d_E, *d_E2, *d_Q, *d_f1, *d_f2, *d_f3;
static cuDoubleComplex *d_Nu, *d_Na, *d_Nb, *d_Nc, *d_a, *d_b, *d_c;
static double *d_k;
static cufftHandle plan;

static const int BLOCK = 256;
static const int GRID  = (N + BLOCK - 1) / BLOCK;

void compute_nonlinear(cuDoubleComplex *u_in_hat, cuDoubleComplex *N_out_hat) {
    kernel_copy_in<<<GRID, BLOCK>>>(d_u_phys, u_in_hat, N);

    CUFFT_CHECK(cufftExecZ2Z(plan, d_u_phys, d_u_phys, CUFFT_INVERSE));

    kernel_square_normalize<<<GRID, BLOCK>>>(d_u_phys, N);

    CUFFT_CHECK(cufftExecZ2Z(plan, d_u_phys, d_u_phys, CUFFT_FORWARD));

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
    static cuDoubleComplex *d_tmp = NULL;
    static cuDoubleComplex *h_tmp = NULL;
    if (!d_tmp) {
        CUDA_CHECK(cudaMalloc(&d_tmp, N * sizeof(cuDoubleComplex)));
        h_tmp = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    }

    CUDA_CHECK(cudaMemcpy(d_tmp, d_u_hat, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
    CUFFT_CHECK(cufftExecZ2Z(plan, d_tmp, d_tmp, CUFFT_INVERSE));
    CUDA_CHECK(cudaMemcpy(h_tmp, d_tmp, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    fprintf(fp, "%d", step);
    for (int i = 0; i < N; i++) {
        double val = h_tmp[i].x / (double)N;
        fprintf(fp, ",%f", val);
    }
    fprintf(fp, "\n");
}

int main(void) {
    host_init_grid();
    host_compute_linear_operator();
    host_compute_etdrk4_coefficients();
    host_init_condition();

    CUDA_CHECK(cudaMalloc(&d_u_hat,  N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_u_phys, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_k,      N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_E,      N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_E2,     N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Q,      N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_f1,     N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_f2,     N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_f3,     N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Nu,     N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Na,     N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Nb,     N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Nc,     N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_a,      N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_b,      N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_c,      N * sizeof(cuDoubleComplex)));

    CUDA_CHECK(cudaMemcpy(d_k,  h_k,  N * sizeof(double),           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,  h_E,  N * sizeof(cuDoubleComplex),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E2, h_E2, N * sizeof(cuDoubleComplex),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q,  h_Q,  N * sizeof(cuDoubleComplex),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f1, h_f1, N * sizeof(cuDoubleComplex),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f2, h_f2, N * sizeof(cuDoubleComplex),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f3, h_f3, N * sizeof(cuDoubleComplex),  cudaMemcpyHostToDevice));

    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1));

    CUDA_CHECK(cudaMemcpy(d_u_phys, h_u_hat_init, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUFFT_CHECK(cufftExecZ2Z(plan, d_u_phys, d_u_phys, CUFFT_FORWARD));
    CUDA_CHECK(cudaMemcpy(d_u_hat, d_u_phys, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    FILE *fp = fopen("csv files/kse_output_cuda.csv", "w");
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
 /*   nvcc -O3 KSE_CUDA_Solver.cu -o kse_cuda -lcufft
   ./kse_cuda
   */