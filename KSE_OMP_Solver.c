#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <omp.h>

#define M_PI     3.14159265358979323846
#define N        32768
#define L        (32.0*M_PI)
#define DT       0.1
#define NSTEPS   3000
#define SAVE_EVERY 10

static double x[N];
static double k[N];
static fftw_complex u_hat[N];
static fftw_complex u_phys[N];

static fftw_plan plan_fwd;
static fftw_plan plan_bwd;

static fftw_complex Lk[N];
static fftw_complex E[N], E2[N];
static fftw_complex Q[N], f1[N], f2[N], f3[N];


void init_fftw_threads(void) {
    if (!fftw_init_threads()) {
        fprintf(stderr, "Error: fftw_init_threads() failed. Multi-threaded FFTW could not be initialized.\n");
        exit(EXIT_FAILURE);
    }
    fftw_plan_with_nthreads(omp_get_max_threads());
}

void init_grid(void) {
    const double dx = L / (double)N;
    const double k_factor = (2.0*M_PI)/L;
    #pragma omp parallel
    {
        #pragma omp for
        for(int i=0; i<N; i++){
            x[i] = i*dx;
        }


        #pragma omp for
        for(int i=0; i<N/2; i++){
            k[i] = i*k_factor;
        }
        
        #pragma omp for
        for(int i=N/2; i<N; i++){
            k[i] = (i-N)*k_factor;
        }
    }
}

void init_condition(void) {
    #pragma omp parallel for
    for(int i=0; i<N; i++){
        u_phys[i] = cos(x[i]/16.0)*(1+sin(x[i]/16.0));
    }
    fftw_execute(plan_fwd);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) u_hat[i] = u_phys[i];
}

void compute_linear_operator(void) {
    #pragma omp parallel 
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            double k2 = k[i]*k[i];
            Lk[i] = k2 - (k2*k2);
        }
    }
}

void compute_etdrk4_coefficients(void) {
    const int M = 32;
    const double R = 1.0;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double complex z = Lk[i] * DT;
        E[i]  = cexp(z);
        E2[i] = cexp(z / 2.0);

        double complex sum_Q  = 0.0;
        double complex sum_f1 = 0.0;
        double complex sum_f2 = 0.0;
        double complex sum_f3 = 0.0;

        for (int j = 0; j < M; j++) {
            double theta = 2.0 * M_PI * ((double)j + 0.5) / (double)M;
            double complex r = z + R * cexp(I * theta);

            double complex exp_r_half = cexp(r / 2.0);
            double complex exp_r      = cexp(r);
            double complex r2         = r * r;
            double complex r3         = r2 * r;

            sum_Q  += (exp_r_half - 1.0) / r;
            sum_f1 += (-4.0 - r + exp_r * (4.0 - 3.0 * r + r2)) / r3;
            sum_f2 += (2.0 + r + exp_r * (-2.0 + r)) / r3;
            sum_f3 += (-4.0 - 3.0 * r - r2 + exp_r * (4.0 - r)) / r3;
        }
        Q[i]  = DT * (sum_Q  / (double)M);
        f1[i] = DT * (sum_f1 / (double)M);
        f2[i] = DT * (sum_f2 / (double)M);
        f3[i] = DT * (sum_f3 / (double)M);
    }
}

void compute_nonlinear(fftw_complex *u_in_hat, fftw_complex *N_out_hat) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            u_phys[i] = u_in_hat[i];
        }
    }

    fftw_execute(plan_bwd);   

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            double u_val = creal(u_phys[i]) / (double)N;
            u_phys[i] = u_val * u_val;
        }
    }
    fftw_execute(plan_fwd);   

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            N_out_hat[i] = -0.5 * I * k[i] * u_phys[i];
        }
    
    int kcut = N / 3;
        #pragma omp for
        for (int i = 0; i < N; i++) {
            int dist = (i <= N/2) ? i : (N - i);
            if (dist > kcut) {
                N_out_hat[i] = 0.0;
            }
        }
    }
}

void etdrk4_step(void) {
    static fftw_complex Nu[N], Na[N], Nb[N], Nc[N];
    static fftw_complex a[N], b[N], c[N];

    compute_nonlinear(u_hat, Nu);
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            a[i] = E2[i] * u_hat[i] + Q[i] * Nu[i];
        }
    }
    compute_nonlinear(a, Na);
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            b[i] = E2[i] * u_hat[i] + Q[i] * Na[i];
        }
    }
    compute_nonlinear(b, Nb);
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            c[i] = E2[i] * a[i] + Q[i] * (2.0 * Nb[i] - Nu[i]);
        }
    }
    compute_nonlinear(c, Nc);

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            u_hat[i] = E[i] * u_hat[i]
                     + Nu[i] * f1[i]
                     + 2.0 * (Na[i] + Nb[i]) * f2[i]
                    + Nc[i] * f3[i];
        }
    }
}

void save_snapshot(FILE *fp, int step) {
    fftw_complex tmp[N];
    #pragma omp parallel for
    for (int i = 0; i < N; i++) tmp[i] = u_hat[i];
    fftw_execute_dft(plan_bwd, tmp, tmp);

    fprintf(fp, "%d", step);
    for (int i = 0; i < N; i++) {
        double val = creal(tmp[i]) / N;
        fprintf(fp, ",%f", val);
    }
    fprintf(fp, "\n");
}

int main(void) {
    double t_start = omp_get_wtime();
    
    init_fftw_threads();
    plan_fwd = fftw_plan_dft_1d(N, u_phys, u_phys, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_bwd = fftw_plan_dft_1d(N, u_phys, u_phys, FFTW_BACKWARD, FFTW_ESTIMATE);

    init_grid();
    compute_linear_operator();
    compute_etdrk4_coefficients();
    init_condition();

    FILE *fp = fopen("csv files/kse_output_omp.csv", "w");
    if (!fp) { perror("fopen"); return 1; }

    for (int step = 0; step < NSTEPS; step++) {
        etdrk4_step();
        if (step % SAVE_EVERY == 0) {
            save_snapshot(fp, step);
        }
    }

    double t_end = omp_get_wtime();

    printf("Threads: %d, Time: %f seconds\n",omp_get_max_threads(), t_end - t_start);
    fclose(fp);
    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_cleanup_threads();   

    printf("Done. Output written to kse_output_omp.csv\n");
    return 0;
}
/*gcc -O3 -fopenmp KSE_OMP_Solver.c -o kse_omp -lfftw3_threads -lfftw3 -lm -lpthread
or, # While inside the 'HPC Project' folder:
gcc -O3 -fopenmp ../KSE_OMP_Solver.c -o kse_omp -lfftw3_threads -lfftw3 -lm -lpthread
OMP_NUM_THREADS=1 ./kse_omp
OMP_NUM_THREADS=2 ./kse_omp
OMP_NUM_THREADS=4 ./kse_omp
OMP_NUM_THREADS=8 ./kse_omp*/