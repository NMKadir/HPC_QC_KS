#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>     
#include <mpi.h>

#define M_PI     3.14159265358979323846
#define N        32768
#define L        (32.0*M_PI)
#define DT       0.1
#define NSTEPS   3000
#define SAVE_EVERY 10

static ptrdiff_t local_n0;        
static ptrdiff_t local_0_start;   
static ptrdiff_t alloc_local;     

static double *x;                 
static double *k;                 
static fftw_complex *u_hat;       
static fftw_complex *u_phys;      

static fftw_plan plan_fwd;
static fftw_plan plan_bwd;

static fftw_complex *Lk;
static fftw_complex *E, *E2;
static fftw_complex *Q, *f1, *f2, *f3;

static int rank, nprocs;          

double global_wavenumber(int g) {
    const double k_factor = (2.0*M_PI)/L;
    if (g < N / 2) {
        return g * k_factor;
    } else {
        return (g - N) * k_factor;
    }
}

void init_grid(void) {
    const double dx = L / (double)N;
    for (ptrdiff_t i = 0; i < local_n0; i++) {
        int g = (int)(local_0_start + i);
        x[i] = g*dx;
        k[i] = global_wavenumber(g);
    }
}

void init_condition(void) {
    for (ptrdiff_t i = 0; i < local_n0; i++) {
        u_phys[i] = cos(x[i]/16.0)*(1+sin(x[i]/16.0));
    }
    fftw_execute(plan_fwd);   
    for (ptrdiff_t i = 0; i < local_n0; i++) u_hat[i] = u_phys[i];
}

void compute_linear_operator(void) {
    for (ptrdiff_t i = 0; i < local_n0; i++) {
        double k2 = k[i]*k[i];
        Lk[i] = k2 - (k2*k2);
    }
}

void compute_etdrk4_coefficients(void) {
    const int M = 32;
    const double R = 1.0;
    for (ptrdiff_t i = 0; i < local_n0; i++) {
        double complex L_val = Lk[i];

        E[i]  = cexp(L_val * DT);
        E2[i] = cexp(L_val * (DT / 2.0));

        double complex q_sum  = 0.0;
        double complex f1_sum = 0.0;
        double complex f2_sum = 0.0;
        double complex f3_sum = 0.0;

        for (int m = 0; m < M; m++) {
            double theta = (m + 0.5) * (2.0 * M_PI / M);
            double complex r = R * cexp(I * theta);
            double complex z = L_val * DT + r;

            q_sum  += (cexp(z / 2.0) - 1.0) / z;
            f1_sum += (-4.0 - z + cexp(z) * (4.0 - 3.0 * z + z * z)) / (z * z * z);
            f2_sum += (2.0 + z + cexp(z) * (-2.0 + z)) / (z * z * z);
            f3_sum += (-4.0 - 3.0 * z - z * z + cexp(z) * (4.0 - z)) / (z * z * z);
        }

        Q[i]  = DT * (q_sum / (double)M);
        f1[i] = DT * (f1_sum / (double)M);
        f2[i] = DT * (f2_sum / (double)M);
        f3[i] = DT * (f3_sum / (double)M);
    }
}

void compute_nonlinear(fftw_complex *u_in_hat, fftw_complex *N_out_hat) {
    for (ptrdiff_t i = 0; i < local_n0; i++) {
        u_phys[i] = u_in_hat[i];
    }
    fftw_execute(plan_bwd);   

    for (ptrdiff_t i = 0; i < local_n0; i++) {
        double u_val = creal(u_phys[i]) / (double)N;
        u_phys[i] = u_val * u_val;
    }

    fftw_execute(plan_fwd);   

    for (ptrdiff_t i = 0; i < local_n0; i++) {
        N_out_hat[i] = -0.5 * I * k[i] * u_phys[i];
    }

    int kcut = N / 3;
    for (ptrdiff_t i = 0; i < local_n0; i++) {
        int g = (int)(local_0_start+i);
        int dist = (g<=N/2) ? g : (N-g);
        if(dist>kcut) N_out_hat[i] = 0.0;
    }
}

void etdrk4_step(void) {
    static fftw_complex *Nu, *Na, *Nb, *Nc, *a, *b, *c;
    static int allocated = 0;
    if (!allocated) {
        Nu = fftw_alloc_complex(local_n0); Na = fftw_alloc_complex(local_n0);
        Nb = fftw_alloc_complex(local_n0); Nc = fftw_alloc_complex(local_n0);
        a  = fftw_alloc_complex(local_n0); b  = fftw_alloc_complex(local_n0);
        c  = fftw_alloc_complex(local_n0);
        allocated = 1;
    }

    compute_nonlinear(u_hat, Nu);
    for (ptrdiff_t i = 0; i < local_n0; i++) {
        a[i] = E2[i]*u_hat[i] + Q[i]*Nu[i];
    }

    compute_nonlinear(a, Na);
    for (ptrdiff_t i = 0; i < local_n0; i++) {
        b[i] = E2[i]*u_hat[i] + Q[i]*Na[i];
    }

    compute_nonlinear(b, Nb);
    for (ptrdiff_t i = 0; i < local_n0; i++) {
        c[i] = E2[i]*a[i] + Q[i]*(2.0*Nb[i] - Nu[i]);
    }

    compute_nonlinear(c, Nc);

    for (ptrdiff_t i = 0; i < local_n0; i++) {
        u_hat[i] = E[i]*u_hat[i] + Nu[i]*f1[i] + 2.0*(Na[i]+Nb[i])*f2[i] + Nc[i]*f3[i]; 
    }
}

void save_snapshot(FILE *fp, int step) {
    for (ptrdiff_t i = 0; i < local_n0; i++) {
        u_phys[i] = u_hat[i];
    }

    fftw_execute(plan_bwd);

    double *local_real = malloc(local_n0 * sizeof(double));
    for (ptrdiff_t i = 0; i < local_n0; i++) {
        local_real[i] = creal(u_phys[i]) / (double)N; 
    }

    double *global_real = NULL;
    if (rank == 0) {
        global_real = malloc(N * sizeof(double));
    }

    MPI_Gather(local_real, local_n0, MPI_DOUBLE, 
               global_real, local_n0, MPI_DOUBLE, 
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        fprintf(fp, "%d", step);
        for (int i = 0; i < N; i++) {
            fprintf(fp, ",%f", global_real[i]);
        }
        fprintf(fp, "\n");
        free(global_real);
    }

    free(local_real);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    ptrdiff_t local_ni, local_i_start, local_no, local_o_start;
    alloc_local = fftw_mpi_local_size_1d(N, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE,
                                        &local_ni, &local_i_start,
                                        &local_no, &local_o_start);

    local_n0      = local_ni;
    local_0_start = local_i_start;

    if (local_ni != local_no || local_i_start != local_o_start) {
        if (rank == 0) {
            fprintf(stderr, "FATAL: input/output distribution mismatch "
                            "(local_ni=%td local_no=%td)\n", local_ni, local_no);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    x   = malloc(local_n0 * sizeof(double));
    k   = malloc(local_n0 * sizeof(double));
    u_hat  = fftw_alloc_complex(alloc_local);
    u_phys = fftw_alloc_complex(alloc_local);
    Lk = fftw_alloc_complex(local_n0);
    E  = fftw_alloc_complex(local_n0);  E2 = fftw_alloc_complex(local_n0);
    Q  = fftw_alloc_complex(local_n0);
    f1 = fftw_alloc_complex(local_n0); f2 = fftw_alloc_complex(local_n0);
    f3 = fftw_alloc_complex(local_n0);

    plan_fwd = fftw_mpi_plan_dft_1d(N, u_phys, u_phys, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_bwd = fftw_mpi_plan_dft_1d(N, u_phys, u_phys, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

    double t_start = MPI_Wtime();

    init_grid();
    compute_linear_operator();
    compute_etdrk4_coefficients();
    init_condition();

    FILE *fp = NULL;
    if (rank == 0) {
        fp = fopen("csv files/kse_output_mpi.csv", "w");
        if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    for (int step = 0; step < NSTEPS; step++) {
        etdrk4_step();
        if (step % SAVE_EVERY == 0) {
            save_snapshot(fp, step);
        }
    }

    double t_end = MPI_Wtime();
    if(rank==0){
        printf("Ranks: %d, Time: %f seconds\n",nprocs, t_end - t_start);
    }

    if (rank == 0) fclose(fp);

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_mpi_cleanup();
    MPI_Finalize();
    
    return 0;
}
/* Build:
 *   mpicc -O3 KSE_MPI_Solver.c -o kse_mpi -lfftw3_mpi -lfftw3 -lm
  mpirun -np 1 ./kse_mpi
  mpirun -np 2 ./kse_mpi
  mpirun -np 4 ./kse_mpi
 */