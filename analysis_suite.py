import numpy as np
import matplotlib.pyplot as plt


def load_trajectory(csv_path):
    data = np.loadtxt(csv_path, delimiter=",")
    steps = data[:, 0]
    U = data[:, 1:]
    return steps, U


# 1. N-scaling: time vs N and speedup vs N
def plot_scaling_and_speedup():
    N_values = [8192, 16384, 32768, 65536]

    TIMINGS = {
        "Serial":      [4.85604, 9.90526, 19.850802, 39.95054],
        "OpenMP":      [5.21213, 8.38824, 15.47395, 34.26238], #2 threads
        "MPI":         [4.20361, 8.51533, 17.52749, 37.95484], #2 cores
        "CUDA_FP64":   [5.13225, 3.02575, 4.59756, 8.47251],
        "CUDA_FP32":   [1.02119, 1.78208, 2.53655, 4.47509],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for backend, times in TIMINGS.items():
        valid = [(n, t) for n, t in zip(N_values, times) if t is not None]
        if not valid:
            continue
        ns, ts = zip(*valid)
        ax1.loglog(ns, ts, marker='o', label=backend)

    ax1.set_xlabel("N")
    ax1.set_ylabel("Execution time (s)")
    ax1.set_title("Execution time vs N (log-log)")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    serial_times = TIMINGS["Serial"]
    for backend, times in TIMINGS.items():
        if backend == "Serial":
            continue
        speedups = [s/t if (s is not None and t is not None and t > 0) else None
                    for s, t in zip(serial_times, times)]
        valid = [(n, sp) for n, sp in zip(N_values, speedups) if sp is not None]
        if not valid:
            continue
        ns, sps = zip(*valid)
        ax2.semilogx(ns, sps, marker='o', label=backend)

    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    ax2.set_xlabel("N")
    ax2.set_ylabel("Speedup ($T_{Serial}/T_{Backend}$)")
    ax2.set_title("Speedup vs N")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("scaling_and_speedup.png", dpi=150)
    print("Saved scaling_and_speedup.png")
    print("\nReminder: fill in TIMINGS with your actual measured times")
    print("before this plot means anything -- currently all None.")

# 2a. FP32 vs FP64 timing bar chart

def plot_fp32_fp64_timing():
    N_values = [8192, 16384, 32768, 65536]
    fp32_ms = [1.02119, 1.78208, 2.53655, 4.47509]
    fp64_ms = [5.13225, 3.02575, 4.59756, 8.47251]

    x = np.arange(len(N_values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, [v if v else 0 for v in fp32_ms], width, label="FP32")
    ax.bar(x + width/2, [v if v else 0 for v in fp64_ms], width, label="FP64")
    ax.set_xticks(x)
    ax.set_xticklabels(N_values)
    ax.set_xlabel("N")
    ax.set_ylabel("Step time (s)")
    ax.set_title("CUDA FP32 vs FP64 step time")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fp32_fp64_timing.png", dpi=150)
    print("Saved fp32_fp64_timing.png (fill in fp32_ms/fp64_ms first)")


# 2b. FP32 vs FP64 error drift over first 300 steps

def plot_fp32_fp64_error_drift(fp32_csv, fp64_csv, n_steps=300):
    steps32, U32 = load_trajectory(fp32_csv)
    steps64, U64 = load_trajectory(fp64_csv)

    common = np.intersect1d(steps32, steps64)
    common = common[common <= n_steps]

    diffs = []
    for s in common:
        u32 = U32[steps32 == s][0]
        u64 = U64[steps64 == s][0]
        diffs.append(np.abs(u32 - u64).max())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(common, diffs, marker='o', markersize=3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Max $|u_{FP32} - u_{FP64}|$")
    ax.set_title("FP32 vs FP64 error drift")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig("fp32_fp64_error_drift_65536.png", dpi=150)
    print("Saved fp32_fp64_error_drift_65536.png")
    print(f"Final drift at step {common[-1]}: {diffs[-1]:.3e}")


# 3a. Energy/norm conservation vs timestep, for a Δt sweep

def plot_energy_conservation(csv_paths_by_dt):

    fig, ax = plt.subplots(figsize=(8, 5))
    for dt, path in csv_paths_by_dt.items():
        steps, U = load_trajectory(path)
        t = steps * dt
        energy = np.sum(np.abs(U)**2, axis=1)
        ax.plot(t, energy, label=f"$\\Delta t$={dt}")

    ax.set_xlabel("Physical time $t$")
    ax.set_ylabel(r"$\sum |u(x,t)|^2$")
    ax.set_title("Energy/norm conservation vs timestep size")
    ax.legend()
    fig.tight_layout()
    fig.savefig("energy_conservation.png", dpi=150)
    print("Saved energy_conservation.png")
    print("Look for: divergence to huge values (instability), or")
    print("visibly different energy levels indicating non-physical")
    print("high-frequency oscillations at large Δt.")


# 3b. Snapshot comparison at t=10 across Δt values

def plot_snapshot_at_time(csv_paths_by_dt, t_target=10.0):
    fig, ax = plt.subplots(figsize=(10, 5))
    for dt, path in csv_paths_by_dt.items():
        steps, U = load_trajectory(path)
        t = steps * dt
        idx = np.argmin(np.abs(t - t_target))
        ax.plot(U[idx], label=f"$\\Delta t$={dt} (t={t[idx]:.2f})")

    ax.set_xlabel("Grid index (space)")
    ax.set_ylabel("u(x)")
    ax.set_title(f"Snapshot comparison near t={t_target}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("snapshot_comparison_dt.png", dpi=150)
    print("Saved snapshot_comparison_dt.png")


# 4. Domain-length heatmaps, side by side

def plot_domain_length_heatmaps(csv_paths_by_L):
    """
    csv_paths_by_L: dict like {'16pi': 'kse_L16pi.csv', '32pi': '...', '64pi': '...'}
    """
    fig, axes = plt.subplots(1, len(csv_paths_by_L), figsize=(6*len(csv_paths_by_L), 6))
    if len(csv_paths_by_L) == 1:
        axes = [axes]

    for ax, (label, path) in zip(axes, csv_paths_by_L.items()):
        steps, U = load_trajectory(path)
        im = ax.imshow(U, aspect='auto', origin='lower', cmap='RdBu_r',
                        extent=[0, U.shape[1], steps.min(), steps.max()])
        ax.set_title(f"L = {label}")
        ax.set_xlabel("grid index (space)")
        ax.set_ylabel("timestep")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig("domain_length_heatmaps.png", dpi=150)
    print("Saved domain_length_heatmaps.png")
    print("Look for: near-periodic/steady structure at small L,")
    print("increasingly rich chaotic branching as L grows.")


# 5. QFT vs FFT trajectory fidelity

def plot_qft_fidelity(classical_csv, quantum_csv, dt=0.1, snapshot_steps=None):
    steps_c, U_c = load_trajectory(classical_csv)
    steps_q, U_q = load_trajectory(quantum_csv)

    common = np.intersect1d(steps_c, steps_q)
    t = common * dt

    errors = []
    for s in common:
        uc = U_c[steps_c == s][0]
        uq = U_q[steps_q == s][0]
        errors.append(np.abs(uc - uq).max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(t, errors, marker='o', markersize=3)
    ax1.set_yscale("log")
    ax1.set_xlabel("Physical time $t$")
    ax1.set_ylabel("Max $|u_{FFT} - u_{QFT}|$")
    ax1.set_title("Pointwise absolute error over time")

    if snapshot_steps is None:
        
        snapshot_steps = [common[0], common[len(common)//2], common[-1]]

    for s in snapshot_steps:
        uc = U_c[steps_c == s][0]
        uq = U_q[steps_q == s][0]
        ax2.plot(uc, '-', label=f"FFT, step {int(s)}")
        ax2.plot(uq, '--', label=f"QFT, step {int(s)}")

    ax2.set_xlabel("grid index (space)")
    ax2.set_ylabel("u(x)")
    ax2.set_title("Overlapped snapshots")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("qft_fidelity.png", dpi=150)
    print("Saved qft_fidelity.png")
    print(f"Error at first common step: {errors[0]:.3e}")
    print(f"Error at last common step:  {errors[-1]:.3e}")
    print("Expect: tiny error early on (simulator floating-point")
    print("precision, ~1e-10 to 1e-12), growing later due to KSE's")
    print("chaotic amplification -- same pattern as MPI/CUDA validation.")


if __name__ == "__main__":
    print("This is a library of functions -- call the ones you need")
    print("with your actual CSV paths, e.g.:\n")
    print("  from analysis_suite import *")
    print("  plot_energy_conservation({0.01: 'kse_dt0.01.csv', 0.1: 'kse_dt0.1.csv'})")
    print("  plot_qft_fidelity('kse_classical_N16.csv', 'kse_quantum_N16.csv')")