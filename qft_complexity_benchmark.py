import numpy as np
import time
import csv
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.synthesis import synth_qft_full

RESULTS_CSV = "qft_complexity_results.csv"


def statevector_memory_bytes(n_qubits):

    return (2 ** n_qubits) * 16  # bytes


def benchmark_qft_at_N(N, n_trials=10):
    n_qubits = int(np.log2(N))
    assert 2**n_qubits == N, "N must be a power of 2"

    x = np.random.randn(N) + 1j * np.random.randn(N)
    norm = np.linalg.norm(x)
    state = x / norm

    #circuit synthesis time
    t0 = time.perf_counter()
    for _ in range(n_trials):
        qft = synth_qft_full(n_qubits, do_swaps=True, inverse=False)
    t_synth = (time.perf_counter() - t0) / n_trials

    #full step time: build circuit, initialize, simulate 
    t0 = time.perf_counter()
    for _ in range(n_trials):
        qc = QuantumCircuit(n_qubits)
        qc.initialize(state, range(n_qubits))
        qc.compose(synth_qft_full(n_qubits, do_swaps=True, inverse=False), inplace=True)
        _ = Statevector(qc).data
    t_step_quantum = (time.perf_counter() - t0) / n_trials

    #classical FFT for comparison 
    t0 = time.perf_counter()
    for _ in range(n_trials):
        np.fft.fft(x)
    t_step_classical = (time.perf_counter() - t0) / n_trials

    mem_bytes = statevector_memory_bytes(n_qubits)

    result = {
        "N": N,
        "n_qubits": n_qubits,
        "synth_time_ms": t_synth * 1e3,
        "step_time_quantum_ms": t_step_quantum * 1e3,
        "step_time_classical_us": t_step_classical * 1e6,
        "statevector_memory_bytes": mem_bytes,
        "statevector_memory_human": human_readable_bytes(mem_bytes),
        "slowdown_factor": t_step_quantum / t_step_classical,
    }
    return result


def human_readable_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB", "PB", "EB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}ZB"


def project_memory_wall(max_qubits=20):
    print("\nProjected statevector memory vs qubit count:")
    print(f"{'qubits':>7} {'N':>8} {'memory':>12}")
    for n_qubits in range(1, max_qubits + 1):
        N = 2 ** n_qubits
        mem = statevector_memory_bytes(n_qubits)
        flag = "  <-- exceeds typical 16GB RAM" if mem > 16 * (1024**3) else ""
        print(f"{n_qubits:>7} {N:>8} {human_readable_bytes(mem):>12}{flag}")


def run_and_save(N_values=(16, 32)):
    results = []
    for N in N_values:
        print(f"Benchmarking N={N}...")
        r = benchmark_qft_at_N(N)
        results.append(r)
        print(f"  synthesis: {r['synth_time_ms']:.3f} ms, "
              f"quantum step: {r['step_time_quantum_ms']:.3f} ms, "
              f"classical step: {r['step_time_classical_us']:.3f} us, "
              f"memory: {r['statevector_memory_human']}, "
              f"slowdown: {r['slowdown_factor']:.0f}x")

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {RESULTS_CSV}")

    return results


if __name__ == "__main__":
    run_and_save(N_values=(16, 32))
    project_memory_wall(max_qubits=20)
    print("\nKey point for your write-up: statevector memory grows as")
    print("O(2^n_qubits), a hard wall independent of implementation")
    print("quality -- this is why N=4096 (12 qubits, 64KB) is already")
    print("noticeable and N~30+ qubits (>8GB) becomes infeasible on a")
    print("laptop, regardless of how the circuit itself is optimized.")