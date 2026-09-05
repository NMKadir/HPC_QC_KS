import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.synthesis import synth_qft_full

N = 16
L = 32.0 * np.pi
DT = 0.1
NSTEPS = 100
SAVE_EVERY = 5
M = 32

def init_grid():
    dx = L / N
    x = np.arange(N) * dx
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    return x, k

def qft_transform(x, inverse=False):
    x = np.asarray(x, dtype=complex)
    norm = np.linalg.norm(x)
    if norm < 1e-15:
        return np.zeros_like(x)

    n_qubits = int(np.log2(len(x)))
    if 2**n_qubits != len(x):
        raise ValueError("N must be a power of 2")

    qc = QuantumCircuit(n_qubits)
    qc.initialize(x / norm, range(n_qubits))

    qft = synth_qft_full(
        n_qubits,
        do_swaps=True,
        inverse=inverse,
        approximation_degree=0,
    )
    qc.compose(qft, inplace=True)

    state = Statevector(qc).data
    return state * norm * np.sqrt(len(x))

def print_quantum_circuit():
    n_qubits = int(np.log2(N))
    qc = synth_qft_full(
        n_qubits,
        do_swaps=True,
        approximation_degree=0,
    )
    print("\nQFT Circuit")
    print(qc.draw(output="text"))
    #qc.draw(output="mpl")

def compute_etdrk4_coefficients(Lk):
    E = np.exp(Lk * DT)
    E2 = np.exp(Lk * DT / 2.0)

    theta = (np.arange(M) + 0.5) * 2.0 * np.pi / M
    r = np.exp(1j * theta)
    z = Lk[:, None] * DT + r[None, :]
    ez = np.exp(z)
    ez2 = np.exp(z / 2.0)

    Q = DT * np.mean((ez2 - 1.0) / z, axis=1)
    f1 = DT * np.mean(
        (-4.0 - z + ez * (4.0 - 3.0 * z + z**2)) / z**3, axis=1
    )
    f2 = DT * np.mean(
        (2.0 + z + ez * (-2.0 + z)) / z**3, axis=1
    )
    f3 = DT * np.mean(
        (-4.0 - 3.0 * z - z**2 + ez * (4.0 - z)) / z**3, axis=1
    )

    return E, E2, Q.real, f1.real, f2.real, f3.real

def nonlinear_classical(u_hat, k):
    u = np.fft.ifft(u_hat).real
    u2_hat = np.fft.fft(u**2)
    n_hat = -0.5j * k * u2_hat

    modes = np.fft.fftfreq(N) * N
    n_hat[np.abs(modes) > N / 3] = 0.0
    return n_hat


def nonlinear_quantum(u_hat, k):
    u = qft_transform(u_hat, inverse=False).real / N

    u2 = u**2

    u2_hat = qft_transform(u2.astype(complex), inverse=True)
    n_hat = -0.5j * k * u2_hat

    modes = np.fft.fftfreq(N) * N
    n_hat[np.abs(modes) > N / 3] = 0.0
    return n_hat

def etdrk4_step(u_hat, k, E, E2, Q, f1, f2, f3, quantum=False):
    nonlinear = nonlinear_quantum if quantum else nonlinear_classical

    Nu = nonlinear(u_hat, k)
    a = E2 * u_hat + Q * Nu

    Na = nonlinear(a, k)
    b = E2 * u_hat + Q * Na

    Nb = nonlinear(b, k)
    c = E2 * a + Q * (2.0 * Nb - Nu)

    Nc = nonlinear(c, k)
    return E * u_hat + f1 * Nu + 2.0 * f2 * (Na + Nb) + f3 * Nc

def run_solver(quantum, csv_file):
    x, k = init_grid()
    Lk = k**2 - k**4
    E, E2, Q, f1, f2, f3 = compute_etdrk4_coefficients(Lk)

    u0 = np.cos(x / 16.0) * (1.0 + np.sin(x / 16.0))
    u_hat = np.fft.fft(u0)

    with open(csv_file, "w") as f:
        f.write("0," + ",".join(f"{v:.12e}" for v in u0) + "\n")

        for step in range(1, NSTEPS + 1):
            u_hat = etdrk4_step(
                u_hat, k, E, E2, Q, f1, f2, f3, quantum=quantum
            )

            if step % SAVE_EVERY == 0 or step == NSTEPS:
                u = np.fft.ifft(u_hat).real
                f.write(
                    str(step) + "," + ",".join(f"{v:.12e}" for v in u) + "\n"
                )

    print(f"Saved: {csv_file}")

if __name__ == "__main__":
    print_quantum_circuit()
    run_solver(False, "kse_classical.csv")
    run_solver(True, "kse_quantum.csv")
