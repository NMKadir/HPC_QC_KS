import sys
import numpy as np
import matplotlib.pyplot as plt

csvfilename = "csv files/kse_output.csv"
snapfilename = "Figures/kse_snapshots.png"
spacefilename = "Figures/kse_spacetime.png"

def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else csvfilename 

    data = np.loadtxt(fname, delimiter=",")
    steps = data[:, 0]        
    u = data[:, 1:]            

    print(f"Loaded {u.shape[0]} snapshots, {u.shape[1]} grid points each")
    print(f"u range: [{u.min():.4f}, {u.max():.4f}]")

    if np.isnan(u).any() or np.isinf(u).any():
        print("WARNING: NaN or Inf detected in output -- solver blew up.")
    if np.max(np.abs(u)) > 1e3:
        print("WARNING: values are very large -- likely unstable / blowing up.")

    plt.figure(figsize=(10, 6))
    plt.imshow(
        u,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        extent=[0, u.shape[1], steps.min(), steps.max()],
    )
    plt.colorbar(label="u(x,t)")
    plt.xlabel("grid index (space)")
    plt.ylabel("timestep")
    plt.title("Kuramoto-Sivashinsky: space-time evolution")
    plt.tight_layout()
    plt.savefig(spacefilename, dpi=150)
    print(spacefilename)
    plt.figure(figsize=(10, 5))
    n_snap = min(5, u.shape[0])
    idxs = np.linspace(0, u.shape[0] - 1, n_snap, dtype=int)
    for i in idxs:
        plt.plot(u[i], label=f"step {int(steps[i])}")
    plt.xlabel("grid index (space)")
    plt.ylabel("u(x)")
    plt.title("Snapshots at a few timesteps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snapfilename, dpi=150)
    print(snapfilename)

    plt.show()

if __name__ == "__main__":
    main()