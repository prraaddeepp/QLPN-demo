#!/usr/bin/env python3
"""
Sanity-check for the Quantum Smoothing Lemma (Lemma 3.1).

– Default n=6, m=8 (so 2^8=256 dimensions)
– Default w=1 so that β+2w/m<1 and the analytic bound decays.
– Partial trace down to n=6 qubits before computing trace distance.
"""

import argparse, math
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

def random_full_rank(n, m):
    """Return a random n×m binary matrix of full rank n."""
    while True:
        M = np.random.randint(0, 2, size=(n, m), dtype=np.uint8)
        if np.linalg.matrix_rank(M) == n:
            return M

def apply_mask(index, mask):
    """X-mask on a basis index (bitwise xor)."""
    return index ^ mask

def state_from_codeword(C, s, x):
    """
    Build |ψ_{s,x}⟩ = 1/√|C| Σ_{c∈C(s)} |c⊕x⟩ exactly by enumerating the coset.
    """
    n, m = C.shape
    amps = np.zeros(2**m, complex)
    for z_int in range(2**n):
        # convert integer to length-n bitvector
        z = np.array(list(map(int, f"{z_int:0{n}b}")), dtype=np.uint8)
        # codeword = z·C + s·C = (z^s)·C mod 2
        c = (z ^ s) @ C % 2
        idx = int("".join(map(str, c ^ x)), 2)
        amps[idx] += 1
    amps /= np.linalg.norm(amps)
    return Statevector(amps)

def trace_distance(rho, sigma):
    """½∥ρ−σ∥₁ via singular values."""
    s = np.linalg.svd(rho - sigma, compute_uv=False)
    return 0.5 * np.sum(np.abs(s))

def experiment(n=6, m=8, w=1, d_max=6, shots=1000, seed=42):
    rng = np.random.default_rng(seed)
    C = random_full_rank(n, m)
    s = rng.integers(0, 2, size=n, dtype=np.uint8)
    x = np.zeros(m, dtype=np.uint8)
    x[rng.choice(m, w, replace=False)] = 1

    psi = state_from_codeword(C, s, x)
    beta = 1 / math.sqrt(n)

    ds, empirical, analytic = [], [], []
    for d in range(1, d_max + 1):
        # Monte Carlo smoothing
        rho_accum = np.zeros((2**m, 2**m), complex)
        for _ in range(shots):
            # random mask of weight ≤ d
            r = 0
            for _ in range(d):
                r ^= 1 << rng.integers(0, m)
            idx = apply_mask(int(np.nonzero(psi.data)[0][0]), r)
            ket = np.zeros(2**m, complex)
            ket[idx] = 1
            rho_accum += np.outer(ket, ket.conj())
        rho_sm = rho_accum / shots

        # Partial trace to first n qubits
        dm_full = DensityMatrix(rho_sm, dims=[2]*m)
        dm_red = partial_trace(dm_full, list(range(n, m)))
        rho_red = dm_red.data

        # ideal maximally mixed on n qubits
        rho_ideal = np.eye(2**n) / (2**n)

        ds.append(d)
        empirical.append(trace_distance(rho_red, rho_ideal))
        beta_prime = beta + 2 * w / m
        analytic.append((2**((n + 1) / 2)) * (beta_prime ** d))

    return ds, empirical, analytic

def main():
    p = argparse.ArgumentParser(description="Quantum smoothing sanity-check")
    p.add_argument("--n",    type=int, default=6,   help="secret length")
    p.add_argument("--m",    type=int, default=8,   help="codeword length")
    p.add_argument("--w",    type=int, default=1,   help="error weight")
    p.add_argument("--dmax", type=int, default=6,   help="max masking depth")
    p.add_argument("--shots",type=int, default=1000,help="MC samples per d")
    p.add_argument("--out",  type=str, default="trace_distance_plot.pdf",
                   help="output filename")
    args = p.parse_args()

    ds, emp, ana = experiment(
        n=args.n, m=args.m, w=args.w,
        d_max=args.dmax, shots=args.shots
    )

    # plot on log scale
    plt.figure(figsize=(6,4))
    plt.semilogy(ds, emp, "o-", label="empirical")
    plt.semilogy(ds, ana, "s--", label="analytical bound")
    plt.xlabel("masking depth $d$")
    plt.ylabel("trace distance (log scale)")
    plt.legend()

    # show decades from 10^1 down to 10^-2
    plt.ylim(1e-2, 1e1)
    plt.yticks([1e1, 1e0, 1e-1, 1e-2])

    plt.grid(which="major", ls=":")
    plt.tight_layout()
    plt.savefig(args.out)
    plt.show()
    print(f"Plot saved to {args.out}")

if __name__ == "__main__":
    main()
