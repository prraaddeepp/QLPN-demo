#!/usr/bin/env python3
"""
Sanity‐check the “piling‐up” lemma for phase‐flip noise under random X‐masking.

Plots the empirical flip probability ε_emp(d) against the analytic formula
ε_ana(d) = ½[1 − (1 − 2w/m)^d].
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def experiment(m=8, w=1, d_max=20, shots=20000, seed=42):
    rng = np.random.default_rng(seed)
    ds = list(range(1, d_max+1))
    emp = []
    ana = []
    for d in ds:
        # simulate shots
        flips = 0
        for _ in range(shots):
            f = 0
            for __ in range(d):
                # pick a qubit to flip uniformly at random
                if rng.integers(0, m) == 0:
                    # flip the parity label bit (empirical tally for ε)
                    f ^= 1
            flips += f
        eps_emp = flips / shots
        eps_ana = 0.5 * (1 - (1 - 2*w/m)**d)
        emp.append(eps_emp)
        ana.append(eps_ana)

    return ds, emp, ana

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m",    type=int, default=8,   help="total qubits")
    p.add_argument("--w",    type=int, default=1,   help="error weight in analytic")
    p.add_argument("--dmax", type=int, default=20,  help="max masking depth")
    p.add_argument("--shots",type=int, default=20000,help="MC trials per d")
    p.add_argument("--out",  type=str, default="noise_rate_vs_d.pdf",
                   help="output filename")
    args = p.parse_args()

    ds, emp, ana = experiment(
        m=args.m, w=args.w,
        d_max=args.dmax, shots=args.shots
    )

    plt.figure(figsize=(6,4))
    plt.plot(ds, emp, 'o-', label=r'$\varepsilon_{\rm emp}(d)$')
    plt.plot(ds, ana, 's--', label=r'$\varepsilon_{\rm ana}(d)$')
    plt.xlabel('masking depth $d$')
    plt.ylabel(r'Noise rate $\varepsilon$')
    plt.legend(loc='upper left')
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.savefig(args.out)
    plt.show()
    print(f"Plot saved to {args.out}")

if __name__ == "__main__":
    main()
