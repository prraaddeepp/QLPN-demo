import argparse
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Kraus, DensityMatrix

def build_qlpn_oracle_unitary(n, s, eps):
    """Exactly the same as your build_qlpn_oracle_circuit up through the final Hadamard,
       but without any measurements attached."""
    qr = QuantumRegister(n, 'data')
    qa = QuantumRegister(1, 'anc')
    qc = QuantumCircuit(qr, qa)

    qc.h(qr)
    for i, bit in enumerate(s):
        if bit:
            qc.cx(qr[i], qa[0])
    qc.z(qa[0])

    # Kraus phase‐flip on the ancilla
    p  = eps
    K0 = np.sqrt(1-p) * np.eye(2, dtype=complex)
    K1 = np.sqrt(p)   * np.array([[1,0],[0,-1]], dtype=complex)
    qc.append(Kraus([K0, K1]), [qa[0]])

    qc.h(qa[0])
    return qc

def sample_quantum(n, s, eps, shots, rng):
    """Simulate exactly once to a density matrix, then sample (a,b) from its probabilities."""
    # 1) build the “unitary” part of your oracle (no measurements)
    qc = build_qlpn_oracle_unitary(n, s, eps)

    # 2) initialize the (n+1)-qubit zero state as a density matrix
    dm = DensityMatrix.from_label('0'*(n+1))

    # 3) evolve under your circuit (this handles both the Kraus noise and the unitaries)
    dm = dm.evolve(qc)

    # 4) get a dictionary mapping bit-strings → probabilities
    probs = dm.probabilities_dict()  
    bitstrs = list(probs.keys())
    ps      = np.array([probs[b] for b in bitstrs])

    # 5) sample `shots` times from that distribution
    picks = rng.choice(len(bitstrs), size=shots, p=ps)
    A, B = [], []
    for idx in picks:
        bstr = bitstrs[idx]
        a_bits = np.array(list(map(int, bstr[:-1])), dtype=np.uint8)
        b_bit  = int(bstr[-1])
        A.append(a_bits)
        B.append(b_bit)

    return np.vstack(A), np.array(B, dtype=np.uint8)

def sample_classical(n, s, eps, shots, rng):
    A = rng.integers(0,2,size=(shots,n), dtype=np.uint8)
    clean = (A.dot(s) & 1)
    flips = rng.random(shots) < eps
    return A, clean ^ flips.astype(np.uint8)

def compare_distributions(Aq, Bq, Ac, Bc):
    """Compute and return the joint probabilities Pq, Pc as flat arrays."""
    from collections import Counter
    Cq = Counter(map(tuple, np.hstack((Aq, Bq[:,None]))))
    Cc = Counter(map(tuple, np.hstack((Ac, Bc[:,None]))))
    keys = sorted(set(Cq) | set(Cc))
    
    Pq = np.array([ Cq[k]/len(Bq) for k in keys ])
    Pc = np.array([ Cc[k]/len(Bc) for k in keys ])
    tv = 0.5 * np.sum(np.abs(Pq - Pc))
    print(f"Total‐variation distance ≈ {tv:.4f}")
    return Pq, Pc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int,   default=3)
    parser.add_argument("--eps",  type=float, default=0.25)
    parser.add_argument("--shots",type=int,   default=20000)
    args = parser.parse_args()

    rng = np.random.default_rng(1234)
    s   = rng.integers(0,2,size=args.n, dtype=np.uint8)
    print("Secret s =", s.tolist())

    Aq, Bq = sample_quantum(   args.n, s, args.eps, args.shots, rng)
    Ac, Bc = sample_classical(args.n, s, args.eps, args.shots, rng)
    
    Pq, Pc = compare_distributions(Aq, Bq, Ac, Bc)

    # --- scatter Pq vs Pc ---
    m = max(Pq.max(), Pc.max())
    plt.figure(figsize=(5,5))
    plt.scatter(Pq, Pc, alpha=0.6)
    # diagonal y=x
    plt.plot([0, m], [0, m], linestyle="--", label="y = x")
    plt.xlabel("Quantum P(a,b)")
    plt.ylabel("Classical P(a,b)")
    plt.title(f"QLPN oracle match (n={args.n}, ε={args.eps:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
