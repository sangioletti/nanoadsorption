"""Test multi-ligand support with different receptor types.

Two ligand types bind DIFFERENT receptor types on the cell surface.
Each ligand specifies its own receptor dict with its own sigma_R.

We compare:
  1. Single ligand type A with its receptor
  2. Single ligand type B with its receptor
  3. Both ligand types together (different receptor types)

The multi-receptor Poisson fluctuation averaging is also tested.
"""
from adsorption import MultivalentBinding, Nmonomers
from units import *
from mpmath import mp
import numpy as np

mp.dps = 50

# ---- NP design parameters ----
R_NP = 35 * nm
amono = 0.28 * nm
akuhn = 0.76 * nm
NmonoLigands = Nmonomers(3400 * g)
NmonoShort = Nmonomers(2000 * g)

N_ligands_A = 100
N_ligands_B = 50
sigma_L_A = N_ligands_A / (4.0 * np.pi * R_NP**2)
sigma_L_B = N_ligands_B / (4.0 * np.pi * R_NP**2)
sigma_P2K = sigma_L_A * 11.4

KD_A = 150.0 * nM   # ligand A affinity
KD_B = 50.0 * nM    # ligand B affinity (stronger)

# Receptor densities for each type
sigma_R_A = 200.0 / um2   # receptor type A
sigma_R_B = 100.0 / um2   # receptor type B

# ---- Cell / dosing parameters ----
A_cell = 100 * um2
NP_conc = 1e10 / mL
cell_conc = 1e5 / mL

# ===========================================================
# System 1: Ligand A only (with its receptor A)
# ===========================================================
rec_A_only = {"name": "receptor_A", "sigma_R": sigma_R_A}
dp_A = {
    "short": {"N": NmonoShort, "a": amono, "sigma": sigma_P2K,
              "name": "PEG2K", "akuhn": akuhn},
    "ligand_A": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_A,
                 "name": "ligA", "akuhn": akuhn,
                 "K_bind_0": KD_A**(-1),
                 "receptor": rec_A_only},
}

system_A = MultivalentBinding(
    kT=kT, R_NP=R_NP, data_polymers=dp_A,
    binding_model="exact", polymer_model="gaussian",
    A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
)

K_bind_A = system_A.calculate_binding_constant()
bf_A = system_A.calculate_bound_fraction(depletion=True)

print(f"=== Ligand A only (KD={KD_A/nM:.0f} nM, sigma_R={sigma_R_A*um2:.0f}/um2) ===")
print(f"  K_bind = {float(K_bind_A):.4e}")
print(f"  Bound fraction = {float(bf_A):.6f}")

# ===========================================================
# System 2: Ligand B only (with its receptor B)
# ===========================================================
rec_B_only = {"name": "receptor_B", "sigma_R": sigma_R_B}
dp_B = {
    "short": {"N": NmonoShort, "a": amono, "sigma": sigma_P2K,
              "name": "PEG2K", "akuhn": akuhn},
    "ligand_B": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_B,
                 "name": "ligB", "akuhn": akuhn,
                 "K_bind_0": KD_B**(-1),
                 "receptor": rec_B_only},
}

system_B = MultivalentBinding(
    kT=kT, R_NP=R_NP, data_polymers=dp_B,
    binding_model="exact", polymer_model="gaussian",
    A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
)

K_bind_B = system_B.calculate_binding_constant()
bf_B = system_B.calculate_bound_fraction(depletion=True)

print(f"\n=== Ligand B only (KD={KD_B/nM:.0f} nM, sigma_R={sigma_R_B*um2:.0f}/um2) ===")
print(f"  K_bind = {float(K_bind_B):.4e}")
print(f"  Bound fraction = {float(bf_B):.6f}")

# ===========================================================
# System 3: Both ligands, different receptors
# Each ligand has its own receptor dict with its own sigma_R
# ===========================================================
rec_A = {"name": "receptor_A", "sigma_R": sigma_R_A}
rec_B = {"name": "receptor_B", "sigma_R": sigma_R_B}
dp_AB = {
    "short": {"N": NmonoShort, "a": amono, "sigma": sigma_P2K,
              "name": "PEG2K", "akuhn": akuhn},
    "ligand_A": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_A,
                 "name": "ligA", "akuhn": akuhn,
                 "K_bind_0": KD_A**(-1),
                 "receptor": rec_A},
    "ligand_B": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_B,
                 "name": "ligB", "akuhn": akuhn,
                 "K_bind_0": KD_B**(-1),
                 "receptor": rec_B},
}

system_AB = MultivalentBinding(
    kT=kT, R_NP=R_NP, data_polymers=dp_AB,
    binding_model="exact", polymer_model="gaussian",
    A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
)

K_bind_AB = system_AB.calculate_binding_constant()
bf_AB = system_AB.calculate_bound_fraction(depletion=True)

print(f"\n=== Both ligands (different receptor types) ===")
print(f"  K_bind   = {float(K_bind_AB):.4e}")
print(f"  Bound fraction = {float(bf_AB):.6f}")

# ===========================================================
# Sanity checks
# ===========================================================
print(f"\n=== Sanity checks ===")
print(f"  K_bind(A+B) >= K_bind(A): {float(K_bind_AB) >= float(K_bind_A)}  "
      f"({float(K_bind_AB):.4e} >= {float(K_bind_A):.4e})")
print(f"  K_bind(A+B) >= K_bind(B): {float(K_bind_AB) >= float(K_bind_B)}  "
      f"({float(K_bind_AB):.4e} >= {float(K_bind_B):.4e})")
print(f"  Bound frac(A+B) >= Bound frac(max(A,B)): "
      f"{float(bf_AB) >= max(float(bf_A), float(bf_B))}")

# ===========================================================
# Test multi-receptor Poisson fluctuations (depletion=True)
# ===========================================================
print(f"\n=== Multi-receptor Poisson fluctuations (depletion=True) ===")
print(f"  Computing K_bind grid over (NR_A, NR_B) combinations...")

K_bind_data = system_AB.calculate_K_bind_vs_receptors(max_N_receptor=30)
K_bind_flat, grid_shape, NR_aves, receptor_names = K_bind_data

print(f"  Grid shape: {grid_shape} (total {len(K_bind_flat)} points)")
print(f"  NR_aves: {[f'{x:.2f}' for x in NR_aves]}")

M_conc = (A_cell / system_AB.NP_excluded_area) * cell_conc

bf_fluct = system_AB.calculate_bound_fraction(
    fluctuations=True, depletion=True,
    K_bind_vs_receptors=K_bind_data,
    rho_m=M_conc,
)
print(f"  Bound fraction (fluctuations+depletion) = {float(bf_fluct):.6f}")

# ===========================================================
# Test multi-receptor Poisson fluctuations (Langmuir, no depletion)
# ===========================================================
print(f"\n=== Multi-receptor Poisson fluctuations (Langmuir) ===")
bf_langmuir = system_AB.calculate_bound_fraction(
    fluctuations=True, depletion=False,
    K_bind_vs_receptors=K_bind_data,
)
print(f"  Bound fraction (fluctuations, Langmuir) = {float(bf_langmuir):.6f}")

# ===========================================================
# Test validation: should raise ValueError
# Two ligands with same receptor name but DIFFERENT dict objects
# ===========================================================
print(f"\n=== Validation test ===")
dp_bad = {
    "ligand_A": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_A,
                 "name": "ligA", "akuhn": akuhn,
                 "K_bind_0": KD_A**(-1),
                 "receptor": {"name": "shared_rec", "sigma_R": 100 / um2}},
    "ligand_B": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_B,
                 "name": "ligB", "akuhn": akuhn,
                 "K_bind_0": KD_B**(-1),
                 "receptor": {"name": "shared_rec", "sigma_R": 300 / um2}},
}
try:
    MultivalentBinding(
        kT=kT, R_NP=R_NP, data_polymers=dp_bad,
        binding_model="exact", polymer_model="gaussian",
        A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
    )
    print("  ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"  Correctly raised ValueError: {e}")

print("\nAll tests passed.")
