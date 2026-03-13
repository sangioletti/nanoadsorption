"""Test multi-ligand support with a single shared receptor type.

Two ligand types bind different epitopes on the SAME receptor.
They share a single receptor dict, so sigma_R is set once and
applies to both ligands automatically.

We compare:
  1. Single ligand type A alone
  2. Single ligand type B alone
  3. Both ligand types A + B together

Since the bonding free energy is additive, K_bind with both ligands
should be larger than with either alone.
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

N_ligands_A = 50
N_ligands_B = 50
sigma_L_A = N_ligands_A / (4.0 * np.pi * R_NP**2)
sigma_L_B = N_ligands_B / (4.0 * np.pi * R_NP**2)
sigma_P2K = sigma_L_A * 11.4

KD_A = 150.0 * nM   # ligand A: moderate affinity
KD_B = 50.0 * nM    # ligand B: higher affinity

# ---- Cell / dosing parameters ----
A_cell = 100 * um2
NP_conc = 1e8 / mL
cell_conc = 1e5 / mL

# ---- Test sigma_R values ----
sigma_R_test = 700.0 / um2

# ===========================================================
# System 1: Ligand A only
# ===========================================================
rec_A = {"name": "shared_rec", "sigma_R": sigma_R_test}
dp_A = {
    "short": {"N": NmonoShort, "a": amono, "sigma": sigma_P2K,
              "name": "PEG2K", "akuhn": akuhn},
    "ligand_A": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_A,
                 "name": "ligA", "akuhn": akuhn,
                 "K_bind_0": KD_A**(-1),
                 "receptor": rec_A},
}

system_A = MultivalentBinding(
    kT=kT, R_NP=R_NP, data_polymers=dp_A,
    binding_model="exact", polymer_model="Flory-exact",
    A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
)

K_bind_A = system_A.calculate_binding_constant()
bf_A = system_A.calculate_bound_fraction(depletion=False)

print(f"=== Ligand A only (KD = {KD_A/nM:.0f} nM, N_lig = {N_ligands_A}) ===")
print(f"  K_bind   = {float(K_bind_A):.4e}")
print(f"  Bound fraction = {float(bf_A):.6f}")

# ===========================================================
# System 2: Ligand B only
# ===========================================================
rec_B = {"name": "shared_rec", "sigma_R": sigma_R_test}
dp_B = {
    "short": {"N": NmonoShort, "a": amono, "sigma": sigma_P2K,
              "name": "PEG2K", "akuhn": akuhn},
    "ligand_B": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_B,
                 "name": "ligB", "akuhn": akuhn,
                 "K_bind_0": KD_B**(-1),
                 "receptor": rec_B},
}

system_B = MultivalentBinding(
    kT=kT, R_NP=R_NP, data_polymers=dp_B,
    binding_model="exact", polymer_model="Flory-exact",
    A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
)

K_bind_B = system_B.calculate_binding_constant()
bf_B = system_B.calculate_bound_fraction(depletion=False)

print(f"\n=== Ligand B only (KD = {KD_B/nM:.0f} nM, N_lig = {N_ligands_B}) ===")
print(f"  K_bind   = {float(K_bind_B):.4e}")
print(f"  Bound fraction = {float(bf_B):.6f}")

# ===========================================================
# System 3: Both ligands A + B (shared receptor)
# Both ligands reference the SAME receptor dict object.
# ===========================================================
rec_shared = {"name": "shared_rec", "sigma_R": sigma_R_test}
dp_AB = {
    "short": {"N": NmonoShort, "a": amono, "sigma": sigma_P2K,
              "name": "PEG2K", "akuhn": akuhn},
    "ligand_A": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_A,
                 "name": "ligA", "akuhn": akuhn,
                 "K_bind_0": KD_A**(-1),
                 "receptor": rec_shared},
    "ligand_B": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_B,
                 "name": "ligB", "akuhn": akuhn,
                 "K_bind_0": KD_B**(-1),
                 "receptor": rec_shared},   # same dict object!
}

system_AB = MultivalentBinding(
    kT=kT, R_NP=R_NP, data_polymers=dp_AB,
    binding_model="exact", polymer_model="Flory-exact",
    A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
)

K_bind_AB = system_AB.calculate_binding_constant()
bf_AB = system_AB.calculate_bound_fraction(depletion=False)

print(f"\n=== Both ligands A + B (shared receptor) ===")
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
print(f"  Bound frac(A+B) >= Bound frac(A): {float(bf_AB) >= float(bf_A)}")
print(f"  Bound frac(A+B) >= Bound frac(B): {float(bf_AB) >= float(bf_B)}")

# ===========================================================
# Test with fluctuations
# ===========================================================
print(f"\n=== With Poisson fluctuations (shared receptor) ===")
max_NR_ave = int(system_AB.NP_excluded_area * sigma_R_test)
max_n_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1
K_bind_vs_NR = system_AB.calculate_K_bind_vs_receptors(max_n_receptor)
M_conc = (A_cell / system_AB.NP_excluded_area) * cell_conc

bf_fluct = system_AB.calculate_bound_fraction(
    fluctuations=True, depletion=True,
    K_bind_vs_receptors=K_bind_vs_NR,
    rho_m=M_conc,
    max_n_receptor=max_n_receptor,
)
print(f"  Bound fraction (fluctuations+depletion) = {float(bf_fluct):.6f}")

# ===========================================================
# Test validation: should raise ValueError
# Two ligands with same receptor name but DIFFERENT dict objects
# ===========================================================
print(f"\n=== Validation test ===")
dp_bad = {
    "ligand_A": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_A,
                 "name": "ligA", "akuhn": akuhn,
                 "K_bind_0": KD_A**(-1),
                 "receptor": {"name": "shared", "sigma_R": 100 / um2}},
    "ligand_B": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_B,
                 "name": "ligB", "akuhn": akuhn,
                 "K_bind_0": KD_B**(-1),
                 "receptor": {"name": "shared", "sigma_R": 200 / um2}},
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
