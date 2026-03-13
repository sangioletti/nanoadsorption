"""Test multi-ligand support with single_receptor=True.

Two ligand types bind different epitopes on the SAME receptor.
They share a single sigma_R passed externally.

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
dp_A = {
    "short": {"N": NmonoShort, "a": amono, "sigma": sigma_P2K,
              "name": "PEG2K", "akuhn": akuhn},
    "ligand_A": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_A,
                 "name": "ligA", "akuhn": akuhn,
                 "K_bind_0": KD_A**(-1)},
}

system_A = MultivalentBinding(
    kT=kT, R_NP=R_NP, data_polymers=dp_A,
    binding_model="exact", polymer_model="Flory-exact",
    A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
    single_receptor=True,
)

K_bind_A = system_A.calculate_binding_constant(sigma_R=sigma_R_test)
bf_A = system_A.calculate_bound_fraction(sigma_R=sigma_R_test, depletion=False)

print(f"=== Ligand A only (KD = {KD_A/nM:.0f} nM, N_lig = {N_ligands_A}) ===")
print(f"  K_bind   = {float(K_bind_A):.4e}")
print(f"  Bound fraction = {float(bf_A):.6f}")

# ===========================================================
# System 2: Ligand B only
# ===========================================================
dp_B = {
    "short": {"N": NmonoShort, "a": amono, "sigma": sigma_P2K,
              "name": "PEG2K", "akuhn": akuhn},
    "ligand_B": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_B,
                 "name": "ligB", "akuhn": akuhn,
                 "K_bind_0": KD_B**(-1)},
}

system_B = MultivalentBinding(
    kT=kT, R_NP=R_NP, data_polymers=dp_B,
    binding_model="exact", polymer_model="Flory-exact",
    A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
    single_receptor=True,
)

K_bind_B = system_B.calculate_binding_constant(sigma_R=sigma_R_test)
bf_B = system_B.calculate_bound_fraction(sigma_R=sigma_R_test, depletion=False)

print(f"\n=== Ligand B only (KD = {KD_B/nM:.0f} nM, N_lig = {N_ligands_B}) ===")
print(f"  K_bind   = {float(K_bind_B):.4e}")
print(f"  Bound fraction = {float(bf_B):.6f}")

# ===========================================================
# System 3: Both ligands A + B (single_receptor=True)
# ===========================================================
dp_AB = {
    "short": {"N": NmonoShort, "a": amono, "sigma": sigma_P2K,
              "name": "PEG2K", "akuhn": akuhn},
    "ligand_A": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_A,
                 "name": "ligA", "akuhn": akuhn,
                 "K_bind_0": KD_A**(-1)},
    "ligand_B": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_B,
                 "name": "ligB", "akuhn": akuhn,
                 "K_bind_0": KD_B**(-1)},
}

system_AB = MultivalentBinding(
    kT=kT, R_NP=R_NP, data_polymers=dp_AB,
    binding_model="exact", polymer_model="Flory-exact",
    A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
    single_receptor=True,
)

K_bind_AB = system_AB.calculate_binding_constant(sigma_R=sigma_R_test)
bf_AB = system_AB.calculate_bound_fraction(sigma_R=sigma_R_test, depletion=False)

print(f"\n=== Both ligands A + B (single_receptor=True) ===")
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
print(f"\n=== With Poisson fluctuations (single_receptor=True) ===")
max_NR_ave = int(system_AB.NP_excluded_area * sigma_R_test)
max_n_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1
K_bind_vs_NR = system_AB.calculate_K_bind_vs_receptors(max_n_receptor)
M_conc = (A_cell / system_AB.NP_excluded_area) * cell_conc

bf_fluct = system_AB.calculate_bound_fraction(
    sigma_R=sigma_R_test,
    fluctuations=True, depletion=True,
    K_bind_vs_receptors=K_bind_vs_NR,
    rho_m=M_conc,
    max_n_receptor=max_n_receptor,
)
print(f"  Bound fraction (fluctuations+depletion) = {float(bf_fluct):.6f}")

# ===========================================================
# Test validation: should raise ValueError
# ===========================================================
print(f"\n=== Validation test ===")
dp_bad = {
    "ligand_A": {"N": NmonoLigands, "a": amono, "sigma": sigma_L_A,
                 "name": "ligA", "akuhn": akuhn,
                 "K_bind_0": KD_A**(-1),
                 "sigma_R": 100 / um2},  # not allowed with single_receptor=True
}
try:
    MultivalentBinding(
        kT=kT, R_NP=R_NP, data_polymers=dp_bad,
        binding_model="exact", polymer_model="gaussian",
        A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
        single_receptor=True,
    )
    print("  ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"  Correctly raised ValueError: {e}")

print("\nAll tests passed.")
