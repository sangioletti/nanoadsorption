from adsorption import *
from units import *
from system_variables_L3 import *
from mpmath import mp
import matplotlib.pyplot as plt
import copy

precision = mp.dps = 50
assert precision >= 30, AssertionError("You need high precision to avoid numerical problem, set to 50")

verbose = False
n_sampling_points = 50

sigma_R_min = 1.0 / um2
sigma_R_max = 1e4 / um2

min_exp = np.log10(sigma_R_min)
max_exp = np.log10(sigma_R_max)
sigma_R_values = np.logspace(min_exp, max_exp, n_sampling_points)

# N_ligands factors: 1/3, 1/2, 1, 3/2, 2, 3
# Base N_ligands from system_variables_L3.py: 150
Nlig_factors = [1/3, 1/2, 1, 3/2, 2, 3]

max_NR_ave = int(mp.pi * R_NP**2 * sigma_R_max)
max_N_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1

results = {}

for nlig_factor in Nlig_factors:
    N_ligands_i = N_ligands * nlig_factor
    sigma_L_i = N_ligands_i / (4.0 * np.pi * R_NP**2)

    # Rebuild data_polymers with the new sigma_L
    data_polymers_i = copy.deepcopy(data_polymers)
    data_polymers_i['ligands']['sigma'] = sigma_L_i

    label = f"N_ligands = {N_ligands_i:g}"
    print(f"\n=== {label} (factor x{nlig_factor:g}) ===")

    # Create system with updated ligand density
    system_i = MultivalentBinding(kT=kT, R_NP=R_NP,
                                  data_polymers=data_polymers_i,
                                  binding_model="exact",
                                  polymer_model="gaussian",
                                  A_cell=A_cell,
                                  NP_conc=NP_conc,
                                  cell_conc=cell_conc)

    # Compute bound_vs_receptor (includes the expensive K_bind calculation)
    print(f"  Computing bound_vs_receptor for NR = 1..{max_N_receptor - 1}")
    bound_vs_receptor = system_i.calculate_bound_vs_receptors(
        K_bind_0, max_N_receptor, depletion=True, verbose=False)

    # Also compute K_bind per sigma_R for the output columns
    cached_K_bind = np.zeros(n_sampling_points)
    for i, sigma_R in enumerate(sigma_R_values):
        cached_K_bind[i] = system_i.calculate_binding_constant(
            K_bind_0, sigma_R, verbose=verbose)
    print(f"  K_bind per sigma_R done.")

    # Compute bound fraction with fluctuations for each sigma_R
    sigma_out = np.zeros(n_sampling_points)
    KD_eff_out = np.zeros(n_sampling_points)
    frac_out = np.zeros(n_sampling_points)
    nads_out = np.zeros(n_sampling_points)

    for i, sigma_R in enumerate(sigma_R_values):
        bound_fraction = system_i.calculate_bound_fraction_with_fluctuations(
            K_bind_0, sigma_R, bound_vs_receptor, verbose=False)
        sigma_out[i] = float(sigma_R / (1 / um2))
        KD_eff_out[i] = float((1 / cached_K_bind[i]) / M)
        frac_out[i] = float(bound_fraction)
        nads_out[i] = float(bound_fraction) * NP_conc

    results[nlig_factor] = (sigma_out, KD_eff_out, frac_out, nads_out)

    # Write per-factor data file
    fname = f"adsorption_Nligands_x{nlig_factor:g}.dat"
    with open(fname, 'w') as f:
        for j in range(n_sampling_points):
            f.write(f"{sigma_out[j]:5.3e} {KD_eff_out[j]:5.3e} {frac_out[j]:5.3e} {nads_out[j]:5.3e}\n")
    print(f"  Written to {fname}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for nlig_factor in Nlig_factors:
    N_ligands_i = N_ligands * nlig_factor
    sigma_out, _, frac_out, nads_out = results[nlig_factor]
    ax1.plot(sigma_out, frac_out, linestyle='solid', label=f"N_lig = {N_ligands_i:g}")
    ax2.plot(sigma_out, nads_out, linestyle='solid', label=f"N_lig = {N_ligands_i:g}")

ax1.set_xscale('log')
ax1.set_xlabel(r'Receptor surface density ($\mu$m$^{-2}$)')
ax1.set_ylabel('Adsorbed fraction')
ax1.legend()

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Receptor surface density ($\mu$m$^{-2}$)')
ax2.set_ylabel('Number of adsorbed particles (nm$^{-3}$)')
ax2.legend()

plt.tight_layout()
plt.savefig('adsorption_scan_Nligands.png')
plt.close()
print("\nPlot saved to adsorption_scan_Nligands.png")
