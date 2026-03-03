from adsorption import *
from units import *
from system_variables import *
from mpmath import mp
import matplotlib.pyplot as plt

precision = mp.dps = 50
assert precision >= 30, AssertionError("You need high precision to avoid numerical problem, set to 50")

verbose = False
n_sampling_points = 50

sigma_R_min = 1.0 / um2
sigma_R_max = 1e4 / um2

min_exp = np.log10(sigma_R_min)
max_exp = np.log10(sigma_R_max)
sigma_R_values = np.logspace(min_exp, max_exp, n_sampling_points)

# KD factors: 10^-3, 10^-2, 10^-1, 1, 10, 100, 1000
# Base KD from system_variables_L3.py: 10000.0 * nM
KD_factors = [10**k for k in range(-12, -8)]

NP_area_val = mp.pi * R_NP**2
M_conc = (A_cell / NP_area_val) * cell_conc

max_NR_ave = int(mp.pi * R_NP**2 * sigma_R_max)
max_N_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1

results = {}
# All cases to run: KD factors + the K_bind_0 = inf limit
all_cases = [(kd_factor, f"KD = {float(KD * kd_factor / nM):g} nM") for kd_factor in KD_factors]
all_cases.append(("limit", "limit"))

# Create one system (reused for all cases since only K_bind_0 changes)
system_i = MultivalentBinding(kT=kT, R_NP=R_NP,
                              data_polymers=data_polymers,
                              binding_model="exact",
                              polymer_model="gaussian",
                              A_cell=A_cell,
                              NP_conc=NP_conc,
                              cell_conc=cell_conc)

for case_key, label in all_cases:
    if case_key == "limit":
        K_bind_0_i = np.inf
    else:
        KD_i = KD * case_key
        K_bind_0_i = KD_i**(-1)

    print(f"\n=== {label} ===")

    # Compute bound_vs_receptor (includes the expensive K_bind calculation)
    print(f"  Computing bound_vs_receptor for NR = 1..{max_N_receptor - 1}")
    bound_vs_receptor = system_i.calculate_bound_vs_receptors(
        K_bind_0_i, max_N_receptor, depletion=True, verbose=False)

    # Compute K_bind per sigma_R for the output columns (skip for limit case)
    cached_K_bind = np.full(n_sampling_points, np.inf)
    if case_key != "limit":
        for i, sigma_R in enumerate(sigma_R_values):
            cached_K_bind[i] = system_i.calculate_binding_constant(
                K_bind_0_i, sigma_R, verbose=verbose)
        print(f"  K_bind per sigma_R done.")

    # Compute bound fraction with fluctuations for each sigma_R
    sigma_out = np.zeros(n_sampling_points)
    KD_eff_out = np.zeros(n_sampling_points)
    frac_out = np.zeros(n_sampling_points)
    nads_out = np.zeros(n_sampling_points)

    for i, sigma_R in enumerate(sigma_R_values):
        bound_fraction = system_i.calculate_bound_fraction_with_fluctuations(
            K_bind_0_i, sigma_R, bound_vs_receptor, verbose=False)
        sigma_out[i] = float(sigma_R / (1 / um2))
        if cached_K_bind[i] != np.inf:
            KD_eff_out[i] = float((1 / cached_K_bind[i]) / M)
        else:
            KD_eff_out[i] = 0.0
        frac_out[i] = float(bound_fraction)
        nads_out[i] = float(bound_fraction) * NP_conc

    results[case_key] = (sigma_out, KD_eff_out, frac_out, nads_out)

    # Write data file
    if case_key == "limit":
        fname = "adsorption_KD_limit.dat"
    else:
        fname = f"adsorption_KD_x{case_key:g}.dat"
    with open(fname, 'w') as f:
        for j in range(n_sampling_points):
            f.write(f"{sigma_out[j]:5.3e} {KD_eff_out[j]:5.3e} {frac_out[j]:5.3e} {nads_out[j]:5.3e}\n")
    print(f"  Written to {fname}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for case_key, label in all_cases:
    sigma_out, _, frac_out, nads_out = results[case_key]
    if case_key == "limit":
        ax1.plot(sigma_out, frac_out, linestyle='dashed', color='black', label="limit")
        ax2.plot(sigma_out, nads_out, linestyle='dashed', color='black', label="limit")
    else:
        KD_nM = float(KD * case_key / nM)
        ax1.plot(sigma_out, frac_out, linestyle='solid', label=f"KD = {KD_nM:g} nM")
        ax2.plot(sigma_out, nads_out, linestyle='solid', label=f"KD = {KD_nM:g} nM")

ax1.set_xscale('linear')
ax1.set_xlabel(r'Receptor surface density ($\mu$m$^{-2}$)')
ax1.set_ylabel('Adsorbed fraction')
ax1.legend(fontsize='small')
ax1.set_xlim(0, 2000)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Receptor surface density ($\mu$m$^{-2}$)')
ax2.set_ylabel('Number of adsorbed particles (nm$^{-3}$)')
ax2.legend(fontsize='small')

plt.tight_layout()
plt.savefig('adsorption_scan_KD.png')
plt.close()
print("\nPlot saved to adsorption_scan_KD.png")
